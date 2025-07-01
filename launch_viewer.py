# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import json
import os
import time
from pathlib import Path
from typing import List, Tuple
import time, types, imageio.v2 as imageio

import einops
import hydra

import numpy as np
import torch
import viser
import viser.transforms as vtf
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from scene.cameras import Camera, create_camera_from_JSON, focal2fov, fov2focal
from utils.image_utils import linear_to_sRGB
from utils.io import load_from_model_path

from utils.render_utils import apply_turbo_colormap, depth_to_normal
from viewer import ViewerCustomized
from scipy.spatial.transform import Rotation, Slerp


def scan_avail_iters(model_path: Path, ply_file: str = "point_cloud.ply"):
    # Scan and get all the checkpoints
    all_paths = glob.glob(str(model_path / "point_cloud" / "iteration_*" / ply_file))
    all_paths = natsorted(all_paths)
    iter_names = [str(p)[len(str(model_path)) + 1:] for p in all_paths]
    return iter_names


def read_cameras_json(cameras_json_path: str):
    print("load {}".format(cameras_json_path))
    with open(cameras_json_path, "r") as f:
        camera_jsons = json.load(f)

    if "train" in camera_jsons.keys():
        cameras = [create_camera_from_JSON(c) for c in camera_jsons["train"]]
    else:
        cameras = [create_camera_from_JSON(c) for c in camera_jsons]
    return cameras


class RenderViewer(ViewerCustomized):
    up_direction = np.asarray([0.0, 0.0, 1.0])

    ply_names: List[str] = None
    real_cameras: List[Camera] = None
    saved_camera_poses: List[Tuple[np.ndarray, float]] = None

    radiance_weight: float = 1.0

    lock_aspect_one = False
    apply_tonemapping = False
    render_depth = False

    show_edit_panel = True
    render_gravity_aligned = False

    camera_modality = "rgb"

    def __init__(
            self,
            cfg: DictConfig,
            server: viser.ViserServer,
    ):
        """
        cfg: the configuration file
        """
        super().__init__(server, self.viewer_render_fn, mode="rendering")

        self.device = torch.device("cuda")
        self.cfg = cfg
        self.model_root = cfg.model_root
        self.sh_degree = cfg.model.sh_degree
        self.enable_transform = cfg.enable_transform
        self.show_cameras = cfg.show_cameras
        self.reorient = cfg.reorient
        self._color_format = cfg.color_format

        # Initialize list to store saved camera poses
        self.saved_camera_poses = []

        self.visualize_raw = cfg.load_raw_ply
        if self.visualize_raw:
            print("Visualize PLY in customized raw configuration")
            self.point_cloud_file = "gaussians_all.ply"
        else:
            print("Visualize PLY file in standard GS configuration.")
            self.point_cloud_file = "point_cloud.ply"

        self.init_from_root(model_root=cfg.model_root)

        if cfg.model.white_background:
            self.background_color = (1.0, 1.0, 1.0)
        else:
            self.background_color = (0.0, 0.0, 0.0)

        self.camera_transform = torch.eye(4, dtype=torch.float)

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self.generate_guis()

    def init_from_root(self, model_root):
        model_paths = glob.glob(os.path.join(model_root, "*/"))
        model_paths = sorted(model_paths)
        model_paths = [Path(x) for x in model_paths]

        # Filter out the invalid model path by validating the existence of point_cloud folder
        valid_model_paths = []
        model_names = []

        print("Load pretrained models in the render")
        model_initialized = False
        for p in model_paths:

            print(
                f"search for all model checkpoints under {p} that contain {self.point_cloud_file}"
            )
            ply_names = scan_avail_iters(p, self.point_cloud_file)

            if len(ply_names) < 1:
                print(f"Did not found target ply file within {p}. Skip!")
                continue
            else:
                print(f"There are {len(ply_names)} target ply files found.")

            if not model_initialized:
                self.init_from_model_path(p, sub_path=ply_names[-1])
                self.ply_names = ply_names
                model_initialized = True

            valid_model_paths.append(p)
            model_names.append(str(p)[len(self.model_root):])

        if len(valid_model_paths) < 1:
            raise RuntimeError(
                f"Does not found valid model names to initialized from {model_root}"
            )

        self.model_paths = valid_model_paths
        self.model_names = model_names

        print("Found {} models".format(len(self.model_paths)))

    def init_from_model_path(self, model_path: Path, sub_path: str = None):

        cfg = copy.copy(self.cfg)
        if (model_path / "train_config.yaml").exists():
            train_cfg = OmegaConf.load(str(f"{model_path}/train_config.yaml"))
            # overwrite certain variables whether the train model needs a different rasterizer
            cfg.train_model = train_cfg.train_model

        # load bilateral filter if they exists and being used
        if (model_path / "bilateral_grid.pth").exists():
            bilateral_grid_path = str(model_path / "bilateral_grid.pth")
        else:
            bilateral_grid_path = None

        # need to read the data from camera json and set the train cameras
        self.viewer_renderer = load_from_model_path(
            cfg,
            model_path,
            ply_subpath=sub_path,
            bilateral_grid_path=bilateral_grid_path,
        )
        self.viewer_renderer = self.viewer_renderer.to(self.device)

        # reorient the scene
        cameras_json_path = model_path / "cameras.json"

        self.real_cameras = read_cameras_json(cameras_json_path)

        # reset the radiance weight to the training camera radiance weight
        self.radiance_weight = self.real_cameras[0].radiance_weight

        # todo: read all cameras and support different modalities in the viewer
        self.camera_modality = self.real_cameras[0].camera_modality

    def get_current_model_path(self) -> Path:
        return Path(self.model_root + self.model_paths_dropdown.value)

    def viewer_render_fn(self, camera_state, img_wh: Tuple[int, int]):
        """
        The render function
        """

        # c2w = self.camera_transform @ camera_state.c2w
        c2w = camera_state.c2w
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)

        fov_x = camera_state.fov
        fov_y = camera_state.fov
        aspect_ratio = self.render_aspect_ratio

        # the camera width & height is not right
        image_width, image_height = img_wh
        render_image = np.ones((image_height, image_width, 3))

        if aspect_ratio != None and image_width != int(image_height * aspect_ratio):
            render_foreground = True
            render_width = int(image_height * aspect_ratio)
            render_height = image_height
            # update fov
            render_fx = fov2focal(fov_x, image_width)
            fov_x = focal2fov(render_fx, render_width)
        else:
            render_foreground = False
            render_width = image_width
            render_height = image_height

        # in case view renderer has not been init
        if not hasattr(self, "viewer_renderer"):
            return render_image

        camera = Camera(
            uid=0,
            w2c=w2c,
            FoVx=fov_x,
            FoVy=fov_y,
            image_width=render_width,
            image_height=render_height,
            image_name="",
            image_path="",
            mask_path="",
            camera_name="render",
            camera_modality=self.radiance_channel,
            camera_projection_model=self.render_camera_model,
            exposure_duration_s=self.exposure,
            gain=self.gain,
            scene_name="",
        )

        camera.radiance_weight = self.radiance_weight

        with torch.no_grad():
            render_pkg = self.viewer_renderer.render(
                camera, scaling_modifier=self.scaling_modifier
            )

            image = camera.expose_image(render_pkg["render"])

            if camera.is_rgb:
                image = self.viewer_renderer.radiance_finishing(image)[0]
            else:
                image = image[0]

            image = einops.rearrange(image, "c h w -> h w c")

            if not camera.is_rgb:
                image = image.repeat(1, 1, 3)

            depth = render_pkg["depth"][0]

            if self.render_modality == "color":
                if self.apply_tonemapping:
                    image_vis = linear_to_sRGB(image, self.gamma)

                    image_vis = torch.clamp(image_vis, max=1.0)
                else:
                    image_vis = image
            elif self.render_modality == "depth":
                min_d, max_d = self.min_depth_range.value, self.max_depth_range.value
                depth_normalized = (depth.clamp(min_d, max_d) - min_d) / (max_d - min_d)
                image_vis = apply_turbo_colormap(depth_normalized[0])
            elif self.render_modality == "normal":
                if "normals" in render_pkg:
                    normal = render_pkg["normals"][0]
                else:
                    # estimate normal from depth
                    normal = depth_to_normal(camera, depth[0])
                image_vis = (normal + 1) / 2
            elif self.render_modality == "normal_from_depth":
                if "normals_from_depth" in render_pkg:
                    normal = render_pkg["normals_from_depth"][0]
                else:
                    # estimate normal from depth
                    normal = depth_to_normal(camera, depth[0])

                image_vis = (normal + 1) / 2
                # normal = einops.rearrange(normal, "c h w -> h w c")

        if render_foreground:
            pad = (image_width - render_width) // 2
            render_image[:, pad: pad + render_width] = image_vis.cpu().numpy()
        else:
            render_image = image_vis.cpu().numpy()

        return render_image

    def add_camera_trajectory_folder(self):
        with self.server.gui.add_folder("Camera Trajectory"):
            self.saved_camera_count_text = self.server.gui.add_text(
                label="Saved Poses", initial_value="0", disabled=True
            )
            save_camera_button = self.server.gui.add_button(
                "Save Camera Pose",
                icon=viser.Icon.PLUS,
            )
            play_button = self.server.gui.add_button(
                "Play Trajectory",
                icon=viser.Icon.PLAYER_PLAY,
            )
            clear_button = self.server.gui.add_button(
                "Clear Trajectory",
                icon=viser.Icon.TRASH,
            )
            self.trajectory_duration_slider = self.server.gui.add_slider(
                "Traj. Duration (sec)", min=1, max=30, step=1, initial_value=5
            )

        @save_camera_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is None:
                return

            R = vtf.SO3.from_quaternion_xyzw(
                np.array(
                    [
                        client.camera.wxyz[1],
                        client.camera.wxyz[2],
                        client.camera.wxyz[3],
                        client.camera.wxyz[0],
                    ]
                )
            ).as_matrix()

            t = client.camera.position

            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t

            self.saved_camera_poses.append((c2w, client.camera.fov))
            self.saved_camera_count_text.value = str(len(self.saved_camera_poses))

        @clear_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.saved_camera_poses.clear()
            self.saved_camera_count_text.value = str(len(self.saved_camera_poses))

        @play_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is None or len(self.saved_camera_poses) < 2:
                return

            play_button.disabled = True
            clear_button.disabled = True
            save_camera_button.disabled = True

            try:
                for i in range(len(self.saved_camera_poses) - 1):
                    start_c2w, start_fov = self.saved_camera_poses[i]
                    end_c2w, end_fov = self.saved_camera_poses[i + 1]

                    start_R = start_c2w[:3, :3]
                    end_R = end_c2w[:3, :3]
                    start_t = start_c2w[:3, 3]
                    end_t = end_c2w[:3, 3]

                    key_rotations = Rotation.from_matrix([start_R, end_R])
                    slerp = Slerp([0, 1], key_rotations)

                    num_steps = int(self.trajectory_duration_slider.value * 60 / (len(self.saved_camera_poses) - 1))
                    if num_steps == 0:
                        num_steps = 1

                    for j in range(num_steps + 1):
                        alpha = j / num_steps

                        # interpolate pose
                        interp_R = slerp([alpha]).as_matrix()[0]  # 3×3
                        interp_t = (1 - alpha) * start_t + alpha * end_t
                        interp_fov = (1 - alpha) * start_fov + alpha * end_fov

                        # Convert rotation matrix to quaternion
                        quat_xyzw = Rotation.from_matrix(interp_R).as_quat()  # (x, y, z, w)
                        wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])  # (w, x, y, z)

                        # Send all camera fields in one atomic update to avoid flicker
                        with client.atomic():
                            client.camera.wxyz = wxyz
                            client.camera.position = interp_t
                            client.camera.fov = interp_fov

                        client.flush()
                        time.sleep(1.0 / 60.0)

            finally:
                play_button.disabled = False
                clear_button.disabled = False
                save_camera_button.disabled = False

        record_button = self.server.gui.add_button(
            "Record MP4",
            icon=viser.Icon.VIDEO,
        )

        @record_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is None or len(self.saved_camera_poses) < 2:
                return

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"trajectory_{ts}.mp4"
            fps = 60
            if hasattr(client.camera, "image_width"):
                width = client.camera.image_width
                height = client.camera.image_height
            else:
                height = 720
                aspect = getattr(client.camera, "aspect", 16 / 9)
                width = int(height * aspect)

            writer = imageio.get_writer(
                out_path,
                fps=fps,
                codec="libx264",
                quality=8,
            )

            try:
                for i in range(len(self.saved_camera_poses) - 1):
                    start_c2w, start_fov = self.saved_camera_poses[i]
                    end_c2w, end_fov = self.saved_camera_poses[i + 1]

                    start_R, start_t = start_c2w[:3, :3], start_c2w[:3, 3]
                    end_R, end_t = end_c2w[:3, :3], end_c2w[:3, 3]
                    key_rots = Rotation.from_matrix([start_R, end_R])
                    slerp = Slerp([0, 1], key_rots)

                    num_steps = int(self.trajectory_duration_slider.value * fps /
                                    (len(self.saved_camera_poses) - 1))
                    num_steps = max(1, num_steps)

                    for j in range(num_steps + 1):
                        α = j / num_steps
                        R = slerp([α]).as_matrix()[0]
                        t = (1 - α) * start_t + α * end_t
                        fov = (1 - α) * start_fov + α * end_fov

                        quat_xyzw = Rotation.from_matrix(R).as_quat()
                        wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])

                        with client.atomic():
                            client.camera.wxyz = wxyz
                            client.camera.position = t
                            client.camera.fov = fov

                        stub = types.SimpleNamespace(c2w=np.linalg.inv(
                            np.column_stack((R, t)).astype(float)
                            .tolist() + [[0, 0, 0, 1]]
                        ), fov=fov)
                        frame = self.viewer_render_fn(stub, (width, height))
                        writer.append_data((frame * 255).astype(np.uint8))

                        time.sleep(1.0 / fps)

            finally:
                writer.close()
                print(f"Saved trajectory to {out_path}")

    def generate_guis(self):
        """
        Update the GUI design
        """

        tabs = self.server.gui.add_tab_group()

        with tabs.add_tab("General"):
            self.model_paths_dropdown = self.server.gui.add_dropdown(
                "Model Paths",
                tuple(self.model_names),
                initial_value=self.model_names[0],
            )

            if self.ply_names is not None:
                self.ply_drop_down = self.server.gui.add_dropdown(
                    "PLY files", tuple(self.ply_names), initial_value=self.ply_names[-1]
                )

                @self.ply_drop_down.on_update
                def _(_) -> None:
                    model_path = self.get_current_model_path()
                    load_iter = self.ply_drop_down.value
                    self.init_from_model_path(model_path, load_iter)
                    self.rerender(_)

            @self.model_paths_dropdown.on_update
            def _(_) -> None:
                model_path = self.get_current_model_path()

                self.server.on_client_disconnect(self._disconnect_client)
                self.server.on_client_connect(self._connect_client)

                if self.ply_names is not None:
                    with self.server.atomic():
                        # if self.model_source == "GS":
                        print(f"search for all model checkpoints under {model_path}")
                        self.ply_drop_down.options = scan_avail_iters(
                            model_path, self.point_cloud_file
                        )
                        print(f"Init model from {self.ply_drop_down.options[-1]}")
                        self.ply_drop_down.value = self.ply_drop_down.options[-1]
                self.init_from_model_path(model_path, self.ply_drop_down.value)
                self.add_frame_slider_folder(
                    self.real_cameras, slider_name="Observations"
                )
                self.rerender(_)

            self.add_camera_reset_button()

            self.add_aria_gravity_align_button()

            self.add_lock_aspect_ratio_button()

            # add cameras
            if self.show_cameras is True:
                self.add_cameras_to_scene(self.server)

            self.add_render_options_button()

            self.add_rgb_postprocessing_button(self.real_cameras)

            self.add_gaussian_model_option_button()

            if self.real_cameras is not None:
                self.add_frame_slider_folder(
                    self.real_cameras, slider_name="Observations"
                )

            self.add_camera_trajectory_folder()



@hydra.main(version_base=None, config_path="conf", config_name="viewer")
def main(cfg: DictConfig) -> None:
    # create viser server
    server = viser.ViserServer(host=cfg.host, port=cfg.port)

    render_viewer = RenderViewer(cfg, server=server)

    print("Viewer running... Ctrl+C to exit.")

    while True:
        time.sleep(999)


if __name__ == "__main__":
    main()