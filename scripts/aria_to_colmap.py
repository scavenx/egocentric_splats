import argparse
import json
import csv
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import gzip
from tqdm import tqdm
from scipy.spatial import KDTree

def convert_aria_to_colmap(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Starting conversion...")
    print(f"Reading data from {input_path}...")

    with open(input_path / 'transforms.json', 'r') as f:
        transforms_data = json.load(f)

    points3d_map = {}
    csv_path = input_path / 'semidense_points.csv'
    gz_path = input_path / 'semidense_points.csv.gz'

    if gz_path.exists():
        print("Reading compressed semidense_points..")
        with gzip.open(gz_path, 'rt') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                points3d_map[i + 1] = {
                    'X': float(row['px_world']),
                    'Y': float(row['py_world']),
                    'Z': float(row['pz_world']),
                    'R': 128, 'G': 128, 'B': 128, 'ERROR': 0.0, 'TRACK': []
                }
    elif csv_path.exists():
        print("Reading semidense_points...")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                points3d_map[i + 1] = {
                    'X': float(row['px_world']),
                    'Y': float(row['py_world']),
                    'Z': float(row['pz_world']),
                    'R': 128, 'G': 128, 'B': 128, 'ERROR': 0.0, 'TRACK': []
                }
    else:
        print("Error: semidense_points.csv or semidense_points.csv.gz not found.")
        return

    image_observations = {}
    frames = sorted(transforms_data['frames'], key=lambda x: x['timestamp'])

    for image_id, frame in enumerate(frames, 1):
        image_name = frame['image_path']
        depth_path = input_path / 'sparse_depth' / f"{Path(image_name).stem}.json"

        if not depth_path.exists():
            continue

        with open(depth_path, 'r') as f:
            sparse_depth = json.load(f)

        observations = []
        for i in range(len(sparse_depth['u'])):
            observations.append({
                'u': sparse_depth['u'][i],
                'v': sparse_depth['v'][i],
                'z': sparse_depth['z'][i]
            })
        image_observations[image_id] = observations

    print("Generating cameras.txt...")
    cam_frame = transforms_data['frames'][0]
    camera_id = 1
    model = "PINHOLE"
    width, height = cam_frame['w'], cam_frame['h']
    fx, fy, cx, cy = cam_frame['fx'], cam_frame['fy'], cam_frame['cx'], cam_frame['cy']

    with open(output_path / 'cameras.txt', 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{camera_id} {model} {width} {height} {fx} {fy} {cx} {cy}\n")

    point3d_coords = np.array([[p['X'], p['Y'], p['Z']] for p in points3d_map.values()])
    point3d_ids = list(points3d_map.keys())

    print("Associating 2D observations with 3D points...")
    for image_id, frame_data in tqdm(enumerate(frames, 1), total=len(frames), desc="Associating 2D-3D"):
        # Invert the c2w matrix to get w2c for projection
        c2w = np.array(frame_data['transform_matrix'])
        w2c = np.linalg.inv(c2w)
        R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]

        K = np.array([
            [frame_data['fx'], 0, frame_data['cx']],
            [0, frame_data['fy'], frame_data['cy']],
            [0, 0, 1]
        ])

        # Project all 3D points into the current camera view
        projected_points = R_w2c @ point3d_coords.T + t_w2c[:, np.newaxis]

        # Filter points that are in front of the camera
        in_front_mask = projected_points[2, :] > 0

        # Perform perspective projection
        projected_2d = K @ projected_points[:, in_front_mask]
        projected_2d /= projected_2d[2, :]

        if image_id not in image_observations:
            continue

        # Get the indices of the 3D points that are in front of the camera
        valid_point_indices = np.where(in_front_mask)[0]
        # Get the 2D coordinates of the projected points
        valid_projected_2d = projected_2d[:2, :].T

        if len(valid_projected_2d) == 0:
            continue

        # Build k-d tree from the projected 3D points
        tree = KDTree(valid_projected_2d)

        # Get all 2D observations for the current image as a numpy array
        obs_2d = np.array([[obs['u'], obs['v']] for obs in image_observations[image_id]])

        # Find the closest projected point for ALL observations at once
        distances, closest_indices = tree.query(obs_2d, k=1)
        match_mask = distances < 2.0  # Filter matches based on tolerance

        # For each valid match, add the observation to the point's track
        for obs_idx, proj_idx in enumerate(closest_indices[match_mask]):
            # Get the original index of the observation that matched
            original_obs_idx = np.where(match_mask)[0][obs_idx]

            # Get the original index of the 3D point that was matched
            point3d_idx = valid_point_indices[proj_idx]
            point3d_id = point3d_ids[point3d_idx]

            # Add the observation to the point's track
            # The 'original_obs_idx' is the point2d_idx for this image
            points3d_map[point3d_id]['TRACK'].append((image_id, original_obs_idx))

    print("Generating images.txt and points3D.txt...")
    with open(output_path / 'images.txt', 'w') as f_images, \
         open(output_path / 'points3D.txt', 'w') as f_points:

        f_images.write("# Image list with two lines of data per image:\n")
        f_images.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_images.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f_points.write("# 3D point list with one line of data per point:\n")
        f_points.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        image_to_point3d_map = {}
        for point3d_id, point_data in points3d_map.items():
            for image_id, point2d_idx in point_data['TRACK']:
                if image_id not in image_to_point3d_map:
                    image_to_point3d_map[image_id] = {}
                image_to_point3d_map[image_id][point2d_idx] = point3d_id

        for image_id, frame in enumerate(frames, 1):
            c2w = np.array(frame['transform_matrix'])
            w2c = np.linalg.inv(c2w)
            R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]
            q = Rotation.from_matrix(R_w2c).as_quat()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            tx, ty, tz = t_w2c
            image_name = frame['image_path']

            f_images.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n")

            points2d_str_list = []
            if image_id in image_observations:
                obs_map = image_to_point3d_map.get(image_id, {})
                for i, obs in enumerate(image_observations[image_id]):
                    point3d_id = obs_map.get(i, -1)
                    points2d_str_list.append(f"{obs['u']} {obs['v']} {point3d_id}")

            f_images.write(" ".join(points2d_str_list) + "\n")

        for point3d_id, data in points3d_map.items():
            if not data['TRACK']:
                continue
            x, y, z = data['X'], data['Y'], data['Z']
            r, g, b = data['R'], data['G'], data['B']
            error = data['ERROR']
            track_str = " ".join([f"{img_id} {p2d_idx}" for img_id, p2d_idx in data['TRACK']])
            f_points.write(f"{point3d_id} {x} {y} {z} {r} {g} {b} {error} {track_str}\n")

    print(f"Conversion complete! Output saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--input', type=str, required=True)
    parser.add_argument(
        '--output', type=str, required=True)
    args = parser.parse_args()
    convert_aria_to_colmap(args.input, args.output)