#!/usr/bin/env python3
"""Visualize EgoVerse human keypoints in synchronized raw RGB-D data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import open3d as o3d
import zarr


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def read_video_frame(path: Path, frame_index: int, rgb: bool) -> np.ndarray:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_ndarray(format="rgb24") if rgb else frame.to_ndarray().squeeze()
    raise IndexError(f"Frame {frame_index} does not exist in {path}")


def find_zarr_episode(root: Path, episode: int) -> Path:
    legacy = root / f"episode_{episode:06d}.zarr"
    if legacy.exists():
        return legacy
    matches = []
    token = f"episode_{episode:06d}"
    for path in root.glob("*.zarr"):
        try:
            group = zarr.open_group(str(path), mode="r")
            attrs = group.attrs
            source_index = attrs.get("source_episode_index")
            source_keypoints = str(attrs.get("source_keypoint_path", ""))
            if (source_index is not None and int(source_index) == episode) or token in source_keypoints:
                matches.append(path)
        except (KeyError, TypeError, ValueError):
            continue
    if not matches:
        # EgoVerse timestamp names use the episode index as the microsecond suffix.
        matches = list(root.glob(f"*-{episode:06d}.zarr"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No Zarr for source episode {episode} under {root}")
    raise ValueError(f"Multiple Zarr episodes match source episode {episode}: {matches}")


def load_intrinsics(path: Path, camera: str | None) -> np.ndarray:
    if path.suffix.lower() == ".json":
        if camera is None:
            raise ValueError("--calibration-camera is required for JSON calibration")
        payload = json.loads(path.read_text())
        if camera not in payload or "intrinsic" not in payload[camera]:
            raise KeyError(f"{path} does not contain {camera}.intrinsic")
        matrix = np.asarray(payload[camera]["intrinsic"], dtype=np.float64)
    else:
        matrix = np.loadtxt(path)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    return matrix


def raw_video_paths(root: Path, episode: int, camera: str) -> tuple[Path, Path]:
    episode_name = f"episode_{episode:06d}"
    episode_dir = root / episode_name
    if episode_dir.is_dir():
        return episode_dir / f"{camera}.mp4", episode_dir / f"{camera}_depth.mkv"
    video_root = root / "videos" / f"chunk-{episode // 1000:03d}"
    return (
        video_root / f"observation.images.{camera}.color" / f"{episode_name}.mp4",
        video_root / f"observation.images.{camera}.transformed_depth" / f"{episode_name}.mkv",
    )


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> o3d.geometry.TriangleMesh:
    direction = end - start
    length = float(np.linalg.norm(direction))
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=12)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 0.15, 0.05])
    z_axis = np.array([0.0, 0.0, 1.0])
    direction /= length
    cross = np.cross(z_axis, direction)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm > 1e-9:
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            cross / cross_norm * np.arccos(dot)
        )
        mesh.rotate(rotation, center=np.zeros(3))
    elif dot < 0:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=np.zeros(3))
    mesh.translate((start + end) / 2)
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0, help="EgoVerse Zarr frame")
    parser.add_argument("--camera", default="cam_azure_kinect_left")
    parser.add_argument("--calibration-camera", help="Camera key when calibration is JSON")
    parser.add_argument("--point-stride", type=int, default=2,
                        help="Scene point-cloud pixel stride (default: 2)")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1.6)
    parser.add_argument("--joint-radius", type=float, default=0.008)
    parser.add_argument("--bone-radius", type=float, default=0.003)
    parser.add_argument("--no-rgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    zarr_path = find_zarr_episode(args.zarr_root.expanduser().resolve(), args.episode)
    rgb_path, depth_path = raw_video_paths(raw_root, args.episode, args.camera)
    for path in (zarr_path, rgb_path, depth_path, args.calibration):
        if not path.exists():
            raise FileNotFoundError(path)

    group = zarr.open_group(str(zarr_path), mode="r")
    total = int(group.attrs["total_frames"])
    if not 0 <= args.frame < total:
        raise IndexError(f"Frame must be in [0, {total - 1}]")
    if "right.obs_keypoints" not in group:
        raise KeyError(f"{zarr_path} does not contain right.obs_keypoints")
    joints = np.asarray(group["right.obs_keypoints"][args.frame], dtype=np.float64).reshape(-1, 3)
    if joints.shape != (21, 3):
        raise ValueError(f"Expected keypoints with shape (21, 3), received {joints.shape}")

    source_frame = args.frame * int(group.attrs.get("temporal_stride", 1))

    depth_m = read_video_frame(depth_path, source_frame, rgb=False).astype(np.float32) / 1000.0
    rgb = read_video_frame(rgb_path, source_frame, rgb=True)
    K = load_intrinsics(args.calibration, args.calibration_camera)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    stride = args.point_stride
    z = depth_m[::stride, ::stride]
    colors = rgb[::stride, ::stride]
    vv, uu = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    valid = (z >= args.min_depth) & (z <= args.max_depth)
    xyz = np.column_stack(((uu[valid] - cx) * z[valid] / fx,
                           (vv[valid] - cy) * z[valid] / fy,
                           z[valid]))

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(xyz)
    if args.no_rgb:
        scene.paint_uniform_color([0.55, 0.55, 0.55])
    else:
        scene.colors = o3d.utility.Vector3dVector(colors[valid].astype(np.float64) / 255.0)

    geometries: list[o3d.geometry.Geometry] = [scene]
    for joint in joints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=args.joint_radius, resolution=12)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.translate(joint)
        geometries.append(sphere)
    for start, end in HAND_EDGES:
        if np.linalg.norm(joints[end] - joints[start]) > 1e-9:
            geometries.append(cylinder_between(joints[start], joints[end], args.bone_radius))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    geometries.append(axes)
    print(
        f"Episode {args.episode}, keypoint frame {args.frame}, "
        f"source RGB-D frame {source_frame}"
    )
    print(f"Scene points: {len(xyz):,}; hand joints: {len(joints)}")
    print("Coordinates: +X right, +Y down, +Z forward; units: metres")
    print("Red spheres/lines are WiLoR joints; RGB points are Kinect depth.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"RGB-D scene + WiLoR hand | episode {args.episode} frame {args.frame}",
        width=1280,
        height=800,
    )


if __name__ == "__main__":
    main()
