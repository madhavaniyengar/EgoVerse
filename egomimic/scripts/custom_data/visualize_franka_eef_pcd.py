"""Visualize a camera-frame Franka EEF pose in its synchronized RGB-D point cloud."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import open3d as o3d
import zarr
from scipy.spatial.transform import Rotation


def read_frame(path: Path, index: int, *, rgb: bool) -> np.ndarray:
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24") if rgb else frame.to_ndarray().squeeze()
    raise IndexError(f"frame {index} does not exist in {path}")


def load_intrinsics(path: Path, camera: str) -> np.ndarray:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        payload = payload[camera]["intrinsic"]
        matrix = np.asarray(payload, dtype=np.float64)
    else:
        matrix = np.loadtxt(path)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    return matrix


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"expected xyz+wxyz pose, got {pose.shape}")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def marker(pose: np.ndarray, size: float, color: tuple[float, float, float]):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axes.transform(pose_matrix(pose))
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.09, resolution=16)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(pose[:3])
    return axes, sphere


def find_zarr_episode(root: Path, episode: int) -> Path:
    legacy = root / f"episode_{episode:06d}.zarr"
    if legacy.exists():
        return legacy
    matches = []
    for path in root.glob("*.zarr"):
        try:
            group = zarr.open_group(str(path), mode="r")
            if int(group.attrs.get("source_episode_index", -1)) == episode:
                matches.append(path)
        except (KeyError, TypeError, ValueError):
            continue
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No Zarr with source_episode_index={episode} found under {root}"
        )
    raise ValueError(f"Multiple Zarr episodes match source episode {episode}: {matches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0, help="15 Hz EgoVerse frame")
    parser.add_argument("--camera", default="cam1", help="camera frame used by the EEF poses")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("/home/madhavan/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json"),
    )
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=1.6)
    parser.add_argument("--axis-size", type=float, default=0.12)
    parser.add_argument("--show-command", action="store_true")
    args = parser.parse_args()

    episode_name = f"episode_{args.episode:06d}"
    episode_path = find_zarr_episode(args.zarr_root, args.episode)
    raw_episode = args.raw_root / episode_name
    rgb_path = raw_episode / f"{args.camera}.mp4"
    depth_path = raw_episode / f"{args.camera}_depth.mkv"
    for path in (episode_path, rgb_path, depth_path, args.intrinsics):
        if not path.exists():
            raise FileNotFoundError(path)

    group = zarr.open_group(str(episode_path), mode="r")
    total = int(group.attrs["total_frames"])
    if not 0 <= args.frame < total:
        raise IndexError(f"frame must be in [0, {total - 1}]")
    stride = int(group.attrs.get("temporal_stride", 1))
    source_frame = args.frame * stride
    rgb = read_frame(rgb_path, source_frame, rgb=True)
    depth = read_frame(depth_path, source_frame, rgb=False).astype(np.float32) / 1000.0
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape} vs {depth.shape}")

    K = load_intrinsics(args.intrinsics, args.camera)
    step = args.point_stride
    z = depth[::step, ::step]
    colors = rgb[::step, ::step]
    vv, uu = np.mgrid[0 : depth.shape[0] : step, 0 : depth.shape[1] : step]
    valid = (z >= args.min_depth) & (z <= args.max_depth)
    xyz = np.column_stack(
        ((uu[valid] - K[0, 2]) * z[valid] / K[0, 0],
         (vv[valid] - K[1, 2]) * z[valid] / K[1, 1], z[valid])
    )
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(xyz)
    scene.colors = o3d.utility.Vector3dVector(colors[valid].astype(np.float64) / 255.0)

    obs = np.asarray(group["right.obs_ee_pose"][args.frame], dtype=np.float64)
    geometries = [scene, *marker(obs, args.axis_size, (1.0, 0.05, 0.05))]
    if args.show_command:
        cmd = np.asarray(group["right.cmd_ee_pose"][args.frame], dtype=np.float64)
        geometries.extend(marker(cmd, args.axis_size * 0.8, (0.1, 0.2, 1.0)))

    u = K[0, 0] * obs[0] / obs[2] + K[0, 2]
    v = K[1, 1] * obs[1] / obs[2] + K[1, 2]
    print(f"Episode {args.episode}, EgoVerse frame {args.frame}, raw RGB-D frame {source_frame}")
    print(f"EEF camera XYZ: {obs[:3].tolist()}; projects to pixel ({u:.1f}, {v:.1f})")
    print(f"Scene points: {len(xyz):,}; pose_frame={group.attrs.get('pose_frame')}")
    print("EEF axes: X=red, Y=green, Z=blue; red sphere=current EEF")
    if args.show_command:
        print("Blue sphere=next-observation command EEF")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Franka EEF in {args.camera} PCD | episode {args.episode} frame {args.frame}",
        width=1280,
        height=800,
    )


if __name__ == "__main__":
    main()
