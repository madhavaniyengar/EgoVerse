"""Run the existing WiLoR RGB-D pipeline over every LeRobot dataset shard.

WiLoR's existing ``demo_lerobot_detectron2.py`` writes metric points in the
calibration world frame.  This wrapper converts them to the front-camera
optical frame required by static-camera human/Franka co-training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_DATASET = Path("/data/madhavan/pick_red_mug_human")
DEFAULT_WILOR_ROOT = Path("/home/madhavan/lerobot/WiLoR")
DEFAULT_WILOR_PYTHON = Path("/home/madhavan/miniconda3/envs/wilor/bin/python")
DEFAULT_CALIBRATION = Path(
    "/home/madhavan/lerobot/lerobot/scripts/franka_2cam_calibration/"
    "calibration_franka_2cam.json"
)
FRONT_CAMERA = "cam_azure_kinect_front"
CAMERAS = (FRONT_CAMERA, "cam_azure_kinect_left")


def _resolve_calibration_path(path: str, calibration_json: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for resolved in (
        calibration_json.parent / candidate,
        calibration_json.parent.parent / candidate,
    ):
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve calibration path {path!r}")


def load_front_from_world(calibration_json: Path) -> np.ndarray:
    calibration = json.loads(calibration_json.read_text())
    front_entry = calibration[FRONT_CAMERA]
    world_from_front = np.loadtxt(
        _resolve_calibration_path(front_entry["extrinsics"], calibration_json)
    )
    if world_from_front.shape != (4, 4):
        raise ValueError(
            f"Front extrinsics must be 4x4, got {world_from_front.shape}"
        )
    return np.linalg.inv(world_from_front)


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shaped (T,N,3), got {points.shape}")
    valid = np.any(points != 0, axis=(1, 2))
    transformed = np.zeros_like(points, dtype=np.float64)
    if valid.any():
        transformed[valid] = (
            target_from_source[:3, :3] @ points[valid].transpose(0, 2, 1)
        ).transpose(0, 2, 1) + target_from_source[:3, 3]
    return transformed.astype(points.dtype, copy=False)


def discover_shards(dataset_root: Path) -> list[Path]:
    if (dataset_root / "meta" / "info.json").is_file():
        return [dataset_root]
    shards = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").is_file()
    ]
    return sorted(
        shards,
        key=lambda path: (0, int(path.name))
        if path.name.isdigit()
        else (1, path.name),
    )


def run(args: argparse.Namespace) -> None:
    shards = discover_shards(args.dataset_root)
    if not shards:
        raise FileNotFoundError(f"No LeRobot shards found under {args.dataset_root}")
    front_from_world = load_front_from_world(args.calibration_json)
    summary: dict[str, dict[str, int | str]] = {}

    for shard in shards:
        shard_name = shard.name
        input_folder = shard / "videos" / "chunk-000"
        raw_output = args.output_root / "world" / shard_name
        camera_output = args.output_root / "front_camera" / shard_name
        raw_output.mkdir(parents=True, exist_ok=True)
        camera_output.mkdir(parents=True, exist_ok=True)

        command = [
            str(args.wilor_python),
            str(args.wilor_root / "demo_lerobot_detectron2.py"),
            "--input_folder",
            str(input_folder),
            "--output_folder",
            str(raw_output),
            "--calibration_json",
            str(args.calibration_json),
            "--cam_names",
            *CAMERAS,
        ]
        if args.no_gsam2:
            command.append("--no_gsam2")
        if args.visualize:
            visualize_dir = args.output_root / "visualizations" / shard_name
            command.extend(["--visualize", "--visualize_dir", str(visualize_dir)])
        if args.episode_start is not None:
            command.extend(["--episode_start", str(args.episode_start)])
        if args.episode_end is not None:
            command.extend(["--episode_end", str(args.episode_end)])

        print(f"Running WiLoR for shard {shard_name}: {input_folder}", flush=True)
        subprocess.run(command, cwd=args.wilor_root, check=True)

        keypoint_files = sorted(raw_output.glob("*.keypoints3d.npy"))
        for source in keypoint_files:
            points_world = np.load(source)
            points_front = transform_points(points_world, front_from_world)
            destination = camera_output / source.name
            np.save(destination, points_front)
            if not np.isfinite(points_front).all():
                raise ValueError(f"Non-finite front-frame keypoints in {destination}")
        summary[shard_name] = {
            "dataset_root": str(shard),
            "keypoint_count": len(keypoint_files),
            "output_dir": str(camera_output),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "cam_azure_kinect_front_optical",
                "units": "metres",
                "mano_order": "0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, "
                "13-16=ring, 17-20=pinky",
                "calibration_json": str(args.calibration_json),
                "shards": summary,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATASET / "wilor_static_camera",
    )
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--wilor-python", type=Path, default=DEFAULT_WILOR_PYTHON)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-gsam2", action="store_true")
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episode-end", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
