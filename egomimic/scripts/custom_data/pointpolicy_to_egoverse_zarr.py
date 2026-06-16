"""Convert PointPolicy human pickle data to EgoVerse Zarr.

Expected PointPolicy pickle layout:
    {
        "observations": [
            {
                "pixels1": (T, H, W, 3) uint8,
                "pixels2": (T, H, W, 3) uint8,
                "human_tracks_3d_pixels1": (T, K, 3),
                ...
            },
            ...
        ],
        ...
    }

The current HPT custom-human config trains on ``right.obs_ee_pose``. PointPolicy
does not provide a wrist orientation, so this converter derives xyz from the
tracked 3D point centroid and writes identity orientation ``[qw, qx, qy, qz]``.
The raw 3D points are also stored in ``right.obs_keypoints`` padded to 21x3.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

LOGGER = logging.getLogger(__name__)


def _episode_hash(base_time: datetime, index: int) -> str:
    return (base_time + timedelta(microseconds=index)).strftime(
        "%Y-%m-%d-%H-%M-%S-%f"
    )


def _base_time_from_file(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    )


def _finite_or_zero(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _tracks_to_keypoints(tracks_3d: np.ndarray, scale: float) -> np.ndarray:
    tracks = _finite_or_zero(np.asarray(tracks_3d, dtype=np.float64)) * scale
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"Expected tracks shape (T, K, 3), got {tracks.shape}")

    total_frames = tracks.shape[0]
    keypoints = np.zeros((total_frames, 21, 3), dtype=np.float64)
    n_copy = min(tracks.shape[1], 21)
    keypoints[:, :n_copy, :] = tracks[:, :n_copy, :]
    return keypoints.reshape(total_frames, 63)


def _tracks_to_pose(tracks_3d: np.ndarray, scale: float) -> np.ndarray:
    tracks = _finite_or_zero(np.asarray(tracks_3d, dtype=np.float64)) * scale
    xyz = np.mean(tracks, axis=1)
    quat_wxyz = np.zeros((tracks.shape[0], 4), dtype=np.float64)
    quat_wxyz[:, 0] = 1.0
    return np.concatenate([xyz, quat_wxyz], axis=-1)


def _front_images(obs: dict[str, Any], camera_key: str) -> np.ndarray:
    if camera_key not in obs:
        raise KeyError(f"Observation is missing image key {camera_key!r}")
    images = np.asarray(obs[camera_key])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected {camera_key} shape (T, H, W, 3), got {images.shape}")
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)
    return images


def _tracks(obs: dict[str, Any], track_key: str) -> np.ndarray:
    if track_key not in obs:
        raise KeyError(f"Observation is missing 3D track key {track_key!r}")
    return np.asarray(obs[track_key])


def convert_pointpolicy_pickle(
    input_path: Path,
    output_dir: Path,
    *,
    task_name: str,
    task_description: str,
    fps: int,
    camera_key: str,
    track_key: str,
    keypoint_scale: float,
    overwrite: bool,
) -> list[Path]:
    with input_path.open("rb") as f:
        payload = pickle.load(f)

    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list):
        raise ValueError("PointPolicy pickle must contain a list at key 'observations'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_time = _base_time_from_file(input_path)
    written: list[Path] = []

    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            raise ValueError(f"Observation {idx} must be a dict, got {type(obs)}")

        images = _front_images(obs, camera_key)
        tracks = _tracks(obs, track_key)
        pose = _tracks_to_pose(tracks, keypoint_scale)
        keypoints = _tracks_to_keypoints(tracks, keypoint_scale)

        episode_hash = _episode_hash(base_time, idx)
        episode_path = output_dir / f"{episode_hash}.zarr"
        if episode_path.exists() and not overwrite:
            raise FileExistsError(
                f"{episode_path} already exists. Pass --overwrite to replace it."
            )

        ZarrWriter.create_and_write(
            episode_path=episode_path,
            numeric_data={
                "right.obs_ee_pose": pose,
                "right.obs_keypoints": keypoints,
            },
            image_data={"images.front_1": images},
            embodiment="custom_human_right_arm",
            fps=fps,
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "pointpolicy_pkl",
                "source_path": str(input_path),
                "source_observation_index": idx,
                "source_camera_key": camera_key,
                "source_track_key": track_key,
                "source_track_points": int(np.asarray(tracks).shape[1]),
                "keypoint_scale": float(keypoint_scale),
            },
        )
        LOGGER.info("Wrote %s", episode_path)
        written.append(episode_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PointPolicy human pickle data to EgoVerse Zarr."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "/home/madhavan/h2r/human_data/processed_data_pkl/stick_in_bowl.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/custom_human_azure_kinect_zarr/stick_in_bowl"),
    )
    parser.add_argument("--task-name", type=str, default="stick_in_bowl")
    parser.add_argument(
        "--task-description",
        type=str,
        default="PointPolicy human demonstration: stick in bowl",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-key", type=str, default="pixels1")
    parser.add_argument("--track-key", type=str, default="human_tracks_3d_pixels1")
    parser.add_argument(
        "--keypoint-scale",
        type=float,
        default=1.0,
        help="Scale applied to PointPolicy 3D tracks before writing Zarr.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = convert_pointpolicy_pickle(
        input_path=args.input_path,
        output_dir=args.output_dir,
        task_name=args.task_name,
        task_description=args.task_description,
        fps=args.fps,
        camera_key=args.camera_key,
        track_key=args.track_key,
        keypoint_scale=args.keypoint_scale,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(written)} episodes to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
