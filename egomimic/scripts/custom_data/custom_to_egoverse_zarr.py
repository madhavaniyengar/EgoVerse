"""Convert custom Azure-Kinect human and Franka logs to EgoVerse Zarr.

This file is intentionally a scaffold: fill in the two ``load_*`` functions
for your raw log format, while keeping their return schema unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

LOGGER = logging.getLogger(__name__)

EMBODIMENTS = {
    "custom_human_right_arm",
    "franka_right_arm",
}


def _episode_hash_from_path(raw_path: Path) -> str:
    try:
        datetime.strptime(raw_path.stem, "%Y-%m-%d-%H-%M-%S-%f")
        return raw_path.stem
    except ValueError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")


def _load_annotations(path: Path | None) -> list[tuple[str, int, int]] | None:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        return [
            (
                str(item["text"]),
                int(item["start_idx"]),
                int(item["end_idx"]),
            )
            for item in payload
        ]
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return [
                (
                    str(row["text"]),
                    int(row["start_idx"]),
                    int(row["end_idx"]),
                )
                for row in csv.DictReader(f)
            ]
    raise ValueError(f"Unsupported annotation file type: {path}")


def _validate_lengths(
    numeric_data: dict[str, np.ndarray],
    image_data: dict[str, np.ndarray],
) -> None:
    lengths = [len(v) for v in numeric_data.values()]
    lengths.extend(len(v) for v in image_data.values())
    if not lengths:
        raise ValueError("No arrays were loaded from the raw episode.")
    if len(set(lengths)) != 1:
        raise ValueError(f"All arrays must share frame count, got {lengths}")


def load_custom_human_episode(raw_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load one Azure-Kinect human episode.

    TODO: replace this placeholder with your actual parser.

    Return:
        numeric_data:
          - "right.obs_ee_pose": float array, shape (T, 7), xyz + quat(wxyz)
          - "right.obs_keypoints": float array, shape (T, 63), optional for
            future keypoint models but recommended to store now
        image_data:
          - "images.front_1": uint8 RGB array, shape (T, H, W, 3)
    """
    del raw_path
    raise NotImplementedError(
        "Fill in load_custom_human_episode() for your Azure Kinect raw format."
    )


def load_franka_episode(raw_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load one Franka right-arm episode.

    TODO: replace this placeholder with your actual parser.

    Return:
        numeric_data:
          - "right.obs_ee_pose": float array, shape (T, 7), xyz + quat(wxyz)
          - "right.obs_gripper": float array, shape (T, 1), normalized aperture
          - "right.cmd_ee_pose": float array, shape (T, 7), xyz + quat(wxyz)
          - "right.cmd_gripper": float array, shape (T, 1), normalized command
        image_data:
          - "images.front_1": uint8 RGB array, shape (T, H, W, 3)
    """
    del raw_path
    raise NotImplementedError(
        "Fill in load_franka_episode() for your Franka raw format."
    )


def load_episode(
    raw_path: Path,
    embodiment: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if embodiment == "custom_human_right_arm":
        return load_custom_human_episode(raw_path)
    if embodiment == "franka_right_arm":
        return load_franka_episode(raw_path)
    raise ValueError(f"Unsupported embodiment: {embodiment}")


def convert_episode(
    raw_path: Path,
    output_dir: Path,
    embodiment: str,
    fps: int,
    task_name: str,
    task_description: str,
    annotations: list[tuple[str, int, int]] | None = None,
    episode_hash: str | None = None,
    chunk_timesteps: int = 100,
) -> Path:
    numeric_data, image_data = load_episode(raw_path, embodiment)
    _validate_lengths(numeric_data, image_data)

    episode_hash = episode_hash or _episode_hash_from_path(raw_path)
    zarr_path = output_dir / f"{episode_hash}.zarr"

    ZarrWriter.create_and_write(
        episode_path=zarr_path,
        numeric_data=numeric_data,
        image_data=image_data,
        embodiment=embodiment,
        fps=fps,
        task_name=task_name,
        task_description=task_description,
        annotations=annotations,
        chunk_timesteps=chunk_timesteps,
    )
    LOGGER.info("Wrote %s", zarr_path)
    return zarr_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert custom Azure-Kinect human or Franka logs to EgoVerse Zarr."
    )
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embodiment", choices=sorted(EMBODIMENTS), required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task-name", type=str, default="custom_task")
    parser.add_argument("--task-description", type=str, default="")
    parser.add_argument("--episode-hash", type=str, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--chunk-timesteps", type=int, default=100)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    annotations = _load_annotations(args.annotations)
    return convert_episode(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        embodiment=args.embodiment,
        fps=args.fps,
        task_name=args.task_name,
        task_description=args.task_description,
        annotations=annotations,
        episode_hash=args.episode_hash,
        chunk_timesteps=args.chunk_timesteps,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main())
