"""Convert a single LeRobot-style Zarr dataset into EgoVerse episode Zarrs.

This handles datasets that store all episodes in one ``dataset.zarr`` with
``episode_index`` and keys such as:

  - observation.images.cam_azure_kinect_front.color/{bytes,lengths}
  - observation.images.cam_azure_kinect_left.color/{bytes,lengths}
  - observation.images.cam_wrist/{bytes,lengths}
  - observation.right_eef_pose
  - action.right_eef_pose

The output is a directory of EgoVerse episode stores compatible with
``train_zarr_custom_franka_3cam``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import simplejpeg
import zarr

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.custom_data.lerobot_2cam_to_egoverse_zarr import (
    _eef10_to_pose_gripper,
)

LOGGER = logging.getLogger(__name__)

FRONT_COLOR_KEY = "observation.images.cam_azure_kinect_front.color"
LEFT_COLOR_KEY = "observation.images.cam_azure_kinect_left.color"
WRIST_COLOR_KEY = "observation.images.cam_wrist"
OBS_EEF_KEY = "observation.right_eef_pose"
ACTION_EEF_KEY = "action.right_eef_pose"

OUTPUT_IMAGE_KEYS = {
    FRONT_COLOR_KEY: "images.front_1",
    LEFT_COLOR_KEY: "images.front_2",
    WRIST_COLOR_KEY: "images.wrist",
}


def _features(source: zarr.Group) -> dict[str, Any]:
    raw = source.attrs.get("features_json")
    if raw:
        return json.loads(raw)
    return dict(source.attrs.get("features", {}))


def _require_keys(source: zarr.Group, keys: list[str]) -> None:
    missing = [key for key in keys if key not in source]
    if missing:
        raise KeyError(f"Source Zarr is missing required keys: {missing}")


def _infer_fps(source: zarr.Group, default: int = 30) -> int:
    timestamp = source.get("timestamp")
    if timestamp is None or timestamp.shape[0] < 2:
        return default
    sample = np.asarray(timestamp[: min(timestamp.shape[0], 2000)], dtype=np.float64)
    diffs = np.diff(sample)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return default
    return int(round(1.0 / float(np.median(diffs))))


def _episode_slices(episode_index: np.ndarray) -> list[tuple[int, int, int]]:
    if episode_index.ndim != 1:
        raise ValueError(f"episode_index must be 1D, got {episode_index.shape}")
    if len(episode_index) == 0:
        raise ValueError("Source dataset contains zero frames")

    boundaries = np.flatnonzero(np.diff(episode_index) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(episode_index)]])

    slices: list[tuple[int, int, int]] = []
    for start, end in zip(starts, ends, strict=True):
        ep = int(episode_index[start])
        if not np.all(episode_index[start:end] == ep):
            raise ValueError(f"Non-contiguous frames found for episode {ep}")
        slices.append((ep, int(start), int(end)))
    return slices


def _image_shape(features: dict[str, Any], key: str) -> list[int]:
    feature = features.get(key)
    if feature is None or "shape" not in feature:
        raise KeyError(f"Could not infer image shape for {key!r} from features_json")
    shape = list(feature["shape"])
    if shape != [720, 1280, 3] and len(shape) != 3:
        raise ValueError(f"Expected HWC image shape for {key!r}, got {shape}")
    return shape


def _encoded_images_for_slice(
    source: zarr.Group,
    source_key: str,
    start: int,
    end: int,
) -> np.ndarray:
    group = source[source_key]
    if "bytes" not in group or "lengths" not in group:
        raise KeyError(f"Image group {source_key!r} must contain bytes and lengths")

    lengths = np.asarray(group["lengths"][start:end], dtype=np.int64)
    byte_array = group["bytes"]
    encoded = np.empty((end - start,), dtype=object)
    if np.any(lengths <= 0):
        bad = int(np.flatnonzero(lengths <= 0)[0])
        raise ValueError(f"Invalid JPEG length {lengths[bad]} for {source_key}[{start + bad}]")

    frames_per_chunk = int(byte_array.chunks[0]) if byte_array.chunks else 64
    for block_start in range(start, end, frames_per_chunk):
        block_end = min(end, block_start + frames_per_chunk)
        out_start = block_start - start
        out_end = block_end - start
        block_lengths = lengths[out_start:out_end]
        max_len = int(block_lengths.max())
        block = np.asarray(byte_array[block_start:block_end, :max_len])
        for offset, nbytes in enumerate(block_lengths):
            encoded[out_start + offset] = bytes(block[offset, : int(nbytes)])
    return encoded


def _validate_source(source: zarr.Group) -> None:
    required = [
        *OUTPUT_IMAGE_KEYS.keys(),
        OBS_EEF_KEY,
        ACTION_EEF_KEY,
        "episode_index",
        "frame_index",
    ]
    _require_keys(source, required)

    total_frames = source["episode_index"].shape[0]
    for key in required:
        obj = source[key]
        if key in OUTPUT_IMAGE_KEYS:
            for image_child in ["bytes", "lengths"]:
                if image_child not in obj:
                    raise KeyError(f"Image group {key!r} is missing {image_child!r}")
                if obj[image_child].shape[0] != total_frames:
                    raise ValueError(
                        f"Key {key}/{image_child} has {obj[image_child].shape[0]} "
                        f"frames, expected {total_frames}"
                    )
        elif obj.shape[0] != total_frames:
            raise ValueError(
                f"Key {key!r} has {obj.shape[0]} frames, expected {total_frames}"
            )

    for key in [OBS_EEF_KEY, ACTION_EEF_KEY]:
        shape = source[key].shape
        if len(shape) != 2 or shape[1] != 10:
            raise ValueError(f"{key!r} must have shape (T, 10), got {shape}")


def _validate_output_episode(path: Path, expected_frames: int) -> None:
    episode = zarr.open_group(str(path), mode="r")
    attrs = dict(episode.attrs)
    if attrs.get("embodiment") != "franka_right_arm":
        raise ValueError(f"{path} has unexpected embodiment {attrs.get('embodiment')!r}")
    if int(attrs.get("total_frames", -1)) != expected_frames:
        raise ValueError(
            f"{path} total_frames={attrs.get('total_frames')} expected {expected_frames}"
        )

    required = [
        "images.front_1",
        "images.front_2",
        "images.wrist",
        "right.obs_ee_pose",
        "right.obs_gripper",
        "right.cmd_ee_pose",
        "right.cmd_gripper",
    ]
    _require_keys(episode, required)

    for key in required:
        if key.startswith("images."):
            if episode[key].shape[0] < expected_frames:
                raise ValueError(f"{path}:{key} has too few frames")
        elif episode[key].shape[0] < expected_frames:
            raise ValueError(f"{path}:{key} has too few frames")

    for key in ["right.obs_ee_pose", "right.cmd_ee_pose"]:
        values = np.asarray(episode[key][:expected_frames])
        if values.shape != (expected_frames, 7):
            raise ValueError(f"{path}:{key} expected {(expected_frames, 7)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}:{key} contains non-finite values")
        quat_norm = np.linalg.norm(values[:, 3:7], axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1e-3):
            raise ValueError(f"{path}:{key} contains non-unit quaternions")

    for key in ["images.front_1", "images.front_2", "images.wrist"]:
        jpeg = episode[key][0:1][0]
        decoded = simplejpeg.decode_jpeg(jpeg, colorspace="RGB")
        if decoded.ndim != 3 or decoded.shape[-1] != 3:
            raise ValueError(f"{path}:{key} did not decode to an RGB image")


def convert_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    task_name: str = "custom_franka_3cam",
    task_description: str = "",
) -> list[Path]:
    source = zarr.open_group(str(source_path), mode="r")
    _validate_source(source)

    features = _features(source)
    fps = _infer_fps(source)
    episode_index = np.asarray(source["episode_index"][:], dtype=np.int64)
    slices = _episode_slices(episode_index)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for episode_id, start, end in slices:
        frame_count = end - start
        episode_name = f"episode_{episode_id:06d}.zarr"
        episode_path = output_dir / episode_name
        if episode_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{episode_path} exists. Pass --overwrite to replace it."
                )
            shutil.rmtree(episode_path)

        obs_pose, obs_gripper = _eef10_to_pose_gripper(
            np.asarray(source[OBS_EEF_KEY][start:end], dtype=np.float64)
        )
        cmd_pose, cmd_gripper = _eef10_to_pose_gripper(
            np.asarray(source[ACTION_EEF_KEY][start:end], dtype=np.float64)
        )

        pre_encoded = {}
        for source_key, output_key in OUTPUT_IMAGE_KEYS.items():
            pre_encoded[output_key] = (
                _encoded_images_for_slice(source, source_key, start, end),
                _image_shape(features, source_key),
            )

        ZarrWriter.create_and_write(
            episode_path=episode_path,
            numeric_data={
                "right.obs_ee_pose": obs_pose,
                "right.obs_gripper": obs_gripper,
                "right.cmd_ee_pose": cmd_pose,
                "right.cmd_gripper": cmd_gripper,
            },
            pre_encoded_image_data=pre_encoded,
            embodiment="franka_right_arm",
            fps=fps,
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "lerobot_zarr_v1",
                "source_path": str(source_path),
                "source_episode_index": episode_id,
                "source_frame_start": start,
                "source_frame_end": end,
            },
        )
        _validate_output_episode(episode_path, frame_count)
        LOGGER.info("Wrote and validated %s (%d frames)", episode_path, frame_count)
        written.append(episode_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single LeRobot-style Zarr into EgoVerse episode Zarrs."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-name", default="custom_franka_3cam")
    parser.add_argument("--task-description", default="")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    written = convert_dataset(
        args.source,
        args.output_dir,
        overwrite=args.overwrite,
        task_name=args.task_name,
        task_description=args.task_description,
    )
    LOGGER.info("Converted %d episodes into %s", len(written), args.output_dir)


if __name__ == "__main__":
    main()
