"""Export metric MANO keypoints as 15 Hz static-camera human episodes.

The input sidecars must contain ``(T, 21, 3)`` MANO-order keypoints in metres
in the front-camera optical frame.  WiLoR/stereo/depth processing is deliberately
upstream of this boundary so this converter can validate its metric contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.custom_data.lerobot_2cam_to_egoverse_zarr import (
    LEFT_COLOR_KEY,
    MANO_CANONICAL_ORDER,
    _base_time_from_dir,
    _episode_hash,
    _keypoints_to_pose_and_flat,
    _load_mano_keypoints,
    _parquet_path,
    _read_jsonl,
    _read_rgb_video,
    _task_for_episode,
    _video_path,
)
from egomimic.scripts.custom_data.run_wilor_static_human import transform_points

LOGGER = logging.getLogger(__name__)


def convert_dataset(
    root: Path,
    keypoint_dir: Path,
    output_dir: Path,
    *,
    target_fps: int = 15,
    video_key: str = LEFT_COLOR_KEY,
    world_from_camera: np.ndarray | None = None,
    keypoint_suffix: str = ".mp4.keypoints3d.npy",
    overwrite: bool = False,
) -> list[Path]:
    info = json.loads((root / "meta/info.json").read_text())
    source_fps = int(round(float(info.get("fps", 30))))
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source fps {source_fps} must be an integer multiple of target fps {target_fps}"
        )
    stride = source_fps // target_fps
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    base_time = _base_time_from_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for episode in episodes:
        episode_id = int(episode["episode_index"])
        path = output_dir / f"{_episode_hash(base_time, episode_id)}.zarr"
        if path.exists():
            if not overwrite:
                LOGGER.info("Skipping existing %s", path)
                continue
            shutil.rmtree(path)
        import pandas as pd

        frame_count = len(pd.read_parquet(_parquet_path(root, info, episode_id)))
        images = _read_rgb_video(
            _video_path(root, info, episode_id, video_key),
            expected_frames=frame_count,
            temporal_stride=stride,
        )
        keypoint_path = keypoint_dir / f"episode_{episode_id:06d}{keypoint_suffix}"
        keypoints = _load_mano_keypoints(keypoint_path, frame_count, scale=1.0)
        if world_from_camera is not None:
            keypoints = transform_points(keypoints, np.linalg.inv(world_from_camera))
        keypoints = keypoints[::stride]
        if len(keypoints) < 2:
            LOGGER.warning(
                "Skipping episode %d: fewer than two frames after downsampling",
                episode_id,
            )
            continue
        if len(images) != len(keypoints):
            raise ValueError(
                f"episode {episode_id}: synchronization mismatch: "
                f"images={len(images)} keypoints={len(keypoints)}"
            )
        if not np.isfinite(keypoints).all():
            raise ValueError(f"episode {episode_id}: keypoints contain NaN/Inf")
        # Generous metric sanity bound; catches millimetres accidentally passed as metres.
        if np.max(np.abs(keypoints)) > 10.0:
            raise ValueError(
                f"episode {episode_id}: keypoints exceed 10 m; expected camera-frame metres"
            )
        pose, flat_keypoints, valid_frames = _keypoints_to_pose_and_flat(
            keypoints, "mecka_right_hand"
        )
        obs_pose = pose[:-1].astype(np.float32)
        action_pose = pose[1:].astype(np.float32)
        np.testing.assert_allclose(action_pose, pose[1:], atol=1e-6, rtol=0)
        task_name, task_description = _task_for_episode(root, episode)
        ZarrWriter.create_and_write(
            episode_path=path,
            numeric_data={
                "right.obs_ee_pose": obs_pose,
                "right.action_ee_pose": action_pose,
                "right.obs_keypoints": flat_keypoints[:-1].astype(np.float32),
            },
            image_data={"images.front_1": images[:-1]},
            embodiment="custom_human_right_arm",
            fps=target_fps,
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "metric_mano_next_observation_v1",
                "source_path": str(root),
                "source_keypoint_path": str(keypoint_path),
                "source_keypoint_order": MANO_CANONICAL_ORDER,
                "source_fps": source_fps,
                "temporal_stride": stride,
                "action_semantics": "next_observation",
                "pose_frame": "selected_camera_optical",
                "source_video_key": video_key,
                "human_pose_valid_frames": int(valid_frames),
            },
        )
        LOGGER.info("Wrote %s with %d aligned samples", path, len(obs_pose))
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--keypoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--video-key", default=LEFT_COLOR_KEY)
    parser.add_argument("--world-from-camera", type=Path)
    parser.add_argument("--keypoint-suffix", default=".mp4.keypoints3d.npy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    world_from_camera = None
    if args.world_from_camera is not None:
        world_from_camera = np.loadtxt(args.world_from_camera)
        if world_from_camera.shape != (4, 4):
            raise ValueError(f"world-from-camera must be 4x4, got {world_from_camera.shape}")
    convert_dataset(
        args.source,
        args.keypoint_dir,
        args.output_dir,
        target_fps=args.target_fps,
        video_key=args.video_key,
        world_from_camera=world_from_camera,
        keypoint_suffix=args.keypoint_suffix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
