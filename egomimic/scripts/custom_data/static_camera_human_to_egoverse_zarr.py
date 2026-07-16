"""Export metric MANO keypoints as static-camera human episodes.

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
    target_fps: int = 30,
    video_key: str = LEFT_COLOR_KEY,
    world_from_camera: np.ndarray | None = None,
    keypoint_suffix: str = ".mp4.keypoints3d.npy",
    keypoint_index_offset: int = 0,
    episode_limit: int | None = None,
    keypoints_at_target_fps: bool = False,
    image_scale_factor: int = 1,
    crop_height: int | None = None,
    crop_width: int | None = None,
    keypoint_source: str = "rgbd_metric",
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
    if episode_limit is not None:
        episodes = episodes[:episode_limit]
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
            spatial_scale_factor=image_scale_factor,
        )
        if (crop_height is None) != (crop_width is None):
            raise ValueError("crop-height and crop-width must be specified together")
        if crop_height is not None and crop_width is not None:
            height, width = images.shape[1:3]
            if crop_height > height or crop_width > width:
                raise ValueError(
                    f"requested crop {crop_height}x{crop_width} exceeds image {height}x{width}"
                )
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
            images = images[:, top : top + crop_height, left : left + crop_width]
        keypoint_index = episode_id + keypoint_index_offset
        keypoint_path = keypoint_dir / f"episode_{keypoint_index:06d}{keypoint_suffix}"
        expected_keypoint_frames = len(range(0, frame_count, stride)) if keypoints_at_target_fps else frame_count
        keypoints = _load_mano_keypoints(keypoint_path, expected_keypoint_frames, scale=1.0)
        if world_from_camera is not None:
            keypoints = transform_points(keypoints, np.linalg.inv(world_from_camera))
        if not keypoints_at_target_fps:
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
                "keypoints_at_target_fps": keypoints_at_target_fps,
                "image_scale_factor": image_scale_factor,
                "center_crop_hw": [crop_height, crop_width],
                "keypoint_3d_source": keypoint_source,
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
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--video-key", default=LEFT_COLOR_KEY)
    parser.add_argument("--world-from-camera", type=Path)
    parser.add_argument("--keypoint-suffix", default=".mp4.keypoints3d.npy")
    parser.add_argument("--keypoint-index-offset", type=int, default=0)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--keypoints-at-target-fps", action="store_true")
    parser.add_argument("--image-scale-factor", type=int, default=1)
    parser.add_argument("--crop-height", type=int)
    parser.add_argument("--crop-width", type=int)
    parser.add_argument("--keypoint-source", default="rgbd_metric")
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
        keypoint_index_offset=args.keypoint_index_offset,
        episode_limit=args.episode_limit,
        keypoints_at_target_fps=args.keypoints_at_target_fps,
        image_scale_factor=args.image_scale_factor,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        keypoint_source=args.keypoint_source,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
