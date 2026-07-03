"""Prepare fixed-camera Franka demonstrations for human/robot co-training.

The source is the consolidated LeRobot Zarr used by
``lerobot_zarr_to_egoverse_zarr``.  Unlike that legacy converter, labels here
are the *next downsampled observation*, never the recorded controller action.
All TCP poses are exported in the front-camera optical frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.custom_data.lerobot_2cam_to_egoverse_zarr import (
    _eef10_to_pose_gripper,
)
from egomimic.scripts.custom_data.lerobot_zarr_to_egoverse_zarr import (
    LEFT_COLOR_KEY,
    OBS_EEF_KEY,
    WRIST_COLOR_KEY,
    _encoded_images_for_slice,
    _episode_slices,
    _features,
    _image_shape,
    _infer_fps,
    _require_keys,
)

LOGGER = logging.getLogger(__name__)


def load_camera_from_base(path: Path) -> np.ndarray:
    """Load and validate a homogeneous ``T_camera_base`` matrix."""
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("camera_from_base", payload.get("T_camera_base"))
        matrix = np.asarray(payload, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera_from_base must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("camera_from_base contains non-finite values")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("camera_from_base must be a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("camera_from_base rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("camera_from_base rotation determinant must be +1")
    return matrix


def transform_wxyz_poses(poses: np.ndarray, camera_from_base: np.ndarray) -> np.ndarray:
    """Apply ``T_camera_base @ T_base_tcp`` to an ``(T, 7)`` pose sequence."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"poses must have shape (T, 7), got {poses.shape}")
    quat_xyzw = poses[:, [4, 5, 6, 3]]
    base_from_tcp = Rotation.from_quat(quat_xyzw).as_matrix()
    camera_from_tcp = camera_from_base[:3, :3][None] @ base_from_tcp
    xyz = (
        camera_from_base[:3, :3] @ poses[:, :3].T
    ).T + camera_from_base[:3, 3]
    out_xyzw = Rotation.from_matrix(camera_from_tcp).as_quat()
    out_wxyz = out_xyzw[:, [3, 0, 1, 2]]
    # q and -q are equivalent; continuity prevents artificial YPR jumps.
    for i in range(1, len(out_wxyz)):
        if np.dot(out_wxyz[i - 1], out_wxyz[i]) < 0:
            out_wxyz[i] *= -1
    return np.concatenate([xyz, out_wxyz], axis=-1)


def next_observation_pairs(
    poses: np.ndarray, gripper: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample, then create exact ``action[t] = observation[t+1]`` pairs."""
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    indices = np.arange(0, len(poses), stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("Episode has fewer than two frames after downsampling")
    sampled_pose = np.asarray(poses)[indices]
    sampled_gripper = np.asarray(gripper)[indices]
    obs_pose, cmd_pose = sampled_pose[:-1].copy(), sampled_pose[1:].copy()
    obs_gripper = sampled_gripper[:-1].copy()
    cmd_gripper = sampled_gripper[1:].copy()
    if not np.allclose(cmd_pose, sampled_pose[1:], atol=1e-6, rtol=0):
        raise AssertionError("next-observation pose alignment failed")
    if not np.allclose(cmd_gripper, sampled_gripper[1:], atol=1e-6, rtol=0):
        raise AssertionError("next-observation gripper alignment failed")
    return indices[:-1], obs_pose, obs_gripper, cmd_pose, cmd_gripper


def convert_dataset(
    source_path: Path,
    output_dir: Path,
    camera_from_base: np.ndarray,
    *,
    target_fps: int = 30,
    camera_key: str = LEFT_COLOR_KEY,
    overwrite: bool = False,
    task_name: str = "static_camera_pick_place",
    task_description: str = "",
) -> list[Path]:
    source = zarr.open_group(str(source_path), mode="r")
    required = [camera_key, WRIST_COLOR_KEY, OBS_EEF_KEY, "episode_index"]
    _require_keys(source, required)
    source_fps = _infer_fps(source)
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source fps {source_fps} must be an integer multiple of target fps {target_fps}"
        )
    stride = source_fps // target_fps
    features = _features(source)
    episodes = _episode_slices(np.asarray(source["episode_index"][:], dtype=np.int64))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for episode_id, start, end in episodes:
        path = output_dir / f"episode_{episode_id:06d}.zarr"
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            shutil.rmtree(path)

        base_pose, gripper = _eef10_to_pose_gripper(
            np.asarray(source[OBS_EEF_KEY][start:end], dtype=np.float64)
        )
        camera_pose = transform_wxyz_poses(base_pose, camera_from_base)
        local_image_indices, obs_pose, obs_gripper, cmd_pose, cmd_gripper = (
            next_observation_pairs(camera_pose, gripper, stride)
        )
        pre_encoded = {}
        for source_key, output_key in (
            (camera_key, "images.front_1"),
            (WRIST_COLOR_KEY, "images.wrist"),
        ):
            encoded = _encoded_images_for_slice(source, source_key, start, end)
            pre_encoded[output_key] = (
                encoded[local_image_indices],
                _image_shape(features, source_key),
            )

        ZarrWriter.create_and_write(
            episode_path=path,
            numeric_data={
                "right.obs_ee_pose": obs_pose.astype(np.float32),
                "right.obs_gripper": obs_gripper.astype(np.float32),
                "right.cmd_ee_pose": cmd_pose.astype(np.float32),
                "right.cmd_gripper": cmd_gripper.astype(np.float32),
            },
            pre_encoded_image_data=pre_encoded,
            embodiment="franka_right_arm",
            fps=target_fps,
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "lerobot_zarr_next_observation_v1",
                "source_path": str(source_path),
                "source_episode_index": episode_id,
                "source_fps": source_fps,
                "temporal_stride": stride,
                "action_semantics": "next_observation",
                "pose_frame": "front_camera_optical",
                "camera_from_base": camera_from_base.tolist(),
            },
        )
        check = zarr.open_group(str(path), mode="r")
        np.testing.assert_allclose(
            check["right.cmd_ee_pose"][: len(cmd_pose)], cmd_pose, atol=1e-6, rtol=0
        )
        LOGGER.info("Wrote %s with %d aligned samples", path, len(obs_pose))
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-from-base",
        type=Path,
        required=True,
        help=".npy or JSON 4x4 transform mapping Franka-base poses to front-camera poses",
    )
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--camera-key", default=LEFT_COLOR_KEY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-name", default="static_camera_pick_place")
    parser.add_argument("--task-description", default="")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    convert_dataset(
        args.source,
        args.output_dir,
        load_camera_from_base(args.camera_from_base),
        target_fps=args.target_fps,
        camera_key=args.camera_key,
        overwrite=args.overwrite,
        task_name=args.task_name,
        task_description=args.task_description,
    )


if __name__ == "__main__":
    main()
