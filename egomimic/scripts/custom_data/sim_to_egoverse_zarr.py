"""Convert simulator trajectory.npz + 3 MP4 episode folders to EgoVerse Zarr.

Expected source layout::

    root/episode_000000/{trajectory.npz,cam0.mp4,cam1.mp4,wrist_cam.mp4}

The source has T observations/video frames and T-1 actions. Output row t pairs
observation[t] with action[t], so the terminal observation is intentionally dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_writer import ZarrWriter


LOGGER = logging.getLogger(__name__)
CAMERAS = {
    # cam1 is the selected static/task camera and therefore occupies EgoVerse's
    # primary front-camera slot. cam0 is intentionally omitted.
    "cam1.mp4": "images.front_1",
    "wrist_cam.mp4": "images.wrist",
}


def _load_world_from_camera(path: Path, camera: str) -> np.ndarray:
    calibration = json.loads(path.read_text())
    if camera not in calibration or "extrinsic" not in calibration[camera]:
        raise KeyError(f"{path} does not contain {camera}.extrinsic")
    transform = np.asarray(calibration[camera]["extrinsic"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{camera}.extrinsic must be 4x4, got {transform.shape}")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{camera}.extrinsic has an invalid homogeneous bottom row")
    return transform


def _poses_world_to_camera(
    poses: np.ndarray, world_from_camera: np.ndarray
) -> np.ndarray:
    """Transform xyz + quaternion(wxyz) poses from world/base into camera frame."""
    poses = np.asarray(poses, dtype=np.float64)
    camera_from_world = np.linalg.inv(world_from_camera)
    world_from_eef = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    world_from_eef[:, :3, 3] = poses[:, :3]
    world_from_eef[:, :3, :3] = Rotation.from_quat(
        poses[:, [4, 5, 6, 3]]
    ).as_matrix()
    camera_from_eef = camera_from_world[None] @ world_from_eef
    quat_xyzw = Rotation.from_matrix(camera_from_eef[:, :3, :3]).as_quat()
    return np.concatenate(
        (camera_from_eef[:, :3, 3], quat_xyzw[:, [3, 0, 1, 2]]), axis=1
    )


def _episode_index(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[1])


def _episode_dirs(root: Path, selected: set[int] | None) -> list[Path]:
    episodes = sorted(
        (p.parent for p in root.glob("episode_*/trajectory.npz")),
        key=_episode_index,
    )
    if selected is not None:
        episodes = [p for p in episodes if _episode_index(p) in selected]
        missing = selected - {_episode_index(p) for p in episodes}
        if missing:
            raise FileNotFoundError(f"Missing episode(s): {sorted(missing)}")
    if not episodes:
        raise FileNotFoundError(f"No episode_*/trajectory.npz under {root}")
    return episodes


def _episode_hash(base_time: datetime, episode_index: int) -> str:
    return (base_time + timedelta(microseconds=episode_index)).strftime(
        "%Y-%m-%d-%H-%M-%S-%f"
    )


def _process_frame(
    frame_bgr: np.ndarray,
    *,
    scale_factor: int,
    crop_shape: tuple[int, int] | None,
    crop_left: int | None,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    frame = cv2.resize(
        frame_bgr,
        (width // scale_factor, height // scale_factor),
        interpolation=cv2.INTER_AREA,
    )
    if crop_shape is not None:
        crop_h, crop_w = crop_shape
        height, width = frame.shape[:2]
        if crop_h > height or crop_w > width:
            raise ValueError(
                f"Crop {crop_h}x{crop_w} does not fit resized frame {height}x{width}"
            )
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2 if crop_left is None else crop_left
        if not 0 <= left <= width - crop_w:
            raise ValueError(
                f"crop_left={left} is outside [0, {width - crop_w}] for "
                f"{height}x{width} and crop {crop_h}x{crop_w}"
            )
        frame = frame[top : top + crop_h, left : left + crop_w]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _read_video(
    path: Path,
    source_action_length: int,
    keep_indices: np.ndarray,
    *,
    scale_factor: int,
    crop_shape: tuple[int, int] | None,
    crop_left: int | None,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if reported_frames != source_action_length + 1:
        cap.release()
        raise ValueError(
            f"{path} has {reported_frames} frames; expected {source_action_length + 1}"
        )

    keep = set(keep_indices.tolist())
    frames = []
    frame_index = 0
    while frame_index < source_action_length:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in keep:
            frames.append(
                _process_frame(
                    frame,
                    scale_factor=scale_factor,
                    crop_shape=crop_shape,
                    crop_left=crop_left,
                )
            )
        frame_index += 1
    cap.release()
    if frame_index != source_action_length or len(frames) != len(keep_indices):
        raise ValueError(
            f"{path}: decoded {frame_index} source and {len(frames)} selected frames; "
            f"expected {source_action_length} and {len(keep_indices)}"
        )
    return np.stack(frames), fps


def _validate_trajectory(data: np.lib.npyio.NpzFile, episode: Path) -> int:
    required = {
        "states_ee",
        "states_joint",
        "action_ee",
        "action_joint",
        "gripper_pcd",
        "goal_gripper_pcd",
        "gripper_width",
        "delta_action",
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"{episode}: missing trajectory arrays {sorted(missing)}")
    action_length = len(data["action_ee"])
    if data["action_ee"].shape != (action_length, 8):
        raise ValueError(f"{episode}: action_ee must be (T-1, 8)")
    if data["states_ee"].shape != (action_length + 1, 8):
        raise ValueError(f"{episode}: states_ee must be (T, 8)")
    return action_length


def convert(
    root: Path,
    output_dir: Path,
    *,
    episodes: list[int] | None,
    temporal_stride: int,
    scale_factor: int,
    crop_shape: tuple[int, int] | None,
    wrist_crop_left: int | None,
    camera_calibration: Path,
    eef_camera: str,
    task_name: str,
    task_description: str,
    overwrite: bool,
) -> list[Path]:
    episode_dirs = _episode_dirs(root, set(episodes) if episodes is not None else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_time = datetime.fromtimestamp(root.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    )
    written = []
    world_from_camera = _load_world_from_camera(camera_calibration, eef_camera)

    for number, episode in enumerate(episode_dirs, start=1):
        episode_index = _episode_index(episode)
        with np.load(episode / "trajectory.npz") as data:
            action_length = _validate_trajectory(data, episode)
            keep = np.arange(0, action_length, temporal_stride)

            # Simulator EEF layout is already xyz + quat(wxyz) + gripper.
            states_ee = np.asarray(data["states_ee"][keep], dtype=np.float64)
            actions_ee = np.asarray(data["action_ee"][keep], dtype=np.float64)
            obs_pose, obs_gripper = states_ee[:, :7], states_ee[:, 7:8]
            cmd_pose, cmd_gripper = actions_ee[:, :7], actions_ee[:, 7:8]
            obs_pose = _poses_world_to_camera(obs_pose, world_from_camera)
            cmd_pose = _poses_world_to_camera(cmd_pose, world_from_camera)

        image_data = {}
        source_fps = None
        for filename, zarr_key in CAMERAS.items():
            crop_left = wrist_crop_left if filename == "wrist_cam.mp4" else None
            images, camera_fps = _read_video(
                episode / filename,
                action_length,
                keep,
                scale_factor=scale_factor,
                crop_shape=crop_shape,
                crop_left=crop_left,
            )
            if source_fps is None:
                source_fps = camera_fps
            elif abs(camera_fps - source_fps) > 1e-3:
                raise ValueError(f"{episode}: camera FPS values do not match")
            image_data[zarr_key] = images

        output = output_dir / f"{_episode_hash(base_time, episode_index)}.zarr"
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"{output} exists; pass --overwrite")
            shutil.rmtree(output)

        output_fps = int(round(source_fps / temporal_stride))
        ZarrWriter.create_and_write(
            episode_path=output,
            numeric_data={
                "right.obs_ee_pose": obs_pose,
                "right.obs_gripper": obs_gripper,
                "right.cmd_ee_pose": cmd_pose,
                "right.cmd_gripper": cmd_gripper,
            },
            image_data=image_data,
            embodiment="franka_right_arm",
            fps=output_fps,
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "sim_trajectory_npz_mp4",
                "source_path": str(root),
                "source_episode_index": episode_index,
                "source_fps": source_fps,
                "terminal_observation_dropped": True,
                "temporal_stride": temporal_stride,
                "spatial_scale_factor": scale_factor,
                "crop_shape": list(crop_shape) if crop_shape is not None else None,
                "wrist_crop_left": wrist_crop_left,
                "pose_frame": f"{eef_camera}_optical",
                "camera_calibration": str(camera_calibration),
                "world_from_pose_camera": world_from_camera.tolist(),
            },
        )
        LOGGER.info("[%d/%d] Wrote %s", number, len(episode_dirs), output)
        written.append(output)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument("--crop-shape", type=int, nargs=2, default=(360, 480), metavar=("H", "W"))
    parser.add_argument("--wrist-crop-left", type=int, default=160)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--eef-camera", default="cam1")
    parser.add_argument("--task-name", default="pick_place_red_mug")
    parser.add_argument("--task-description", default="pick and place the red mug")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.temporal_stride < 1 or args.scale_factor < 1:
        parser.error("--temporal-stride and --scale-factor must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    written = convert(
        args.root,
        args.output_dir,
        episodes=args.episodes,
        temporal_stride=args.temporal_stride,
        scale_factor=args.scale_factor,
        crop_shape=tuple(args.crop_shape) if args.crop_shape is not None else None,
        wrist_crop_left=args.wrist_crop_left,
        camera_calibration=args.camera_calibration,
        eef_camera=args.eef_camera,
        task_name=args.task_name,
        task_description=args.task_description,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(written)} EgoVerse Zarr episodes to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
