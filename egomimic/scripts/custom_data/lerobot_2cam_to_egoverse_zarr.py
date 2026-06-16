"""Convert local LeRobot 2-camera human/Franka datasets to EgoVerse Zarr.

Robot episodes use LeRobot ``action.right_eef_pose`` and
``observation.right_eef_pose`` fields. These are stored as
``[rot6d(6), xyz(3), gripper(1)]`` and are converted to EgoVerse
``xyz + quat(wxyz)`` pose arrays.

Human episodes use a sidecar ``episode_XXXXXX.mp4.keypoints3d.npy`` with shape
``(T, 21, 3)`` in the canonical MANO/EgoVerse order:
``0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky``.
The converter stores those points directly as ``right.obs_keypoints`` and
derives ``right.obs_ee_pose`` with the same 21-keypoint hand-frame convention
used by the Mecka converter by default.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

LOGGER = logging.getLogger(__name__)
FRONT_COLOR_KEY = "observation.images.cam_azure_kinect_front.color"
LEFT_COLOR_KEY = "observation.images.cam_azure_kinect_left.color"
MANO_CANONICAL_ORDER = (
    "0=wrist,1-4=thumb,5-8=index,9-12=middle,13-16=ring,17-20=pinky"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _episode_hash(base_time: datetime, episode_index: int) -> str:
    return (base_time + timedelta(microseconds=episode_index)).strftime(
        "%Y-%m-%d-%H-%M-%S-%f"
    )


def _base_time_from_dir(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    )


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _parquet_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = _episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    rel = info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    return root / rel


def _video_path(
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str = FRONT_COLOR_KEY,
) -> Path:
    chunk = _episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    rel = info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return root / rel


def _read_rgb_video(path: Path, expected_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Video had no readable frames: {path}")
    arr = np.stack(frames, axis=0)
    if expected_frames is not None and len(arr) != expected_frames:
        raise ValueError(
            f"{path} has {len(arr)} frames but parquet has {expected_frames}"
        )
    return arr


def _stack_array_column(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"Missing parquet column {key!r}")
    return np.stack(df[key].to_numpy()).astype(np.float64)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    c1 = rot6d[..., 0:3]
    c2 = rot6d[..., 3:6]
    eps = 1e-8
    c1 = c1 / np.clip(np.linalg.norm(c1, axis=-1, keepdims=True), eps, None)
    c2 = c2 - np.sum(c2 * c1, axis=-1, keepdims=True) * c1
    c2 = c2 / np.clip(np.linalg.norm(c2, axis=-1, keepdims=True), eps, None)
    c3 = np.cross(c1, c2)
    return np.stack([c1, c2, c3], axis=-1)


def _eef10_to_pose_gripper(eef10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if eef10.ndim != 2 or eef10.shape[-1] != 10:
        raise ValueError(f"Expected EEF shape (T, 10), got {eef10.shape}")
    rot = _rot6d_to_matrix(eef10[:, :6])
    xyz = eef10[:, 6:9]
    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    pose = np.concatenate([xyz, quat_wxyz], axis=-1).astype(np.float64)
    gripper = eef10[:, 9:10].astype(np.float64)
    return pose, gripper


def _load_mano_keypoints(
    path: Path,
    expected_frames: int,
    scale: float,
) -> np.ndarray:
    keypoints = np.nan_to_num(np.asarray(np.load(path), dtype=np.float64)) * scale
    if keypoints.shape != (expected_frames, 21, 3):
        raise ValueError(
            f"Expected MANO/EgoVerse keypoints shape ({expected_frames}, 21, 3), "
            f"got {keypoints.shape} from {path}"
        )
    return keypoints


def _safe_normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = np.linalg.norm(v)
    if norm < eps:
        return None
    return v / norm


def _mano_right_hand_pose_from_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute a right-hand 7D pose from canonical 21-point MANO keypoints.

    This mirrors ``compute_hand_pose_xyzquat`` in the Mecka converter for the
    right hand. The pose position is the palm centroid, while the returned
    orientation is built from wrist, middle-base, index-base, and pinky-base.
    """
    quat_wxyz_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rot_right = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

    if np.allclose(keypoints, 0) or not np.isfinite(keypoints).all():
        fallback = np.concatenate([np.zeros(3), quat_wxyz_identity])
        return fallback, False

    wrist = keypoints[0]
    position = np.mean(
        [keypoints[0], keypoints[17], keypoints[13], keypoints[9], keypoints[5]],
        axis=0,
    )

    forward = _safe_normalize(keypoints[9] - wrist)
    if forward is None:
        fallback = np.concatenate([position, quat_wxyz_identity])
        return fallback, False

    thumb_dir = keypoints[5] - wrist
    pinky_dir = keypoints[17] - wrist
    up = _safe_normalize(np.cross(thumb_dir, pinky_dir))
    if up is None:
        fallback = np.concatenate([position, quat_wxyz_identity])
        return fallback, False

    right = _safe_normalize(np.cross(forward, up))
    if right is None:
        fallback = np.concatenate([position, quat_wxyz_identity])
        return fallback, False
    up = _safe_normalize(np.cross(right, forward))
    if up is None:
        fallback = np.concatenate([position, quat_wxyz_identity])
        return fallback, False

    right = right * -1.0
    rot_matrix = np.column_stack([forward, right, up]) @ rot_right
    quat_xyzw = R.from_matrix(rot_matrix).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate([position, quat_wxyz]).astype(np.float64), True


def _mano_keypoints_to_mecka_right_hand_poses(keypoints: np.ndarray) -> tuple[np.ndarray, int]:
    poses = np.zeros((keypoints.shape[0], 7), dtype=np.float64)
    valid_count = 0
    last_valid_pose: np.ndarray | None = None
    for i, frame_keypoints in enumerate(keypoints):
        pose, valid = _mano_right_hand_pose_from_keypoints(frame_keypoints)
        if valid:
            last_valid_pose = pose
            valid_count += 1
        elif last_valid_pose is not None:
            pose = last_valid_pose.copy()
            pose[:3] = np.mean(
                [
                    frame_keypoints[0],
                    frame_keypoints[17],
                    frame_keypoints[13],
                    frame_keypoints[9],
                    frame_keypoints[5],
                ],
                axis=0,
            )
        poses[i] = pose
    return poses, valid_count


def _keypoints_to_pose_and_flat(
    keypoints: np.ndarray,
    pose_from: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 3):
        raise ValueError(f"Expected keypoints shape (T, 21, 3), got {keypoints.shape}")
    if pose_from == "mecka_right_hand":
        pose, valid_count = _mano_keypoints_to_mecka_right_hand_poses(keypoints)
        return pose, keypoints.reshape(keypoints.shape[0], 63), valid_count
    if pose_from == "wrist":
        xyz = keypoints[:, 0, :]
    elif pose_from == "centroid":
        xyz = np.mean(keypoints, axis=1)
    else:
        raise ValueError(f"Unknown pose_from mode: {pose_from!r}")

    quat = np.zeros((keypoints.shape[0], 4), dtype=np.float64)
    quat[:, 0] = 1.0
    pose = np.concatenate([xyz, quat], axis=-1)
    return pose, keypoints.reshape(keypoints.shape[0], 63), 0


def _task_for_episode(root: Path, episode_meta: dict[str, Any]) -> tuple[str, str]:
    tasks = episode_meta.get("tasks") or []
    if tasks:
        return str(tasks[0]).lower().replace(" ", "_").replace(".", ""), str(tasks[0])
    task_rows = _read_jsonl(root / "meta/tasks.jsonl")
    task_idx = int(episode_meta.get("task_index", 0))
    for row in task_rows:
        if int(row.get("task_index", -1)) == task_idx:
            text = str(row.get("task", "custom_task"))
            return text.lower().replace(" ", "_").replace(".", ""), text
    return "custom_task", ""


def convert_robot_dataset(
    root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    video_key: str = FRONT_COLOR_KEY,
    second_video_key: str = LEFT_COLOR_KEY,
) -> list[Path]:
    info = json.loads((root / "meta/info.json").read_text())
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_time = _base_time_from_dir(root)
    written = []

    for ep in episodes:
        episode_index = int(ep["episode_index"])
        df = pd.read_parquet(_parquet_path(root, info, episode_index))
        images = _read_rgb_video(
            _video_path(root, info, episode_index, video_key),
            expected_frames=len(df),
        )
        second_video_path = _video_path(root, info, episode_index, second_video_key)
        images_2 = _read_rgb_video(second_video_path, expected_frames=len(df)) if second_video_path.exists() else None

        obs_pose, obs_gripper = _eef10_to_pose_gripper(
            _stack_array_column(df, "observation.right_eef_pose")
        )
        cmd_pose, cmd_gripper = _eef10_to_pose_gripper(
            _stack_array_column(df, "action.right_eef_pose")
        )

        task_name, task_description = _task_for_episode(root, ep)
        episode_hash = _episode_hash(base_time, episode_index)
        ep_path = output_dir / f"{episode_hash}.zarr"
        if ep_path.exists() and not overwrite:
            raise FileExistsError(f"{ep_path} exists. Pass --overwrite to replace it.")

        image_data = {"images.front_1": images}
        if images_2 is not None:
            image_data["images.front_2"] = images_2

        ZarrWriter.create_and_write(
            episode_path=ep_path,
            numeric_data={
                "right.obs_ee_pose": obs_pose,
                "right.obs_gripper": obs_gripper,
                "right.cmd_ee_pose": cmd_pose,
                "right.cmd_gripper": cmd_gripper,
            },
            image_data=image_data,
            embodiment="franka_right_arm",
            fps=int(info.get("fps", 30)),
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "lerobot_v2.1",
                "source_path": str(root),
                "source_episode_index": episode_index,
                "source_video_key": video_key,
                "source_second_video_key": second_video_key,
            },
        )
        LOGGER.info("Wrote %s", ep_path)
        written.append(ep_path)
    return written


def convert_human_dataset(
    root: Path,
    keypoint_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    video_key: str = FRONT_COLOR_KEY,
    keypoint_scale: float = 1.0,
    keypoint_suffix: str = ".mp4.keypoints3d.npy",
    pose_from: str = "mecka_right_hand",
) -> list[Path]:
    info = json.loads((root / "meta/info.json").read_text())
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_time = _base_time_from_dir(root)
    written = []

    for ep in episodes:
        episode_index = int(ep["episode_index"])
        df = pd.read_parquet(_parquet_path(root, info, episode_index))
        images = _read_rgb_video(
            _video_path(root, info, episode_index, video_key),
            expected_frames=len(df),
        )
        keypoint_path = keypoint_dir / f"episode_{episode_index:06d}{keypoint_suffix}"
        source_keypoints = _load_mano_keypoints(keypoint_path, len(df), keypoint_scale)
        pose, keypoints, pose_valid_frames = _keypoints_to_pose_and_flat(
            source_keypoints, pose_from
        )

        task_name, task_description = _task_for_episode(root, ep)
        episode_hash = _episode_hash(base_time, episode_index)
        ep_path = output_dir / f"{episode_hash}.zarr"
        if ep_path.exists() and not overwrite:
            raise FileExistsError(f"{ep_path} exists. Pass --overwrite to replace it.")

        ZarrWriter.create_and_write(
            episode_path=ep_path,
            numeric_data={
                "right.obs_ee_pose": pose,
                "right.obs_keypoints": keypoints,
            },
            image_data={"images.front_1": images},
            embodiment="custom_human_right_arm",
            fps=int(info.get("fps", 30)),
            task_name=task_name,
            task_description=task_description,
            metadata_override={
                "source_format": "lerobot_v2.1+mano_keypoints3d",
                "source_path": str(root),
                "source_keypoint_path": str(keypoint_path),
                "source_episode_index": episode_index,
                "source_video_key": video_key,
                "source_keypoint_order": MANO_CANONICAL_ORDER,
                "source_keypoint_points": int(source_keypoints.shape[1]),
                "keypoint_scale": float(keypoint_scale),
                "human_pose_from": pose_from,
                "human_pose_valid_frames": int(pose_valid_frames),
            },
        )
        LOGGER.info("Wrote %s", ep_path)
        written.append(ep_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot 2-camera human/Franka datasets to EgoVerse Zarr."
    )
    parser.add_argument("--mode", choices=["robot", "human", "both"], required=True)
    parser.add_argument(
        "--robot-root",
        type=Path,
        default=Path("/home/madhavan/lerobot/data/franka_2cam_stick"),
    )
    parser.add_argument(
        "--human-root",
        type=Path,
        default=Path("/home/madhavan/lerobot/data/human_mug_table_2cam_depth"),
    )
    parser.add_argument(
        "--human-keypoint-dir",
        "--hand-pose-dir",
        dest="human_keypoint_dir",
        type=Path,
        default=Path(
            "/home/madhavan/lerobot/data/human_mug_table_2cam_depth/keypoints"
        ),
    )
    parser.add_argument(
        "--robot-output-dir",
        type=Path,
        default=Path("./data/custom_franka_zarr/franka_2cam_stick"),
    )
    parser.add_argument(
        "--human-output-dir",
        type=Path,
        default=Path("./data/custom_human_azure_kinect_zarr/human_mug_table_2cam_depth"),
    )
    parser.add_argument("--video-key", type=str, default=FRONT_COLOR_KEY)
    parser.add_argument(
        "--keypoint-suffix",
        type=str,
        default=".mp4.keypoints3d.npy",
        help="Suffix after episode_XXXXXX for human keypoint sidecars.",
    )
    parser.add_argument(
        "--keypoint-scale",
        "--hand-pose-scale",
        dest="keypoint_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--human-pose-from",
        choices=["mecka_right_hand", "wrist", "centroid"],
        default="mecka_right_hand",
        help="Which point summary becomes right.obs_ee_pose xyz.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    if args.mode in ("robot", "both"):
        total += len(
            convert_robot_dataset(
                args.robot_root,
                args.robot_output_dir,
                overwrite=args.overwrite,
                video_key=args.video_key,
            )
        )
    if args.mode in ("human", "both"):
        total += len(
            convert_human_dataset(
                args.human_root,
                args.human_keypoint_dir,
                args.human_output_dir,
                overwrite=args.overwrite,
                video_key=args.video_key,
                keypoint_scale=args.keypoint_scale,
                keypoint_suffix=args.keypoint_suffix,
                pose_from=args.human_pose_from,
            )
        )
    print(f"Wrote {total} EgoVerse Zarr episodes.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
