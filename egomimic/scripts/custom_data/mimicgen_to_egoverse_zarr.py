"""Convert paired MimicGen HDF5 data to EgoVerse Zarr format.

Each worker directory contains a `panda/` and a `sawyer/` subdirectory.
- panda/ episodes are converted as `franka_right_arm` embodiment using
  EEF pose from the 4x4 datagen_info/eef_pose matrix and gripper from
  datagen_info/gripper_action.
- sawyer/ episodes are converted as `sawyer_as_human` embodiment. Instead
  of storing raw Sawyer actions, we compute the delta DINOv2-B CLS token
  between consecutive frames as the action representation:
    action_dino[t] = dino_cls[t+1] - dino_cls[t]
  (last frame uses zeros).

Usage
-----
source emimic/bin/activate

# Convert all workers (both embodiments):
python egomimic/scripts/custom_data/mimicgen_to_egoverse_zarr.py \\
    --input-dir /scratch/madhavai/paired_pick_place_robot_transfer_100seeds_4starts/PickPlace_D0 \\
    --franka-output-dir /scratch/madhavai/mimicgen_pickplace_franka_zarr \\
    --sawyer-output-dir /scratch/madhavai/mimicgen_pickplace_sawyer_zarr \\
    --mode both

# Convert only franka:
python egomimic/scripts/custom_data/mimicgen_to_egoverse_zarr.py \\
    --input-dir ... --franka-output-dir ... --mode franka

# Convert only sawyer (requires GPU for DINOv2):
python egomimic/scripts/custom_data/mimicgen_to_egoverse_zarr.py \\
    --input-dir ... --sawyer-output-dir ... --mode sawyer
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

LOGGER = logging.getLogger(__name__)

DINO_MODEL_NAME = "dinov2_vitb14"  # 768-D CLS token
DINO_IMG_SIZE = 224  # DINOv2 expects 224x224
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]

FRANKA_EMBODIMENT = "franka_right_arm"
SAWYER_EMBODIMENT = "sawyer_as_human"

# Mimicgen stores gripper qpos in meters for Panda (~[-0.04, 0.04]).
GRIPPER_QPOS_HALF_RANGE = 0.04


# ---------------------------------------------------------------------------
# Quaternion / pose helpers
# ---------------------------------------------------------------------------


def _mat4_to_xyz_wxyz(mat4: np.ndarray) -> np.ndarray:
    """Convert (N, 4, 4) homogeneous matrices to (N, 7) xyz + quat(wxyz)."""
    xyz = mat4[:, :3, 3]
    rot_mats = mat4[:, :3, :3]
    quat_xyzw = R.from_matrix(rot_mats).as_quat()  # scipy: xyzw
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([xyz, quat_wxyz], axis=-1).astype(np.float64)


def _normalize_gripper_obs(qpos: np.ndarray) -> np.ndarray:
    """Normalize Panda gripper qpos (T, 2) to (T, 1) in [0, 1]."""
    finger = qpos[:, 0:1]  # one finger position (meters)
    normalized = (finger + GRIPPER_QPOS_HALF_RANGE) / (2 * GRIPPER_QPOS_HALF_RANGE)
    return np.clip(normalized, 0.0, 1.0).astype(np.float64)


def _parse_gripper_cmd(gripper_action: np.ndarray) -> np.ndarray:
    """Map gripper action {-1, 1} → [0, 1] for consistent normalization."""
    return ((gripper_action.astype(np.float64) + 1.0) / 2.0).clip(0.0, 1.0)


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert robosuite observation quaternions from xyzw to wxyz."""
    return quat[:, [3, 0, 1, 2]]


def _natural_demo_key(key: str) -> tuple[str, int]:
    """Sort demo_2 before demo_10 while tolerating non-standard names."""
    match = re.match(r"^(.*?)(\d+)$", key)
    return (match.group(1), int(match.group(2))) if match else (key, -1)


def _split_demo_data(
    demo: h5py.Group,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load every observation and trajectory field from a MimicGen demo."""
    numeric_data: dict[str, np.ndarray] = {}
    image_data: dict[str, np.ndarray] = {}

    for key, dataset in demo["obs"].items():
        arr = dataset[:]
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4) and arr.dtype == np.uint8:
            # Keep canonical image names used by the training keymaps while
            # recording the original HDF5 key in episode metadata.
            image_key = {
                "agentview_image": "images.front_1",
                "robot0_eye_in_hand_image": "images.wrist_1",
            }.get(key, f"images.{key}")
            image_data[image_key] = arr
        else:
            numeric_data[f"obs.{key}"] = arr

    # These are not observations, but retaining them makes the conversion
    # lossless for downstream users that need the original MimicGen signals.
    for key, dataset in demo.items():
        if isinstance(dataset, h5py.Dataset):
            numeric_data[key] = dataset[:]

    return numeric_data, image_data


def _standard_obs_pose(numeric_data: dict[str, np.ndarray]) -> np.ndarray:
    pos = numeric_data["obs.robot0_eef_pos"]
    quat_wxyz = _xyzw_to_wxyz(numeric_data["obs.robot0_eef_quat"])
    return np.concatenate([pos, quat_wxyz], axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------


def _load_dino_model(device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading %s from torch.hub ...", DINO_MODEL_NAME)
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL_NAME, pretrained=True)
    model.eval()
    model.to(device)
    LOGGER.info("DINOv2 model loaded on %s", device)
    return model


def _preprocess_images_for_dino(images: np.ndarray) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 array to (T, 3, 224, 224) float tensor."""
    import torchvision.transforms.functional as TF

    T = images.shape[0]
    tensors = []
    for i in range(T):
        img = images[i]  # (H, W, 3) uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        t = TF.resize(t, [DINO_IMG_SIZE, DINO_IMG_SIZE], antialias=True)
        t = TF.normalize(t, DINO_MEAN, DINO_STD)
        tensors.append(t)
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def _extract_dino_cls_tokens(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run DINOv2 on (T, H, W, 3) images and return (T, D) CLS tokens."""
    T = images.shape[0]
    img_tensor = _preprocess_images_for_dino(images)  # (T, 3, 224, 224)
    cls_tokens = []
    for start in range(0, T, batch_size):
        batch = img_tensor[start : start + batch_size].to(device)
        feats = model(batch)  # (B, D) CLS token
        cls_tokens.append(feats.cpu().numpy())
    return np.concatenate(cls_tokens, axis=0).astype(np.float32)  # (T, D)


def _compute_dino_delta(cls_tokens: np.ndarray) -> np.ndarray:
    """Compute delta CLS tokens: delta[t] = cls[t+1] - cls[t]; delta[T-1] = 0."""
    T = cls_tokens.shape[0]
    delta = np.zeros_like(cls_tokens)
    delta[: T - 1] = cls_tokens[1:] - cls_tokens[: T - 1]
    return delta


# ---------------------------------------------------------------------------
# Episode loading / writing
# ---------------------------------------------------------------------------


def _episode_hash(ts: datetime, worker_idx: int, file_stem: str) -> str:
    """Create a unique episode identifier from worker index and filename stem."""
    return f"{worker_idx:04d}_{file_stem}"


def _load_and_write_franka_episode(
    hdf5_path: Path,
    output_dir: Path,
    worker_idx: int,
    task_name: str = "pick_place",
    task_description: str = "MimicGen paired pick-place (Franka/Panda)",
    overwrite: bool = False,
) -> Path | None:
    """Convert one panda HDF5 episode to a franka_right_arm zarr."""
    hash_str = _episode_hash(datetime.now(timezone.utc), worker_idx, hdf5_path.stem)
    zarr_path = output_dir / f"{hash_str}.zarr"
    if zarr_path.exists() and not overwrite:
        LOGGER.info("Skipping existing %s", zarr_path)
        return zarr_path

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = list(f["data"].keys())
        if not demo_keys:
            LOGGER.warning("No demos in %s, skipping", hdf5_path)
            return None
        demo = f[f"data/{demo_keys[0]}"]

        eef_poses = demo["datagen_info/eef_pose"][:]  # (T, 4, 4)
        gripper_qpos = demo["obs/robot0_gripper_qpos"][:]  # (T, 2)
        gripper_action = demo["datagen_info/gripper_action"][:]  # (T, 1)
        images = demo["obs/agentview_image"][:]  # (T, H, W, 3)

    T = len(images)

    obs_ee_pose = _mat4_to_xyz_wxyz(eef_poses)  # (T, 7)

    # Command pose = next-frame absolute EEF pose (last frame repeats itself)
    cmd_ee_pose = np.concatenate([obs_ee_pose[1:], obs_ee_pose[-1:]], axis=0)  # (T, 7)

    obs_gripper = _normalize_gripper_obs(gripper_qpos)  # (T, 1)
    cmd_gripper = _parse_gripper_cmd(gripper_action)  # (T, 1)

    ZarrWriter.create_and_write(
        episode_path=zarr_path,
        numeric_data={
            "right.obs_ee_pose": obs_ee_pose,
            "right.obs_gripper": obs_gripper,
            "right.cmd_ee_pose": cmd_ee_pose,
            "right.cmd_gripper": cmd_gripper,
        },
        image_data={"images.front_1": images},
        embodiment=FRANKA_EMBODIMENT,
        fps=20,
        task_name=task_name,
        task_description=task_description,
        metadata_override={
            "source_format": "mimicgen_hdf5",
            "source_path": str(hdf5_path),
            "worker_idx": worker_idx,
        },
    )
    LOGGER.info("Wrote franka zarr: %s (%d frames)", zarr_path, T)
    return zarr_path


def _load_and_write_sawyer_episode(
    hdf5_path: Path,
    output_dir: Path,
    worker_idx: int,
    dino_model: torch.nn.Module,
    device: torch.device,
    task_name: str = "pick_place",
    task_description: str = "MimicGen paired pick-place (Sawyer as human, DINOv2 delta action)",
    overwrite: bool = False,
) -> Path | None:
    """Convert one sawyer HDF5 episode to a sawyer_as_human zarr."""
    hash_str = _episode_hash(datetime.now(timezone.utc), worker_idx, hdf5_path.stem)
    zarr_path = output_dir / f"{hash_str}.zarr"
    if zarr_path.exists() and not overwrite:
        LOGGER.info("Skipping existing %s", zarr_path)
        return zarr_path

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = list(f["data"].keys())
        if not demo_keys:
            LOGGER.warning("No demos in %s, skipping", hdf5_path)
            return None
        demo = f[f"data/{demo_keys[0]}"]

        eef_poses = demo["datagen_info/eef_pose"][:]  # (T, 4, 4)
        images = demo["obs/agentview_image"][:]  # (T, H, W, 3)

    T = len(images)

    obs_ee_pose = _mat4_to_xyz_wxyz(eef_poses)  # (T, 7)

    # Compute DINOv2 CLS token deltas as the "human" action
    cls_tokens = _extract_dino_cls_tokens(dino_model, images, device)  # (T, D)
    action_dino = _compute_dino_delta(cls_tokens).astype(np.float32)  # (T, D)

    ZarrWriter.create_and_write(
        episode_path=zarr_path,
        numeric_data={
            "right.obs_ee_pose": obs_ee_pose,
            "right.action_dino": action_dino,
        },
        image_data={"images.front_1": images},
        embodiment=SAWYER_EMBODIMENT,
        fps=20,
        task_name=task_name,
        task_description=task_description,
        metadata_override={
            "source_format": "mimicgen_hdf5",
            "source_path": str(hdf5_path),
            "worker_idx": worker_idx,
            "dino_model": DINO_MODEL_NAME,
            "dino_feature_dim": action_dino.shape[-1],
        },
    )
    LOGGER.info(
        "Wrote sawyer zarr: %s (%d frames, dino_dim=%d)",
        zarr_path,
        T,
        action_dino.shape[-1],
    )
    return zarr_path


# ---------------------------------------------------------------------------
# Dataset-level conversion
# ---------------------------------------------------------------------------


def convert_dataset(
    input_dir: Path,
    franka_output_dir: Path | None,
    sawyer_output_dir: Path | None,
    mode: str,
    device: torch.device,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> dict[str, list[Path]]:
    """Convert all worker directories in input_dir."""
    worker_dirs = sorted(input_dir.glob("worker_*"))
    if max_workers is not None:
        worker_dirs = worker_dirs[:max_workers]

    if not worker_dirs:
        raise RuntimeError(f"No worker_* directories found under {input_dir}")

    dino_model = None
    if mode in ("sawyer", "both"):
        if sawyer_output_dir is None:
            raise ValueError("--sawyer-output-dir required for mode 'sawyer' or 'both'")
        sawyer_output_dir.mkdir(parents=True, exist_ok=True)
        dino_model = _load_dino_model(device)

    if mode in ("franka", "both"):
        if franka_output_dir is None:
            raise ValueError("--franka-output-dir required for mode 'franka' or 'both'")
        franka_output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {"franka": [], "sawyer": []}

    for worker_idx, worker_dir in enumerate(worker_dirs):
        LOGGER.info(
            "Processing %s (worker %d/%d)",
            worker_dir.name,
            worker_idx + 1,
            len(worker_dirs),
        )

        if mode in ("franka", "both"):
            panda_dir = worker_dir / "panda"
            for hdf5_path in sorted(panda_dir.glob("*.hdf5")):
                result = _load_and_write_franka_episode(
                    hdf5_path=hdf5_path,
                    output_dir=franka_output_dir,
                    worker_idx=worker_idx,
                    overwrite=overwrite,
                )
                if result is not None:
                    written["franka"].append(result)

        if mode in ("sawyer", "both"):
            sawyer_dir = worker_dir / "sawyer"
            for hdf5_path in sorted(sawyer_dir.glob("*.hdf5")):
                result = _load_and_write_sawyer_episode(
                    hdf5_path=hdf5_path,
                    output_dir=sawyer_output_dir,
                    worker_idx=worker_idx,
                    dino_model=dino_model,
                    device=device,
                    overwrite=overwrite,
                )
                if result is not None:
                    written["sawyer"].append(result)

    return written


def convert_aggregate_hdf5(
    hdf5_path: Path,
    output_dir: Path,
    embodiment: str,
    device: torch.device,
    overwrite: bool = False,
    max_episodes: int | None = None,
    start_episode: int = 0,
) -> list[Path]:
    """Convert a Robomimic-style aggregate HDF5 file (one group per demo).

    Unlike the legacy worker-directory path above, this preserves every
    dataset under ``obs`` as well as the original actions, rewards, dones,
    and simulator states.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    is_franka = embodiment == "franka"
    dino_model = None if is_franka else _load_dino_model(device)
    written: list[Path] = []

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=_natural_demo_key)
        demo_keys = demo_keys[start_episode:]
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]

        for episode_idx, demo_key in enumerate(demo_keys):
            zarr_path = output_dir / f"{hdf5_path.stem}_{demo_key}.zarr"
            if zarr_path.exists() and not overwrite:
                LOGGER.info("Skipping existing %s", zarr_path)
                written.append(zarr_path)
                continue

            demo = f["data"][demo_key]
            numeric_data, image_data = _split_demo_data(demo)
            obs_ee_pose = _standard_obs_pose(numeric_data)
            numeric_data["right.obs_ee_pose"] = obs_ee_pose

            if is_franka:
                numeric_data["right.obs_gripper"] = _normalize_gripper_obs(
                    numeric_data["obs.robot0_gripper_qpos"]
                )
                numeric_data["right.cmd_ee_pose"] = np.concatenate(
                    [obs_ee_pose[1:], obs_ee_pose[-1:]], axis=0
                )
                numeric_data["right.cmd_gripper"] = _parse_gripper_cmd(
                    numeric_data["actions"][:, -1:]
                )
                zarr_embodiment = FRANKA_EMBODIMENT
                description = "MimicGen Square D0 (Panda/Franka)"
            else:
                images = image_data["images.front_1"]
                cls_tokens = _extract_dino_cls_tokens(dino_model, images, device)
                numeric_data["right.action_dino"] = _compute_dino_delta(cls_tokens)
                zarr_embodiment = SAWYER_EMBODIMENT
                description = (
                    "MimicGen Square D0 (Sawyer as human, DINOv2 delta action)"
                )

            partial_path = output_dir / f".{zarr_path.name}.partial"
            if partial_path.exists():
                import shutil

                shutil.rmtree(partial_path)
            ZarrWriter.create_and_write(
                episode_path=partial_path,
                numeric_data=numeric_data,
                image_data=image_data,
                embodiment=zarr_embodiment,
                fps=20,
                task_name="square_d0",
                task_description=description,
                metadata_override={
                    "source_format": "mimicgen_hdf5",
                    "source_path": str(hdf5_path),
                    "source_demo_key": demo_key,
                    "source_obs_to_zarr_images": {
                        "agentview_image": "images.front_1",
                        "robot0_eye_in_hand_image": "images.wrist_1",
                    },
                    **(
                        {}
                        if is_franka
                        else {
                            "dino_model": DINO_MODEL_NAME,
                            "dino_feature_dim": numeric_data["right.action_dino"].shape[
                                -1
                            ],
                        }
                    ),
                },
            )
            partial_path.rename(zarr_path)
            written.append(zarr_path)
            LOGGER.info(
                "Wrote %s episode %d/%d: %s (%d frames)",
                embodiment,
                episode_idx + 1,
                len(demo_keys),
                zarr_path,
                len(obs_ee_pose),
            )

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired MimicGen HDF5 data to EgoVerse Zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Root of MimicGen task dir containing worker_* subdirs (e.g. PickPlace_D0/).",
    )
    parser.add_argument(
        "--panda-hdf5",
        type=Path,
        default=None,
        help="Aggregate Panda HDF5 file containing data/demo_* groups.",
    )
    parser.add_argument(
        "--sawyer-hdf5",
        type=Path,
        default=None,
        help="Aggregate Sawyer HDF5 file containing data/demo_* groups.",
    )
    parser.add_argument(
        "--franka-output-dir",
        type=Path,
        default=None,
        help="Output directory for franka_right_arm zarr episodes.",
    )
    parser.add_argument(
        "--sawyer-output-dir",
        type=Path,
        default=None,
        help="Output directory for sawyer_as_human zarr episodes.",
    )
    parser.add_argument(
        "--mode",
        choices=["franka", "sawyer", "both"],
        default="both",
        help="Which embodiment(s) to convert.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DINOv2 inference.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Limit number of worker dirs processed (for debugging).",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Start index for aggregate HDF5 conversion (for sharding/resume).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zarr episodes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device)
    LOGGER.info("Using device: %s", device)

    if args.panda_hdf5 or args.sawyer_hdf5:
        written = {"franka": [], "sawyer": []}
        if args.mode in ("franka", "both"):
            if args.panda_hdf5 is None or args.franka_output_dir is None:
                raise ValueError("--panda-hdf5 and --franka-output-dir are required")
            written["franka"] = convert_aggregate_hdf5(
                args.panda_hdf5,
                args.franka_output_dir,
                "franka",
                device,
                args.overwrite,
                args.max_workers,
                args.start_episode,
            )
        if args.mode in ("sawyer", "both"):
            if args.sawyer_hdf5 is None or args.sawyer_output_dir is None:
                raise ValueError("--sawyer-hdf5 and --sawyer-output-dir are required")
            written["sawyer"] = convert_aggregate_hdf5(
                args.sawyer_hdf5,
                args.sawyer_output_dir,
                "sawyer",
                device,
                args.overwrite,
                args.max_workers,
                args.start_episode,
            )
    elif args.input_dir is not None:
        written = convert_dataset(
            input_dir=args.input_dir,
            franka_output_dir=args.franka_output_dir,
            sawyer_output_dir=args.sawyer_output_dir,
            mode=args.mode,
            device=device,
            overwrite=args.overwrite,
            max_workers=args.max_workers,
        )
    else:
        raise ValueError("Provide --input-dir or at least one aggregate HDF5 input")

    LOGGER.info(
        "Done. Wrote %d franka and %d sawyer episodes.",
        len(written["franka"]),
        len(written["sawyer"]),
    )


if __name__ == "__main__":
    main()
