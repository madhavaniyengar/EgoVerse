from __future__ import annotations

import numpy as np
import zarr

from egomimic.rldb.embodiment.custom import CustomHumanAzureKinect, Franka
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def _pose_sequence(num_frames: int, offset: float = 0.0) -> np.ndarray:
    xyz = np.stack(
        [
            np.linspace(0.0 + offset, 0.1 + offset, num_frames),
            np.linspace(0.2 + offset, 0.3 + offset, num_frames),
            np.linspace(0.4 + offset, 0.5 + offset, num_frames),
        ],
        axis=-1,
    )
    quat_wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_frames, 1))
    return np.concatenate([xyz, quat_wxyz], axis=-1).astype(np.float64)


def _images(num_frames: int) -> np.ndarray:
    images = np.zeros((num_frames, 16, 16, 3), dtype=np.uint8)
    images[..., 0] = 64
    images[..., 1] = 128
    images[..., 2] = 192
    return images


def _write_human_episode(root, name: str, num_frames: int = 60) -> None:
    ZarrWriter.create_and_write(
        episode_path=root / f"{name}.zarr",
        numeric_data={
            "right.obs_ee_pose": _pose_sequence(num_frames),
            "right.obs_keypoints": np.zeros((num_frames, 63), dtype=np.float64),
        },
        image_data={"images.front_1": _images(num_frames)},
        embodiment="custom_human_right_arm",
        fps=30,
        task_name="test_task",
    )


def _write_franka_episode(root, name: str, num_frames: int = 60) -> None:
    gripper = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)[:, None]
    ZarrWriter.create_and_write(
        episode_path=root / f"{name}.zarr",
        numeric_data={
            "right.obs_ee_pose": _pose_sequence(num_frames),
            "right.obs_gripper": gripper,
            "right.cmd_ee_pose": _pose_sequence(num_frames, offset=0.01),
            "right.cmd_gripper": gripper,
        },
        image_data={"images.front_1": _images(num_frames)},
        embodiment="franka_right_arm",
        fps=30,
        task_name="test_task",
    )


def _load_custom_dataset(root, embodiment_name: str, embodiment_cls) -> MultiDataset:
    resolver = LocalEpisodeResolver(
        folder_path=root,
        key_map=embodiment_cls.get_keymap(keymap_mode="cartesian"),
        transform_list=embodiment_cls.get_transform_list(mode="cartesian"),
    )
    filters = DatasetFilter(
        filter_lambdas=[f"lambda row: row.get('embodiment') == '{embodiment_name}'"]
    )
    return MultiDataset._from_resolver(resolver=resolver, filters=filters, mode="total")


def test_custom_human_and_franka_local_zarr_shapes(tmp_path) -> None:
    human_root = tmp_path / "human"
    franka_root = tmp_path / "franka"
    human_root.mkdir()
    franka_root.mkdir()

    _write_human_episode(human_root, "human_episode")
    _write_franka_episode(franka_root, "franka_episode")

    human_store = zarr.open_group(str(human_root / "human_episode.zarr"), mode="r")
    assert "right.obs_keypoints" in human_store.attrs["features"]
    assert human_store["right.obs_keypoints"].shape == (100, 63)

    human_ds = _load_custom_dataset(
        human_root, "custom_human_right_arm", CustomHumanAzureKinect
    )
    franka_ds = _load_custom_dataset(franka_root, "franka_right_arm", Franka)

    human_sample = human_ds[0]
    franka_sample = franka_ds[0]

    assert tuple(human_sample["actions_cartesian"].shape) == (100, 6)
    assert tuple(human_sample["observations.state.ee_pose"].shape) == (6,)
    assert tuple(franka_sample["actions_cartesian"].shape) == (100, 7)
    assert tuple(franka_sample["observations.state.ee_pose"].shape) == (7,)
