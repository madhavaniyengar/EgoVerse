from __future__ import annotations

import numpy as np
import zarr

from egomimic.rldb.embodiment.custom import (
    StaticCameraFranka15Hz,
    StaticCameraHuman15Hz,
)
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.custom_data.run_wilor_static_human import transform_points
from egomimic.scripts.custom_data.static_camera_franka_to_egoverse_zarr import (
    next_observation_pairs,
    transform_wxyz_poses,
)
from egomimic.scripts.serve_egoverse_policy import _transform_xyzypr


def _poses(n: int) -> np.ndarray:
    pose = np.zeros((n, 7), dtype=np.float64)
    pose[:, 0] = np.arange(n)
    pose[:, 3] = 1.0
    return pose


def test_next_observation_pairs_downsample_before_shift() -> None:
    poses = _poses(8)
    gripper = np.arange(8, dtype=np.float64)[:, None]
    indices, obs, obs_grip, action, action_grip = next_observation_pairs(
        poses, gripper, stride=2
    )

    np.testing.assert_array_equal(indices, [0, 2, 4])
    np.testing.assert_array_equal(obs[:, 0], [0, 2, 4])
    np.testing.assert_array_equal(action[:, 0], [2, 4, 6])
    np.testing.assert_array_equal(obs_grip[:, 0], [0, 2, 4])
    np.testing.assert_array_equal(action_grip[:, 0], [2, 4, 6])
    assert len(obs) == 3  # final downsampled observation was dropped


def test_pose_transform_applies_camera_from_base_and_preserves_unit_quaternion() -> None:
    camera_from_base = np.eye(4)
    camera_from_base[:3, 3] = [1.0, 2.0, 3.0]
    transformed = transform_wxyz_poses(_poses(3), camera_from_base)

    np.testing.assert_allclose(transformed[:, :3], [[1, 2, 3], [2, 2, 3], [3, 2, 3]])
    np.testing.assert_allclose(np.linalg.norm(transformed[:, 3:], axis=1), 1.0)


def test_human_dataset_first_action_is_next_observation(tmp_path) -> None:
    # Exported arrays have equal length; action[i] is the pre-drop pose[i+1].
    obs = _poses(30)
    action = _poses(30)
    action[:, 0] += 1.0
    images = np.zeros((30, 8, 8, 3), dtype=np.uint8)
    ZarrWriter.create_and_write(
        episode_path=tmp_path / "human.zarr",
        numeric_data={
            "right.obs_ee_pose": obs,
            "right.action_ee_pose": action,
        },
        image_data={"images.front_1": images},
        embodiment="custom_human_right_arm",
        fps=15,
        task_name="test",
    )
    store = zarr.open_group(str(tmp_path / "human.zarr"), mode="r")
    np.testing.assert_allclose(
        store["right.action_ee_pose"][:30, :3],
        store["right.obs_ee_pose"][:30, :3] + [1, 0, 0],
    )

    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=StaticCameraHuman15Hz.get_keymap("cartesian"),
        transform_list=StaticCameraHuman15Hz.get_transform_list("cartesian"),
    )
    dataset = MultiDataset._from_resolver(resolver=resolver, mode="total")
    sample = dataset[0]
    assert sample["actions_cartesian"].shape == (100, 6)
    assert sample["observations.state.ee_pose"].shape == (6,)
    assert np.isclose(sample["actions_cartesian"][0, 0].item(), 1.0)
    assert np.isclose(sample["observations.state.ee_pose"][0].item(), 0.0)


def test_static_franka_uses_one_front_camera_and_wrist() -> None:
    keymap = StaticCameraFranka15Hz.get_keymap("cartesian")
    assert "observations.images.front_img_1" in keymap
    assert "observations.images.front_img_2" not in keymap
    assert "observations.images.wrist_img" in keymap
    assert keymap["right.cmd_ee_pose"]["horizon"] == 23
    assert keymap["right.cmd_gripper"]["horizon"] == 23


def test_wilor_world_points_transform_to_front_and_preserve_missing_frames() -> None:
    points = np.zeros((2, 21, 3), dtype=np.float32)
    points[0, :, 0] = 1.0
    front_from_world = np.eye(4)
    front_from_world[:3, 3] = [0.0, 2.0, 3.0]
    result = transform_points(points, front_from_world)

    np.testing.assert_allclose(result[0, :, 0], 1.0)
    np.testing.assert_allclose(result[0, :, 1], 2.0)
    np.testing.assert_allclose(result[0, :, 2], 3.0)
    np.testing.assert_array_equal(result[1], 0.0)


def test_eval_pose_frame_transform_round_trip() -> None:
    camera_from_base = np.array(
        [
            [0.0, -1.0, 0.0, 0.2],
            [1.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    base_poses = np.array(
        [
            [0.5, -0.2, 0.3, 0.1, -0.2, 0.3, 0.0],
            [0.6, 0.1, 0.4, -0.3, 0.15, -0.1, 1.0],
        ],
        dtype=np.float32,
    )
    camera_poses = _transform_xyzypr(base_poses, camera_from_base)
    recovered = _transform_xyzypr(camera_poses, np.linalg.inv(camera_from_base))
    np.testing.assert_allclose(recovered, base_poses, atol=1e-5)
