import numpy as np

from egomimic.rldb.embodiment.custom import (
    StaticCameraFranka30Hz,
    StaticCameraHuman30Hz,
)


def test_native_45_step_chunk_is_not_interpolated() -> None:
    horizon = 45
    cmd_pose = np.zeros((horizon, 7), dtype=np.float32)
    cmd_pose[:, 0] = np.arange(horizon, dtype=np.float32)
    cmd_pose[:, 3] = 1.0  # identity quaternion in wxyz order
    batch = {
        "right.cmd_ee_pose": cmd_pose,
        "right.cmd_gripper": np.arange(horizon, dtype=np.float32)[:, None],
        "right.obs_ee_pose": np.array(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "right.obs_gripper": np.array([0.25], dtype=np.float32),
    }

    transforms = StaticCameraFranka30Hz.get_transform_list()
    assert not any(
        type(transform).__name__.startswith("Interpolate")
        for transform in transforms
    )
    for transform in transforms:
        batch = transform.transform(batch)

    action = batch["actions_cartesian"].numpy()
    state = batch["observations.state.ee_pose"].numpy()
    assert action.shape == (45, 7)
    assert state.shape == (7,)
    np.testing.assert_array_equal(action[:, 0], np.arange(horizon))
    np.testing.assert_array_equal(action[:, 6], np.arange(horizon))


def test_human_native_45_step_chunk_is_not_interpolated() -> None:
    horizon = 45
    action_pose = np.zeros((horizon, 7), dtype=np.float32)
    action_pose[:, 1] = np.arange(horizon, dtype=np.float32)
    action_pose[:, 3] = 1.0
    batch = {
        "right.action_ee_pose": action_pose,
        "right.obs_ee_pose": np.array(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    }

    transforms = StaticCameraHuman30Hz.get_transform_list()
    assert not any(
        type(transform).__name__.startswith("Interpolate")
        for transform in transforms
    )
    for transform in transforms:
        batch = transform.transform(batch)

    action = batch["actions_cartesian"].numpy()
    state = batch["observations.state.ee_pose"].numpy()
    assert action.shape == (45, 6)
    assert state.shape == (6,)
    np.testing.assert_array_equal(action[:, 1], np.arange(horizon))
