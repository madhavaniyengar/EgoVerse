from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    Transform,
    XYZWXYZ_to_XYZYPR,
)


class CustomHumanAzureKinect(Embodiment):
    """Single right-hand Azure Kinect data adapter.

    The converter is expected to write all poses in one consistent task/world
    frame. Unlike Aria data, there is no egocentric head pose transform here.
    """

    VIZ_INTRINSICS_KEY = "identity"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @classmethod
    def _get_keymap(cls, keymap_mode: Literal["cartesian"] = "cartesian"):
        if keymap_mode != "cartesian":
            raise ValueError(
                f"Unsupported keymap_mode '{keymap_mode}' for {cls.__name__}"
            )
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "right.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.obs_ee_pose",
                "horizon": 45,
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
        }

    @staticmethod
    def get_transform_list(
        mode: Literal["cartesian"] = "cartesian",
    ) -> list[Transform]:
        if mode != "cartesian":
            raise ValueError(f"Unsupported transform mode '{mode}'")
        return _build_single_arm_human_cartesian_transform_list()


class Franka(Embodiment):
    """Single right-arm Franka Panda data adapter."""

    VIZ_INTRINSICS_KEY = "identity"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @classmethod
    def _get_keymap(cls, keymap_mode: Literal["cartesian"] = "cartesian"):
        if keymap_mode != "cartesian":
            raise ValueError(
                f"Unsupported keymap_mode '{keymap_mode}' for {cls.__name__}"
            )
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.front_img_2": {
                "key_type": "camera_keys",
                "zarr_key": "images.front_2",
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_gripper",
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose",
                "horizon": 45,
            },
            "right.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_gripper",
                "horizon": 45,
            },
        }

    @staticmethod
    def get_transform_list(
        mode: Literal["cartesian"] = "cartesian",
    ) -> list[Transform]:
        if mode != "cartesian":
            raise ValueError(f"Unsupported transform mode '{mode}'")
        return _build_franka_right_arm_cartesian_transform_list()


def _build_single_arm_human_cartesian_transform_list(
    *,
    right_action_world: str = "right.action_ee_pose",
    right_obs_world: str = "right.obs_ee_pose",
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
) -> list[Transform]:
    return [
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_action_world,
            output_action_key=right_action_world,
            stride=stride,
            mode="xyzwxyz",
        ),
        XYZWXYZ_to_XYZYPR(keys=[right_action_world, right_obs_world]),
        ConcatKeys(
            key_list=[right_action_world],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[right_obs_world],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        NumpyToTensor(keys=[action_key, obs_key]),
    ]


def _build_franka_right_arm_cartesian_transform_list(
    *,
    right_cmd_world: str = "right.cmd_ee_pose",
    right_obs_world: str = "right.obs_ee_pose",
    right_cmd_gripper: str = "right.cmd_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
) -> list[Transform]:
    return [
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_world,
            output_action_key=right_cmd_world,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_cmd_gripper,
            output_action_key=right_cmd_gripper,
            stride=stride,
        ),
        XYZWXYZ_to_XYZYPR(keys=[right_cmd_world, right_obs_world]),
        ConcatKeys(
            key_list=[right_cmd_world, right_cmd_gripper],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[right_obs_world, right_obs_gripper],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        DeleteKeys(keys_to_delete=[right_cmd_gripper, right_obs_gripper]),
        NumpyToTensor(keys=[action_key, obs_key]),
    ]
