"""Validate static-camera human/Franka EgoVerse Zarr outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def validate_episode(path: Path, robot: bool) -> int:
    group = zarr.open_group(str(path), mode="r")
    total = int(group.attrs["total_frames"])
    obs = np.asarray(group["right.obs_ee_pose"][:total])
    action_key = "right.cmd_ee_pose" if robot else "right.action_ee_pose"
    action = np.asarray(group[action_key][:total])
    if obs.shape != action.shape or obs.shape[1] != 7:
        raise ValueError(f"{path}: pose shapes {obs.shape} vs {action.shape}")
    if not np.isfinite(obs).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: non-finite pose")
    if not np.allclose(np.linalg.norm(obs[:, 3:], axis=1), 1, atol=1e-3):
        raise ValueError(f"{path}: non-unit observation quaternion")
    if not np.allclose(np.linalg.norm(action[:, 3:], axis=1), 1, atol=1e-3):
        raise ValueError(f"{path}: non-unit action quaternion")
    if total > 1 and not np.allclose(action[:-1], obs[1:], atol=1e-5, rtol=0):
        error = float(np.max(np.abs(action[:-1] - obs[1:])))
        raise ValueError(f"{path}: action[t] != observation[t+1], max error={error}")
    required_images = ["images.front_1"] + (["images.wrist"] if robot else [])
    for key in required_images:
        if key not in group or group[key].shape[0] < total:
            raise ValueError(f"{path}: missing/short image key {key}")
    if robot:
        obs_gripper = np.asarray(group["right.obs_gripper"][:total])
        cmd_gripper = np.asarray(group["right.cmd_gripper"][:total])
        if total > 1 and not np.allclose(cmd_gripper[:-1], obs_gripper[1:], atol=1e-6):
            raise ValueError(f"{path}: gripper action[t] != observation[t+1]")
    return total


def validate_root(root: Path, expected: int, robot: bool) -> None:
    episodes = sorted(root.glob("*.zarr"))
    if len(episodes) != expected:
        raise ValueError(f"{root}: expected {expected} episodes, found {len(episodes)}")
    frames = sum(validate_episode(path, robot) for path in episodes)
    print(f"OK: {root}: episodes={len(episodes)}, frames={frames}, robot={robot}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-root", type=Path, required=True)
    parser.add_argument("--robot-root", type=Path, required=True)
    args = parser.parse_args()
    validate_root(args.human_root, 218, robot=False)
    validate_root(args.robot_root, 100, robot=True)


if __name__ == "__main__":
    main()
