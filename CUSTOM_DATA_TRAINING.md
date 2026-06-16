# Training HPT on Custom Azure-Kinect Human Data and Franka Data

This guide covers the local training path for:

- `custom_human_right_arm`: right-hand pose from two Azure Kinect cameras
- `franka_right_arm`: single-arm Franka Panda demonstrations

The first supported model path is HPT flow matching with one RGB view and
cartesian end-effector actions.

## 1. Convert Raw Logs to EgoVerse Zarr

Fill in the placeholder loaders in:

```bash
egomimic/scripts/custom_data/custom_to_egoverse_zarr.py
```

The loader functions must return dictionaries that can be passed directly to
`ZarrWriter`.

Human episode output:

| Key | Shape | Dtype | Notes |
|---|---:|---|---|
| `images.front_1` | `(T, H, W, 3)` | `uint8` | RGB frames from one Azure Kinect stream |
| `right.obs_ee_pose` | `(T, 7)` | float | `[x, y, z, qw, qx, qy, qz]` |
| `right.obs_keypoints` | `(T, 63)` | float | 21 hand keypoints flattened as xyz |

Franka episode output:

| Key | Shape | Dtype | Notes |
|---|---:|---|---|
| `images.front_1` | `(T, H, W, 3)` | `uint8` | RGB frames aligned to robot state |
| `right.obs_ee_pose` | `(T, 7)` | float | observed EEF pose, `[x, y, z, qw, qx, qy, qz]` |
| `right.obs_gripper` | `(T, 1)` | float | normalized aperture, usually `[0, 1]` |
| `right.cmd_ee_pose` | `(T, 7)` | float | commanded EEF pose |
| `right.cmd_gripper` | `(T, 1)` | float | commanded gripper |

All arrays in one episode must have the same frame count. Poses should already
be expressed in one consistent task/world frame shared by human and robot data.
The training adapter converts quaternions to `[x, y, z, yaw, pitch, roll]` and
interpolates action chunks to 100 steps.

Example conversion commands:

```bash
source emimic/bin/activate

python egomimic/scripts/custom_data/custom_to_egoverse_zarr.py \
  --raw-path /path/to/raw_human_episode \
  --output-dir ./data/custom_human_azure_kinect_zarr \
  --embodiment custom_human_right_arm \
  --task-name pick_place \
  --task-description "right hand pick-place with Azure Kinect"

python egomimic/scripts/custom_data/custom_to_egoverse_zarr.py \
  --raw-path /path/to/raw_franka_episode \
  --output-dir ./data/custom_franka_zarr \
  --embodiment franka_right_arm \
  --task-name pick_place \
  --task-description "Franka right-arm pick-place"
```

For the PointPolicy human pickle format, use the dedicated converter:

```bash
python egomimic/scripts/custom_data/pointpolicy_to_egoverse_zarr.py \
  --input-path /home/madhavan/h2r/human_data/processed_data_pkl/stick_in_bowl.pkl \
  --output-dir ./data/custom_human_azure_kinect_zarr/stick_in_bowl \
  --overwrite
```

This writes one `custom_human_right_arm` Zarr episode per PointPolicy observation
clip. Since PointPolicy provides 11 tracked 3D points and no wrist orientation,
the converter derives `right.obs_ee_pose` from the point centroid with identity
orientation and stores the 11 points padded into `right.obs_keypoints`.

For local LeRobot v2.1 two-camera datasets with Franka robot episodes and human
3D keypoint sidecars, use:

```bash
python egomimic/scripts/custom_data/lerobot_2cam_to_egoverse_zarr.py \
  --mode both \
  --robot-root /home/madhavan/lerobot/data/franka_2cam_stick \
  --human-root /home/madhavan/lerobot/data/human_mug_table_2cam_depth \
  --human-keypoint-dir /home/madhavan/lerobot/data/human_mug_table_2cam_depth/keypoints \
  --robot-output-dir ./data/custom_franka_zarr/franka_2cam_stick \
  --human-output-dir ./data/custom_human_azure_kinect_zarr/human_mug_table_2cam_depth \
  --overwrite
```

This maps the LeRobot front Azure Kinect RGB video to `images.front_1`. Franka
uses `observation.right_eef_pose` and `action.right_eef_pose`, converting the
LeRobot `[rot6d, xyz, gripper]` layout into EgoVerse `xyz + quat(wxyz)` pose and
gripper keys. Human episodes use `episode_XXXXXX.mp4.keypoints3d.npy` sidecars
with shape `(T, 21, 3)`. These keypoints must already be in the canonical
MANO/EgoVerse order: `0=wrist`, `1-4=thumb`, `5-8=index`, `9-12=middle`,
`13-16=ring`, and `17-20=pinky`. The converter stores them directly in
`right.obs_keypoints`. By default, `right.obs_ee_pose` is derived with the
Mecka-style right-hand frame: palm centroid position plus orientation from the
wrist, middle-finger base, index-side base, and pinky base. This gives the HPT
cartesian action space non-constant `[x, y, z, yaw, pitch, roll]` targets.

Optional annotations may be JSON or CSV with `text`, `start_idx`, and `end_idx`.

## 2. Configure Dataset Paths

Defaults live in:

```bash
egomimic/hydra_configs/paths/default.yaml
```

Override paths inline when launching:

```bash
paths.custom_human_dataset_dir=/path/to/human_zarr \
paths.custom_franka_dataset_dir=/path/to/franka_zarr
```

Each directory should contain episode folders like:

```text
2026-06-12-15-22-01-123456.zarr/
```

The default custom data config uses `mode: total` for both train and validation
so one-episode bring-up works. After you have enough episodes, switch the
dataset entries in `egomimic/hydra_configs/data/custom_human_franka.yaml` to
`mode: train` and `mode: valid` to use the configured `valid_ratio`.

## 3. Validate a Local Batch

Use the included tests as a reference for expected post-transform shapes:

- human `actions_cartesian`: `(100, 6)`
- Franka `actions_cartesian`: `(100, 7)`
- human `right.obs_keypoints` is stored in Zarr but not consumed by the first cartesian HPT config

Compose the Hydra config before launching a real run:

```bash
source emimic/bin/activate
python egomimic/trainHydra.py --config-name=train_zarr_custom_human_franka --cfg job
```

## 4. Train

Interactive single-GPU run:

```bash
source emimic/bin/activate
python egomimic/trainHydra.py \
  --config-name=train_zarr_custom_human_franka \
  paths.custom_human_dataset_dir=/path/to/human_zarr \
  paths.custom_franka_dataset_dir=/path/to/franka_zarr \
  data.train_dataloader_params.custom_human_right_arm.batch_size=16 \
  data.train_dataloader_params.franka_right_arm.batch_size=16 \
  data.valid_dataloader_params.custom_human_right_arm.batch_size=16 \
  data.valid_dataloader_params.franka_right_arm.batch_size=16
```

For a small debugging run, use:

```bash
logger=debug trainer=debug norm_stats.sample_frac=0.1
```

On `sky1` or `sky2`, request a GPU before running or testing training:

```bash
salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G
```

## 5. Notes

- `right.obs_keypoints` is preserved for future keypoint models, but the first
  HPT config trains on EEF pose only.
- The model config uses identity camera intrinsics/extrinsics so training can
  instantiate without custom calibration. Replace these entries in
  `egomimic/utils/egomimicUtils.py` if you need accurate trajectory
  visualization over camera images.
- If your human and Franka coordinate frames differ, align them during
  conversion before writing Zarr. The custom transforms do not apply a hidden
  hand-to-robot calibration.
