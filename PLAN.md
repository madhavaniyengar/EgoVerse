# Static-Camera Human–Franka Co-training Plan

## Final Training Contract

Train both domains at native 30 Hz as future-state prediction:

```text
action[t] = observation[t+1]
```

Use 45 future states, covering targets from `t+1` through `t+45` (1.5 seconds
at 30 Hz). Do not interpolate action chunks to 100 model steps.

The fixed left/front camera optical frame is the canonical Cartesian frame for
both human palms and Franka TCP poses. Store positions in metres and
quaternions as `wxyz`. The dataset adapters convert poses to
`xyz+yaw/pitch/roll` for the model without changing the time axis.

## Dataset Preparation

### Shared rules

- Export observations at 30 Hz (`temporal_stride: 1` for a 30 Hz source).
- Construct actions only after episode separation; never cross boundaries.
- Drop the final observation because it has no `t+1` target.
- Store explicit shifted targets and assert before writing:

```python
action_pose[t] == observation_pose[t + 1]
action_gripper[t] == observation_gripper[t + 1]
```

- Pad a 45-step chunk only at the episode tail by repeating its final valid
  action. Padding is performed by the dataset loader, not by temporal
  interpolation.
- Enforce unit quaternions and quaternion sign continuity during conversion.
- Reject invalid tracking during preprocessing; use `check_bounds: false` at
  training time.

### Human episodes

Run WiLoR on the calibrated human views and obtain metric MANO keypoints using
stereo triangulation, with calibrated depth as verification/fallback. Derive
the right palm pose using the existing EgoVerse MANO convention and express it
in the selected fixed camera optical frame.

Export:

```text
images.front_1       uint8   (T-1,H,W,3)
right.obs_ee_pose    float32 (T-1,7)  xyz+quat(wxyz)
right.action_ee_pose float32 (T-1,7)  next observed palm pose
```

Use `StaticCameraHuman30Hz`. It loads a 45-step action chunk, converts
`xyz+wxyz` to `xyz+yaw-pitch-roll`, and returns:

```text
observations.state.ee_pose (6,)
actions_cartesian          (45,6)
```

No human gripper dimension is synthesized.

Conversion and split commands:

```bash
bash scripts/convert_pick_red_mug_human_left_to_egoverse.sh

python -m egomimic.scripts.custom_data.prepare_static_human_splits \
  --source /data/madhavan/pick_red_mug_human/egoverse_human_left_30hz \
  --output /data/madhavan/pick_red_mug_human/egoverse_human_splits_30hz
```

The deterministic split contains nested 50-, 100-, 150-, and 200-episode
training sets plus 18 disjoint validation episodes:

```text
train_50 ⊂ train_100 ⊂ train_150 ⊂ train_200
```

### Franka episodes

Read the consolidated 30 Hz LeRobot Zarr. Use the calibrated left/front image
as `front_img_1` and retain the wrist camera. Transform every observed TCP pose:

```text
T_camera_tcp = T_camera_base @ T_base_tcp
```

Do not use the recorded controller action. Explicitly construct:

```text
right.cmd_ee_pose[t] = transformed_observation_ee_pose[t+1]
right.cmd_gripper[t] = observation_gripper[t+1]
```

Export:

```text
images.front_1       uint8   (T-1,H,W,3)
images.wrist         uint8   (T-1,H,W,3)
right.obs_ee_pose    float32 (T-1,7)
right.obs_gripper    float32 (T-1,1)
right.cmd_ee_pose    float32 (T-1,7)
right.cmd_gripper    float32 (T-1,1)
```

The `cmd_*` names are retained for compatibility but represent the next
observed state. Use `StaticCameraFranka30Hz`, which returns native `(45,7)`
actions containing `xyz+yaw/pitch/roll+gripper` without interpolation.

Convert with:

```bash
bash scripts/convert_pick_red_mug_franka_left_to_egoverse.sh
```

The output root is:

```text
/data/madhavan/pick_red_mug_franka_egoverse_left_30hz
```

## Model and Training

- Domains: `custom_human_right_arm`, `franka_right_arm`.
- Shared visual input and encoder: `front_img_1`.
- Franka-only visual stem and encoder: `wrist_img`.
- Shared transformer trunk with domain embeddings.
- Human head: `(45,6)`.
- Franka head: `(45,7)`.
- Trunk, flow-policy head, and CrossTransformer horizons are all 45.
- Use continuous observed gripper values; do not add binary classification or
  special loss weighting.
- Normalize each embodiment independently with quantile normalization.
- Use batch size 32 per domain initially.
- Keep photometric augmentation only. Do not apply geometric image transforms
  unless the calibration and Cartesian targets are transformed consistently.

Train after obtaining the required GPU allocation:

```bash
source emimic/bin/activate

python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human100_franka100

python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human200_franka100
```

For the robot-only baseline, use:

```bash
python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_franka30 \
  paths.custom_franka_dataset_dir=/data/madhavan/pick_red_mug_franka_egoverse_left_30hz
```

## Validation and Acceptance Tests

- Assert every converted episode reports `fps: 30`, `temporal_stride: 1`,
  `action_semantics: next_observation`, and the expected camera optical frame.
- Verify the first raw action is exactly observation 1 and that shifting never
  crosses an episode boundary.
- Assert dataset-pipeline shapes: human `(45,6)`, Franka `(45,7)`.
- Assert neither 30 Hz adapter contains `InterpolatePose` or
  `InterpolateLinear`.
- Verify the trunk horizon, policy horizon, and denoiser `act_seq` all equal 45.
- Project human palms and robot TCPs onto `front_img_1` to check calibration.
- Verify train and validation episode hashes are disjoint.
- Overfit a small two-domain subset before the full run.
- At evaluation, unnormalize the first predicted action and compare it with the
  actual next observed EEF pose in the camera frame.

## Assumptions

- Both source datasets and corresponding camera streams are synchronized at
  30 Hz.
- The fixed camera remains rigidly calibrated relative to the Franka base.
- Human and robot targets use the same selected camera optical-frame convention.
- Next-observation prediction is intentional for both embodiments.
