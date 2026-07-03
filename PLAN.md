# Static-Camera Human–Franka Co-training Plan

## Summary

Use the fixed front camera’s optical frame as the canonical frame for both human and Franka EEF poses:

- Position in metres.
- Quaternion ordered `wxyz`.
- Camera axes follow the calibrated camera convention consistently.
- Human and robot are normalized separately and use separate action heads.
- `front_img_1` is shared; `wrist_img` exists only for Franka.
- Convert both datasets to 15 Hz and predict a 1.5-second trajectory, interpolated to the model’s 100-step action horizon.

WiLoR provides MANO hand geometry and orientation, but its weak-perspective reconstruction is not inherently metric. Metric human translation will therefore come from stereo triangulation, checked against front-camera depth, with depth used as fallback. [Official WiLoR implementation](https://github.com/rolpotamias/WiLoR).

## Data Preparation

### Canonical human episode format

Each EgoVerse Zarr episode will contain:

```text
images.front_1          uint8  (T,H,W,3), RGB
right.obs_ee_pose       float32 (T,7), [x,y,z,qw,qx,qy,qz]
```

Processing:

- Synchronize both calibrated RGB views and front depth.
- Run WiLoR on both RGB views and retain right-hand MANO-order 21-point keypoints.
- Triangulate corresponding joints into the front-camera optical frame.
- Accept triangulation when reprojection error is ≤3 pixels, both depths are positive, and the reconstructed hand scale is plausible.
- Use median-filtered front depth at the WiLoR wrist/palm projection when triangulation is invalid.
- Derive the palm EEF using EgoVerse’s existing MANO convention: palm centroid for translation and wrist/middle/index/pinky directions for orientation.
- Enforce quaternion sign continuity and reject degenerate or implausible frames.
- Interpolate gaps of at most five 15-Hz frames; reject demonstrations with longer gaps or excessive invalid tracking.
- Downsample synchronized data from its source rate to 15 Hz before export.

Human future observations serve as actions. A 23-frame future window—approximately 1.5 seconds—is interpolated to:

```text
observations.state.ee_pose  (6,)     xyz + yaw/pitch/roll
actions_cartesian           (100,6)  future xyz + yaw/pitch/roll
```

Do not synthesize a human gripper scalar; retain the existing separate 6D human head.

### Canonical Franka episode format

Convert `/home/madhavan/lerobot/data/pick_place_red_mug_100/dataset.zarr`, using:

```text
observation.images.cam_azure_kinect_front.color → images.front_1
observation.images.cam_wrist                    → images.wrist
observation.right_eef_pose                      → right.obs_ee_pose
action.right_eef_pose                           → right.cmd_ee_pose
```

The source EEF arrays are `[rot6d(6), xyz(3), gripper(1)]`. Conversion will:

- Convert rotation-6D to a proper rotation matrix and then `wxyz`.
- Apply the calibrated transform:

```text
T_camera_tcp = T_camera_base @ T_base_tcp
```

- Store camera-frame pose and gripper separately.
- Normalize gripper observation and command consistently to `[0,1]`.
- Downsample the 30-Hz source by two to 15 Hz while preserving `observation[t] ↔ action[t]` alignment.
- Ignore the unused second static camera.

Output:

```text
images.front_1          uint8   (T,H,W,3)
images.wrist            uint8   (T,H,W,3)
right.obs_ee_pose       float32 (T,7)
right.obs_gripper       float32 (T,1)
right.cmd_ee_pose       float32 (T,7)
right.cmd_gripper       float32 (T,1)
```

The dataset transform will use a 23-frame command window and produce:

```text
observations.state.ee_pose  (7,)
actions_cartesian           (100,7)
```

## Model and Training Configuration

- Add a one-front-camera co-training data config using `CustomHumanAzureKinect` and `FrankaWrist`.
- Use deterministic, disjoint 90/10 episode splits independently for human and robot; never use `mode: total` for both train and validation.
- Derive a model config from `hpt_cotrain_custom_human_franka`:
  - Shared modality: `front_img_1`.
  - Franka-only modality: `wrist_img`.
  - Shared front-camera ResNet, shared transformer trunk, and domain embeddings.
  - Separate human 6D and Franka 7D state stems and flow-matching heads.
  - Preserve the 100-step output horizon.
- Use equal per-domain batches, initially 32 human and 32 robot samples per optimizer step. `CombinedLoader(max_size_cycle)` will balance domains by cycling the smaller dataset.
- Compute quantile normalization independently per embodiment for state and action.
- Disable runtime percentile rejection with `check_bounds: false`; perform data rejection explicitly during conversion instead of silently substituting another sample.
- Keep photometric color jitter and ImageNet normalization. Do not use geometric image augmentation unless EEF coordinates and camera calibration are transformed identically.
- Request an A40 through the repository’s prescribed Slurm allocation before training.

Training sequence:

1. Load one episode per domain and verify complete batches.
2. Overfit a tiny human/robot subset to validate targets and loss wiring.
3. Run the full co-training experiment.
4. Compare against a Franka-only model using the same robot split and architecture.

## Validation and Acceptance Tests

- Project human palm and Franka TCP positions into `front_img_1`; overlays must track the visible hand/gripper.
- Verify all poses are in metres, quaternions are unit length and continuous, and rotations round-trip through `rot6d → matrix → quaternion`.
- Check stereo reprojection error, depth-versus-triangulation disagreement, invalid-frame percentage, and trajectory velocity/acceleration outliers per episode.
- Confirm every episode has synchronized array lengths and monotonic 15-Hz timestamps.
- Assert sample shapes: human `(100,6)` actions and robot `(100,7)` actions.
- Assert train/validation episode hashes are disjoint.
- Visualize unnormalized model targets after the dataset pipeline to confirm normalization is reversible.
- Require the tiny-subset run to overfit before launching full training.
- Evaluate robot validation action error in camera-frame translation, rotation, and gripper dimensions, plus task rollout success when available.

## Assumptions

- The front camera remains rigidly mounted and uses the same calibration during human and robot collection.
- `action.right_eef_pose` is the intended commanded Franka TCP target; the existing dataset records it synchronously with each observation.
- Stereo triangulation is primary for human metric position; calibrated depth is validation and fallback.
- Human orientation follows EgoVerse’s existing right-hand MANO palm-frame convention.
- Human and Franka retain separate action heads; no artificial human gripper label or shared 7D action head is introduced.
