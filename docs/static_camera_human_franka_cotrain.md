# Static-Camera Human–Franka Co-training

This workflow prepares the `pick_red_mug` human and Franka demonstrations for
EgoVerse HPT co-training. Both domains use the Azure Kinect **left camera** as
the single shared third-person view. Robot data additionally uses its wrist
camera.

## Conventions

- Dataset camera: `cam_azure_kinect_left`
- Task calibration camera: `cam0`
- Pose frame: left-camera optical frame
- Pose storage: metres and quaternion `wxyz`
- Training rate: 30 Hz
- Raw action semantics: `action[t] = observation[t+1]`
- Model action chunk: 45 native source steps with no temporal interpolation
- Human action dimension: 6 (`xyz + yaw/pitch/roll`)
- Franka action dimension: 7 (human dimensions + gripper)

## 1. Environment

From the EgoVerse checkout:

```bash
cd /home/madhavan/EgoVerse
source emimic/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

On another checkout, replace `/home/madhavan/EgoVerse` with that checkout's
absolute path. `PYTHONPATH` is important when the shared virtual environment
has an editable installation pointing to a different checkout.

## 2. Generate Human Keypoints with WiLoR

The input human dataset is:

```text
/data/madhavan/pick_red_mug_human/{0..6}
```

The task-specific calibration is:

```text
/home/madhavan/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json
```

Run the compute-only WiLoR launcher:

```bash
cd /home/madhavan/lerobot/WiLoR
./run_pick_red_mug_left_4workers.sh
```

The launcher uses four workers, two per GPU, left RGB-D only, and
`--no_gsam2`. It processes shards 0–5 and shard 6 episodes 0–19.

Shard 6 contains 100 episodes. Process its remaining episodes 20–99:

```bash
./run_pick_red_mug_shard6_remaining_4workers.sh
```

Confirm all 220 keypoint files exist:

```bash
find /data/madhavan/pick_red_mug_human/wilor_task_calibration_left/world \
  -type f -name '*.keypoints3d.npy' | wc -l
```

Expected: `220`.

The WiLoR output is in calibration-world coordinates. The human Zarr converter
applies `inverse(T_world_from_cam0)` to express it in the left-camera frame.

## 3. Convert Human Data to EgoVerse Zarr

```bash
cd /home/madhavan/EgoVerse
./scripts/convert_pick_red_mug_human_left_to_egoverse.sh
```

Output:

```text
/data/madhavan/pick_red_mug_human/egoverse_human_left_30hz
```

Two source demonstrations contain only one frame and are intentionally
rejected because they cannot define a next-frame action:

- shard 2, episode 6
- shard 3, episode 8

Expected valid human episodes: `218`.

The Zarr `VariableLengthBytes` warning is expected for JPEG storage and is not
a conversion failure.

## 4. Convert Franka Data to EgoVerse Zarr

Source:

```text
/home/madhavan/lerobot/data/pick_place_red_mug_100/dataset.zarr
```

Convert it:

```bash
cd /home/madhavan/EgoVerse
./scripts/convert_pick_red_mug_franka_left_to_egoverse.sh
```

Output:

```text
/data/madhavan/pick_red_mug_franka_egoverse_left_30hz
```

The converter ignores recorded controller actions. It retains the observed
TCP trajectory at 30 Hz and creates exact next-observation targets.

Expected Franka episodes: `100`.

## 5. Validate Converted Data

```bash
cd /home/madhavan/EgoVerse
source emimic/bin/activate

python -m egomimic.scripts.custom_data.validate_static_cotrain_zarr \
  --human-root /data/madhavan/pick_red_mug_human/egoverse_human_left_30hz \
  --robot-root /data/madhavan/pick_red_mug_franka_egoverse_left_30hz
```

Expected output:

```text
OK: ...human...: episodes=218, ... robot=False
OK: ...franka...: episodes=100, ... robot=True
```

Validation checks finite poses, unit quaternions, required images, gripper
alignment, and `action[t] == observation[t+1]`.

## 6. Create Human-100 and Human-200 Splits

The two experiments use exact human training counts with one shared validation
set:

- Human-100: 100 training episodes
- Human-200: 200 training episodes, containing the Human-100 subset
- Human validation: 18 held-out episodes

```bash
python -m egomimic.scripts.custom_data.prepare_static_human_splits \
  --source /data/madhavan/pick_red_mug_human/egoverse_human_left_30hz \
  --output /data/madhavan/pick_red_mug_human/egoverse_human_splits_30hz
```

Verify:

```bash
find /data/madhavan/pick_red_mug_human/egoverse_human_splits_30hz/train_100 \
  -maxdepth 1 -name '*.zarr' | wc -l
find /data/madhavan/pick_red_mug_human/egoverse_human_splits_30hz/train_200 \
  -maxdepth 1 -name '*.zarr' | wc -l
find /data/madhavan/pick_red_mug_human/egoverse_human_splits_30hz/valid \
  -maxdepth 1 -name '*.zarr' | wc -l
```

Expected: `100`, `200`, and `18`.

The split directories contain absolute symlinks. Recreate them after moving
data to another machine; do not copy the symlink directories unchanged.

## 7. Training on Another Machine

Copy the converted Zarr roots, then recreate the human split links with the
paths on that machine. Update these paths in the data config if necessary:

```text
egomimic/hydra_configs/data/static_camera_human100_franka.yaml
```

Verify imports come from the intended checkout:

```bash
python -c "
import egomimic
import egomimic.utils.utils as u
import egomimic.rldb.embodiment.custom as c
print(egomimic.__file__)
print(u.__file__)
print(c.__file__)
print(c.StaticCameraHuman30Hz)
"
```

All printed source paths must belong to the same checkout.

## 8. Train Human-100 + Franka

Single GPU:

```bash
python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human100_franka100
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human100_franka100 \
  launch_params.gpus_per_node=2
```

## 9. Train Human-200 + Franka

Single GPU:

```bash
python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human200_franka100
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human200_franka100 \
  launch_params.gpus_per_node=2
```

The Franka dataset uses a deterministic 90/10 train/validation split. Both
human experiments use the same 18 human validation episodes. Numeric
validation is enabled; legacy language/camera validation-video rendering is
disabled for these static-camera experiments.

Current validation schedule is defined by the experiment configs. Confirm the
resolved values before a long run:

```bash
python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human100_franka100 \
  --cfg job | grep -A12 '^trainer:'
```

## 10. Slurm GPU Allocation

On `sky1` or `sky2`, request a GPU before training:

```bash
salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G
```

For two-GPU DDP, request two GPUs in the allocation and launch with
`launch_params.gpus_per_node=2`.

## Outputs and Resuming

Hydra writes runs under:

```text
logs/<name>/<description>_<timestamp>/
```

W&B uses project `franka`. `last.ckpt` is updated each epoch. To resume a run,
pass its checkpoint explicitly:

```bash
python egomimic/trainHydra.py \
  --config-name train_zarr_static_camera_human100_franka100 \
  ckpt_path=/absolute/path/to/checkpoints/last.ckpt
```
