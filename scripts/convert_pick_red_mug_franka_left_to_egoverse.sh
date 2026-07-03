#!/usr/bin/env bash
set -euo pipefail

cd /home/madhavan/EgoVerse
source emimic/bin/activate

SOURCE=/home/madhavan/lerobot/data/pick_place_red_mug_100/dataset.zarr
CALDIR=/data/madhavan/pick_red_mug_human/wilor_task_calibration_left/calibration
WORLD_FROM_LEFT="$CALDIR/left_extrinsics.txt"
LEFT_FROM_BASE="$CALDIR/left_from_base.npy"
OUTPUT=/data/madhavan/pick_red_mug_franka_egoverse_left_15hz
CAMERA_KEY=observation.images.cam_azure_kinect_left.color

python -c "import numpy as np; np.save(\"$LEFT_FROM_BASE\", np.linalg.inv(np.loadtxt(\"$WORLD_FROM_LEFT\")))"

python -m egomimic.scripts.custom_data.static_camera_franka_to_egoverse_zarr \
  --source "$SOURCE" \
  --output-dir "$OUTPUT" \
  --camera-from-base "$LEFT_FROM_BASE" \
  --camera-key "$CAMERA_KEY" \
  --target-fps 15

COUNT=$(find "$OUTPUT" -maxdepth 1 -type d -name "*.zarr" | wc -l)
echo "Converted EgoVerse Franka episodes: $COUNT/100"
[[ "$COUNT" -eq 100 ]]
