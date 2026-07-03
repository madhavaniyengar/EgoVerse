#!/usr/bin/env bash
set -euo pipefail

WILOR=/home/madhavan/lerobot/WiLoR
PYTHON=/home/madhavan/miniconda3/envs/wilor/bin/python
DATA=/data/madhavan/pick_red_mug_human
OUT="$DATA/wilor_task_calibration_left"
CAL="$OUT/calibration/wilor_calibration.json"
LOGDIR="$OUT/logs"
mkdir -p "$OUT/world/6" "$LOGDIR"

run_range() {
  local gpu=$1 start=$2 end=$3 log=$4
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=3 \
    "$PYTHON" "$WILOR/demo_lerobot_detectron2.py" \
      --input_folder "$DATA/6/videos/chunk-000" \
      --output_folder "$OUT/world/6" \
      --calibration_json "$CAL" \
      --cam_names cam_azure_kinect_left \
      --no_gsam2 \
      --episode_start "$start" \
      --episode_end "$end" >> "$log" 2>&1
}

run_range 0 20 39 "$LOGDIR/shard6_20_39_gpu0.log" & P0=$!
run_range 0 40 59 "$LOGDIR/shard6_40_59_gpu0.log" & P1=$!
run_range 1 60 79 "$LOGDIR/shard6_60_79_gpu1.log" & P2=$!
run_range 1 80 99 "$LOGDIR/shard6_80_99_gpu1.log" & P3=$!

echo "Started shard 6 remaining workers: $P0 $P1 $P2 $P3"
status=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "Worker failure; inspect $LOGDIR" >&2; exit 1; }
COUNT=$(find "$OUT/world" -type f -name "*.keypoints3d.npy" | wc -l)
echo "WiLoR keypoint files: $COUNT/220"
[[ "$COUNT" -eq 220 ]]
