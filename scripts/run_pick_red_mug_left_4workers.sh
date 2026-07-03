#!/usr/bin/env bash
set -euo pipefail

WILOR=/home/madhavan/lerobot/WiLoR
PYTHON=/home/madhavan/miniconda3/envs/wilor/bin/python
DATA=/data/madhavan/pick_red_mug_human
OUT="$DATA/wilor_task_calibration_left"
SOURCE_CAL=/home/madhavan/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json
CALDIR="$OUT/calibration"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing WiLoR Python: $PYTHON" >&2; exit 1; }
[[ -f "$SOURCE_CAL" ]] || { echo "Missing calibration: $SOURCE_CAL" >&2; exit 1; }

mkdir -p "$CALDIR" "$OUT/world" "$OUT/logs"

# The dataset's left camera is cam0 in the task-specific calibration.
jq -r '.cam0.intrinsic[] | @tsv' "$SOURCE_CAL" > "$CALDIR/left_intrinsics.txt"
jq -r '.cam0.extrinsic[] | @tsv' "$SOURCE_CAL" > "$CALDIR/left_extrinsics.txt"

cat > "$CALDIR/wilor_calibration.json" <<EOF
{
  "cam_azure_kinect_left": {
    "intrinsics": "$CALDIR/left_intrinsics.txt",
    "extrinsics": "$CALDIR/left_extrinsics.txt"
  }
}
EOF

run_range() {
  local gpu=$1 shard=$2 start=$3 end=$4 log=$5
  mkdir -p "$OUT/world/$shard"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=3 \
    "$PYTHON" "$WILOR/demo_lerobot_detectron2.py" \
      --input_folder "$DATA/$shard/videos/chunk-000" \
      --output_folder "$OUT/world/$shard" \
      --calibration_json "$CALDIR/wilor_calibration.json" \
      --cam_names cam_azure_kinect_left \
      --no_gsam2 \
      --episode_start "$start" \
      --episode_end "$end" >> "$log" 2>&1
}

# Each worker owns 35 episodes. Ranges never overlap.
(
  run_range 0 0 0 19 "$OUT/logs/worker0_gpu0.log"
  run_range 0 4 0 14 "$OUT/logs/worker0_gpu0.log"
) &
PID0=$!

(
  run_range 0 1 0 19 "$OUT/logs/worker1_gpu0.log"
  run_range 0 4 15 19 "$OUT/logs/worker1_gpu0.log"
  run_range 0 5 0 9 "$OUT/logs/worker1_gpu0.log"
) &
PID1=$!

(
  run_range 1 2 0 19 "$OUT/logs/worker2_gpu1.log"
  run_range 1 5 10 19 "$OUT/logs/worker2_gpu1.log"
  run_range 1 6 0 4 "$OUT/logs/worker2_gpu1.log"
) &
PID2=$!

(
  run_range 1 3 0 19 "$OUT/logs/worker3_gpu1.log"
  run_range 1 6 5 19 "$OUT/logs/worker3_gpu1.log"
) &
PID3=$!

echo "Workers started: GPU0=($PID0 $PID1), GPU1=($PID2 $PID3)"
echo "Progress: find '$OUT/world' -name '*.keypoints3d.npy' | wc -l"

status=0
for pid in "$PID0" "$PID1" "$PID2" "$PID3"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one worker failed. Inspect $OUT/logs; rerunning is safe." >&2
  exit "$status"
fi

COUNT=$(find "$OUT/world" -type f -name '*.keypoints3d.npy' | wc -l)
echo "All workers completed. Keypoint files: $COUNT/140"
[[ "$COUNT" -eq 140 ]] || exit 1
find "$OUT/world" -type f -name 'episode_*.mp4.npy' -delete
echo "Removed saved MANO vertex arrays; retained only EgoVerse keypoints."
