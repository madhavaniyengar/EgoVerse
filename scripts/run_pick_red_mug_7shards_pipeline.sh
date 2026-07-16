#!/usr/bin/env bash
set -euo pipefail
cd /home/madhavan/EgoVerse

LOG_ROOT=/data/madhavan/pick_red_mug_human/pipeline_logs
mkdir -p "$LOG_ROOT"

run_shard() {
  local shard=$1 gpu=$2
  echo "Starting shard $shard on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    SOURCE="/data/madhavan/pick_red_mug_human/$shard" \
    WORK="/home/madhavan/lerobot/extradata/human_redmug_picknplace/shard_${shard}_pipeline_15hz" \
    OUTPUT="/data/madhavan/pick_red_mug_human/$shard/egoverse_human_left_camera_rgbd_15hz_480x360" \
    TARGET_FPS=15 WILOR_MODE=rgbd DEPTH_MODE=sam_calibrated \
    ./scripts/run_human_lerobot_to_egoverse_pipeline.sh all \
    >"$LOG_ROOT/shard_${shard}.log" 2>&1
  echo "Finished shard $shard"
}

run_wave() {
  local specs=("$@") pids=() failures=0
  for spec in "${specs[@]}"; do
    read -r shard gpu <<<"$spec"
    run_shard "$shard" "$gpu" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failures=$((failures+1)); done
  (( failures == 0 )) || { echo "$failures shard job(s) failed; inspect $LOG_ROOT" >&2; exit 1; }
}

run_wave "0 0" "1 0" "2 1" "3 1"
run_wave "4 0" "5 0" "6 1"
echo "All seven shards finished successfully."
