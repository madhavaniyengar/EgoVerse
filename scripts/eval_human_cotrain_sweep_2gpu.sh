#!/usr/bin/env bash
set -euo pipefail

# Evaluate static-camera EgoVerse checkpoints with two parallel GPU workers.
# Each policy is evaluated for ROLLOUTS rollouts across each seed in EVAL_SEEDS.
# Default: 20 rollouts x 3 seeds.

EGOVERSE_ROOT=/home/madhavan/EgoVerse
POLARIS_ROOT=/home/madhavan/polaris
SERVER_PY="$EGOVERSE_ROOT/emimic/bin/python"
EVAL_PY="$POLARIS_ROOT/.venv/bin/python"
CAMERA_FROM_BASE="$POLARIS_ROOT/PolaRiS-Hub/put_red_cup_no_curtain/cam1_from_base.txt"

# Calling the venv Python by absolute path does not prepend the venv's bin
# directory to PATH. PyTorch's C++ extension loader searches PATH for the
# ninja executable, so make the activation-equivalent PATH explicit.
export PATH="$POLARIS_ROOT/.venv/bin:$PATH"

ROLLOUTS="${ROLLOUTS:-20}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-23}"
CONTROL_FREQUENCY_HZ="${CONTROL_FREQUENCY_HZ:-15}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-180}"
EPOCH="${EPOCH:-799}"
SWEEP_NAME="${SWEEP_NAME:-egoverse_rectified200_400_sweep_${EPOCH}_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-$POLARIS_ROOT/runs/$SWEEP_NAME}"
LOG_DIR="$SWEEP_ROOT/logs"
PREVIEW_DIR="$SWEEP_ROOT/previews"

CKPT_50="$EGOVERSE_ROOT/logs/static_camera_human50_franka100_shared_head/hpt_flow_shared_head_human50_robot100_nextobs_30hz_2026-07-14_15-30-49/checkpoints/epoch_epoch=$EPOCH.ckpt"
CKPT_100="$EGOVERSE_ROOT/logs/static_camera_human100_franka100/hpt_flow_human100_robot100_nextobs_30hz_2026-07-03_17-15-59/checkpoints/epoch_epoch=$EPOCH.ckpt"
CKPT_150="$EGOVERSE_ROOT/logs/static_camera_human150_franka100_shared_head/hpt_flow_shared_head_human150_robot100_nextobs_30hz_2026-07-14_15-35-15/checkpoints/epoch_epoch=$EPOCH.ckpt"
CKPT_200="$EGOVERSE_ROOT/logs/static_camera_human200_franka100_shared_head/hpt_flow_shared_head_human200_robot100_nextobs_30hz_2026-07-14_15-36-35/checkpoints/last.ckpt"
CKPT_FILTERED_50="$EGOVERSE_ROOT/logs/static_camera_filtered_human50_franka100_shared_head/hpt_flow_shared_head_filtered_human50_robot100_nextobs_30hz_2026-07-14_15-30-40/checkpoints/epoch_epoch=$EPOCH.ckpt"
CKPT_FRANKA_30="$EGOVERSE_ROOT/logs/static_camera_franka30/hpt_flow_next_observation_30hz_45_front_wrist_2026-07-03_17-14-42/checkpoints/epoch_epoch=$EPOCH.ckpt"
HUMAN_400="/home/madhavan/EgoVerse/logs/old+new_400human_15hz/hpt_flow_human100_robot100_nextobs_15hz_2026-07-16_21-56-03/checkpoints/epoch_epoch=999.ckpt"
REC_200="/home/madhavan/EgoVerse/logs/rectified_old_15hz/hpt_flow_human100_robot100_nextobs_15hz_2026-07-16_20-52-28/checkpoints/epoch_epoch=999.ckpt"
for required in \
  "$SERVER_PY" \
  "$EVAL_PY" \
  "$CAMERA_FROM_BASE" \
  "$CKPT_50" \
  "$CKPT_100" \
  "$CKPT_150" \
  "$CKPT_200" \
  "$CKPT_FILTERED_50" \
  "$CKPT_FRANKA_30"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

# Running another Isaac evaluation or policy server on the same GPUs can cause
# severe slowdown or OOM. Override only when that sharing is intentional.
if [[ "${ALLOW_EXISTING_EVAL:-0}" != "1" ]] && \
   pgrep -f 'scripts/eval_policy.py|src/polaris/server/egoverse_server.py' >/dev/null; then
  echo "Another EgoVerse server/eval is active. Stop it first, or set ALLOW_EXISTING_EVAL=1." >&2
  pgrep -af 'scripts/eval_policy.py|src/polaris/server/egoverse_server.py' >&2 || true
  exit 1
fi

mkdir -p "$LOG_DIR" "$PREVIEW_DIR"

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local waited=0

  while (( waited < SERVER_READY_TIMEOUT )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server process $server_pid exited before opening port $port" >&2
      return 1
    fi
    if ss -ltn | grep -Eq ":${port}[[:space:]]"; then
      return 0
    fi
    sleep 1
    ((waited += 1))
  done

  echo "Timed out waiting ${SERVER_READY_TIMEOUT}s for server port $port" >&2
  return 1
}

run_eval() {
  local gpu="$1"
  local port="$2"
  local policy_label="$3"
  local checkpoint="$4"
  local seed="$5"
  local run_name="egoverse_${policy_label}_h${OPEN_LOOP_HORIZON}_r${ROLLOUTS}_seed${seed}"
  local run_folder="$SWEEP_ROOT/$run_name"
  local log_dir="$LOG_DIR"
  local server_log="$log_dir/${run_name}_server.log"
  local eval_log="$log_dir/${run_name}_eval.log"
  local preview="$PREVIEW_DIR/${run_name}_crop_preview.jpg"

  echo "[GPU $gpu] Starting $policy_label seed $seed on port $port"
  echo "[GPU $gpu] Checkpoint: $checkpoint"
  echo "[GPU $gpu] Output: $run_folder"

  CUDA_VISIBLE_DEVICES="$gpu" "$SERVER_PY" \
    "$POLARIS_ROOT/src/polaris/server/egoverse_server.py" \
    --egoverse-root "$EGOVERSE_ROOT" \
    --checkpoint "$checkpoint" \
    --camera-from-base "$CAMERA_FROM_BASE" \
    --host 127.0.0.1 \
    --port "$port" \
    --device cuda:0 \
    --image-scale-factor 2 \
    --crop-shape 360 480 \
    --wrist-crop-left 160 \
    --resampled-action-len 0 \
    --crop-viz-output "$preview" \
    >"$server_log" 2>&1 &
  local server_pid=$!

  cleanup_server() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_server EXIT

  wait_for_server "$port" "$server_pid"

  (
    cd "$POLARIS_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$EVAL_PY" scripts/eval_policy.py \
      --policy.client EgoVerse \
      --policy.host 127.0.0.1 \
      --policy.port "$port" \
      --policy.open-loop-horizon "$OPEN_LOOP_HORIZON" \
      --policy.device cuda:0 \
      --control-frequency-hz "$CONTROL_FREQUENCY_HZ" \
      --environment DROID-PutRedCup-no-curtain \
      --run-folder "$run_folder" \
      --rollouts "$ROLLOUTS" \
      --seed "$seed" \
      2>&1 | tee "$eval_log"
  )

  cleanup_server
  trap - EXIT
  echo "[GPU $gpu] Finished $policy_label seed $seed"
}

run_policy_all_seeds() {
  local gpu="$1"
  local port="$2"
  local policy_label="$3"
  local checkpoint="$4"
  local seed

  for seed in $EVAL_SEEDS; do
    run_eval "$gpu" "$port" "$policy_label" "$checkpoint" "$seed" || return 1
  done
}

gpu0_queue() {
  # run_policy_all_seeds 0 5560 human50_franka100_epoch$EPOCH "$CKPT_50" || return 1
  run_policy_all_seeds 0 5560 human400_mix_$EPOCH "$HUMAN_400" || return 1
  # run_policy_all_seeds 0 5560 human100_franka100_epoch$EPOCH "$CKPT_100" || return 1
  # run_policy_all_seeds 0 5560 filtered_human50_franka100_last "$CKPT_FILTERED_50" || return 1
}

gpu1_queue() {
  # run_policy_all_seeds 1 5561 human150_franka100_epoch$EPOCH "$CKPT_150" || return 1
  run_policy_all_seeds 1 5561 rectified_human200_$EPOCH "$REC_200" || return 1
  # run_policy_all_seeds 1 5561 human200_franka100_epoch$EPOCH "$CKPT_200" || return 1
  # run_policy_all_seeds 1 5561 franka30_epoch$EPOCH "$CKPT_FRANKA_30" || return 1
}

# A single queue can be resumed while the other GPU continues, for example:
#   ALLOW_EXISTING_EVAL=1 GPU_QUEUE=1 ./scripts/eval_human_cotrain_sweep_2gpu.sh
case "${GPU_QUEUE:-both}" in
  0)
    gpu0_queue
    exit $?
    ;;
  1)
    gpu1_queue
    exit $?
    ;;
  both) ;;
  *)
    echo "GPU_QUEUE must be 0, 1, or both" >&2
    exit 2
    ;;
esac

gpu0_queue &
gpu0_pid=$!
gpu1_queue &
gpu1_pid=$!

cleanup_workers() {
  kill "$gpu0_pid" "$gpu1_pid" 2>/dev/null || true
  wait "$gpu0_pid" "$gpu1_pid" 2>/dev/null || true
}
trap cleanup_workers INT TERM

status=0
wait "$gpu0_pid" || status=1
wait "$gpu1_pid" || status=1
trap - INT TERM

if (( status != 0 )); then
  echo "At least one GPU evaluation queue failed. Check $LOG_DIR" >&2
  exit "$status"
fi

echo "All evaluations completed successfully (${ROLLOUTS} rollouts per seed; seeds: ${EVAL_SEEDS})."
echo "Sweep output: $SWEEP_ROOT"
