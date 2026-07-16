#!/usr/bin/env bash
set -euo pipefail

cd /home/madhavan/EgoVerse
source emimic/bin/activate

SOURCE=${SOURCE:-/home/madhavan/lerobot/extradata/human_redmug_picknplace/shard_0_pipeline_15hz/lerobot_15hz} 
WORK=${WORK:-/home/madhavan/lerobot/extradata/human_redmug_picknplace/pipeline_15hz}
OUTPUT=${OUTPUT:-$SOURCE/egoverse_human_left_camera_15hz_480x360}
CAMERA=${CAMERA:-cam_azure_kinect_left}
TARGET_FPS=${TARGET_FPS:-15}
DEPTH_MODE=${DEPTH_MODE:-sam_calibrated}
SCALE=${SCALE:-2}
CROP_HEIGHT=${CROP_HEIGHT:-360}
CROP_WIDTH=${CROP_WIDTH:-480}
STAGE=${1:-all}
VIDEO_KEY=observation.images.$CAMERA.color
DATASET=$WORK/lerobot_${TARGET_FPS}hz
RGB=$WORK/rgb/$VIDEO_KEY
RGBD=$WORK/rgbd
KEYPOINTS=$WORK/wilor
WILOR=/home/madhavan/lerobot/WiLoR
WILOR_PYTHON=${WILOR_PYTHON:-/home/madhavan/miniconda3/envs/wilor/bin/python}
INTRINSICS=${INTRINSICS:-/home/madhavan/lerobot/data/human_redmug_picknplace/wilor_depthanything_sam/${CAMERA}_intrinsics.txt}
CALIBRATION_EPISODE=${CALIBRATION_EPISODE:-0}
DEPTH_MODEL=${DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf}
DEPTH_CALIBRATION_JSON=${DEPTH_CALIBRATION_JSON:-}

run_stage() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

if run_stage prepare; then
  if [[ ! -d "$DATASET" ]]; then
    python -m egomimic.scripts.custom_data.subsample_lerobot_dataset_local \
      --source "$SOURCE" --output "$DATASET" --target-fps "$TARGET_FPS"
  elif [[ ! -f "$DATASET/.complete" ]]; then
    echo "Resuming incomplete prepared dataset: $DATASET"
    python -m egomimic.scripts.custom_data.subsample_lerobot_dataset_local \
      --source "$SOURCE" --output "$DATASET" --target-fps "$TARGET_FPS"
  else
    echo "Using existing prepared dataset: $DATASET"
  fi
  python -m egomimic.scripts.custom_data.prepare_lerobot_rgb_for_wilor \
    --source "$DATASET" --output "$RGB" --video-key "$VIDEO_KEY" --target-fps "$TARGET_FPS"
fi

if run_stage depth; then
  case "$DEPTH_MODE" in
    recorded)
      python -m egomimic.scripts.custom_data.prepare_lerobot_recorded_depth \
        --source "$DATASET" --output "$RGBD" --camera "$CAMERA" --target-fps "$TARGET_FPS"
      ;;
    depthanything)
      "$WILOR_PYTHON" "$WILOR/generate_depthanything_lerobot.py" \
        --input_folder "$WORK/rgb" --output_folder "$RGBD" --camera_name "$CAMERA" \
        --fps "$TARGET_FPS" --model "$DEPTH_MODEL"
      ;;
    calibrated_depthanything)
      [[ -f "$DEPTH_CALIBRATION_JSON" ]] || { echo "Set DEPTH_CALIBRATION_JSON to an existing SAM calibration report" >&2; exit 2; }
      "$WILOR_PYTHON" "$WILOR/generate_depthanything_lerobot.py" \
        --input_folder "$WORK/rgb" --output_folder "$RGBD" --camera_name "$CAMERA" \
        --fps "$TARGET_FPS" --model "$DEPTH_MODEL" --calibration_json "$DEPTH_CALIBRATION_JSON"
      ;;
    sam_calibrated)
      RECORDED=$WORK/recorded_rgbd
      RAW_DEPTH=$WORK/depthanything_raw
      CALIBRATION=$WORK/sam_calibration
      python -m egomimic.scripts.custom_data.prepare_lerobot_recorded_depth \
        --source "$DATASET" --output "$RECORDED" --camera "$CAMERA" --target-fps "$TARGET_FPS"
      "$WILOR_PYTHON" "$WILOR/generate_depthanything_lerobot.py" \
        --input_folder "$WORK/rgb" --output_folder "$RAW_DEPTH" --camera_name "$CAMERA" \
        --fps "$TARGET_FPS" --model "$DEPTH_MODEL"
      "$WILOR_PYTHON" "$WILOR/calibrate_depthanything_sam.py" \
        --dataset_root "$DATASET" --recorded_root "$RECORDED" --depthanything_root "$RAW_DEPTH" \
        --output_dir "$CALIBRATION" --episode "$CALIBRATION_EPISODE" --cameras "$CAMERA"
      "$WILOR_PYTHON" "$WILOR/generate_depthanything_lerobot.py" \
        --input_folder "$WORK/rgb" --output_folder "$RGBD" --camera_name "$CAMERA" \
        --fps "$TARGET_FPS" --model "$DEPTH_MODEL" \
        --calibration_json "$CALIBRATION/sam_depth_calibration.json"
      ;;
    *) echo "DEPTH_MODE must be recorded, depthanything, calibrated_depthanything, or sam_calibrated" >&2; exit 2 ;;
  esac
fi

if run_stage wilor; then
  "$WILOR_PYTHON" "$WILOR/extract_lerobot_keypoints_camera_frame.py" \
    --input_folder "$RGBD" --output_folder "$KEYPOINTS" --camera_name "$CAMERA" \
    --method depth --fps "$TARGET_FPS" --intrinsics "$INTRINSICS" --no_gsam2
fi

if run_stage zarr; then
  python -m egomimic.scripts.custom_data.static_camera_human_to_egoverse_zarr \
    --source "$DATASET" --keypoint-dir "$KEYPOINTS" --output-dir "$OUTPUT" \
    --target-fps "$TARGET_FPS" --video-key "$VIDEO_KEY" --keypoints-at-target-fps \
    --image-scale-factor "$SCALE" --crop-height "$CROP_HEIGHT" --crop-width "$CROP_WIDTH"
fi

if run_stage validate; then
  python -m egomimic.scripts.custom_data.visualize_human_egoverse_pipeline \
    --source "$DATASET" --keypoint-dir "$KEYPOINTS" --zarr-dir "$OUTPUT" \
    --video-key "$VIDEO_KEY" --episode 0 --frame 20 --output "$WORK/validation_ep000000_f000020.png"
fi
