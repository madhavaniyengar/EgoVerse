#!/usr/bin/env bash
set -euo pipefail

cd /home/madhavan/EgoVerse
source emimic/bin/activate

SOURCE=/data/madhavan/pick_red_mug_human
KEYPOINTS="$SOURCE/wilor_task_calibration_left/world"
WORLD_FROM_LEFT="$SOURCE/wilor_task_calibration_left/calibration/left_extrinsics.txt"
OUTPUT="$SOURCE/egoverse_human_left_30hz"
VIDEO_KEY=observation.images.cam_azure_kinect_left.color

mkdir -p "$OUTPUT"

for shard in {0..6}; do
  echo "Converting human shard $shard"
  python -m egomimic.scripts.custom_data.static_camera_human_to_egoverse_zarr \
    --source "$SOURCE/$shard" \
    --keypoint-dir "$KEYPOINTS/$shard" \
    --output-dir "$OUTPUT" \
    --target-fps 30 \
    --video-key "$VIDEO_KEY" \
    --world-from-camera "$WORLD_FROM_LEFT"
done

COUNT=$(find "$OUTPUT" -maxdepth 1 -type d -name "*.zarr" | wc -l)
echo "Converted EgoVerse human episodes: $COUNT/218"
[[ "$COUNT" -eq 218 ]]
