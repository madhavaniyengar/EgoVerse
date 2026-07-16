#!/usr/bin/env bash
set -euo pipefail

cd /home/madhavan/EgoVerse
source emimic/bin/activate

SOURCE=/home/madhavan/lerobot/data/human_redmug_picknplace
KEYPOINTS=/home/madhavan/lerobot/data/human_redmug_picknplace/wilor_depthanything_sam
OUTPUT="$SOURCE/egoverse_human_left_camera_additional"
VIDEO_KEY=observation.images.cam_azure_kinect_left.color

mkdir -p "$OUTPUT"

EXPECTED=$(find "$KEYPOINTS" -maxdepth 1 -type f -name 'episode_*.mp4.keypoints3d.npy' | wc -l)
SOURCE_EPISODES=$(wc -l < "$SOURCE/meta/episodes.jsonl")
[[ "$SOURCE_EPISODES" -eq "$EXPECTED" ]]

echo "Converting $EXPECTED human episodes"
python -m egomimic.scripts.custom_data.static_camera_human_to_egoverse_zarr \
  --source "$SOURCE" \
  --keypoint-dir "$KEYPOINTS" \
  --output-dir "$OUTPUT" \
  --target-fps 15 \
  --video-key "$VIDEO_KEY"

COUNT=$(find "$OUTPUT" -maxdepth 1 -type d -name "*.zarr" | wc -l)
echo "Converted EgoVerse human episodes: $COUNT/$EXPECTED"
[[ "$COUNT" -eq "$EXPECTED" ]]
