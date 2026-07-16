# Human LeRobot → EgoVerse pipeline

This is the canonical processing order:

```text
LeRobot dataset
  → temporal subsampling (only when source FPS > target FPS)
  → metric Depth Anything RGB-D
  → WiLoR 21-joint keypoints in the selected camera optical frame
  → image scale and center crop
  → EgoVerse Zarr
  → alignment validation image
```

The source dataset is never modified. The prepare stage creates a complete
standalone LeRobot dataset at `$WORK/lerobot_${TARGET_FPS}hz`; intermediate
RGB, depth, and WiLoR files also go under `WORK`.

## Defaults

```bash
cd /home/madhavan/EgoVerse

export SOURCE=/home/madhavan/lerobot/data/human_redmug_picknplace
export WORK=/home/madhavan/lerobot/extradata/human_redmug_picknplace/pipeline_15hz
export OUTPUT=$SOURCE/egoverse_human_left_camera_15hz_480x360
export CAMERA=cam_azure_kinect_left
export TARGET_FPS=15
export DEPTH_MODE=depthanything
export WILOR_MODE=rgbd
export SCALE=2
export CROP_HEIGHT=360
export CROP_WIDTH=480
export INTRINSICS=$SOURCE/wilor_depthanything_sam/cam_azure_kinect_left_intrinsics.txt
```

`INTRINSICS` must be the 3×3 calibration matrix for the camera that recorded
the videos. Do not substitute intrinsics from a different Kinect or resolution.

## Choosing the depth source

Set exactly one `DEPTH_MODE` before running the `depth` stage.
Use a distinct `WORK` directory when changing modes; depth stages are resumable
and intentionally do not overwrite existing episode files.

## WiLoR with or without external depth

The default depth-anchored mode is:

```bash
export WILOR_MODE=rgbd
```

It uses the selected `DEPTH_MODE` to anchor the hand reconstruction to measured
or generated depth.

For WiLoR-only monocular reconstruction:

```bash
export WILOR_MODE=monocular
export WORK=/path/to/a/separate/monocular_work_directory
export OUTPUT=/path/to/egoverse_monocular_zarr
scripts/run_human_lerobot_to_egoverse_pipeline.sh prepare
scripts/run_human_lerobot_to_egoverse_pipeline.sh wilor
scripts/run_human_lerobot_to_egoverse_pipeline.sh zarr
scripts/run_human_lerobot_to_egoverse_pipeline.sh validate
```

The `depth` stage becomes a no-op in monocular mode. WiLoR predicts the MANO
mesh, 21 joints, scale, and camera translation from RGB. Intrinsics are still
used for the camera projection, but no recorded depth, Depth Anything, or SAM
is used. Output filenames remain compatible with the Zarr converter.

Monocular XYZ is substantially less reliable for co-training with robot metric
poses: absolute Z and scale are inferred by the model and can drift between
frames or episodes. Zarr metadata records `keypoint_3d_source=monocular` so
these episodes can be distinguished from `rgbd` episodes. Always inspect RGB
reprojection and trajectory scale before training.

### Existing recorded depth (recommended when available)

```bash
export DEPTH_MODE=recorded
scripts/run_human_lerobot_to_egoverse_pipeline.sh prepare
scripts/run_human_lerobot_to_egoverse_pipeline.sh depth
```

The LeRobot dataset must contain
`observation.images.$CAMERA.transformed_depth`. RGB and uint16-millimetre depth
are subsampled with the same frame selection. No learned depth model is used.

### Metric Depth Anything without calibration

```bash
export DEPTH_MODE=depthanything
scripts/run_human_lerobot_to_egoverse_pipeline.sh prepare
scripts/run_human_lerobot_to_egoverse_pipeline.sh depth
```

Use this when recorded depth is unavailable. Override `DEPTH_MODEL` if needed.

### SAM-calibrated Depth Anything

```bash
export DEPTH_MODE=sam_calibrated
export CALIBRATION_EPISODE=0
scripts/run_human_lerobot_to_egoverse_pipeline.sh prepare
scripts/run_human_lerobot_to_egoverse_pipeline.sh depth
```

This mode requires recorded depth for calibration. It prepares synchronized
recorded RGB-D, generates uncalibrated Depth Anything predictions, segments the
hand with SAM/GSAM2, robustly fits predicted hand depth to recorded hand depth,
and regenerates all depth videos using the selected calibration model. The
report is saved at:

```text
$WORK/sam_calibration/sam_depth_calibration.json
```

Inspect its held-out `median_abs_m`, `p90_abs_m`, and `bias_m` before accepting
the calibration. Choose a calibration episode with clear hand visibility and
valid recorded depth. Calibration and prediction use the same target-FPS frame
stream, avoiding an easy frame-pairing error.

If a new dataset has no recorded depth, `sam_calibrated` cannot establish
metric scale by itself; use `depthanything` or provide a separate metric-depth
reference dataset captured with the same camera setup.

### Reusing an existing SAM calibration

For a new dataset recorded with the same camera, resolution, depth model, and
scene/depth configuration:

```bash
export DEPTH_MODE=calibrated_depthanything
export DEPTH_CALIBRATION_JSON=/path/to/sam_depth_calibration.json
scripts/run_human_lerobot_to_egoverse_pipeline.sh prepare
scripts/run_human_lerobot_to_egoverse_pipeline.sh depth
```

Do not reuse a report after changing the camera, resolution, Depth Anything
model, or physical depth setup without validating it against metric depth.

## Execution order

Run all stages:

```bash
scripts/run_human_lerobot_to_egoverse_pipeline.sh all
```

Or run/resume one stage at a time:

```bash
scripts/run_human_lerobot_to_egoverse_pipeline.sh prepare
scripts/run_human_lerobot_to_egoverse_pipeline.sh depth
scripts/run_human_lerobot_to_egoverse_pipeline.sh wilor
scripts/run_human_lerobot_to_egoverse_pipeline.sh zarr
scripts/run_human_lerobot_to_egoverse_pipeline.sh validate
```

Stages are resumable. The `prepare` stage selects every Nth row and video frame
when `source_fps / TARGET_FPS = N`, and rewrites all Parquet tables, timestamps,
frame/global indices, episode lengths, FPS, frame totals, RGB videos, and depth
videos. It verifies each video frame count against its episode Parquet length.
WiLoR therefore runs against the complete prepared dataset at the final rate.
The Zarr converter uses `--keypoints-at-target-fps` to prevent a second
keypoint subsampling pass.

The default spatial transform takes 1280×720 input, downsizes it by 2 to
640×360, then center-crops it to 480×360. This matches the Franka front-image
shape while retaining the center of the left-camera view. Keypoints remain in
metric camera coordinates; image resizing/cropping does not alter XYZ poses.

## Validation

The validation stage writes:

```text
$WORK/validation_ep000000_f000020.png
```

Its left panel is the source RGB frame with the 3D joints projected through
the camera intrinsics. Its right panel is the final Zarr image. Check that:

- wrist and finger joints lie on the hand;
- finger ordering is anatomically correct;
- the Zarr image has the intended center crop;
- metadata reports 15 FPS and a camera-optical pose frame.

To inspect another episode/frame:

```bash
source emimic/bin/activate
python -m egomimic.scripts.custom_data.visualize_human_egoverse_pipeline \
  --source "$SOURCE" \
  --keypoint-dir "$WORK/wilor" \
  --zarr-dir "$OUTPUT" \
  --video-key observation.images.cam_azure_kinect_left.color \
  --intrinsics "$INTRINSICS" \
  --episode 42 --frame 30 \
  --output "$WORK/validation_ep000042_f000030.png"
```

The useful WiLoR output is `*.mp4.keypoints3d.npy` with shape `(T, 21, 3)`.
The companion `*.mp4.npy` contains 778 MANO mesh vertices and is not consumed
by the Zarr converter.
