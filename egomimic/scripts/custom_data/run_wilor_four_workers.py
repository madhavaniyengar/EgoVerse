"""Run four resumable WiLoR workers: two on each of GPUs 0 and 1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from egomimic.scripts.custom_data.run_wilor_static_human import transform_points

DATASET_ROOT = Path("/data/madhavan/pick_red_mug_human")
TASK_CALIBRATION = Path(
    "/home/madhavan/polaris/PolaRiS-Hub/put_red_cup_no_curtain/"
    "cam_calibration.json"
)
WILOR_ROOT = Path("/home/madhavan/lerobot/WiLoR")
WILOR_PYTHON = Path("/home/madhavan/miniconda3/envs/wilor/bin/python")
FRONT = "cam_azure_kinect_front"
LEFT = "cam_azure_kinect_left"

# Each worker receives exactly 35 of the 140 episodes. Ranges are inclusive.
WORK = {
    0: [("0", 0, 19), ("4", 0, 14)],
    1: [("1", 0, 19), ("4", 15, 19), ("5", 0, 9)],
    2: [("2", 0, 19), ("5", 10, 19), ("6", 0, 4)],
    3: [("3", 0, 19), ("6", 5, 19)],
}
WORKER_GPU = {0: 0, 1: 0, 2: 1, 3: 1}


def write_wilor_calibration(source: Path, output_dir: Path) -> Path:
    """Adapt the inline task calibration to WiLoR's file-based schema.

    Dataset camera mapping was established from exact intrinsic matches:
    front -> task cam1, left -> task cam0.
    """
    calibration = json.loads(source.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {FRONT: "cam1", LEFT: "cam0"}
    adapted = {}
    for dataset_camera, task_camera in mapping.items():
        entry = calibration[task_camera]
        stem = dataset_camera.replace("cam_azure_kinect_", "")
        intrinsics = output_dir / f"{stem}_intrinsics.txt"
        extrinsics = output_dir / f"{stem}_extrinsics.txt"
        np.savetxt(intrinsics, np.asarray(entry["intrinsic"], dtype=np.float64))
        np.savetxt(extrinsics, np.asarray(entry["extrinsic"], dtype=np.float64))
        adapted[dataset_camera] = {
            "intrinsics": str(intrinsics),
            "extrinsics": str(extrinsics),
        }
    path = output_dir / "wilor_calibration.json"
    path.write_text(json.dumps(adapted, indent=2))
    return path


def available_cameras(input_folder: Path) -> list[str]:
    cameras = []
    for camera in (FRONT, LEFT):
        color = input_folder / f"observation.images.{camera}.color"
        depth = input_folder / f"observation.images.{camera}.transformed_depth"
        if color.is_dir() and depth.is_dir():
            cameras.append(camera)
    if not cameras:
        raise FileNotFoundError(f"No complete RGB-D stream under {input_folder}")
    return cameras


def run_worker(
    worker: int,
    dataset_root: Path,
    output_root: Path,
    calibration: Path,
    no_gsam2: bool,
) -> None:
    gpu = WORKER_GPU[worker]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for shard, start, end in WORK[worker]:
        input_folder = dataset_root / shard / "videos" / "chunk-000"
        raw_output = output_root / "world" / shard
        raw_output.mkdir(parents=True, exist_ok=True)
        cameras = available_cameras(input_folder)
        command = [
            str(WILOR_PYTHON),
            str(WILOR_ROOT / "demo_lerobot_detectron2.py"),
            "--input_folder",
            str(input_folder),
            "--output_folder",
            str(raw_output),
            "--calibration_json",
            str(calibration),
            "--cam_names",
            *cameras,
            "--episode_start",
            str(start),
            "--episode_end",
            str(end),
        ]
        if no_gsam2:
            command.append("--no_gsam2")
        log_path = log_dir / f"worker{worker}_gpu{gpu}_shard{shard}_{start}_{end}.log"
        print(
            f"worker={worker} gpu={gpu} shard={shard} episodes={start}-{end} "
            f"cameras={cameras} log={log_path}",
            flush=True,
        )
        with log_path.open("a") as log:
            subprocess.run(
                command,
                cwd=WILOR_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def convert_outputs(output_root: Path, calibration: Path) -> None:
    adapted = json.loads(calibration.read_text())
    world_from_front = np.loadtxt(adapted[FRONT]["extrinsics"])
    front_from_world = np.linalg.inv(world_from_front)
    for source in sorted((output_root / "world").glob("*/*.keypoints3d.npy")):
        destination = output_root / "front_camera" / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, transform_points(np.load(source), front_from_world))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--task-calibration", type=Path, default=TASK_CALIBRATION)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATASET_ROOT / "wilor_task_calibration",
    )
    parser.add_argument(
        "--no-gsam2",
        action="store_true",
        help="Reduce VRAM use, at a potential depth-alignment quality cost.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = write_wilor_calibration(
        args.task_calibration, args.output_root / "calibration"
    )
    failures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_worker,
                worker,
                args.dataset_root,
                args.output_root,
                calibration,
                args.no_gsam2,
            ): worker
            for worker in range(4)
        }
        for future in as_completed(futures):
            worker = futures[future]
            try:
                future.result()
            except Exception as error:  # preserve other workers and report all failures
                failures.append((worker, error))
                print(f"worker {worker} failed: {error}", flush=True)
    if failures:
        raise RuntimeError(f"WiLoR worker failures: {failures}")
    convert_outputs(args.output_root, calibration)
    print(f"Completed all 140 episodes: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
