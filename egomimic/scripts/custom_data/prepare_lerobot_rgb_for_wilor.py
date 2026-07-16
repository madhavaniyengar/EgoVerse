"""Prepare a flat, optionally subsampled RGB video directory for WiLoR."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--target-fps", type=int, required=True)
    args = parser.parse_args()

    info = json.loads((args.source / "meta/info.json").read_text())
    source_fps = int(round(float(info["fps"])))
    if source_fps % args.target_fps:
        raise ValueError(f"source FPS {source_fps} is not divisible by target FPS {args.target_fps}")
    stride = source_fps // args.target_fps
    episodes = [json.loads(line) for line in (args.source / "meta/episodes.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        index = int(episode["episode_index"])
        chunk = index // int(info.get("chunks_size", 1000))
        rel = info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=args.video_key)
        source = args.source / rel
        output = args.output / f"episode_{index:06d}.mp4"
        if output.exists():
            continue
        if stride == 1:
            os.symlink(source.resolve(), output)
        else:
            subprocess.run([
                "/usr/bin/ffmpeg", "-loglevel", "error", "-i", str(source),
                "-vf", f"select=not(mod(n\\,{stride})),setpts=N/({args.target_fps}*TB)",
                "-an", "-r", str(args.target_fps), "-fps_mode", "cfr", str(output),
            ], check=True)
        print(output)


if __name__ == "__main__":
    main()
