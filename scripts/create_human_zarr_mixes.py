#!/usr/bin/env python3
"""Create reproducible symlink-based train/valid mixes of human Zarr episodes."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import zarr


@dataclass(frozen=True)
class Episode:
    source: Path
    domain: str
    shard: int | None
    episode: int

    @property
    def link_name(self) -> str:
        prefix = "new" if self.shard is None else f"old_s{self.shard}"
        return f"{prefix}_e{self.episode:06d}_{self.source.name}"


def episode_id(path: Path) -> int:
    attrs = zarr.open_group(str(path), mode="r").attrs
    source = str(attrs.get("source_keypoint_path", ""))
    match = re.search(r"episode_(\d+)", source)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{6})\.zarr$", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot determine episode index for {path}")


def inventory_new(root: Path) -> list[Episode]:
    episodes = [Episode(p.resolve(), "new", None, episode_id(p)) for p in root.glob("*.zarr")]
    episodes.sort(key=lambda x: x.episode)
    if len({x.episode for x in episodes}) != len(episodes):
        raise ValueError("new dataset contains duplicate source episode indices")
    return episodes


def inventory_old(pattern: str) -> list[Episode]:
    episodes = []
    for shard in range(7):
        root = Path(pattern.format(shard=shard))
        shard_eps = [Episode(p.resolve(), "old", shard, episode_id(p)) for p in root.glob("*.zarr")]
        shard_eps.sort(key=lambda x: x.episode)
        episodes.extend(shard_eps)
    return episodes


def write_mix(root: Path, train: list[Episode], valid: list[Episode], *, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    rows = []
    for split, episodes in (("train", train), ("valid", valid)):
        folder = root / split
        folder.mkdir(parents=True)
        for item in episodes:
            (folder / item.link_name).symlink_to(item.source, target_is_directory=True)
            rows.append({
                "split": split,
                "domain": item.domain,
                "shard": item.shard,
                "episode": item.episode,
                "link_name": item.link_name,
                "source": str(item.source),
            })
    summary = {
        "train_episodes": len(train),
        "valid_episodes": len(valid),
        "train_new": sum(x.domain == "new" for x in train),
        "train_old": sum(x.domain == "old" for x in train),
        "valid_new": sum(x.domain == "new" for x in valid),
        "valid_old": sum(x.domain == "old" for x in valid),
    }
    (root / "manifest.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
    print(root, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("/home/madhavan/lerobot/data/human_redmug_picknplace/egoverse_human_left_camera_rgbd_15hz_480x360"),
    )
    parser.add_argument(
        "--old-root-pattern",
        default="/data/madhavan/pick_red_mug_human/{shard}/egoverse_human_left_camera_rgbd_15hz_480x360",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/madhavan/pick_red_mug_human/egoverse_human_mixes_15hz"),
    )
    parser.add_argument("--old-train-count", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    new = inventory_new(args.new_root)
    old = inventory_old(args.old_root_pattern)
    if len(new) != 200:
        raise ValueError(f"expected 200 new episodes, found {len(new)}")
    if len(old) <= args.old_train_count:
        raise ValueError(f"need more than {args.old_train_count} old episodes, found {len(old)}")
    old_train, old_valid = old[: args.old_train_count], old[args.old_train_count :]
    selected_new = [x for x in new if 0 <= x.episode <= 39 or 120 <= x.episode <= 139]
    if len(selected_new) != 60:
        raise ValueError(f"expected 62 range-selected new episodes, found {len(selected_new)}")

    write_mix(args.output_root / "mix1_old200_new200", old_train + new, old_valid, overwrite=args.overwrite)
    write_mix(args.output_root / "mix2_new200", new, old_valid, overwrite=args.overwrite)
    write_mix(args.output_root / "mix3_new_000_040_120_140", selected_new, old_valid, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
