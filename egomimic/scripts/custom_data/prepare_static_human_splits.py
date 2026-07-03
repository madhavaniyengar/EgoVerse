"""Create nested 100/200 human training sets and a shared validation set."""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def link_set(destination: Path, sources: list[Path]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {source.name for source in sources}
    existing = {path.name for path in destination.iterdir()}
    unexpected = existing - expected
    if unexpected:
        raise RuntimeError(f"{destination} contains unexpected entries: {sorted(unexpected)}")
    for source in sources:
        link = destination / source.name
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/madhavan/pick_red_mug_human/egoverse_human_left_15hz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/madhavan/pick_red_mug_human/egoverse_human_splits"),
    )
    args = parser.parse_args()
    episodes = sorted(args.source.glob("*.zarr"))
    if len(episodes) != 218:
        raise RuntimeError(f"Expected 218 validated episodes, found {len(episodes)}")
    random.Random(42).shuffle(episodes)
    valid = episodes[:18]
    train_200 = episodes[18:]
    train_100 = train_200[:100]
    assert len(train_100) == 100 and len(train_200) == 200 and len(valid) == 18
    assert not (set(train_200) & set(valid))
    link_set(args.output / "train_100", train_100)
    link_set(args.output / "train_200", train_200)
    link_set(args.output / "valid", valid)
    print(f"Created train_100=100, train_200=200, valid=18 under {args.output}")


if __name__ == "__main__":
    main()
