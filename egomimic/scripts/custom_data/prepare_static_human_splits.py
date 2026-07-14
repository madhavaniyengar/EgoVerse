"""Create nested 50/100/150/200 human sets and shared validation data."""
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
        default=Path("/data/madhavan/pick_red_mug_human/egoverse_human_left_30hz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/madhavan/pick_red_mug_human/egoverse_human_splits_30hz"),
    )
    args = parser.parse_args()
    episodes = sorted(args.source.glob("*.zarr"))
    if len(episodes) != 218:
        raise RuntimeError(f"Expected 218 validated episodes, found {len(episodes)}")
    random.Random(42).shuffle(episodes)
    valid = episodes[:18]
    train_200 = episodes[18:]
    train_50 = train_200[:50]
    train_100 = train_200[:100]
    train_150 = train_200[:150]
    assert (
        len(train_50) == 50
        and len(train_100) == 100
        and len(train_150) == 150
        and len(train_200) == 200
        and len(valid) == 18
    )
    assert set(train_50) < set(train_100) < set(train_150) < set(train_200)
    assert not (set(train_200) & set(valid))
    link_set(args.output / "train_50", train_50)
    link_set(args.output / "train_100", train_100)
    link_set(args.output / "train_150", train_150)
    link_set(args.output / "train_200", train_200)
    link_set(args.output / "valid", valid)
    print(
        "Created train_50=50, train_100=100, train_150=150, "
        f"train_200=200, valid=18 under {args.output}"
    )


if __name__ == "__main__":
    main()
