#!/usr/bin/env python3
r"""Create a Zarr episode mix from explicit human and/or robot sources.

Examples:
    python scripts/create_human_zarr_mixes.py \
        --source-zarr /data/human/zarr "0-39,120-139" \
        --source-zarr /data/robot/zarr "0-199" \
        --output-folder /data/mixes/human60_robot200

    python scripts/create_human_zarr_mixes.py \
        --source-zarr /data/human/zarr all \
        --output-folder /data/mixes/all_human

Episode ranges are inclusive. The output contains symlinks, so creating a mix
does not duplicate the (usually large) Zarr stores.
"""
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
    source_root: Path
    source_path: Path
    source_number: int
    episode_index: int

    @property
    def link_name(self) -> str:
        # Prefixing prevents collisions when different sources use the same
        # episode filename. Keep .zarr at the end for dataset discovery.
        return f"source_{self.source_number:02d}__{self.source_path.name}"


def episode_index(path: Path) -> int:
    """Read an episode index without assuming a human or robot embodiment."""
    # Most EgoVerse exports include episode_N in the store name.
    matches = re.findall(r"episode[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])

    # Older exports use a bare numeric Zarr filename.
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1))

    # Human conversion outputs may retain the original episode in this attr.
    attrs = zarr.open_group(str(path), mode="r").attrs
    for key in ("episode_index", "episode_id", "source_keypoint_path"):
        value = attrs.get(key)
        if isinstance(value, int):
            return value
        if value is not None:
            match = re.search(r"episode[_-]?(\d+)", str(value), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
    raise ValueError(f"cannot determine episode index for {path}")


def parse_episode_spec(spec: str) -> set[int] | None:
    """Parse 'all' or comma-separated IDs/inclusive ranges."""
    if spec.strip().lower() == "all":
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"empty item in episode specification {spec!r}")
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise ValueError(
                f"invalid episode item {token!r}; use IDs/ranges such as 0-39,120-139"
            )
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if end < start:
            raise ValueError(f"descending episode range is not allowed: {token!r}")
        selected.update(range(start, end + 1))
    return selected


def select_episodes(root: Path, spec: str, source_number: int) -> list[Episode]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"source Zarr folder does not exist: {root}")

    discovered: list[tuple[int, Path]] = []
    for path in sorted(root.glob("*.zarr")):
        index = episode_index(path)
        # Preserve the visible filename when the source is itself a symlink mix.
        # Resolving here discards its source_NN__ prefix and can make distinct
        # mixed episodes collide when constructing the next mix's link names.
        discovered.append((index, path))
    if not discovered:
        raise FileNotFoundError(f"no .zarr episodes found directly under {root}")

    requested = parse_episode_spec(spec)
    if requested is None:
        # A source may itself be a mix. In that case episode indices legitimately
        # repeat because each original source starts numbering at zero. Selecting
        # "all" must preserve every store rather than collapsing by index.
        selected = discovered
    else:
        by_index: dict[int, Path] = {}
        for index, path in discovered:
            if index in by_index:
                raise ValueError(
                    f"source {root} contains duplicate episode index {index}, so "
                    f"selection {spec!r} is ambiguous: {by_index[index].name} and {path.name}; "
                    "use 'all' for an already-mixed source"
                )
            by_index[index] = path
        missing = sorted(requested - by_index.keys())
        if missing:
            preview = ", ".join(map(str, missing[:20]))
            suffix = " ..." if len(missing) > 20 else ""
            raise ValueError(f"source {root} is missing requested episodes: {preview}{suffix}")
        selected = [(index, by_index[index]) for index in sorted(requested)]

    return [
        Episode(root, path, source_number, index)
        for index, path in selected
    ]


def write_mix(output: Path, episodes: list[Episode], *, overwrite: bool) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        if output.is_symlink() or output.is_file():
            output.unlink()
        else:
            shutil.rmtree(output)
    output.mkdir(parents=True)

    rows = []
    for item in episodes:
        link = output / item.link_name
        link.symlink_to(item.source_path, target_is_directory=True)
        rows.append(
            {
                "source_number": item.source_number,
                "source_root": str(item.source_root),
                "episode_index": item.episode_index,
                "source": str(item.source_path),
                "link_name": item.link_name,
            }
        )

    counts: dict[str, int] = {}
    for item in episodes:
        key = str(item.source_root)
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "summary": {"total_episodes": len(episodes), "episodes_per_source": counts},
        "episodes": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {len(episodes)}-episode mix at {output}")
    for source, count in counts.items():
        print(f"  {count:6d}  {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-zarr",
        nargs=2,
        action="append",
        required=True,
        metavar=("FOLDER", "EPISODES"),
        help="source folder and episode selection (all or e.g. 0-39,120-139); repeatable",
    )
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    episodes: list[Episode] = []
    seen_roots: set[Path] = set()
    for source_number, (folder, spec) in enumerate(args.source_zarr, start=1):
        root = Path(folder).expanduser().resolve()
        if root in seen_roots:
            parser.error(f"source folder was supplied more than once: {root}")
        seen_roots.add(root)
        try:
            episodes.extend(select_episodes(root, spec, source_number))
        except (FileNotFoundError, NotADirectoryError, ValueError) as error:
            parser.error(str(error))

    try:
        write_mix(args.output_folder, episodes, overwrite=args.overwrite)
    except FileExistsError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
