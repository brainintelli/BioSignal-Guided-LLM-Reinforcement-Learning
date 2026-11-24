"""Rebuild the small paired outcome table from an authorized original CSV."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def render(source: Path) -> bytes:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Source CSV is empty.")
    coords = [(int(r["训练种子"]), int(r["地图种子"])) for r in rows]
    seeds = {v: i for i, v in enumerate(sorted({s for s, _ in coords}), 1)}
    maps = {v: i for i, v in enumerate(sorted({m for _, m in coords}), 1)}
    if len(coords) != len(seeds) * len(maps) or len(set(coords)) != len(coords):
        raise ValueError("Incomplete or duplicate seed-map pairs in the original table.")
    converted = []
    for r, (s, m) in zip(rows, coords, strict=True):
        values = {"seed_index": seeds[s], "map_index": maps[m]}
        for label, arm in (("屏蔽", "all_zero"), ("自发", "intact"), ("零输出", "unstimulated")):
            for time, field in (("d6", "新增格"), ("t6", "第6刻新增格")):
                values[arm + "_" + time] = int(r[label + "组" + field])
        converted.append(values)
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, list(converted[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(converted, key=lambda r: (r["seed_index"], r["map_index"])))
    return buf.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="Path for a new output file")
    args = parser.parse_args()
    try:
        data = render(args.source)
        want = json.loads(
            (Path(__file__).resolve().parents[1] / "data/input_hashes.json").read_bytes()
        )["data/exploration.csv"]
        if hashlib.sha256(data).hexdigest() != want:
            raise ValueError("The recalculated CSV differs from the released paired data.")
        args.output.parent.mkdir(exist_ok=True, parents=True)
        with args.output.open("xb") as stream:
            stream.write(data)
        print(json.dumps({"sha256": want, "file": str(args.output)}))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Pair packing failed: {exc}\n")


if __name__ == "__main__":
    main()
