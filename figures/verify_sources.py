"""Check archived inputs for the corresponding marks in the published figures.

The figure renderer works without these archives. Supply only authorized files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11), label)


def inspect_h5(path, d, value):
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "Install h5py==3.15.1 to check the original recording."
        ) from exc
    window = d["raw_window"]
    require(digest(path) == window["h5_sha256"], "Recording SHA-256 mismatch.")
    rank = {channel: i + 1 for i, channel in enumerate(window["roi_channels"])}
    with h5py.File(path, "r") as src:
        group = src["data_store/data0000"]
        hz = int(group["settings/sampling"][0])
        origin = int(group["groups/routed/frame_nos"][0])
        start, end = [origin + int(value(k)) * hz for k in d["window"]]
        points, events = [], []
        sites, bins = Counter(), Counter()
        for row in group["spikes"][:]:
            channel, frame = int(row["channel"]), int(row["frameno"])
            if channel in rank and start <= frame < end:
                events.append((frame, channel, float(row["amplitude"])))
                points.append([(frame - start) / hz, rank[channel]])
                sites[channel] += 1
                bins[(frame - start) // (hz // 10)] += 1
    state = json.dumps(sorted(events), separators=(",", ":"), allow_nan=False).encode()
    require(
        hashlib.sha256(state).hexdigest() == window["event_digest"],
        "Spike event digest mismatch.",
    )
    require(points == d["matrix"], "Figure 1A event coordinates differ.")
    count = len(events)
    require(
        count == int(value(d["events"])) and len(sites) == int(value(d["active"])),
        "Event count mismatch.",
    )

    def entropy(hist):
        return -math.fsum((n / count) * math.log(n / count) for n in hist.values())

    raw = [
        math.log1p(count / len(rank)),
        len(sites) / len(rank),
        1 - entropy(bins) / math.log(10),
        entropy(sites) / math.log(len(rank)),
    ]
    norm = window["normalization"]
    clip = norm["clip"]
    for observation, center, scale, key in zip(
        raw, norm["centers"], norm["scales"], d["features"], strict=True
    ):
        result = round(
            min(clip, max(-clip, (observation - center) / scale)) / clip,
            norm["decimals"],
        )
        equal(result, value(key), "Figure 1A normalized readout mismatch.")


def inspect_pairs(path, d, value):
    with path.open(encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
    columns = {"baseline": "自发组新增格", "sham": "零输出组新增格"}
    require(
        len({(r["训练种子"], r["地图种子"]) for r in rows})
        == len(rows)
        == int(value(d["pairs"])),
        "The paired source has duplicate or missing seed/map keys.",
    )
    for arm, column in columns.items():
        differences = [int(r[column]) - int(r["屏蔽组新增格"]) for r in rows]
        counts = [
            sum(x < 0 for x in differences),
            sum(x == 0 for x in differences),
            sum(x > 0 for x in differences),
        ]
        require(
            counts == [int(value(k)) for k in d["frequency"][arm]],
            "Figure 5B paired counts differ.",
        )
        row = next(r for r in d["results"] if r["arm"] == arm)
        equal(
            mean(differences), value(row["mean"]), "Figure 5A short-run mean mismatch."
        )


def inspect_logs(roots, figures, value):
    f2, f3, f6, f7 = [figures[str(i)] for i in (2, 3, 6, 7)]
    needed = {f2["decision_log"]["sha256"]}
    needed.update(t["log_sha256"] for t in f3["arms"].values())
    for files in f6["log_sources"].values():
        needed.update(files.values())
    for items in f7["log_sources"].values():
        needed.update(items)
    available = {}
    for root in roots:
        files = (
            [root]
            if root.is_file()
            else (
                p
                for pattern in ("holdout.jsonl", "train-1.jsonl")
                for p in root.rglob(pattern)
            )
        )
        for p in files:
            signature = digest(p)
            if signature in needed:
                available[signature] = p
            if len(available) == len(needed):
                break
    missing = sorted(needed - available.keys())
    require(not missing, "Missing archive files with SHA-256: " + ", ".join(missing))

    def load(signature):
        with available[signature].open(encoding="utf-8") as src:
            return [json.loads(line) for line in src]

    decision = load(f2["decision_log"]["sha256"])[f2["decision_log"]["line"] - 1]
    require(
        decision["action"]["verb"] == f2["decision_log"]["action"],
        "Figure 2 action mismatch.",
    )
    before = decision["transition"]["before"]["player"]
    after = decision["transition"]["after"]["player"]
    require(
        sum(abs(after[k] - before[k]) for k in ("x", "y")) == 1,
        "Figure 2 movement distance mismatch.",
    )
    require(
        decision["prepared"]["wetware_features"]
        == decision["prepared"]["critic_input"][-4:]
        == decision["snapshot"]["normalized"]
        == [value(k) for k in f2["features"]],
        "Figure 2 feature mismatch.",
    )
    for field, key in (
        ("seed", "map_seed"),
        ("duration", "duration_game_ticks"),
        ("reward", "native_R1"),
        ("value", "value"),
        ("advantage", "whole_batch_advantage"),
        ("return", "return_target"),
    ):
        equal(decision[field], value(f2[key]), "Figure 2 field mismatch: " + field)
    for track in f3["arms"].values():
        rows = load(track["log_sha256"])[track["line"] - 1]["rows"]
        require(
            [r["action"]["verb"] for r in rows] == track["actions"]
            and [r["after"]["tick"] for r in rows] == track["end_ticks"],
            "Figure 3A trajectory mismatch.",
        )
    score = defaultdict(dict)
    discount, award = value(f6["timing_discount"]), value(f6["timing_award"])
    for arm, files in f6["log_sources"].items():
        for seed, signature in files.items():
            games = {}
            for game in load(signature):
                clock, reward, prev = 0.0, 0.0, 0
                for row in game["rows"]:
                    after = row["after"]["tick"]
                    clock += sum(award * discount**j for j in range(after - prev))
                    prev = after
                    reward += row["r1"]
                require(game["world"] not in games, "Duplicate world in holdout.")
                games[game["world"]] = (reward, clock)
            score[arm][seed] = games
    for arm, expected in f6["raw_seed_differences"].items():
        require(
            sorted(score[arm]) == f6["seed_order"][arm],
            "Figure 6 row-to-seed binding differs.",
        )
        require(
            sum(row[0] > 0 for row in expected)
            == int(value(f6["positive_total"][arm])),
            "Figure 6 caption count differs.",
        )
        for seed, means in zip(f6["seed_order"][arm], expected, strict=True):
            a, ref = score[arm][seed], score["masked"][seed]
            require(
                a.keys() == ref.keys() and len(a) == int(value(f6["maps"])),
                "Unpaired game keys.",
            )
            total = mean(a[m][0] - ref[m][0] for m in a)
            deduction = mean(ref[m][1] - a[m][1] for m in a)
            for actual, target in zip(
                (total, deduction, total + deduction), means, strict=True
            ):
                equal(actual, target, "Figure 6 seed mark mismatch.")
    for arm, files in f7["log_sources"].items():
        counts = Counter(
            r["action"]["verb"]
            for signature in files
            for g in load(signature)
            for r in g["rows"]
        )
        require(
            sum(counts.values()) == int(value(f7["denominator"])),
            "Figure 7 total choices mismatch.",
        )
        for action, (count, pct) in f7["actions"][arm].items():
            equal(counts[action], value(count), "Figure 7 action count mismatch.")
            equal(
                100 * counts[action] / sum(counts.values()),
                value(pct),
                "Figure 7 share mismatch.",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, help="Original chip warm-up H5.")
    parser.add_argument(
        "--paired-csv", type=Path, help="Original first-organoid paired CSV."
    )
    parser.add_argument(
        "--log-root",
        action="append",
        type=Path,
        help="A log file or directory containing original holdout or training logs (repeatable).",
    )
    args = parser.parse_args()
    if not any((args.h5, args.paired_csv, args.log_root)):
        parser.error("Choose --h5, --paired-csv, or --log-root.")
    folder = Path(__file__).resolve().parent / "data"
    packets = folder / "figure_inputs.json"
    index = json.loads((folder / "source_index.json").read_bytes())
    require(
        digest(packets) == index["input_file_sha256"], "Figure input SHA-256 mismatch."
    )
    bundle = json.loads(packets.read_bytes())
    f = bundle["figures"]

    def value(key):
        return float(bundle["values"][key].replace(",", "").replace("−", "-"))

    if args.h5:
        inspect_h5(args.h5, f["1"], value)
        print(
            "Figure 1A spike raster and standardized values agree with the original H5."
        )
    if args.paired_csv:
        expected = {s["sha256"] for s in index["figures"]["5"]["sources"]}
        require(digest(args.paired_csv) in expected, "Unrecognized paired CSV.")
        inspect_pairs(args.paired_csv, f["5"], value)
        print("Figure 5 paired counts and short-run means agree with the archived CSV.")
    if args.log_root:
        inspect_logs(args.log_root, f, value)
        print(
            "Figure 2 decision, Figure 3A trajectories, Figure 6 seed means, and Figure 7 choices agree with the original logs."
        )


if __name__ == "__main__":
    main()
