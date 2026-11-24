"""Link each archived evaluation decision to a verified acquisition window."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .access import raw_windows, verify_archive
from .files import Release, csv_bytes, sha256


def checked(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(release: Release, data_dir: Path, archive: Path, dest: Path) -> dict[str, Any]:
    archive_info = verify_archive(release, archive)
    features = raw_windows(release, data_dir)
    feature_file = dest / "features.csv"
    feature_file.write_bytes(csv_bytes(features.pop("features")))
    verified = {
        int(r["diagnostic_window_id"]): r
        for r in release.rows("windows")
        if r["phase"] == "heldout"
    }
    with ZipFile(archive) as z:
        design = json.loads(z.read("stress/design.json"))
        for w in design["windows"]:
            saved = verified[w["index"]]
            checked(
                w["events_sha"] == saved["spike_event_sha256"]
                and w["normalized"] == json.loads(saved["standardized_vector"])
                and w["features"] == json.loads(saved["feature_vector"]),
                "The design's evaluation window differs from the verified HDF5 extraction.",
            )
        windows = design["windows"]
        n = len(windows)
        checked(
            n == design["horizon"] == len(verified),
            "Coverage of evaluation ticks differs from the design.",
        )
        for order in design["shuffle"].values():
            checked(sorted(order) == list(range(n)), "Archived window permutation is incomplete.")
        labels = {"ordered": "intact", "masked": "all-zero", "shuffled": "shuffled"}
        seeds = {s: i for i, s in enumerate(design["training_seeds"], 1)}
        maps = {m: i for i, m in enumerate(design["worlds"], 1)}
        with z.open("stress/episodes.csv") as stream:
            episodes = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
        tallies = []
        seen = set()
        action_count = 0
        for e in episodes:
            seed, world, scene, arm = int(e["seed"]), int(e["world"]), e["scene"], e["arm"]
            key = (seed, world, scene, arm)
            checked(key not in seen, "A seed, map, scene and arm key is duplicated.")
            seen.add(key)
            doc = json.loads(z.read(f"stress/games/{seed}/{scene}/{arm}/{world}.json"))
            g = doc["game"]
            checked(
                doc["seed"] == seed
                and doc["scene"] == scene
                and g["world"] == world
                and g["mode"] == arm
                and doc["frozen"] == design["states"][f"{seed}/{arm}"],
                "Unexpected archived model or game identity.",
            )
            order = design["shuffle"][str(world)] if arm == "shuffled" else range(n)
            first: dict[str, int | None] = {"drink": None, "eat": None}
            end = 0
            for step in g["rows"]:
                tick = step["tick"]
                checked(
                    tick == end and 0 <= tick < n, "Receipt's decision clock is not contiguous."
                )
                end = step["end_tick"]
                window = windows[order[tick]]
                values = (
                    [0.0] * len(window["normalized"]) if arm == "masked" else window["normalized"]
                )
                event = step["resource_event"]
                checked(
                    step["window"] == window["index"]
                    and step["events"] == window["events_sha"]
                    and step["features"] == values
                    and step["r2_contribution"] == 0
                    and step["transition_sha"] == event["tr_sha"]
                    and event["start"] == tick
                    and event["end"] == end
                    and tick < end <= n,
                    "A receipt's feature, transition or timing differs.",
                )
                for verb, v in first.items():
                    when = event[verb]
                    if when is not None:
                        checked(
                            tick < when <= end and step["action"]["verb"] == verb,
                            "Misaligned resource event in a step.",
                        )
                        if v is None:
                            first[verb] = when
            waits = {v: t if t is not None else n for v, t in first.items()}
            checked(
                g["rows"]
                and len(g["rows"]) == g["decisions"] == int(e["decisions"])
                and g["first_success_tick"] == first
                and g["restricted_wait"] == waits
                and end == g["end_tick"]
                and g["new_cells"] == g["rows"][-1]["new_cells_at_end"],
                "Episode summary and selected-action receipts disagree.",
            )
            outcomes = dict(
                cells=g["new_cells"],
                survived=g["survived"],
                end_tick=end,
                drank=int(first["drink"] is not None),
                ate=int(first["eat"] is not None),
                drink_wait=waits["drink"],
                eat_wait=waits["eat"],
            )
            checked(
                all(e[k] == str(v) for k, v in outcomes.items()),
                "Source endpoint CSV disagrees with the game receipt.",
            )
            tallies.append(
                dict(
                    scene=scene,
                    seed_index=seeds[seed],
                    map_index=maps[world],
                    arm=labels[arm],
                    **outcomes,
                )
            )
            action_count += g["decisions"]
    result_file = dest / "stress.csv"
    result_file.write_bytes(csv_bytes(tallies))
    checked(
        sha256(result_file) == sha256(release.locate("stress")),
        "The rebuilt endpoint table differs from the distributed table.",
    )
    return {
        "scope": "archived evaluation receipts and their inputs; no policy training or game engine run",
        "windows": features,
        "archive_sha256": archive_info["sha256"],
        "episodes": len(seen),
        "recorded_decisions": action_count,
        "stress_csv_sha256": sha256(result_file),
    }
