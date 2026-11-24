"""Inspect a controlled local export by SHA-256. No network transfer occurs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any
from zipfile import ZipFile

from .files import Release, root_path, sha256


def locate_h5(release: Release, directory: Path, stress_only: bool) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError("Supply an existing local H5 directory with --data-dir.")
    wanted = {
        r["sha256"]: int(r["bytes"])
        for r in release.rows("h5_manifest")
        if not stress_only or r["stress_record"] == "True"
    }
    sizes = set(wanted.values())
    found: dict[str, Path] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"} and p.stat().st_size in sizes:
            digest = sha256(p)
            if digest in wanted:
                found.setdefault(digest, p)
    if missing := set(wanted) - set(found):
        raise ValueError(
            f"Missing {len(missing)} checksum-verified recordings. First missing digest: {min(missing)}. "
            "Use the access-request command to create an inventory for the data holder."
        )
    return found


def request(release: Release) -> dict[str, Any]:
    rows = release.rows("h5_manifest")
    return {
        "purpose_to_be_supplied_by_requester": "",
        "institution_and_contact_to_be_supplied_privately": "",
        "data_use_documentation_to_be_supplied_privately": "",
        "workflow": "See data/README.md for routing and restrictions.",
        "recordings": [r["sha256"] for r in rows if r["stress_record"] == "True"],
        "archive_sha256": release.spec["archived_model_code"]["sha256"],
    }


def verify_archive(release: Release, archive: Path) -> dict[str, Any]:
    accepted = [release.spec["archived_model_code"]]
    if "public_receipt_archive" in release.spec:
        accepted.append(release.spec["public_receipt_archive"])
    if archive.is_file():
        candidates = [info for info in accepted if info["bytes"] == archive.stat().st_size]
        if candidates:
            digest = sha256(archive)
            for info in candidates:
                if info["sha256"] == digest:
                    return info
    raise ValueError(
        "Use a training/evaluation ZIP whose size and SHA-256 match configs/study.yaml."
    )


def archive_code(release: Release, archive: Path, destination: Path) -> dict[str, Any]:
    info = verify_archive(release, archive)
    dest = destination / "archived-code"
    dest.mkdir()
    with ZipFile(archive) as z:
        for member, digest in info["members"].items():
            raw = z.read(member)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"Unexpected bytes for archived source {member}.")
            out = root_path(dest, member)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
    return {"archive_sha256": info["sha256"], "extracted": info["members"], "executed": False}


def raw_windows(release: Release, directory: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    from .files import csv_bytes

    files = locate_h5(release, directory, True)
    channels = {int(r["channel"]): int(r["electrode"]) for r in release.rows("roi")}
    windows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in release.rows("windows"):
        windows[row["raw_sha256"]].append(row)
    features = []
    cal = []
    norm = release.spec["raw_h5"]["normalization"]
    for key, lookup in sorted(windows.items()):
        with h5py.File(files[key], "r") as f:
            group = f["data_store/data0000"]
            rate = int(group["settings/sampling"][0])
            mapping = {int(r["channel"]): int(r["electrode"]) for r in group["settings/mapping"][:]}
            if any(mapping.get(c) != e for c, e in channels.items()):
                raise ValueError(f"ROI/electrode map mismatch in {key}")
            raw = group["spikes"][:]
            raw = raw[np.isin(raw["channel"], list(channels))]
            for row in lookup:
                start, end = int(row["device_first_frame"]), int(row["device_exclusive_end_frame"])
                if end - start != rate or rate % 10:
                    raise ValueError("Expected a complete one-second window.")
                selected = raw[(raw["frameno"] >= start) & (raw["frameno"] < end)]
                events = sorted(
                    (int(e["frameno"]), int(e["channel"]), float(e["amplitude"])) for e in selected
                )
                event_bytes = json.dumps(
                    events, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
                if hashlib.sha256(event_bytes).hexdigest() != row["spike_event_sha256"]:
                    raise ValueError(
                        f"Event digest mismatch for {row['phase']} window {row['diagnostic_window_id']}."
                    )
                sites = Counter(e[1] for e in events)
                times = Counter((e[0] - start) // (rate // 10) for e in events)
                n = len(events)

                def entropy(counts: Counter) -> float:
                    return -math.fsum((k / n) * math.log(k / n) for k in counts.values())

                vals = (
                    [
                        math.log1p(n / len(channels)),
                        len(sites) / len(channels),
                        min(1.0, max(0.0, 1 - entropy(times) / math.log(10))),
                        min(1.0, max(0.0, entropy(sites) / math.log(len(channels)))),
                    ]
                    if n
                    else [0.0] * 4
                )
                if not np.allclose(vals, json.loads(row["feature_vector"]), atol=1e-14, rtol=1e-12):
                    raise ValueError("Feature mismatch.")
                if row["phase"] == "calibration":
                    cal.append(vals)
                else:
                    z = [
                        round(
                            min(norm["clip"], max(-norm["clip"], (v - c) / s)) / norm["clip"],
                            norm["decimals"],
                        )
                        for v, c, s in zip(vals, norm["centers"], norm["scales"], strict=True)
                    ]
                    if z != json.loads(row["standardized_vector"]):
                        raise ValueError("Standardized feature mismatch.")
                features.append(
                    dict(
                        raw_sha256=key,
                        phase=row["phase"],
                        window=row["diagnostic_window_id"],
                        **dict(zip(norm["names"], vals, strict=True)),
                    )
                )
    centers = [median(col) for col in zip(*cal, strict=True)]
    scales = [
        max(1.4826 * median(abs(row[k] - c) for row in cal), floor)
        for k, (c, floor) in enumerate(zip(centers, norm["floors"], strict=True))
    ]
    if not np.allclose(centers, norm["centers"], atol=1e-14, rtol=1e-12) or not np.allclose(
        scales, norm["scales"], atol=1e-14, rtol=1e-12
    ):
        raise ValueError("Calibration parameters differ from the frozen context.")
    features.sort(key=lambda r: (r["phase"], int(r["window"])))
    return {
        "verified_files": len(files),
        "windows": dict(Counter(r["phase"] for r in features)),
        "intermediate_features_sha256": hashlib.sha256(csv_bytes(features)).hexdigest(),
        "features": features,
    }
