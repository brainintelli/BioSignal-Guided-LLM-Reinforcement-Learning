"""Run a checksum-verified offline analysis and save its inputs and outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from . import access, source_checks
from .exploration import exploration
from .files import Release, csv_bytes, csv_rows, encoded, read_json, sha256

NUMERICAL = (
    "exploration",
    "exploration-intervals",
    "score",
    "stress",
    "stimulation",
    "cross-chip",
    "heldout",
    "boundary",
    "budgets",
    "actions",
)
COMMANDS = (
    "smoke",
    "all",
    *NUMERICAL,
    "deviations",
    "access-request",
    "verify-h5",
    "raw-windows",
    "audit-chain",
    "archived-code",
)


def matches(actual: Any, expected: Any, label: str = "result") -> None:
    if isinstance(expected, dict) and isinstance(actual, dict) and actual.keys() == expected.keys():
        for key in expected:
            matches(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            matches(a, b, f"{label}[{i}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if (
            not isinstance(actual, (float, int))
            or isinstance(actual, bool)
            or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11)
        ):
            raise ValueError(
                f"Archived numerical mismatch at {label}: measured {actual!r}, expected {expected!r}."
            )
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f"Result differs from the archived reference at {label}.")


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "h5py"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "packages": versions,
    }


def commit(root: Path) -> str | None:
    if not shutil.which("git"):
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != root.resolve():
        return None
    rev = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        encoding="utf-8",
    )
    return rev.stdout.strip() if not rev.returncode else None


def analysis(release: Release, command: str) -> Any:
    result: Any
    if command in ("smoke", "exploration"):
        result = exploration(release)
        matches(result, read_json(release.root / "results/exploration.json"))
        if command == "smoke":
            arm = result["d6"]["intact"]
            return {
                "scope": "first 22021 recording; six decisions; intact minus all-zero",
                "pairs": arm["pairs"],
                "mean_difference": arm["mean_difference"],
                "wins": arm["wins"],
                "ties": arm["ties"],
                "losses": arm["losses"],
            }
        return result
    if command == "deviations":
        return {
            "scope": "retrieval of archived integrity summary",
            "record": read_json(release.locate("deviations")),
        }
    if command == "access-request":
        return access.request(release)
    if command in {"boundary", "budgets", "actions"}:
        return {
            "boundary": source_checks.boundary,
            "budgets": source_checks.budgets,
            "actions": source_checks.actions,
        }[command](release)
    from . import intervals

    routines = {
        "exploration-intervals": intervals.exploration_intervals,
        "score": intervals.score,
        "stress": intervals.stress,
        "stimulation": intervals.stimulation,
        "cross-chip": intervals.cross_chip,
        "heldout": intervals.heldout,
    }
    refs = {"cross-chip": "first-day-22554", "heldout": "heldout"}
    result = routines[command](release)
    if command == "heldout":
        archived = csv_rows(release.root / "results/heldout.csv")
        # The published table is keyed by endpoint and budget, independent of display order.
        expected = {(r["tick"], r["endpoint"]): r for r in archived}
        if len(archived) != len(result) or len(expected) != len(archived):
            raise ValueError("Incomplete held-out results.")
        for actual in result:
            saved = expected[str(actual["tick"]), actual["endpoint"]]
            matches(actual, {k: str(v) if k == "endpoint" else float(v) for k, v in saved.items()})
    else:
        matches(
            result, read_json(release.root / "results" / (refs.get(command, command) + ".json"))
        )
    return result


def run(args: argparse.Namespace) -> tuple[Path, Any]:
    release = Release(config=args.config)
    release.check_inputs()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    name = started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    target = output_root / name
    with tempfile.TemporaryDirectory(dir=output_root, prefix=".partial-") as tmp:
        staging = Path(tmp)
        payload = staging / "run"
        payload.mkdir()
        if args.command == "raw-windows":
            result = access.raw_windows(release, args.data_dir)
            (payload / "features.csv").write_bytes(csv_bytes(result.pop("features")))
        elif args.command == "audit-chain":
            from .receipt_chain import audit

            result = audit(release, args.data_dir, args.archive, payload)
            result["results"] = analysis(release, "stress")
        elif args.command == "verify-h5":
            files = access.locate_h5(release, args.data_dir, args.stress_only)
            result = {"verified_file_count": len(files), "sha256": sorted(files)}
        elif args.command == "archived-code":
            result = access.archive_code(release, args.archive, payload)
        elif args.command == "all":
            result = {cmd: analysis(release, cmd) for cmd in (*NUMERICAL, "deviations")}
        else:
            result = analysis(release, args.command)
        (payload / "result.json").write_bytes(encoded(result))
        shutil.copyfile(release.config_path, payload / "study.yaml")
        env = environment()
        pinned = {
            "exploration-intervals": {
                "supplement": release.spec["short"]["bootstrap_seed"],
                "archive_tick": release.spec["short"]["seed_tick_archived"],
                "archive_decisions": release.spec["short"]["seed_decisions_archived"],
            },
            "stress": release.spec["stress"]["bootstrap_seed"],
            "audit-chain": release.spec["stress"]["bootstrap_seed"],
            "score": release.spec["score"]["bootstrap_seed"],
            "stimulation": {d: s["seed"] for d, s in release.spec["stimulation"].items()},
            "cross-chip": release.spec["first_day_22554"]["seeds"],
            "heldout": release.spec["day19"]["bootstrap_seed"],
        }
        outputs = {
            p.relative_to(payload).as_posix(): sha256(p)
            for p in sorted(payload.rglob("*"))
            if p.is_file()
        }
        external = {}
        if args.command in {"raw-windows", "audit-chain", "verify-h5"}:
            use_all = args.command == "verify-h5" and not args.stress_only
            external["recording_sha256"] = [
                row["sha256"]
                for row in release.rows("h5_manifest")
                if use_all or row["stress_record"] == "True"
            ]
        if args.command in {"archived-code", "audit-chain"}:
            external["program_and_receipt_archive_sha256"] = result["archive_sha256"]
        manifest = {
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "git_commit": commit(release.root),
            "config_sha256": sha256(release.config_path),
            "input_sha256": release.pins,
            "external_inputs": external,
            "outputs": outputs,
            "seeds": pinned if args.command == "all" else pinned.get(args.command),
            "environment": env,
            "environment_sha256": hashlib.sha256(encoded(env)).hexdigest(),
            "maintained_code_sha256": {
                p.relative_to(release.root).as_posix(): sha256(p)
                for p in sorted((release.root / "src").glob("*.py"))
            },
        }
        (payload / "run_manifest.json").write_bytes(encoded(manifest))
        payload.rename(target)
    return target, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--config", default="configs/study.yaml", help="Release-relative JSON-compatible YAML"
    )
    parser.add_argument(
        "--output-dir", default="runs", help="Parent of a new isolated run directory"
    )
    parser.add_argument("--data-dir", type=Path, help="Authorized local H5 directory")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Public receipts.zip or the matching original training/evaluation ZIP",
    )
    parser.add_argument(
        "--stress-only",
        action="store_true",
        help="Verify the stress-record subset of the H5 manifest",
    )
    args = parser.parse_args()
    if args.command in {"raw-windows", "verify-h5", "audit-chain"} and not args.data_dir:
        parser.error("This command requires --data-dir with locally authorized recordings.")
    if args.command in {"archived-code", "audit-chain"} and not args.archive:
        parser.error("Supply --archive with receipts.zip or the original training/evaluation ZIP.")
    if args.stress_only and args.command != "verify-h5":
        parser.error("--stress-only applies to verify-h5.")
    # Restrict CPU matrix products to one thread before importing NumPy for the analysis.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        out, result = run(args)
    except (ValueError, KeyError, OSError, ImportError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        if isinstance(error, ImportError):
            print(
                "Install the pinned requirements first. H5 analyses also require configs/raw-requirements.txt.",
                file=sys.stderr,
            )
        return 1
    key_result = result if args.command == "smoke" else {"result": str(out / "result.json")}
    print(
        json.dumps(
            {"command": args.command, "run_directory": str(out), "summary": key_result},
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
