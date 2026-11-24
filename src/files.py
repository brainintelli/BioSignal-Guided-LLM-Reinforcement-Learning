"""Local inputs, numerical-series identifiers and release configuration."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def root_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a path relative to the release directory: {relative}")
    return path


def csv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Cannot write a table with no observations.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class Release:
    def __init__(self, root: Path = ROOT, config: str = "configs/study.yaml"):
        self.root = root.resolve()
        self.config_path = root_path(self.root, config)
        self.spec = read_json(self.config_path)
        self.pins = read_json(self.root / "data/input_hashes.json")
        if self.spec["schema_version"] != 1:
            raise ValueError("Unsupported configuration schema. Use the included study.yaml.")

    def locate(self, key: str) -> Path:
        return root_path(self.root, self.spec["inputs"][key])

    def rows(self, key: str) -> list[dict[str, str]]:
        return csv_rows(self.locate(key))

    def check_inputs(self) -> None:
        for name, expected in self.pins.items():
            source = root_path(self.root, name)
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"Input missing or changed: {name}. Restore it from this release.")
