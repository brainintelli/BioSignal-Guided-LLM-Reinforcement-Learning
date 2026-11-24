"""Verify archived sources or restore one historical directory layout."""

import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent


def below(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unexpected archive path: {name}")
    local = root.joinpath(*path.parts).resolve()
    if not local.is_relative_to(root):
        raise ValueError(f"Archive path escapes its directory: {name}")
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("verify", help="check every shipped file against the repository digest")
    actions.add_parser("list", help="show available source collections")
    extract = actions.add_parser("extract", help="restore a collection to a new directory")
    extract.add_argument("collection")
    extract.add_argument("destination", type=Path)
    args = parser.parse_args()
    with (ROOT / "source-index.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = sorted({r["collection"] for r in rows})
    if args.command == "list":
        print("\n".join(names))
        return
    if args.command == "extract":
        if args.collection not in names:
            parser.error(f"Unknown collection. Choose from {', '.join(names)}.")
        destination = args.destination.resolve()
        if destination.exists():
            parser.error("Choose a new destination directory.")
        rows = [r for r in rows if r["collection"] == args.collection]
    checked = {}
    for row in rows:
        stored = below(ROOT, row["stored_as"])
        if stored not in checked:
            checked[stored] = hashlib.sha256(stored.read_bytes()).hexdigest()
        if checked[stored] != row["repository_sha256"]:
            raise ValueError(f"Checksum mismatch: {row['stored_as']}")
        if row["status"] == "exact" and checked[stored] != row["source_sha256"]:
            raise ValueError(f"Source identity mismatch: {row['source_path']}")
    if args.command == "verify":
        print(f"Verified {len(checked)} files for {len(rows)} source entries.")
        return
    destination.mkdir(parents=True)
    for row in rows:
        target = below(destination, row["source_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(below(ROOT, row["stored_as"]).read_bytes())
    print(f"Restored {len(rows)} files to {destination}")


if __name__ == "__main__":
    main()
