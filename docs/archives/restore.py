"""Verify and join a project's GitHub Release volumes. Python standard library only."""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = "brainintelli/BioSignal-Guided-LLM-Reinforcement-Learning"
TAG = "research-artifacts-v1"


def local_name(name):
    if Path(name).name != name or name in {"", ".", ".."} or any(c in name for c in "/\\:"):
        raise ValueError(f"Invalid filename in manifest: {name!r}")
    return name


def sha256(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_missing(parts, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for part in parts:
        name = local_name(part["name"])
        path = directory / name
        if path.is_file():
            if path.stat().st_size == part["bytes"] and sha256(path) == part["sha256"]:
                continue
            raise ValueError(f"Cached volume failed verification: {path}. Move it aside and rerun.")
        url = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{urllib.parse.quote(name)}"
        request = urllib.request.Request(url, headers={"User-Agent": "Brainintelli-archive-restore"})
        temp = None
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=180) as body, tempfile.NamedTemporaryFile(
                dir=directory, prefix="volume-", suffix=".partial", delete=False
            ) as stream:
                temp = Path(stream.name)
                while chunk := body.read(4 * 1024**2):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                raise ValueError(f"Downloaded volume failed verification: {name}. Rerun to retry.")
            temp.replace(path)
            print(f"Downloaded and verified {name}")
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


def restore(item, directory, output):
    destination = output / local_name(item["archive_name"])
    output.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == item["bytes"] and sha256(destination) == item["sha256"]:
            print(f"Already verified {destination}")
            return
        raise ValueError(f"Existing file differs from the manifest: {destination}. Choose a different output directory.")
    full = hashlib.sha256()
    count = 0
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix="restore-", suffix=".partial", delete=False) as assembled:
            temp = Path(assembled.name)
            for entry in item["parts"]:
                part = directory / local_name(entry["name"])
                if not part.is_file() or part.stat().st_size != entry["bytes"]:
                    raise ValueError(f"Missing or incomplete volume: {part}. Download that volume again.")
                digest = hashlib.sha256()
                with part.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                        digest.update(chunk)
                        full.update(chunk)
                        count += len(chunk)
                        assembled.write(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {part}. Replace that volume with a fresh download.")
        if count != item["bytes"] or full.hexdigest() != item["sha256"]:
            raise ValueError("Reassembled file failed the complete archive checksum.")
        # Windows rename is atomic and rejects an existing destination, including
        # on external drives without hard-link support. POSIX rename can replace
        # a concurrent writer, so use a no-clobber hard link on POSIX.
        if os.name == "nt":
            os.rename(temp, destination)
        else:
            os.link(temp, destination)
        print(f"Verified {destination} ({count} bytes, SHA-256 {full.hexdigest()})")
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", nargs="?", help="Archive key; omit to list the available collections.")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--parts", type=Path, default=Path("downloaded-parts"))
    parser.add_argument("--output", type=Path, default=Path("restored-archives"))
    parser.add_argument("--download", action="store_true", help="Download missing volumes from the public GitHub release.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["repository"] != REPOSITORY or manifest["tag"] != TAG:
        raise ValueError("Manifest belongs to a different repository or release.")
    items = {item["key"]: item for item in manifest["archives"]}
    if args.key is None:
        for item in items.values():
            print(f"{item['key']:23} {len(item['parts']):4d} parts    {item['bytes']:14d} bytes    {item['archive_name']}")
        return
    if args.key not in items:
        parser.error("Unknown archive. Run the command without a key to list the choices.")
    item = items[args.key]
    for part in item["parts"]:
        local_name(part["name"])
    if args.download:
        fetch_missing(item["parts"], args.parts)
    restore(item, args.parts, args.output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
