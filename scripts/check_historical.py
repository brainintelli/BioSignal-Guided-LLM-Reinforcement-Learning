"""Check archived inspection source without importing its historical dependencies."""

import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "historical"
    manifest = json.loads((root / "source_manifest.json").read_bytes())
    verified = []
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source manifest must refer to its own directory.")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["distributed_sha256"] or (
            item["kind"] == "verbatim file" and digest != item["upstream_sha256"]
        ):
            raise ValueError(f"Changed historical source: {item['path']}")
        tree = ast.parse(data)
        if item.get("symbol") and [getattr(n, "name", None) for n in tree.body] != [item["symbol"]]:
            raise ValueError(f"Archived function identity differs: {item['path']}")
        verified.append(item["path"])
    print(json.dumps({"verified": verified, "executed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
