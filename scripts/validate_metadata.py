"""Read all local record sets with the official Croissant implementation."""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import mlcroissant as mlc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = root / "data/croissant.json"
    schema = json.loads(p.read_bytes())
    ds = mlc.Dataset(p)
    objects = {o["@id"]: o for o in schema["distribution"]}
    tally = {}
    for rs in schema["recordSet"]:
        fields = rs["field"]
        fid = fields[0]["source"]["fileObject"]["@id"]
        resource = objects[fid]
        source = (p.parent / resource["contentUrl"]).resolve()
        if (
            not source.is_relative_to(root)
            or hashlib.sha256(source.read_bytes()).hexdigest() != resource["sha256"]
        ):
            raise ValueError(f"Unresolved path or file checksum mismatch: {fid}")
        actual = ds.records(record_set=rs["@id"])
        if fields[0]["source"]["extract"].get("fileProperty") == "filename":
            (item,) = actual
            v = item[fields[0]["@id"]]
            if v not in {source.name, source.name.encode()}:
                raise ValueError("The documentation fileProperty did not resolve.")
            tally[rs["@id"]] = 1
            continue
        with source.open(encoding="utf-8-sig", newline="") as stream:
            expected = csv.DictReader(stream)
            if [f["source"]["extract"]["column"] for f in fields] != expected.fieldnames:
                raise ValueError(f"Croissant/source CSV header mismatch for {source.name}.")
            count = 0
            for num, (r, e) in enumerate(itertools.zip_longest(actual, expected), 1):
                if r is None or e is None:
                    raise ValueError(f"Croissant/source row count mismatch for {rs['@id']}.")
                if {f["@id"] for f in fields} != set(r):
                    raise ValueError(f"Unexpected decoded fields in {source.name}, record {num}.")
                if rs["@id"] == "data-exploration" and any(
                    r[f["@id"]] != int(e[f["name"]]) for f in fields
                ):
                    raise ValueError(f"Primary paired data cell mismatch, record {num}.")
                count = num
        tally[rs["@id"]] = count
    print(json.dumps({"record_sets": len(tally), "decoded_rows": tally}, sort_keys=True))


if __name__ == "__main__":
    main()
