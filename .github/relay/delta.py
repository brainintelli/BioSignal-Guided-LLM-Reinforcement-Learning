"""Mirror reviewed bytes to one existing draft. Uses only the Python standard library."""

import csv
import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit


REPO = "brainintelli/BioSignal-Guided-LLM-Reinforcement-Learning"
TAG = "research-artifacts-v1"
DRAFT_ID = 456282580
SAS_HOST = "release-assets.githubusercontent.com"
API = "api.github.com"


class Halt(Exception):
    """Safe diagnostic for the Actions log."""


def approvals(path):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["name", "bytes", "sha256"]:
            raise Halt("Invalid approval manifest header")
        for row in reader:
            name = row["name"]
            if (None in row or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", name)
                    or name.lower() in rows or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    or not re.fullmatch(r"[1-9][0-9]*", row["bytes"]) or int(row["bytes"]) >= 2**31):
                raise Halt("Invalid or duplicate approval manifest entry")
            rows[name.lower()] = (name, int(row["bytes"]), row["sha256"])
    return rows


def select(rows, names):
    chosen = names.split(",")
    if not 1 <= len(chosen) <= 4 or len(set(chosen)) != len(chosen):
        raise Halt("Select one to four distinct approved filenames")
    result = []
    for name in chosen:
        row = rows.get(name.lower())
        if not row or row[0] != name:
            raise Halt("Filename is absent from the public approval manifest")
        result.append(row)
    return result


def signed_path(url):
    parts = urlsplit(url)
    q = parse_qs(parts.query)
    if (parts.scheme != "https" or parts.netloc != SAS_HOST or parts.fragment
            or not parts.path.startswith("/github-production-release-asset/")
            or not q.get("sig") or len(q.get("se", [])) != 1 or q.get("sp") != ["r"]):
        raise Halt("Source is not the expected read-only GitHub release-asset URL")
    try:
        expires = datetime.fromisoformat(q["se"][0].replace("Z", "+00:00"))
        seconds = (expires - datetime.now(timezone.utc)).total_seconds()
    except (TypeError, ValueError):
        raise Halt("Invalid source expiration") from None
    if not 0 < seconds <= 2 * 3600:
        raise Halt("Source URL expired or has an unexpectedly long lifetime")
    return parts.path + "?" + parts.query


def request_json(token, method, host, path, file=None, length=None):
    h = {"User-Agent": "approved-asset-stage", "Accept": "application/vnd.github+json",
         "Authorization": "Bearer " + token, "X-GitHub-Api-Version": "2022-11-28"}
    if file is not None:
        h.update({"Content-Type": "application/octet-stream", "Content-Length": str(length)})
    conn = http.client.HTTPSConnection(host, timeout=180)
    try:
        conn.request(method, path, body=file, headers=h)
        res = conn.getresponse()
        if res.status not in (200, 201):
            raise Halt(f"Destination {method} returned HTTP {res.status}; inspect the draft before retrying")
        return json.load(res)
    finally:
        conn.close()


def copy_verified(stream, out, size, expected):
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(1024 * 1024):
        total += len(chunk)
        if total > size:
            raise Halt("Source is larger than the approved byte count")
        digest.update(chunk)
        out.write(chunk)
    if total != size or digest.hexdigest() != expected:
        raise Halt("Downloaded bytes failed the public size or SHA-256 check")


def verified_download(url, file, size, expected):
    path = signed_path(url)
    conn = http.client.HTTPSConnection(SAS_HOST, timeout=180)
    try:
        conn.request("GET", path, headers={"User-Agent": "approved-asset-stage"})
        res = conn.getresponse()
        if res.status != 200:
            raise Halt(f"Source returned HTTP {res.status}; generate a fresh signed URL")
        copy_verified(res, file, size, expected)
        file.seek(0)
    finally:
        conn.close()


def asset_matches(asset, row):
    name, size, digest = row
    return (asset.get("name") == name and asset.get("size") == size and asset.get("state") == "uploaded"
            and asset.get("digest") == "sha256:" + digest)


def require_draft(token, base):
    release = request_json(token, "GET", API, base)
    if release.get("draft") is not True or release.get("tag_name") != TAG:
        raise Halt("Destination must be the existing research-artifacts-v1 draft")


def inventory(token, base):
    result = {}
    for page in range(1, 12):
        assets = request_json(token, "GET", API, f"{base}/assets?per_page=100&page={page}")
        for item in assets:
            result[item["name"]] = item
        if len(assets) < 100:
            return result
    raise Halt("Unexpectedly large destination inventory")


def upload_verified_file(token, base, row, file):
    name, size, digest = row
    require_draft(token, base)
    created = request_json(token, "POST", "uploads.github.com",
                           f"{base}/assets?name={quote(name, safe='')}", file, size)
    if not asset_matches(created, row):
        raise Halt(f"{name}: upload response failed verification; inspect the draft")
    stored = request_json(token, "GET", API, f"/repos/{REPO}/releases/assets/{int(created['id'])}")
    if not asset_matches(stored, row):
        raise Halt(f"{name}: GitHub asset readback failed")
    print(f"{name}: staged SHA-256 {digest}", flush=True)


def item(row):
    name, size, digest = row["name"], row["bytes"], row["sha256"]
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", name) or type(size) is not int
            or not 0 < size < 2**31 or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise Halt("Invalid source declaration in recipe")
    return name, size, digest


def bounded_copy(source, out, offset, size, digest):
    if type(size) is not int or type(offset) is not int or offset < 0 or size <= 0:
        raise Halt("Invalid byte range")
    source.seek(offset)
    while size:
        chunk = source.read(min(size, 1024 * 1024))
        if not chunk:
            raise Halt("Byte range extends past its source")
        digest.update(chunk)
        out.write(chunk)
        size -= len(chunk)


def reconstruct(recipe, literal, sources, row, output):
    if recipe.get("schema") != "organoid-replay.release-volume-delta.v1" or item(recipe["part"]) != row:
        raise Halt("Recipe target differs from the approved part")
    actual = hashlib.sha256()
    total = 0
    for op in recipe["ops"]:
        if set(op) == {"literal","bytes","offset"} and op["literal"] is True:
            stream = literal
        elif set(op) == {"source","bytes","offset"} and op["source"] in sources:
            stream = sources[op["source"]]
        else:
            raise Halt("Invalid recipe operation")
        total += op["bytes"]
        if total > row[1]:
            raise Halt("Operation exceeds the approved part size")
        bounded_copy(stream, output, op["offset"], op["bytes"], actual)
    if total != row[1] or actual.hexdigest() != row[2]:
        raise Halt("Reconstructed part does not match the public digest")
    output.seek(0)


def run():
    from contextlib import ExitStack
    env = os.environ
    if (env.get("GITHUB_REPOSITORY") != REPO or env.get("GITHUB_REF") != "refs/heads/main"
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[0-9A-F]{32}", env.get("SLOT",""))):
        raise Halt("Invalid job identity or parameters")
    rows = select(approvals(Path(__file__).with_name("approved.csv")), env["SELECTED"])
    helper_row = item({"name":"helper.zip","bytes":int(env["HELPER_BYTES"]),"sha256":env["HELPER_SHA"]})
    links = json.loads(env.pop("SIGNED_SOURCES"))
    token = env.pop("DESTINATION_TOKEN")
    repo = request_json(token, "GET", API, f"/repos/{REPO}")
    if repo.get("visibility") != "public" or repo.get("full_name") != REPO:
        raise Halt("Unexpected destination")
    base = f"/repos/{REPO}/releases/{DRAFT_ID}"
    require_draft(token, base)
    existing = inventory(token, base)
    with ExitStack() as stack:
        def spool():
            return stack.enter_context(tempfile.TemporaryFile())
        helper = spool()
        verified_download(links.pop("__helper__"), helper, helper_row[1], helper_row[2])
        outer = stack.enter_context(zipfile.ZipFile(helper))
        if sorted(outer.namelist()) != sorted(r[0]+".delta.zip" for r in rows):
            raise Halt("Helper contents differ from requested targets")
        sources = {}
        definitions = {}
        recipes = []
        for row in rows:
            name = row[0]
            with outer.open(name+".delta.zip") as packed:
                archive = stack.enter_context(zipfile.ZipFile(packed))
                if sorted(archive.namelist()) != ["literal.bin","recipe.json"]:
                    raise Halt("Unexpected recipe contents")
                if archive.getinfo("recipe.json").file_size > 8 * 1024**2:
                    raise Halt("Unexpectedly large recipe")
                recipe = json.loads(archive.read("recipe.json"))
                if item(recipe["part"]) != row or recipe["schema"] != "organoid-replay.release-volume-delta.v1":
                    raise Halt("Unapproved recipe target")
                literal = spool()
                ls = recipe["literal_bytes"]
                if type(ls) is not int or not 0 <= ls <= row[1]:
                    raise Halt("Invalid literal size")
                with archive.open("literal.bin") as stream:
                    copy_verified(stream,literal,ls,recipe["literal_sha256"])
                for declared in recipe["sources"]:
                    source = item(declared)
                    if source != definitions.setdefault(source[0], source):
                        raise Halt("Inconsistent source declaration")
                recipes.append((row,recipe,literal))
        if len(definitions)>12 or set(links) != set(definitions):
            raise Halt("Asset references do not match the restricted source grants")
        for name, (_,length,sha) in definitions.items():
            sources[name] = spool()
            verified_download(links.pop(name),sources[name],length,sha)
        for row, recipe, literal in recipes:
            if row[0] in existing:
                if not asset_matches(existing[row[0]], row): raise Halt("Conflicting destination asset")
                print(f"{row[0]}: already verified")
                continue
            with tempfile.TemporaryFile() as output:
                reconstruct(recipe,literal,sources,row,output)
                # A concurrent local uploader can finish while this job downloads its sources.
                current = inventory(token, base).get(row[0])
                if current:
                    if not asset_matches(current,row): raise Halt("Concurrent destination is incomplete or conflicting")
                    print(f"{row[0]}: already verified")
                else:
                    upload_verified_file(token,base,row,output)
        require_draft(token,base)


if __name__ == "__main__":
    try:
        run()
    except Halt as exc:
        sys.exit(str(exc))
    except Exception as exc:
        sys.exit(f"Staging stopped ({type(exc).__name__}); no source URLs are logged")
