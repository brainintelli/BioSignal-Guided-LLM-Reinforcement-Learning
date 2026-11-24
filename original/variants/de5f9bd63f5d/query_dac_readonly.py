"""Run in memory with vendor Python. All server messages are allowlisted reads."""
import ast
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import socket
import sys

sys.dont_write_bytecode = True
import maxlab as mx

host = socket.gethostname()
profile = {
    "researcher-biolab-3": {"expected_chip_id": "22021", "ip": "192.0.2.101", "sites": [7587, 2047, 4267]},
    "researcher-biolab-1": {"expected_chip_id": "22554", "ip": "192.0.2.21", "sites": [9764, 1388, 9985]},
}[host]
config_query = "mea_array_command online\nmea_array_get_config"
route_queries = {e: f"mea_array_command online\nmea_array_query_stimulation_at_electrode {e}" for e in profile["sites"]}
allowed = {"system_query_dac_lsb_mv", "wellplate_query_version", "wellplate_query_rows", "wellplate_query_columns", config_query, *route_queries.values()}
commands = []


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_only_send(message):
    if message not in allowed:
        raise ValueError("Non-allowlisted server message rejected")
    entry = {"command": message, "started_utc": now()}
    commands.append(entry)
    api = mx.comm.ApiComm(timeout=4)
    try:
        reply = api.send(message, timeout=4)
        entry.update(ended_utc=now())
        if message == config_query:
            entry.update(reply_sha256=hashlib.sha256(reply.encode()).hexdigest(), reply_length=len(reply))
        else:
            entry["reply"] = reply
        if reply.strip() == "Error":
            raise RuntimeError("Server query returned Error")
        return reply
    finally:
        api.shutdown()


def process_snapshot():
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            name = (proc / "comm").read_text().strip()
            if name in ("mxwserver", "scope"):
                rows.append({"pid": int(proc.name), "name": name, "executable": os.readlink(proc / "exe")})
        except OSError:
            pass
    return sorted(rows, key=lambda r: r["pid"])


source = inspect.getsource(mx.util.query_DAC_lsb_mV)
assert "system_query_dac_lsb_mv" in source
mx.util.send_raw = read_only_send
result = dict(schema="mea.dac.readonly_query.v1", hostname=host, profile=profile,
              started_utc=now(), python=sys.executable, maxlab_api_version=mx.__version__,
              maxlab_git_hash=getattr(mx, "__git_hash__", None),
              query_function_source=source, queries=[], commands=commands,
              processes_before=process_snapshot(), stimulation_sent=False,
              initialization_sent=False, routing_changed=False,
              device_chip_id_directly_queried=False,
              current_stimulation_mode_verified=False,
              physical_waveform_measured=False)
before = read_only_send(config_query)
mapping = []
for part in before.split()[0].rstrip(";").split(";"):
    match = re.fullmatch(r"(\d+)\((\d+)\)(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", part)
    if not match:
        raise ValueError("Could not parse current mapping")
    mapping.append([int(match[1]), int(match[2])])
mapping.sort()
result["mapped_channels"] = len(mapping)
result["channel_map_id"] = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
result["candidate_current_recording_channels"] = {str(e): [c for c, electrode in mapping if electrode == e] for e in profile["sites"]}
for _ in range(3):
    raw = mx.query_DAC_lsb_mV()
    value = float(raw.strip())
    assert math.isfinite(value) and value > 0
    result["queries"].append({"query_utc": now(), "raw_reply": raw, "dac_lsb_mv": value})
result["wellplate_geometry"] = {key: read_only_send("wellplate_query_" + key) for key in ("version", "rows", "columns")}
result["candidate_stimulation_routes"] = {str(e): read_only_send(q) for e, q in route_queries.items()}
after = read_only_send(config_query)
result.update(config_before_sha256=hashlib.sha256(before.encode()).hexdigest(),
              config_after_sha256=hashlib.sha256(after.encode()).hexdigest(),
              config_unchanged=before == after,
              processes_after=process_snapshot(), ended_utc=now())
result["dac_queries_identical"] = len({r["dac_lsb_mv"] for r in result["queries"]}) == 1
result["physical_step_interpretation"] = "Device-reported voltage-mode DAC step, not an oscilloscope verification."
print("DAC_READONLY_BEGIN")
encoded = json.dumps(result, ensure_ascii=True).encode().hex()
for offset in range(0, len(encoded), 64):
    print("J:" + encoded[offset:offset + 64])
print("DAC_READONLY_END")
