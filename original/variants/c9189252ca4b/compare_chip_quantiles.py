"""Offline amplitude quantiles, with provenance; no device or stimulation API."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / ".tools/mea_baseline_deps"))
import h5py
import numpy as np

OUT = HERE / "p50_p90_comparison"
OUT.mkdir(exist_ok=True)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    assert np.isfinite(values).all() and np.all(values >= 0)
    q = np.quantile(values, [.5, .9], method="linear")
    # Independent rank/interpolation check against the library calculation.
    ordered = sorted(float(v) for v in values)
    for probability, expected in zip((.5, .9), q):
        rank = probability * (len(ordered) - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        actual = ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])
        assert abs(actual - expected) < 1e-10
    return dict(event_count=len(values), p50_uv=float(q[0]), p90_uv=float(q[1]),
                p50_target_phase_mv=float(q[0] / 1000),
                p90_target_phase_mv=float(q[1] / 1000))


def summarize(chip, frames, channels, uv, mapping, lo, hi, rate, provenance, fixed_sites=None):
    midpoint = lo + (hi - lo) // 2
    rows = []
    for channel, electrode in sorted(mapping.items()):
        mask = channels == channel
        first = mask & (frames < midpoint)
        second = mask & (frames >= midpoint)
        stat = quantiles(uv[mask])
        row = dict(chip_id=chip, recording_channel=channel, electrode_id=electrode,
                   event_count=int(mask.sum()), first_half_events=int(first.sum()),
                   second_half_events=int(second.sum()))
        for key in ("p50_uv", "p90_uv", "p50_target_phase_mv", "p90_target_phase_mv"):
            row[key] = stat[key] if stat else None
        row["first_half_p50_uv"] = quantiles(uv[first])["p50_uv"] if first.any() else None
        row["first_half_p90_uv"] = quantiles(uv[first])["p90_uv"] if first.any() else None
        row["second_half_p50_uv"] = quantiles(uv[second])["p50_uv"] if second.any() else None
        row["second_half_p90_uv"] = quantiles(uv[second])["p90_uv"] if second.any() else None
        rows.append(row)
    path = OUT / f"chip_{chip}_per_electrode.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if fixed_sites:
        candidates = [next(row for row in rows if row["electrode_id"] == e) for e in fixed_sites]
        rule = "Previously selected recording candidates, not validated stimulation routes."
    else:
        candidates = sorted(rows, key=lambda r: (-min(r["first_half_events"], r["second_half_events"]),
                                                 -r["event_count"], r["electrode_id"]))[:3]
        rule = "Rank by minimum event count across two time halves, then total event count. Recording candidates only; spatial independence, tissue coverage and stimulation routing unverified."
    return dict(chip_id=chip, scope_frames=[int(lo), int(hi)], duration_seconds=(hi-lo)/rate,
                mapped_channels=len(mapping), active_channels=len(set(channels.tolist())),
                pooled_event_quantiles=quantiles(uv),
                first_half=quantiles(uv[frames < midpoint]), second_half=quantiles(uv[frames >= midpoint]),
                candidate_selection=rule, candidate_sites=candidates,
                per_electrode_csv=str(path), provenance=provenance)


# A line: use the finalized, audited event export, already converted to microvolts.
a_path = ROOT / "output/mea_timing/timing_delivery_20261116/timing_20261116_run02/recording.jsonl"
a_sha = checksum(a_path)
assert a_sha == "6efe0203806fcffa602b66e78d595a39aa42e74d2cd327c16d5f1e55489aa2c6"
a_rows = [json.loads(line) for line in a_path.read_text(encoding="utf-8").splitlines()]
a_audit = json.loads(a_path.with_name("audit.json").read_text(encoding="utf-8"))
a_events = np.array([e for row in a_rows for e in row.get("events", [])], dtype=np.float64)
assert len(a_events) == 555 and a_audit["offline_integrity_passed"]
a_lo, a_hi = a_audit["audit_interval"]
a = summarize("22021", a_events[:, 0].astype(np.int64), a_events[:, 1].astype(np.int64),
              np.abs(a_events[:, 2]), dict(a_rows[0]["channels"]), a_lo, a_hi, 20000,
              dict(source=str(a_path), sha256=a_sha, units="microvolts, already converted",
                   source_full_structural_passed=True, paired_sdk_event_agreement=True,
                   one_second_boundary_guards=True), [7587, 2047, 4267])

prior_path = ROOT / "output/mea_p1/p1_delivery_20261116/baseline_p1_20261116_run06/recording.jsonl"
prior_rows = [json.loads(line) for line in prior_path.read_text(encoding="utf-8").splitlines()]
a["earlier_recording_reference"] = dict(source=str(prior_path), sha256=checksum(prior_path),
    pooled_event_quantiles=quantiles([abs(e[2]) for row in prior_rows for e in row.get("events", [])]))

# B line: preserve source QC failure; map/filter events using prior registered rules.
b_path = ROOT / "output/mea_22554_baseline_20261116/source/22554_baseline.raw.h5"
b_sha = checksum(b_path)
assert b_sha == "45a341c80d7ac60eb2b9ba846dce926f9e7fd4b6fb27f0b7b8c963798640f84c"
with h5py.File(b_path, "r") as file:
    assert file["wellplate/id"][0] == b"22554"
    store = file["data_store/data0000"]
    mapping = {int(r["channel"]): int(r["electrode"]) for r in store["settings/mapping"]}
    raw_frames = store["groups/routed/frame_nos"][:]
    assert len(raw_frames) == 6001600 and np.all(np.diff(raw_frames) == 1)
    lo, hi = int(raw_frames[0]), int(raw_frames[-1]) + 1
    rate = float(store["settings/sampling"][0])
    lsb_volts = float(store["settings/lsb"][0])
    events = store["spikes"][:]
frames, channels = events["frameno"], events["channel"]
mapped = np.isin(channels, list(mapping))
in_range = (frames >= lo) & (frames < hi)
valid = mapped & in_range
assert len(events) == 57652 and int((~mapped).sum()) == 133 and int((~in_range).sum()) == 2
assert int(valid.sum()) == 57517 and len(mapping) == 850
scoped_prior = json.loads((b_path.parent.parent / "analysis_routed_v1/analysis.json").read_text())
assert np.flatnonzero(~valid).tolist() == scoped_prior["scope_exclusions"]["event_indices"]
b_lo, b_hi = lo + int(rate), hi - int(rate)
selected = valid & (frames >= b_lo) & (frames < b_hi)
assert int(selected.sum()) == 57302
uv = np.abs(events["amplitude"].astype(np.float64)) * lsb_volts * 1e6
assert abs(float(uv[valid].mean()) - scoped_prior["qc"]["mean_abs_event_amplitude_uv"]) < 1e-10
b = summarize("22554", frames[selected], channels[selected], uv[selected], mapping, b_lo, b_hi, rate,
    dict(source=str(b_path), sha256=b_sha, recorded_lsb_volts=lsb_volts,
         conversion="abs(native spike amplitude) * recorded_lsb_volts * 1e6",
         source_full_structural_passed=False, paired_sdk_event_agreement=False,
         source_event_count=len(events), unmapped_events=133, outside_raw_frames=2,
         excluded_union=135, valid_in_raw_range=57517, boundary_guard_events=215,
         one_second_boundary_guards=True, scope="exploratory mapped in-range events with guards"))
b["full_mapped_in_range_sensitivity"] = quantiles(uv[valid])

report = dict(schema="mea.two_chip.amplitude_quantiles.v1", date="2026-11-16",
              amplitude_definition="absolute amplitude of detected extracellular spike events",
              quantile_method="linear interpolation, rank=(n-1)*p; NumPy linear/type 7",
              chips_not_pooled=True, biological_n_per_chip=1,
              requested_multiplier=1, target_scope="per_electrode; pooled values descriptive",
              p50_role="baseline-amplitude target", p90_role="higher observed-amplitude target, not a safety limit",
              actual_dac_lsb_mv={"22021":None,"22554":None},
              documented_example_dac_lsb_mv=2.981901,
              execution_enabled=False, stimulation_executed=False, chips=[a,b])
(OUT / "quantiles.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=True, indent=2))
