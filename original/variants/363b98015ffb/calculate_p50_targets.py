"""Describe detected-event amplitude medians; never connect to hardware."""
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "p1": ROOT / "output/mea_p1/p1_delivery_20261116/baseline_p1_20261116_run06/recording.jsonl",
    "latest_audited": ROOT / "output/mea_timing/timing_delivery_20261116/timing_20261116_run02/recording.jsonl",
}
results = []
for name, source in SOURCES.items():
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    header = rows[0]
    mapping = dict(header["channels"])
    events = [event for row in rows for event in row.get("events", [])]
    by_channel = defaultdict(list)
    for frame, channel, amplitude_uv in events:
        by_channel[channel].append(abs(amplitude_uv))
    sites = []
    for channel in (840, 628, 888):
        values = by_channel[channel]
        median_uv = statistics.median(values)
        sites.append(dict(electrode_id=mapping[channel], recording_channel=channel,
                          detected_events=len(values), p50_abs_amplitude_uv=median_uv,
                          target_phase_amplitude_mv=median_uv / 1000))
    results.append(dict(recording=name, source=str(source),
                        source_sha256=hashlib.sha256(raw).hexdigest(),
                        channel_map_id=header["channel_map_id"],
                        detected_events=len(events), active_channels=len(by_channel),
                        pooled_event_p50_abs_uv=statistics.median(abs(e[2]) for e in events),
                        median_of_active_channel_medians_uv=statistics.median(
                            statistics.median(values) for values in by_channel.values()),
                        candidate_sites=sites))

assert [r["detected_events"] for r in results] == [608, 555]
assert len({r["channel_map_id"] for r in results}) == 1
report = dict(
    schema="mea.stimulation.p50_design.v1",
    definition="Per-electrode median of absolute amplitudes of audited detected spikes; native recorded sign removed after conversion to uV. Not raw waveform median, peak-to-peak voltage, or membrane voltage.",
    selected_recording="latest_audited",
    target_scope="per_candidate_electrode",
    pooled_p50_role="descriptive_only",
    target_multiplier=1.0,
    user_requirement="自身记录信号幅值的中位数（P50）",
    local_dac_lsb_mv=None,
    hardware_output_feasibility="unverified; targets below the documented example DAC step",
    hardware_execution_enabled=False,
    stimulation_executed=False,
    recordings=results,
)
destination = HERE / "p50_targets.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(dict(output=str(destination), selected=results[-1],
                     hardware_execution_enabled=False), ensure_ascii=True, indent=2))
