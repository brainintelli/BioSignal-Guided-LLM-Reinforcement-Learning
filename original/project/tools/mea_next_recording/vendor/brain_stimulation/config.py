"""Validate explicit JSON values before the desktop's permissive normalizer."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .plan import REQUIRED_BACKEND_FIELDS, StimulationParameters


def number(value, name, *, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    if value < 0 or (positive and value == 0) or (integer and int(value) != value):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def load_config(path: Path) -> StimulationParameters:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    source = data.get("params", data)
    if not isinstance(source, dict):
        raise ValueError("params must be an object")
    missing = REQUIRED_BACKEND_FIELDS - source.keys()
    unknown = source.keys() - REQUIRED_BACKEND_FIELDS - {"assay_path", "record_only_spikes"}
    if missing or unknown:
        raise ValueError(f"Invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key in ("pulses", "pulse_trains"):
        number(source[key], key, positive=True, integer=True)
    for key in ("ipi", "ipti", "pre_stim_period", "post_stim_period", "phase"):
        number(source[key], key, positive=(key == "phase"))
    for key in ("randomize", "control", "record_only_spikes"):
        if key in source and not isinstance(source[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    amplitude = source["amplitude_range"]
    if not isinstance(amplitude, list) or len(amplitude) != 3:
        raise ValueError("amplitude_range must be [min_mv, max_mv, step_mv]")
    for i, value in enumerate(amplitude):
        number(value, f"amplitude_range[{i}]", positive=(i == 2))
    effective_step = max(0.001, amplitude[2])
    largest = max(amplitude[:2])
    if largest + effective_step <= largest:
        raise ValueError("amplitude_range step is too small to advance at this magnitude")
    burst_count = math.floor(abs(amplitude[1] - amplitude[0]) / effective_step) + 1
    if (burst_count + int(source["control"])) * source["pulse_trains"] * source["pulses"] > 100_000:
        raise ValueError("A single run supports at most 100000 pulses; split this experiment")
    electrodes = source["electrodes_selection"]
    if not isinstance(electrodes, dict) or set(electrodes) != {"recording", "stimulation"}:
        raise ValueError("electrodes_selection requires recording and stimulation lists")
    for key, values in electrodes.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} electrodes must be a nonempty JSON list")
        for electrode in values:
            number(electrode, f"{key} electrode", integer=True)
            if electrode >= 26400:
                raise ValueError(f"MaxOne electrode out of range: {electrode}")
    source = dict(source)
    if source.get("assay_path") is not None:
        if not isinstance(source["assay_path"], str):
            raise ValueError("assay_path must be a .cfg filename or null")
        assay = Path(source["assay_path"])
        if not assay.is_absolute():
            assay = path.parent / assay
        if assay.suffix.lower() != ".cfg" or not assay.is_file():
            raise ValueError(f"Missing .cfg file: {assay}")
        source["assay_path"] = str(assay.resolve())
    params = StimulationParameters.from_params(source)
    params.validated_electrodes()
    return params


def describe(params, schedule):
    timing = params.timing()
    return {
        "amplitude_schedule_mv": schedule,
        "burst_count": len(schedule),
        "pulses_per_burst": timing.pulses,
        "actual_phase_us": timing.phase_samples * 50,
        "actual_ipi_ms": (timing.pulse_duration_samples + timing.interpulse_delay_samples) * 0.05,
        "actual_ipti_seconds": timing.actual_interburst_start_samples * 0.00005,
        "pre_recording_seconds": params.pre_recording_seconds(),
        "stimulation_seconds": params.stimulation_seconds(),
        "post_recording_seconds": params.post_recording_seconds(),
        "recording_seconds": params.record_time_seconds(),
        "record_only_spikes": params.record_only_spikes,
        "recording_electrodes": params.validated_electrodes()[0],
        "stimulation_electrodes": list(params.stimulation_electrodes),
    }
