"""Canonical stimulation parameters and pure schedule/timing calculations."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .parameter_utils import (
    as_bool,
    as_float,
    as_int,
    as_int_list,
    extract_params,
    samples_from_seconds,
    samples_from_us,
    samples_to_seconds,
)


SAMPLE_PERIOD_US = 50.0
SEQUENCE_INITIAL_DELAY_SAMPLES = 100
REQUIRED_BACKEND_FIELDS = {
    "pulses",
    "ipi",
    "pulse_trains",
    "ipti",
    "amplitude_range",
    "randomize",
    "control",
    "phase",
    "electrodes_selection",
    "pre_stim_period",
    "post_stim_period",
}


@dataclass(frozen=True)
class StimulationTiming:
    pulses: int
    phase_samples: int
    pulse_duration_samples: int
    interpulse_delay_samples: int
    burst_duration_samples: int
    actual_interburst_start_samples: int
    interburst_delay_samples: int


@dataclass(frozen=True)
class StimulationParameters:
    """Typed view of the backend stimulation ``params`` contract.

    Units follow the backend/UI contract used by this client: ``ipi`` is in
    milliseconds, ``ipti`` is in seconds, ``phase`` is in microseconds, and
    pre/post stimulation periods are in minutes.
    """

    pulses: int
    ipi: float
    pulse_trains: int
    ipti: float
    amplitude_range: tuple[float, float, float]
    randomize: bool
    control: bool
    phase: float
    recording_electrodes: tuple[int, ...]
    stimulation_electrodes: tuple[int, ...]
    pre_stim_period: float
    post_stim_period: float
    record_only_spikes: bool = False
    assay_path: str | None = None

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "StimulationParameters":
        return cls.from_params(extract_params(data))

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> "StimulationParameters":
        source = dict(params) if isinstance(params, Mapping) else {}
        missing = sorted(REQUIRED_BACKEND_FIELDS.difference(source))
        if missing:
            raise ValueError(
                "missing backend stimulation params: " + ", ".join(missing)
            )

        electrodes = source.get("electrodes_selection")
        if not isinstance(electrodes, Mapping):
            raise ValueError("electrodes_selection must be an object")

        return cls(
            pulses=max(1, as_int(source["pulses"], 3)),
            ipi=as_float(source["ipi"], 10.0),
            pulse_trains=max(1, as_int(source["pulse_trains"], 5)),
            ipti=as_float(source["ipti"], 1.0),
            amplitude_range=cls._read_amplitude_range(source),
            randomize=as_bool(source["randomize"]),
            control=as_bool(source["control"]),
            phase=as_float(source["phase"], 200.0),
            recording_electrodes=tuple(as_int_list(electrodes.get("recording"))),
            stimulation_electrodes=tuple(as_int_list(electrodes.get("stimulation"))),
            pre_stim_period=max(
                0.0,
                as_float(source["pre_stim_period"], 0.0),
            ),
            post_stim_period=max(
                0.0,
                as_float(source["post_stim_period"], 0.0),
            ),
            record_only_spikes=as_bool(source.get("record_only_spikes"), False),
            assay_path=str(source["assay_path"]) if source.get("assay_path") else None,
        )

    @staticmethod
    def _read_amplitude_range(source: Mapping[str, Any]) -> tuple[float, float, float]:
        canonical = source["amplitude_range"]
        if not isinstance(canonical, (list, tuple)) or len(canonical) != 3:
            raise ValueError("amplitude_range must be [min_mv, max_mv, step_mv]")
        start, stop, step = canonical
        return as_float(start, 200.0), as_float(stop, 400.0), as_float(step, 50.0)

    def validated_electrodes(self) -> tuple[list[int], list[int]]:
        stimulation = list(self.stimulation_electrodes)
        recording = list(self.recording_electrodes)
        if not stimulation:
            raise ValueError("Stimulation experiment requires stimulation electrodes.")
        if len(stimulation) > 32:
            raise ValueError("MaxLab supports at most 32 stimulation electrodes.")
        if not recording:
            raise ValueError("Stimulation experiment requires recording electrodes.")
        return list(dict.fromkeys(recording + stimulation)), stimulation

    def amplitude_values(self) -> list[float]:
        start, stop, step = self.amplitude_range
        if stop < start:
            start, stop = stop, start
        step = max(0.001, step)

        values: list[float] = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 6))
            current += step
        if self.control:
            values = [0.0] + [value for value in values if value != 0.0]
        return values or [200.0]

    def amplitude_schedule(self) -> list[float]:
        schedule = [
            amplitude
            for amplitude in self.amplitude_values()
            for _ in range(self.pulse_trains)
        ]
        if self.randomize:
            random.shuffle(schedule)
        return schedule

    def timing(self) -> StimulationTiming:
        phase_samples = samples_from_us(self.phase, SAMPLE_PERIOD_US)
        pulse_duration_samples = phase_samples * 2
        interpulse_start_samples = samples_from_seconds(self.ipi / 1000.0, SAMPLE_PERIOD_US)
        actual_interpulse_start_samples = max(
            interpulse_start_samples,
            pulse_duration_samples,
        )
        interpulse_delay_samples = max(
            0,
            interpulse_start_samples - pulse_duration_samples,
        )
        burst_duration_samples = (
            max(0, self.pulses - 1) * actual_interpulse_start_samples
            + pulse_duration_samples
        )
        interburst_start_samples = samples_from_seconds(self.ipti, SAMPLE_PERIOD_US)
        actual_interburst_start_samples = max(
            interburst_start_samples,
            burst_duration_samples,
        )
        return StimulationTiming(
            pulses=self.pulses,
            phase_samples=phase_samples,
            pulse_duration_samples=pulse_duration_samples,
            interpulse_delay_samples=interpulse_delay_samples,
            burst_duration_samples=burst_duration_samples,
            actual_interburst_start_samples=actual_interburst_start_samples,
            interburst_delay_samples=max(
                0,
                interburst_start_samples - burst_duration_samples,
            ),
        )

    def stimulation_seconds(self) -> float:
        timing = self.timing()
        # Duration depends only on the number of bursts.  Do not shuffle the
        # schedule here: randomization belongs to sequence construction and
        # should happen exactly once for each experiment run.
        burst_count = len(self.amplitude_values()) * self.pulse_trains
        duration_samples = (
            SEQUENCE_INITIAL_DELAY_SAMPLES
            + timing.burst_duration_samples
            + max(0, burst_count - 1) * timing.actual_interburst_start_samples
        )
        return samples_to_seconds(duration_samples, SAMPLE_PERIOD_US)

    def pre_recording_seconds(self) -> float:
        return self.pre_stim_period * 60.0

    def post_recording_seconds(self) -> float:
        return self.post_stim_period * 60.0

    def record_time_seconds(self) -> float:
        return (
            self.pre_recording_seconds()
            + self.stimulation_seconds()
            + self.post_recording_seconds()
        )

    def assay_inputs(self) -> dict[str, str]:
        """Return MaxLab assay inputs in the VoltageStimulation H5 format."""
        recording, stimulation = self.validated_electrodes()
        amplitude_min, amplitude_max, amplitude_step = self.amplitude_range
        electrodes = {
            "electrodes": {
                "0": {
                    "selected": recording,
                    "stimulation": stimulation,
                }
            },
            "properties": {},
        }
        return {
            "amplitude_range": ",".join(
                format_seconds(value)
                for value in (amplitude_min, amplitude_max, amplitude_step)
            ),
            "electrodes": json.dumps(
                electrodes,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "ipi": format_seconds(self.ipi),
            "ipti": format_seconds(self.ipti),
            "phase": format_seconds(self.phase),
            "post_stim_period": format_seconds(self.post_stim_period),
            "pre_stim_period": format_seconds(self.pre_stim_period),
            "pulse_trains": str(self.pulse_trains),
            "pulses": str(self.pulses),
            "randomize": str(self.randomize).lower(),
        }


def format_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return str(int(seconds))
    return f"{seconds:.6f}".rstrip("0").rstrip(".")


def format_progress_matrix(
    params: StimulationParameters,
    amplitude_schedule: list[float],
) -> str:
    """Format the exact programmed burst order as a completion-time matrix."""
    timing = params.timing()
    total = len(amplitude_schedule)
    pre_seconds = params.pre_recording_seconds()
    burst_seconds = samples_to_seconds(
        timing.burst_duration_samples,
        SAMPLE_PERIOD_US,
    )

    lines = [
        f"刺激实验进度点明细（共 {total} 点，时间相对记录开始）",
        "Point | Progress | Amplitude(mV) | Start(s) | End(s) | Duration(ms) | Pulses | Phase(us) | IPI(ms)",
        "------|----------|---------------|----------|--------|--------------|--------|-----------|--------",
    ]
    for index, amplitude_mv in enumerate(amplitude_schedule, start=1):
        start_samples = (
            SEQUENCE_INITIAL_DELAY_SAMPLES
            + (index - 1) * timing.actual_interburst_start_samples
        )
        start_seconds = pre_seconds + samples_to_seconds(
            start_samples,
            SAMPLE_PERIOD_US,
        )
        end_seconds = start_seconds + burst_seconds
        progress = int(round(index / total * 100)) if total else 100
        lines.append(
            f"{index:>5} | {progress:>7}% | {amplitude_mv:>13.3f} | "
            f"{start_seconds:>8.6f} | {end_seconds:>6.6f} | "
            f"{burst_seconds * 1000.0:>12.3f} | {params.pulses:>6} | "
            f"{params.phase:>9.3f} | {params.ipi:>7.3f}"
        )
    return "\n".join(lines)
