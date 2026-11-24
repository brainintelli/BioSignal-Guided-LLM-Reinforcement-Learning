"""MaxOne stimulation lifecycle, independent of Qt, cloud and realtime TCP."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import signal
import sys
import threading
import time
import uuid

from . import maxlab as mx
from .config import describe
from .plan import SEQUENCE_INITIAL_DELAY_SAMPLES
from .transport import bind_sdk


class ManagedSequence(mx.Sequence):
    # The SDK destructor opens a connection even when persistent=True. Cleanup
    # belongs to the lifecycle below, never to GC or interpreter shutdown.
    def __del__(self):
        pass


def dac_values(params):
    lsb = float(mx.query_DAC_lsb_mV())
    if not math.isfinite(lsb) or lsb <= 0:
        raise RuntimeError(f"Invalid DAC LSB from MaxLab: {lsb}")
    amplitudes = params.amplitude_values()
    if any(int(value / lsb) > 511 for value in amplitudes):
        raise ValueError(f"Amplitude exceeds the bipolar 10-bit DAC range (LSB={lsb} mV)")
    for value in amplitudes:
        if value > 0 and int(value / lsb) == 0:
            raise ValueError(f"Nonzero amplitude {value:g} mV quantizes to zero "
                             f"(DAC LSB={lsb:g} mV); refusing a zero-output stimulation run")
    return lsb


def check_connection(params, endpoint):
    with bind_sdk(mx, endpoint):
        return {"host": endpoint.host, "port": endpoint.port,
                "dac_lsb_mv": dac_values(params), "status": "api_ready"}


@contextmanager
def handle_signals(handler):
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def build_sequence(params, schedule, lsb, sequence):
    timing = params.timing()
    event_id = 1
    for burst, amplitude in enumerate(schedule):
        bits = int(amplitude / lsb)
        for pulse in range(timing.pulses):
            event_id += 1
            sequence.append(mx.Event(0, 1, event_id, f"amplitude {bits} event_id {event_id}"))
            sequence.append(mx.DAC(0, 512 - bits))
            sequence.append(mx.DelaySamples(timing.phase_samples))
            sequence.append(mx.DAC(0, 512 + bits))
            sequence.append(mx.DelaySamples(timing.phase_samples))
            sequence.append(mx.DAC(0, 512))
            if pulse < timing.pulses - 1:
                sequence.append(mx.DelaySamples(timing.interpulse_delay_samples))
        if burst < len(schedule) - 1:
            sequence.append(mx.DelaySamples(timing.interburst_delay_samples))


def run_experiment(params, endpoint, output_dir, *, recording_dir=None,
                   schedule=None, wait=time.sleep):
    local_host = endpoint.host.lower() in {"localhost", "127.0.0.1", "::1"}
    if not local_host and not recording_dir:
        raise ValueError("Remote MaxLab requires --recording-dir with a server-side absolute path")
    if recording_dir:
        if not PurePosixPath(recording_dir).is_absolute():
            raise ValueError("--recording-dir must be an absolute path on the Ubuntu MaxLab server")
        if any(char in recording_dir for char in "\r\n\x00") or "###" in recording_dir:
            raise ValueError("Invalid recording directory")
        try:
            recording_dir.encode("latin1")
        except UnicodeEncodeError as exc:
            raise ValueError("MaxLab API requires a Latin-1 recording path; use an ASCII directory") from exc
    recording, stimulation = params.validated_electrodes()
    schedule = params.amplitude_schedule() if schedule is None else schedule
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    output = Path(output_dir).resolve() / run_id
    h5_directory = str(PurePosixPath(recording_dir) / run_id) if recording_dir else str(output / "h5")
    if not recording_dir:
        try:
            h5_directory.encode("latin1")
        except UnicodeEncodeError as exc:
            raise ValueError("Use an ASCII --output-dir for the MaxLab recording path") from exc
    if any(char in h5_directory for char in "\r\n\x00") or "###" in h5_directory:
        raise ValueError("Invalid H5 path")
    output.mkdir(parents=True, exist_ok=False)
    if local_host:
        Path(h5_directory).mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": asdict(endpoint), "parameters": asdict(params),
        "plan": describe(params, schedule), "h5_directory_on_server": h5_directory,
        "h5_file_stem": "stimulation", "cleanup_errors": [],
    }

    def save_report():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "run.json")

    def log(message):
        print(message, file=sys.stderr, flush=True)

    save_report()
    log(f"本次实验目录：{output}")
    touched = file_attempted = recording_attempted = False
    array_token = sequence_token = None
    saving = mx.Saving()
    code = 0
    with handle_signals(interrupt), bind_sdk(mx, endpoint):
        try:
            lsb = dac_values(params)
            report["dac_lsb_mv"] = lsb
            report["actual_amplitude_schedule_mv"] = [int(value / lsb) * lsb for value in schedule]
            save_report()
            touched = True
            log("初始化 MaxLab，配置电极路由…")
            mx.initialize()
            mx.send(mx.Core().enable_stimulation_power(True))
            # Register known tokens before creation: the server may create the
            # resource and then lose the reply, leaving the constructor failed.
            array_token = "standalone_" + uuid.uuid4().hex
            array = mx.Array(array_token, persistent=True)
            if params.assay_path:
                array.load_config(params.assay_path)
            else:
                array.reset()
                array.clear_selected_electrodes()
                array.select_electrodes(recording + stimulation)
                array.route()
            units = []
            for electrode in stimulation:
                array.connect_electrode_to_stimulation(electrode)
                result = array.query_stimulation_at_electrode(electrode).strip()
                if not result or not result.isdigit() or not 0 <= int(result) < 32:
                    raise RuntimeError(f"No stimulation unit connected to electrode {electrode}: {result!r}")
                unit = int(result)
                if unit in units:
                    raise RuntimeError(f"Electrode {electrode} shares stimulation unit {unit}; select a neighbor")
                units.append(unit)
            report["stimulation_units"] = dict(zip(map(str, stimulation), units))
            mx.activate([0])
            array.download([0])
            wait(mx.Timing.waitAfterDownload)
            mx.offset()
            array.save_config(str(output / "routing.cfg"))
            # Prepare the full sequence before recording, so API upload time is
            # not silently added to the requested pre-stimulation period.
            sequence_token = "standalone_" + uuid.uuid4().hex
            sequence = ManagedSequence(token=sequence_token, initial_delay=0, persistent=True)
            sequence_token = sequence.token
            sequence.append(mx.DelaySamples(SEQUENCE_INITIAL_DELAY_SAMPLES))
            build_sequence(params, schedule, lsb, sequence)
            mx.send(mx.DAC(0, 512))
            for unit in units:
                mx.send(mx.StimulationUnit(unit).power_up(True).connect(True).set_voltage_mode().dac_source(0))

            saving.open_directory(h5_directory)
            saving.set_legacy_format(False)
            file_attempted = True  # A timeout can occur after a server-side start.
            saving.start_file("stimulation")
            for key, value in params.assay_inputs().items():
                saving.write_assay_input(key, value)
            saving.group_delete_all()
            if not params.record_only_spikes:
                saving.group_define(0, "routed", list(range(1024)))
            mx.clear_events()
            recording_attempted = True
            saving.start_recording([0])
            report["status"] = "recording"
            save_report()
            log(f"刺激前记录 {params.pre_recording_seconds():g} 秒")
            wait(params.pre_recording_seconds())
            log(f"发送刺激序列，预计 {params.stimulation_seconds():g} 秒")
            sequence.send()
            wait(params.stimulation_seconds())
            log(f"刺激后记录 {params.post_recording_seconds():g} 秒")
            wait(params.post_recording_seconds())
            report["status"] = "completed"
        except KeyboardInterrupt as exc:
            code = 130
            report.update(status="interrupted", error=str(exc) or "KeyboardInterrupt")
            log("收到停止请求，正在断开刺激单元并关闭记录…")
        except Exception as exc:
            code = 1
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log(f"实验失败：{exc}")
        finally:
            def attempt(label, action):
                try:
                    action()
                except Exception as exc:
                    report["cleanup_errors"].append(f"{label}: {exc}")
                    log(f"清理失败 {label}：{exc}")

            # Every cleanup operation gets its own chance even if an earlier
            # command fails. Bound each request so a disconnected server cannot
            # block teardown forever. Keep the endpoint bound until all finish.
            with handle_signals(signal.SIG_IGN), bind_sdk(mx, replace(endpoint, timeout=min(endpoint.timeout, 2.0))):
                if touched:
                    for unit in range(32):
                        attempt(f"poweroff unit {unit}", lambda unit=unit:
                                mx.send(mx.StimulationUnit(unit).power_up(False).connect(False)))
                    attempt("reset DAC", lambda: mx.send(mx.DAC(0, 512)))
                if recording_attempted:
                    attempt("stop recording", saving.stop_recording)
                if file_attempted:
                    attempt("stop file", saving.stop_file)
                    attempt("delete recording groups", saving.group_delete_all)
                if sequence_token is not None:
                    attempt("delete sequence", lambda: mx.send_raw("sequence_delete " + sequence_token))
                if array_token is not None:
                    attempt("delete array", lambda: mx.send_raw("mea_array_delete " + array_token))
            if report["cleanup_errors"]:
                code = 1
                report["status"] = "failed"
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save_report()
    log(f"实验状态：{report['status']}；详情：{output / 'run.json'}")
    return code
