"""Single authorized first probe using the unchanged, tested standalone package."""
import datetime as dt
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

from probe_guard import ProbeGuard


def write_audit(audit, record, *, strict=False):
    try:
        audit.write(json.dumps(record) + '\n')
        audit.flush()
        return True
    except Exception:
        if strict:
            raise
        return False


def deadline_safe_signals(original_context, cancel_alarm):
    @contextmanager
    def guarded(handler):
        if handler == signal.SIG_IGN:
            cancel_alarm()
        with original_context(handler):
            yield
    return guarded


def validate_profile(profile, template):
    special = {'electrodes_selection', 'assay_path'}
    if set(profile) != set(template):
        raise ValueError('Unexpected profile fields')
    if any(profile[k] != template[k] for k in template if k not in special):
        raise ValueError('First-probe fixed parameters changed')
    e = profile['electrodes_selection']
    if set(e) != {'recording', 'stimulation'} or len(e['stimulation']) != 1:
        raise ValueError('Exactly one stimulation electrode required')
    if e['stimulation'][0] not in e['recording']:
        raise ValueError('Stimulus site must be recorded in the qualified mapping')
    if not profile['assay_path']:
        raise ValueError('Qualified current routing is required')


def main():
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base / 'BrainStimulation'))
    from brain_stimulation import maxlab as mx
    from brain_stimulation import runner
    from brain_stimulation.config import load_config
    from brain_stimulation.transport import Api, Endpoint
    registration = json.loads((base / 'session.json').read_text())
    if registration['device_host'] != '192.0.2.21' or registration['chip_id'] != '22554':
        raise ValueError('Unexpected device/chip registration')
    if registration['r2_enabled'] or registration['realtime_training']:
        raise ValueError('R2 and training must remain off')
    started = dt.datetime.fromisoformat(registration['session_started_at']).timestamp()
    guard = ProbeGuard(base, started_wall=started, wall_now=time.time(), monotonic_now=time.monotonic())
    if guard.claim.exists():
        raise RuntimeError('A previous send attempt exists; refusing this run')
    config_path = base / 'first_probe.json'
    profile = json.loads(config_path.read_text())
    template = json.loads((base / 'BrainStimulation/profiles/maxone_first_probe.template.json').read_text())
    validate_profile(profile, template)
    params = load_config(config_path)
    qualification = json.loads((base / 'qualification_qc.json').read_text())
    if qualification['status'] != 'pass_for_first_probe':
        raise RuntimeError('Qualification has not passed')
    if qualification['selected_electrode'] != profile['electrodes_selection']['stimulation'][0]:
        raise RuntimeError('Qualified stimulus site mismatch')
    expected_cfg = (base / 'qualified_route.cfg').read_bytes()
    if Path(profile['assay_path']).read_bytes() != expected_cfg:
        raise RuntimeError('Qualified route file mismatch')
    remaining = guard.remaining(wall_now=time.time(), monotonic_now=time.monotonic())
    if remaining < params.record_time_seconds() + 300:
        raise RuntimeError('Insufficient time for recording, setup and closeout')
    # A launched run is never automatically restarted, including before send.
    with (base / 'first_probe_launch.json').open('x') as f:
        json.dump({'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'profile_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}, f)
        f.flush()
        os.fsync(f.fileno())
    audit = (base / 'api_audit.jsonl').open('a', buffering=1)
    real_send = Api.send
    original_signal_context = runner.handle_signals
    runner.handle_signals = deadline_safe_signals(original_signal_context,
                            lambda: signal.setitimer(signal.ITIMER_REAL, 0))
    route_checked = False
    downloaded_token = None
    audit_failed = False
    def event(kind, message, *, strict=False, **extra):
        nonlocal audit_failed
        ok = write_audit(audit, {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'kind': kind, 'command': message[:150],
            'command_sha256': hashlib.sha256(message.encode()).hexdigest(), **extra}, strict=strict)
        audit_failed = audit_failed or not ok
    def observed(api, message, timeout=None):
        nonlocal downloaded_token
        if message.startswith('sequence_send '):
            if not route_checked:
                raise RuntimeError('Downloaded route not checked')
            guard.reserve_send(wall_now=time.time(), monotonic_now=time.monotonic())
            event('send_attempt_reserved', message, strict=True)
            os.fsync(audit.fileno())
        event('request', message)
        try:
            reply = real_send(api, message, timeout)
        except BaseException as exc:
            event('error', message, error=repr(exc))
            raise
        event('ack', message, reply=reply[:120])
        if message.startswith('mea_array_command standalone_') and '\nmea_array_download' in message:
            downloaded_token = message.splitlines()[0].split()[1]
        return reply
    Api.send = observed
    def bounded_wait(seconds):
        nonlocal route_checked
        if guard.remaining(wall_now=time.time(), monotonic_now=time.monotonic()) < seconds + 120:
            raise RuntimeError('Insufficient time remaining')
        if not route_checked:
            if not downloaded_token:
                raise RuntimeError('No owned array download acknowledged')
            current = mx.send_raw('mea_array_command ' + downloaded_token + '\nmea_array_get_config')
            (base / 'downloaded_owned_array.cfg').write_text(current)
            before_map = sorted(re.findall(r'(\d+)\((\d+)\)', expected_cfg.decode().split()[0]))
            after_map = sorted(re.findall(r'(\d+)\((\d+)\)', current.split()[0]))
            if before_map != after_map:
                raise RuntimeError('Downloaded recording mapping differs from qualified mapping')
            route_checked = True
            event('route_verified', 'owned downloaded array recording map equals qualification')
        time.sleep(seconds)
    def deadline_stop(signum, frame):
        raise KeyboardInterrupt('Biological session deadline; reserving 120 seconds for cleanup')
    signal.signal(signal.SIGALRM, deadline_stop)
    signal.setitimer(signal.ITIMER_REAL, remaining - 120)
    try:
        code = runner.run_experiment(params, Endpoint(timeout=20), base / 'results',
                                     wait=bounded_wait)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        runner.handle_signals = original_signal_context
        Api.send = real_send
        audit.close()
    if audit_failed:
        print('AUDIT_WRITE_FAILURE: inspect hardware result and files; no automatic retry', file=sys.stderr)
        return 1
    return code


if __name__ == '__main__':
    if sys.argv[1:] != ['--run']:
        raise SystemExit('Use --run only for the authorized fixed physical first probe')
    raise SystemExit(main())
