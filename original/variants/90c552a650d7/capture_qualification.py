"""Single 120 s saving-only acquisition; no initialize, routing or stimulus calls."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

BASE = Path('/home/researcher/mea_first_probe_20261117')
sys.path.insert(0, str(BASE / 'BrainStimulation'))
from brain_stimulation import maxlab as mx
from brain_stimulation.transport import Api, Endpoint, bind_sdk

ROOT = BASE / 'qualification_120s_v2'
ALLOWED = {
    'mea_array_command online\nmea_array_get_config',
    'system_query_dac_lsb_mv',
    'saving_open_dir ' + str(ROOT),
    'saving_use_old_file_format 0', 'saving_start_file qualification',
    'saving_group_clear',
    'saving_group_define 0 routed ' + ','.join(map(str, range(1024))),
    'saving_set_recording_wells 0', 'saving_start_recording',
    'saving_stop_recording', 'saving_stop_file',
}


def main():
    ROOT.mkdir(exist_ok=False)
    report = {'status': 'starting', 'stimulation_sent': False, 'duration_requested_s': 120,
              'commands': [], 'cleanup_errors': []}
    def flush():
        (ROOT / 'capture.json').write_text(json.dumps(report, indent=2))
    original = Api.send
    def guarded(api, message, timeout=None):
        if message not in ALLOWED:
            raise RuntimeError('Saving-only allowlist rejected: ' + message[:100])
        report['commands'].append({'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                   'message': message})
        flush()
        return original(api, message, timeout)
    Api.send = guarded
    def stop(signum, frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(240)
    file_attempted = recording_attempted = False
    saving = mx.Saving()
    try:
        # The same host must have no currently open H5 in the device server.
        opened = []
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                if (p / 'comm').read_text().strip() != 'mxwserver':
                    continue
                for fd in (p / 'fd').iterdir():
                    target = os.readlink(fd)
                    if '.h5' in target:
                        opened.append(target)
            except (FileNotFoundError, PermissionError):
                continue
        if opened:
            raise RuntimeError('Existing device H5 open: ' + repr(opened))
        with bind_sdk(mx, Endpoint(timeout=10)):
            cfg = mx.send_raw('mea_array_command online\nmea_array_get_config')
            (ROOT / 'before.cfg').write_text(cfg)
            report['mapping_sha256'] = hashlib.sha256(cfg.encode()).hexdigest()
            report['dac_lsb_mv'] = float(mx.query_DAC_lsb_mV())
            saving.open_directory(str(ROOT))
            saving.set_legacy_format(False)
            file_attempted = True
            saving.start_file('qualification')
            saving.group_delete_all()
            saving.group_define(0, 'routed', list(range(1024)))
            recording_attempted = True
            saving.start_recording([0])
            report['recording_start_ack_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            report['status'] = 'recording'
            flush()
            print('QUALIFICATION_RECORDING_STARTED', flush=True)
            start = time.monotonic()
            time.sleep(120)
            report['wall_recorded_seconds'] = time.monotonic() - start
            report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
    finally:
        signal.alarm(0)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        with bind_sdk(mx, Endpoint(timeout=2)):
            for label, should, fn in [('stop_recording', recording_attempted, saving.stop_recording),
                                      ('stop_file', file_attempted, saving.stop_file),
                                      ('delete_groups', file_attempted, saving.group_delete_all)]:
                if should:
                    try:
                        fn()
                    except Exception as exc:
                        report['cleanup_errors'].append([label, repr(exc)])
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        flush()
        Api.send = original
    print(json.dumps({k: v for k, v in report.items() if k != 'commands'}), flush=True)
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    if sys.argv[1:] != ['--run']:
        raise SystemExit('Use --run for the explicitly authorized 120 s saving-only capture')
    raise SystemExit(main())
