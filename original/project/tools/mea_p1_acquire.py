"""One bounded, unstimulated recording. Run with MaxLab vendor Python -s."""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

LEGACY = '/opt/braindesktop/bin/tcp_server_debug'
LEGACY_SHA = 'd08981e1f86332c0cd404fb6386472ecd1c6690d6f61855900899e78851a008b'
CONFIG_QUERY = 'mea_array_command online\nmea_array_get_config'


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_mapping(config):
    tokens = config.split()
    if not tokens or not tokens[0].endswith(';'): raise ValueError('incomplete mapping')
    mapping = []
    for part in tokens[0][:-1].split(';'):
        match = re.fullmatch(r'(\d+)\((\d+)\)(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)', part)
        if not match: raise ValueError('invalid mapping')
        channel, electrode = int(match[1]), int(match[2])
        if channel >= 1024: raise ValueError('invalid channel')
        mapping.append([channel, electrode])
    if any(len({p[i] for p in mapping}) != len(mapping) for i in (0, 1)):
        raise ValueError('mapping is not one-to-one')
    return sorted(mapping)


class CheckedApi:
    patterns = (r'saving_open_dir /[A-Za-z0-9_./-]+', r'saving_start_file [A-Za-z0-9_-]+',
                r'saving_stop_file', r'saving_start_recording', r'saving_stop_recording',
                r'saving_set_recording_wells 0', r'saving_use_old_file_format 0',
                r'saving_group_define 0 p1_[A-Za-z0-9_]+ [0-9,]+',
                r'saving_group_delete 0 p1_[A-Za-z0-9_]+')

    def __init__(self, api, log_path=None):
        self.api = api; self.log = []; self.log_path = log_path; self.directory = None

    def send(self, command):
        query = command in (CONFIG_QUERY, 'saving_group_info 0')
        if not query and not any(re.fullmatch(p, command) for p in self.patterns):
            raise ValueError('command outside recording allowlist')
        row = dict(time_utc=utc(), command=command)
        try:
            reply = self.api.send(command, timeout=5)
            row['reply'] = reply
            if not isinstance(reply, str) or reply.strip().lower().startswith('error'):
                raise RuntimeError('MaxLab API error: '+command.split()[0])
            accepted = {'ok'}
            if command == 'saving_use_old_file_format 0': accepted.add('using new file format')
            if command.startswith('saving_open_dir '): accepted.add(command.split(' ', 1)[1].lower())
            if command.startswith('saving_start_file ') and self.directory:
                accepted.add((self.directory+'/'+command.split(' ', 1)[1]+'.raw.h5').lower())
            # ApiComm returns '' only after receiving its response end marker;
            # EOF/timeout raise. This server's start-recording acknowledgement is empty.
            if command == 'saving_start_recording': accepted.add('')
            if command == 'saving_stop_recording': accepted.add('stopped recording')
            if command in ('saving_stop_recording', 'saving_stop_file'): accepted.add('stopped')
            if not query and reply.strip().lower() not in accepted:
                raise RuntimeError('Unexpected acknowledgement: '+repr(reply))
            if query and not reply.strip(): raise RuntimeError('Empty query result')
            if command.startswith('saving_open_dir '): self.directory = command.split(' ', 1)[1]
            return reply
        except Exception as exc:
            row['error'] = str(exc); raise
        finally:
            self.log.append(row)
            if self.log_path:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())


class RecordingOwner:
    def __init__(self, api):
        self.api = api; self.file_started = False; self.recording_started = False; self.group_name = None

    def cleanup(self):
        errors = []
        for attr, command in (('recording_started', 'saving_stop_recording'),
                              ('file_started', 'saving_stop_file'),
                              ('group_name', 'saving_group_delete 0 '+str(self.group_name))):
            if not getattr(self, attr): continue
            try:
                self.api.send(command); setattr(self, attr, None if attr == 'group_name' else False)
            except Exception as exc: errors.append(str(exc))
        if errors: raise RuntimeError('; '.join(errors))


def exclusive_directory(path): Path(path).mkdir(parents=True, exist_ok=False)


def capture_budget(seconds):
    if type(seconds) is not int or not 5 <= seconds <= 1800:
        raise ValueError('record duration must be an integer in 5..1800 seconds')
    raw_bytes=(seconds+20)*20_000*1024*2
    return dict(collector_seconds=seconds+20,
                required_free_bytes=5_000_000_000+(raw_bytes*11+9)//10)


def write_capture_progress(root,seconds,elapsed):
    free=shutil.disk_usage(root).free
    if free<2_000_000_000:raise RuntimeError('free disk below2GB during capture')
    h5=root/'baseline.raw.h5'
    status=dict(state='recording',checked_utc=utc(),requested_seconds=seconds,elapsed_seconds=elapsed,
        h5_bytes=h5.stat().st_size if h5.exists() else 0,free_bytes=free,
        stimulation=False,alpha=1,r2_reward_enabled=False,realtime_training=False,peak_windows_complete=False)
    temporary=root/'acquisition_progress.partial.json'
    temporary.write_text(json.dumps(status,indent=2)+'\n')
    temporary.replace(root/'acquisition_progress.json')


def proc_identity(pid):
    base = Path('/proc')/str(pid)
    return dict(exe=os.readlink(base/'exe'), cmd=(base/'cmdline').read_bytes().split(b'\0')[:-1],
                start=(base/'stat').read_text().rsplit(')', 1)[1].split()[19])


def alive(pid, identity):
    try: return proc_identity(pid) == identity
    except FileNotFoundError: return False


def preflight(pid):
    identity = proc_identity(pid)
    if identity['exe'] != LEGACY or identity['cmd'] != [LEGACY.encode(), b'5555']:
        raise RuntimeError('unexpected legacy process identity')
    if hashlib.sha256(Path(LEGACY).read_bytes()).hexdigest() != LEGACY_SHA:
        raise RuntimeError('legacy binary changed')
    servers = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit(): continue
        try:
            exe = os.readlink(item/'exe')
            if exe.endswith('/mxwserver'): servers.append(int(item.name))
            if 'BrainDesktop' in exe: raise RuntimeError('BrainDesktop is running')
        except (FileNotFoundError, PermissionError): continue
    if len(servers) != 1: raise RuntimeError('expected one MaxLab server')
    server = servers[0]
    for fd in (Path('/proc')/str(server)/'fd').iterdir():
        try: target = os.readlink(fd)
        except FileNotFoundError: continue
        if re.search(r'\.h5(?:$| \(deleted\)$)', target):
            raise RuntimeError('MaxLab already has an H5 open')
    connections = subprocess.check_output(['ss', '-Htn', 'state', 'established', '( sport = :5555 )'], text=True)
    if connections.strip(): raise RuntimeError('legacy receiver occupied')
    return identity, server, proc_identity(server)


@contextmanager
def cleanup_step(record, label):
    try: yield
    except Exception as exc: record['errors'].append(label+': '+str(exc))


def start_legacy_process(command, log):
    # POSIX ignores survive exec. The single-threaded coordinator ignores signals
    # during cleanup; its replacement service must receive normal stop signals.
    def reset_child_signals():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            preexec_fn=reset_child_signals)


def finalize_capture(owner, collector, collector_log, api, record, root, legacy_pid, legacy_identity, legacy_stopped):
    """Attempt every independent cleanup step and always attempt the final ledger write."""
    try:
        with cleanup_step(record, 'recording cleanup'):
            if owner: owner.cleanup()
        with cleanup_step(record, 'collector cleanup'):
            if collector and collector.poll() is None:
                collector.send_signal(signal.SIGINT)
                record['collector_exit'] = collector.wait(timeout=10)
        with cleanup_step(record, 'collector log close'):
            if collector_log: collector_log.close()
        with cleanup_step(record, 'API shutdown'):
            if api: api.api.shutdown()
        with cleanup_step(record, 'legacy restoration'):
            if legacy_stopped:
                # Failed SDK close requires operator action, never an automatic MaxLab restart.
                safe_restore = collector is None
                if collector and collector.poll() is not None:
                    footer = json.loads((root/'events.jsonl').read_bytes().splitlines()[-1])
                    safe_restore = footer.get('type') == 'end' and footer.get('sdk_close_status') == 0
                if safe_restore and not alive(legacy_pid, legacy_identity):
                    with (root/'legacy_restored.log').open('xb') as log:
                        restored = start_legacy_process([LEGACY, '5555'], log)
                    time.sleep(1)
                    record['restored_legacy_pid'] = restored.pid
                    record['legacy_restored'] = restored.poll() is None
                    if not record['legacy_restored']: raise RuntimeError('restored process exited')
                else: raise RuntimeError('held; inspect SDK cleanup/process status')
    finally:
        if record['errors']: record['success'] = False
        record['finished_utc'] = utc()
        (root/'acquisition.json').write_text(json.dumps(record, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--collector', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--legacy-pid', type=int, required=True)
    p.add_argument('--chip-id', default='22021')
    args = p.parse_args()
    budget=capture_budget(args.seconds)
    root = args.output.resolve(); exclusive_directory(root)
    if shutil.disk_usage(root).free < budget['required_free_bytes']: raise RuntimeError('insufficient free disk for uncompressed recording budget')
    record = dict(schema='mea.acquisition.v1', started_utc=utc(), expected_chip_id=args.chip_id,
                  organoid_id=None, organoid_type=None, culture_days=None, stimulation=False,
                  alpha=1, r2_reward_enabled=False, success=False, errors=[],
                  requested_seconds=args.seconds, original_legacy_pid=args.legacy_pid,capacity_budget=budget)
    api = owner = collector = legacy_identity = None
    legacy_stopped = False
    collector_log = None
    def interrupted(signum, frame): raise InterruptedError('coordinator signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        from maxlab.comm import ApiComm
        legacy_identity, server, server_identity = preflight(args.legacy_pid)
        record['maxlab_pid'] = server
        api = CheckedApi(ApiComm(timeout=5), root/'api_commands.jsonl'); owner = RecordingOwner(api)
        if api.send('saving_group_info 0').strip().lower() != 'no groups defined':
            raise RuntimeError('existing recording groups present')
        config = api.send(CONFIG_QUERY)
        (root/'mapping_before.cfg').write_text(config)
        record['mapping_before'] = parse_mapping(config)
        record['collector_sha256'] = hashlib.sha256(args.collector.read_bytes()).hexdigest()
        if not alive(args.legacy_pid, legacy_identity): raise RuntimeError('legacy changed')
        os.kill(args.legacy_pid, signal.SIGTERM); legacy_stopped = True
        for _ in range(50):
            if not alive(args.legacy_pid, legacy_identity): break
            time.sleep(.1)
        else: raise RuntimeError('legacy did not exit; no forced termination')
        collector_log = (root/'collector.log').open('xb')
        collector = subprocess.Popen([str(args.collector.resolve()), str(root/'events.jsonl'), str(budget['collector_seconds'])],
                                     stdout=collector_log, stderr=subprocess.STDOUT)
        for _ in range(100):
            if collector.poll() is not None: raise RuntimeError('collector exited during startup')
            if (root/'events.jsonl').exists():
                first = (root/'events.jsonl').read_bytes().split(b'\n')[0]
                if first and json.loads(first).get('sdk_open_status') == 0: break
            time.sleep(.1)
        else: raise RuntimeError('collector did not become ready')
        time.sleep(2)
        api.send('saving_use_old_file_format 0')
        api.send('saving_open_dir '+str(root))
        owner.file_started = True  # also clean up an ambiguous start acknowledgement
        api.send('saving_start_file baseline')
        owner.group_name = 'p1_raw'
        api.send('saving_group_define 0 p1_raw '+','.join(str(c) for c, e in record['mapping_before']))
        api.send('saving_set_recording_wells 0')
        owner.recording_started = True
        api.send('saving_start_recording'); record['recording_started_utc'] = utc()
        recording_start=time.monotonic();deadline = recording_start+args.seconds;next_check=recording_start
        while time.monotonic() < deadline:
            if collector.poll() is not None: raise RuntimeError('collector exited while recording')
            if time.monotonic()>=next_check:
                write_capture_progress(root,args.seconds,time.monotonic()-recording_start)
                next_check=time.monotonic()+5
            time.sleep(min(.2, max(0, deadline-time.monotonic())))
        owner.cleanup(); record['recording_stopped_utc'] = utc()
        time.sleep(2)
        collector.send_signal(signal.SIGINT)
        record['collector_exit'] = collector.wait(timeout=10)
        if record['collector_exit'] != 0: raise RuntimeError('collector returned failure')
        config = api.send(CONFIG_QUERY)
        (root/'mapping_after.cfg').write_text(config)
        record['mapping_after'] = parse_mapping(config)
        if record['mapping_after'] != record['mapping_before']: raise RuntimeError('mapping changed')
        if not alive(server, server_identity): raise RuntimeError('MaxLab server changed')
        record['success'] = True
    except Exception as exc:
        record['errors'].append(str(exc))
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        finalize_capture(owner, collector, collector_log, api, record, root,
                         args.legacy_pid, legacy_identity, legacy_stopped)
    print(json.dumps({k:v for k,v in record.items() if not k.startswith('mapping_')}))
    return 0 if record['success'] else 2


if __name__ == '__main__': raise SystemExit(main())
