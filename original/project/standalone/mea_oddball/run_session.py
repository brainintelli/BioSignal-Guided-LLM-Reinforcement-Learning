"""Run the oddball habituation session on the bound MaxOne acquisition host.

Order on the acquisition host, all inside the two hours after loading the sample:

    python quiet_baseline.py < baseline-binding.json        (mea_live_pair, 65 s passive)
    python current_chip_route.py STAGE                      (mea_live_pair, writes session-route.json)
    python run_session.py STAGE --output oddball-v1         (this file)

STAGE holds baseline-binding.json and session-route.json. PYTHONPATH needs
mea_oddball, mea_live_pair, mea_online, mea_binary_r2, mea_state2_r2 and
mea_stimulation_v2. OPERATOR_MANUAL.md has the full procedure.

`--plan` prints the frozen plan and budgets and contacts nothing.
Creating a file named STOP in the run directory ends stimulation at the next
chunk boundary (at most 20 s); the recovery recording and cleanup still run.
"""
import argparse
import json
from pathlib import Path
import signal
import shutil
import time
from types import SimpleNamespace
import uuid

from oddball import design
from oddball.session import START_LIMIT_S, TEARDOWN_S, Session, write_once

SEED = 2026112301
MIN_FREE_BYTES = 80_000_000_000


class Interrupted(BaseException):
    """Signal or deadline: skip the recovery recording, clean up and leave."""


class MaxOneRig:
    """The three routed sites of one MaxOne chip, through the audited backend."""

    def __init__(self, binding, route, out):
        from livepair.device import Device
        from mea_stim_v2 import live
        from mea_stim_v2.recording import H5Recording
        self.mx, self.Sequence, self.Recording = live.mx, live.ManagedSequence, H5Recording
        self.binding, self.route = binding, route
        self.dev = Device(binding, out)
        self.dev.before_sequence_send = None
        self.lsb_mv = None
        self.site_channels = {}

    @property
    def faulted(self):
        return self.dev._faulted

    def prepare(self):
        dev = self.dev
        dev.lsb = self.lsb_mv = dev.preflight()['lsb_mv']
        for amplitude in design.AMPLITUDES_MV:
            design.pulse_bits(self.lsb_mv, amplitude)
        dev.prepare_sites()
        self.site_channels = dict(dev.sites)
        return dict(units=dev.units, channels=dev.sites, lsb_mv=self.lsb_mv,
                    actual_mv={str(a): design.pulse_bits(self.lsb_mv, a) * self.lsb_mv for a in design.AMPLITUDES_MV})

    def open(self, name):
        self.dev.open_record(name)

    def units(self, on):
        mx = self.mx
        with self.dev._sdk():
            for dac in range(len(design.SITES)):
                mx.send(mx.DAC(dac, 512))
            for dac, site in enumerate(design.SITES):
                unit = mx.StimulationUnit(self.dev.units[site])
                mx.send(unit.power_up(on).connect(on).set_voltage_mode().dac_source(dac))

    def build(self, block, bits):
        """Download nothing yet: create every chunk of the block on the server, ready to send."""
        mx, dev = self.mx, self.dev
        if not dev.file_open or not dev.recording:
            raise RuntimeError('Sequences are built inside an open recording')
        handles = []
        with dev._sdk():
            if float(mx.query_DAC_lsb_mV()) != self.lsb_mv:
                raise RuntimeError('DAC step changed')
            for trials in design.chunks(block):
                token = 'odd_seq_' + uuid.uuid4().hex
                dev.sequences.append(token)  # owned before the constructor; its ACK may be lost
                sequence = self.Sequence(token=token, initial_delay=100, persistent=True)
                if sequence.token != token:
                    raise RuntimeError('Server returned a different sequence token')
                for kind, value in design.chunk_codes(block, trials, bits):
                    if kind == 'marker':
                        number = design.event_id(block, value)
                        if number in dev._events_attempted:
                            raise ValueError('Event ID already used')
                        dev._events_attempted.add(number)
                        properties = dict(design.event_properties(block, value), session=dev.config.session_id)
                        text = ' '.join('%s %s' % item for item in sorted(properties.items()))
                        sequence.append(mx.Event(0, 1, number, text))
                    elif kind == 'dac':
                        sequence.append(mx.DAC(*value))
                    else:
                        sequence.append(mx.DelaySamples(value))
                handles.append(sequence)
        return handles

    def send(self, handle, event, gate):
        dev = self.dev
        dev.before_sequence_send, dev._pending_event, dev._pending_pulse = gate, dict(event), True
        try:
            with dev._sdk():
                handle.send()
        finally:
            dev.before_sequence_send, dev._pending_event, dev._pending_pulse = None, None, False

    def close(self):
        return self.dev.close_record()['path']

    def read(self, path):
        return self.Recording(path, self.dev.config)

    def verify(self, rec):
        b = self.binding
        expected = dict(chip_id=b['chip_id'], fs=20000., gain=512., hpf=300., frame_gaps=0,
                        threshold_value=b['threshold_value'])
        if any(rec.metadata[k] != v for k, v in expected.items()) or sorted(rec.mapping) != self.dev.config.validate_binding():
            raise ValueError('Chip, recording settings or route differ from the binding')
        for site, channel in self.site_channels.items():
            if tuple(self.route['coordinates'][site]) != tuple(rec.channel_xy_um[channel]):
                raise ValueError('Physical coordinates differ from the selected sites')

    def cleanup(self):
        return self.dev.cleanup()


def show_plan(seed):
    value = design.plan(seed)
    rows = design.summary(value)
    print('seed %d  plan %s' % (seed, value['sha256']))
    for r in rows:
        print('%2d %-16s trials %3d  H %2d W %2d F %2d sham %2d' % (r['index'], r['name'], r['trials'], r['H'], r['W'], r['F'], r['sham']))
    main = sum(r['real'] for r in rows if r['stage'] == 'main')
    screen = sum(r['real'] for r in rows if r['stage'] == 'screen')
    print('real pulses: screen %d per amplitude, main %d, at most %d (budget %d)'
          % (screen // len(design.AMPLITUDES_MV), main, screen + main, design.MAX_REAL_PULSES))


def run(stage, output, seed):
    import fcntl
    from livepair.protocol import binding
    from mea_stim_v2.live import _server_snapshot
    original = json.loads((stage / 'baseline-binding.json').read_text())
    route = json.loads((stage / 'session-route.json').read_text())
    if any(original[k] != route[k] for k in ('chip_id', 'organoid_id', 'session_id')):
        raise ValueError('Route belongs to a different sample')
    start = binding(original)
    age = time.time() - start
    if not 0 <= age < START_LIMIT_S:
        raise ValueError('Too late after loading for two complete rounds before the stimulation cutoff')
    root = Path(original['record_root'])
    selection = json.loads((root / (original['session_id'] + '-site-selection') / 'site-candidates.json').read_text())
    if (selection['chip'] != original['chip_id'] or selection['source'] != route['baseline']
            or {s: v['electrode'] for s, v in selection['suggested'].items()} != route['electrodes']):
        raise ValueError('Routed electrodes do not match the quiet baseline selection')
    if Path(output).name != output or not output.startswith('oddball-'):
        raise ValueError('Run directory name must start with oddball-')
    b = dict(original, electrodes=route['electrodes'], route_file=route['route_file'], route_sha256=route['route_sha256'],
             material='Pt-Black', prior_nonzero_attempts=0, evidence_mode='oddball_habituation')
    out = root / output
    value = design.plan(seed)
    lease = Path.home() / '.local/state/mea_photo_full/device.lock'
    with lease.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid, opened = _server_snapshot()
        if opened:
            raise ValueError('Acquisition server has an open recording')
        if list(root.rglob('attempt-*.json')) or list(root.rglob('dose.jsonl')):
            raise ValueError('This session directory already holds stimulation records')
        if shutil.disk_usage(root).free < MIN_FREE_BYTES:
            raise ValueError('Less than 80 GB free on the recording volume')
        out.mkdir(exist_ok=False)
        write_once(out / 'binding.json', b)
        write_once(out / 'plan.json', value)
        rig = MaxOneRig(b, route, out)
        signal.alarm(max(1, int(design.HARD_STOP_S - TEARDOWN_S - (time.time() - start))))
        clock = SimpleNamespace(wall=time.time, mono=time.monotonic, sleep=time.sleep)
        session = Session(rig, out, value, start, b['session_id'], b['bad_channels'], clock)
        status = session.run()
        signal.alarm(0)
        print(json.dumps(dict(status, server_closed=_server_snapshot() == (pid, []))), flush=True)
    session.manifest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', type=Path, nargs='?')
    p.add_argument('--output', default='oddball-v1')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--plan', action='store_true')
    args = p.parse_args()
    if args.plan or args.stage is None:
        show_plan(args.seed)
    else:
        def stop(*_):
            raise Interrupted('Signal or hard deadline')
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            signal.signal(sig, stop)
        run(args.stage, args.output, args.seed)
