"""Run the second oddball habituation session on the bound MaxOne acquisition host.

Order on the acquisition host, all inside the two hours after loading the sample:

    python quiet_baseline.py < baseline-binding.json                  (mea_live_pair, 65 s passive)
    python route_sites.py STAGE --electrodes-per-site 5               (this directory, software only)
    python run_session.py STAGE --output oddball2-v1 --ceiling-mv 600 (this file)

STAGE holds baseline-binding.json and session-route.json. PYTHONPATH needs
mea_oddball2, mea_live_pair, mea_online, mea_binary_r2, mea_state2_r2 and
mea_stimulation_v2. OPERATOR_MANUAL.md has the full procedure.

`--plan --ceiling-mv N` prints the frozen plan and budgets and contacts nothing.
Creating a file named STOP in the run directory ends stimulation at the next
chunk boundary (at most 20 s); the recovery recording and cleanup still run.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import shutil
import time
from types import SimpleNamespace

from oddball2 import design, sites as geometry
from oddball2.session import START_LIMIT_S, TEARDOWN_S, Session, write_once

SEED = 2026112401
MIN_FREE_BYTES = 80_000_000_000
PREFIX = 'oddball2-'


class Interrupted(BaseException):
    """Signal or deadline: skip the recovery recording, clean up and leave."""


def show_plan(seed, ceiling_mv):
    value = design.plan(seed, ceiling_mv)
    print('seed %d  ceiling %d mV  plan %s' % (seed, ceiling_mv, value['sha256']))
    for r in design.summary(value):
        print('%2d %-16s trials %3d  H %2d W %2d F %2d sham %2d'
              % (r['index'], r['name'], r['trials'], r['H'], r['W'], r['F'], r['sham']))
    print('amplitudes tried in order: %s mV; a screen that tells two sites from sham is followed by one '
          'confirmation block, and the main blocks run only if that confirms' % value['ladder_mv'])
    print('site pulses: at most %d (budget %d); every pulse drives all electrodes of one site'
          % (design.worst_case_pulses(value), design.MAX_REAL_PULSES))


def check_route(original, route, selection):
    """The route must come from this sample's quiet baseline and follow the site rules."""
    if any(original[k] != route[k] for k in ('chip_id', 'organoid_id', 'session_id')):
        raise ValueError('Route belongs to a different sample')
    if selection['chip'] != original['chip_id'] or selection['source'] != route['baseline']:
        raise ValueError('Route was not built from this quiet baseline')
    electrodes = route['electrodes']
    if set(electrodes) != set(design.SITES) or any(
            not 1 <= len(electrodes[s]) <= design.MAX_SITE_ELECTRODES for s in design.SITES):
        raise ValueError('One to %d electrodes per site required' % design.MAX_SITE_ELECTRODES)
    stable = {r['electrode'] for r in selection['candidates']}
    centres = {}
    for site in design.SITES:
        centre, *others = electrodes[site]
        if centre not in stable or any(e not in geometry.neighbours(centre) for e in others):
            raise ValueError('Site %s is not a stable baseline electrode with its neighbours' % site)
        centres[site] = geometry.electrode_xy(centre)
    for a in design.SITES:
        for b in design.SITES:
            if a < b and math.dist(centres[a], centres[b]) < geometry.MIN_SEPARATION_UM:
                raise ValueError('Site centres are closer than %d um' % geometry.MIN_SEPARATION_UM)


def check_history(root):
    """A sample is stimulated in one session only, and never after a run that left the device in doubt.

    A run that ended before its first real pulse with a clean device does not
    count: the sample is untouched, so the session may be started again.
    """
    if list(root.rglob('attempt-*.json')):
        raise ValueError('This session directory already holds stimulation records')
    for ledger in root.rglob('dose.jsonl'):
        if any(json.loads(line).get('real', 1) for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip()):
            raise ValueError('This sample has already received pulses')
    for audit in root.rglob('api_audit.jsonl'):
        end = audit.with_name('end.json')
        if not end.is_file() or json.loads(end.read_text(encoding='utf-8')).get('cleanup') != []:
            raise ValueError('An earlier run on this sample did not end with a clean device')


def run(stage, output, seed, ceiling_mv):
    import fcntl
    from livepair.protocol import binding
    from mea_stim_v2.live import _server_snapshot
    from oddball2.rig import MaxOneRig
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    original = read(stage / 'baseline-binding.json')
    route = read(stage / 'session-route.json')
    start = binding(original)
    age = time.time() - start
    if not 0 <= age < START_LIMIT_S:
        raise ValueError('Too late after loading for two complete rounds before the stimulation cutoff')
    root = Path(original['record_root'])
    selection = read(root / (original['session_id'] + '-site-selection') / 'site-candidates.json')
    check_route(original, route, selection)
    if Path(output).name != output or not output.startswith(PREFIX):
        raise ValueError('Run directory name must start with ' + PREFIX)
    value = design.plan(seed, ceiling_mv)
    b = dict(original, electrodes=route['electrodes'], route_file=route['route_file'], route_sha256=route['route_sha256'],
             material='Pt-Black', evidence_mode='oddball2_habituation', ceiling_mv=ceiling_mv,
             electrodes_per_site={s: len(v) for s, v in route['electrodes'].items()})
    out = root / output
    lease = Path.home() / '.local/state/mea_photo_full/device.lock'
    with lease.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid, opened = _server_snapshot()
        if opened:
            raise ValueError('Acquisition server has an open recording')
        check_history(root)
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
    p.add_argument('--output', default=PREFIX + 'v1')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--ceiling-mv', type=int, required=True, choices=design.LADDER_MV,
                   help='highest amplitude this session may reach; agreed with the person responsible for the sample')
    p.add_argument('--plan', action='store_true')
    args = p.parse_args()
    if args.plan or args.stage is None:
        show_plan(args.seed, args.ceiling_mv)
    else:
        def stop(*_):
            raise Interrupted('Signal or hard deadline')
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            signal.signal(sig, stop)
        run(args.stage, args.output, args.seed, args.ceiling_mv)
