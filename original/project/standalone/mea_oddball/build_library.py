"""Offline: re-check every main block from its closed H5, apply the two gates, build the library.

    python build_library.py RUN_DIR [--out DIR] [--trust-session-qc]

RUN_DIR is the oddball-* directory written by run_session.py. Reading raw ADC
samples needs the MaxWell HDF5 filter (HDF5_PLUGIN_PATH); `--trust-session-qc`
reuses the on-site block checks instead and reads no raw samples.

Writes trials.csv, gates.json, psth.json and, only when both gates pass on the
held-out round, library.json. A failed gate is the result; no library is written.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oddball import analysis, design, library
from oddball.session import write_once


def load(run):
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    if not (run / 'decision.json').is_file():
        raise SystemExit('The session ended before the main blocks; there is nothing to build from.')
    return dict(plan=design.check_plan(read('plan.json')), binding=read('binding.json'), route=read('route.json'),
                decision=read('decision.json'), warmup=read('warmup.json'))


def recheck(run, block, ctx):
    from mea_stim_v2.recording import H5Recording
    path = run / block['name'] / (block['name'] + '.raw.h5')
    config = SimpleNamespace(threshold_h5_dataset=ctx['binding']['threshold_h5_dataset'])
    with H5Recording(path, config) as rec:
        if rec.metadata['chip_id'] != ctx['binding']['chip_id'] or rec.metadata['frame_gaps']:
            raise ValueError('Recording belongs to another chip or has frame gaps: ' + block['name'])
        qc = analysis.block_qc(rec, block, ctx['binding']['session_id'], ctx['route']['channels'],
                               ctx['decision']['roi'], ctx['decision']['reference_hz'])
        counts = analysis.psth(rec, [t['frame'] for t in qc['trials']], ctx['decision']['roi']) if qc['passed'] else None
    return qc, counts


def collect(run, ctx, trust):
    rows, blocks, sums = [], [], {}
    for block in (b for b in ctx['plan']['blocks'] if b['stage'] == 'main'):
        onsite = run / (block['name'] + '-qc.json')
        raw = run / block['name'] / (block['name'] + '.raw.h5')
        if trust:
            if not onsite.is_file():
                continue
            qc, counts = json.loads(onsite.read_text(encoding='utf-8')), None
        elif raw.is_file():
            qc, counts = recheck(run, block, ctx)
        else:
            continue
        blocks.append(dict(name=block['name'], round=block['round'], kind=block['kind'], passed=qc['passed'],
                           errors=qc['errors'], missed_onsets=qc.get('missed_onsets')))
        if not qc['passed']:
            continue
        rows += analysis.table(block, qc)
        if counts is not None:
            for trial, row in zip(qc['trials'], counts):
                if trial['valid']:
                    total, n = sums.get(trial['site'], (0, 0))
                    sums[trial['site']] = (total + row, n + 1)
    edges = np.arange(*analysis.PSTH[:2], analysis.PSTH[2]) / design.RATE * 1000
    psth = dict(bin_start_ms=edges.tolist(), trials={s: n for s, (_, n) in sums.items()},
                spikes_per_trial={s: (total / n).tolist() for s, (total, n) in sums.items()})
    return rows, blocks, psth


def main(run, out, trust):
    ctx = load(run)
    rows, blocks, psth = collect(run, ctx, trust)
    gates = analysis.offline_gates(rows, ctx['plan']['seed'])
    out.mkdir(parents=True, exist_ok=False)
    keys = ('block', 'round', 'kind', 'frequent', 'trial', 'frame', 'site', 'real', 'count', 'bin', 'valid',
            'before', 'after', 'response', 'after_20ms', 'artifact')
    with (out / 'trials.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    report = dict(run=str(run), plan_sha256=ctx['plan']['sha256'], amplitude_mv=ctx['decision']['amplitude_mv'],
                  actual_mv=ctx['decision']['actual_mv'], readout_channels=len(ctx['decision']['roi']),
                  raw_rechecked=not trust, blocks=blocks, gates=gates)
    write_once(out / 'gates.json', report)
    write_once(out / 'psth.json', psth)
    if gates['training_allowed']:
        kept, _, _ = analysis.split_rounds(rows)
        write_once(out / 'library.json', library.build(kept))
    print('blocks passed: %d of %d' % (sum(b['passed'] for b in blocks), len(blocks)))
    for name in ('evoked', 'habituation'):
        if name in gates:
            for site, r in gates[name]['sites'].items():
                print('%-11s %s  AUC %s  p %s  %s' % (name, site, None if r['auc'] is None else round(r['auc'], 3),
                      None if r['p'] is None else round(r['p'], 4), 'pass' if r['passed'] else 'fail'))
    print('training allowed: %s%s' % (gates['training_allowed'], '' if gates['training_allowed'] else ' (' + gates['reason'] + ')'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--trust-session-qc', action='store_true')
    args = p.parse_args()
    main(args.run, args.out or args.run / 'analysis', args.trust_session_qc)
