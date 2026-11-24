"""Offline: recount every main block from raw samples, apply the registered gates, build the library.

    python build_library.py RUN_DIR [--out DIR] [--rho-min X] [--preview]

RUN_DIR is the oddball2-* directory written by run_session.py. Reading raw ADC
samples needs the MaxWell HDF5 filter (HDF5_PLUGIN_PATH).

Each main block is checked again from its closed file, and the 5-50 ms counts
are taken from raw samples with the pulse blanked and a causal filter (see
oddball2/detect.py); real and sham trials are treated identically. The gates
are evaluated on the held-out round only.

Writes trials.csv, gates.json, psth.json and, when at least two sites pass every
response gate, library.json. `training_allowed` in gates.json additionally needs
`--rho-min`, the reliability threshold measured by stage 0; without it the
library is written but not cleared for training. A failed gate is the result.

`--preview` uses the on-site checks and the device's own spike detection, reads
no raw samples and never writes a library.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oddball2 import analysis, design, detect, library
from oddball2.session import write_once


def load(run):
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    if not (run / 'decision.json').is_file():
        raise SystemExit('The session ended before the main blocks; there is nothing to build from.')
    return dict(plan=design.check_plan(read('plan.json')), binding=read('binding.json'), route=read('route.json'),
                decision=read('decision.json'))


def opened(run, name, ctx):
    from mea_stim_v2.recording import H5Recording
    rec = H5Recording(run / name / (name + '.raw.h5'),
                      SimpleNamespace(threshold_h5_dataset=ctx['binding']['threshold_h5_dataset']))
    if rec.metadata['chip_id'] != ctx['binding']['chip_id'] or rec.metadata['frame_gaps']:
        rec.close()
        raise ValueError('Recording belongs to another chip or has frame gaps: ' + name)
    return rec


def collect(run, ctx, preview):
    roi = dict(sites=ctx['decision']['roi']['sites'])
    rows, blocks = [], []
    sums = {s: dict(real=[0, 0], sham=[0, 0]) for s in design.SITES}
    sos = threshold = None
    if not preview:
        with opened(run, 'warmup', ctx) as rec:
            sos = detect.band_pass(rec.fs)
            threshold = detect.thresholds(rec, analysis.union(roi), sos)
    for block in (b for b in ctx['plan']['blocks'] if b['stage'] == 'main'):
        name = block['name']
        if preview:
            if not (run / (name + '-qc.json')).is_file():
                continue
            qc = json.loads((run / (name + '-qc.json')).read_text(encoding='utf-8'))
            table = analysis.table(block, qc) if qc['passed'] else []
        elif (run / name / (name + '.raw.h5')).is_file():
            with opened(run, name, ctx) as rec:
                qc = analysis.block_qc(rec, block, ctx['binding']['session_id'], ctx['route']['channels'], roi,
                                       ctx['decision']['reference_hz'])
                table = []
                if qc['passed']:
                    table = analysis.table(block, qc)
                    frames = [t['frame'] for t in qc['trials']]
                    for row, counts in zip(table, detect.recount(rec, qc, roi, threshold, sos)):
                        row['device_r'], row['r'] = row['r'], counts
                    for site in design.SITES:
                        if roi['sites'][site]:
                            histogram = analysis.psth(rec, frames, roi['sites'][site])
                            for trial, line in zip(qc['trials'], histogram):
                                kind = 'real' if trial['real'] and trial['valid'] and trial['site'] == site else \
                                    'sham' if not trial['real'] else None
                                if kind:
                                    sums[site][kind][0] = sums[site][kind][0] + line
                                    sums[site][kind][1] += 1
        else:
            continue
        blocks.append(dict(name=name, round=block['round'], kind=block['kind'], passed=qc['passed'],
                           errors=qc['errors'], missed_onsets=qc.get('missed_onsets')))
        rows += table
    edges = np.arange(*analysis.PSTH[:2], analysis.PSTH[2]) / design.RATE * 1000
    psth = dict(bin_start_ms=edges.tolist(), detection='device', note='device-detected spikes; unreliable from -1 to +2.5 ms',
                sites={s: {k: dict(trials=n, spikes_per_trial=(total / n).tolist()) for k, (total, n) in v.items() if n}
                       for s, v in sums.items()})
    return rows, blocks, psth, threshold


def main(run, out, rho_min=None, preview=False):
    ctx = load(run)
    if out.exists():
        raise FileExistsError('Output directory already exists: %s' % out)
    rows, blocks, psth, threshold = collect(run, ctx, preview)
    gates = analysis.offline_gates(rows, ctx['plan']['seed'], sites=ctx['decision']['readable'], rho_min=rho_min)
    out.mkdir(parents=True, exist_ok=False)
    keys = ['block', 'round', 'kind', 'frequent', 'trial', 'frame', 'site', 'real', 'count', 'bin', 'valid', 'rate_hz']
    flat = []
    for row in rows:
        line = {k: row[k] for k in keys}
        for site in design.SITES:
            line['r_' + site] = row['r'][site]
            if 'device_r' in row:
                line['device_r_' + site] = row['device_r'][site]
        flat.append(line)
    extra = ['r_' + s for s in design.SITES] + ([] if preview else ['device_r_' + s for s in design.SITES])
    with (out / 'trials.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, keys + extra)
        writer.writeheader()
        writer.writerows(flat)
    report = dict(run=str(run), plan_sha256=ctx['plan']['sha256'], amplitude_mv=ctx['decision']['amplitude_mv'],
                  actual_mv=ctx['decision']['actual_mv'], readable_sites=ctx['decision']['readable'],
                  readout_channels={s: len(v) for s, v in ctx['decision']['roi']['sites'].items()},
                  detection='device spike detection (preview only)' if preview else
                  'raw samples, pulse blanked, causal 300-3000 Hz band-pass, -5 sigma of the warmup noise',
                  window_ms=[1000 * f / design.RATE for f in analysis.RESPONSE], preview=preview, blocks=blocks, gates=gates)
    if gates['library_allowed'] and not preview:
        kept, _, _ = analysis.split_rounds(rows)
        try:
            built = library.build(kept, gates['usable_sites'], gates['reliability'])
        except ValueError as error:
            report['library_error'] = str(error)
        else:
            write_once(out / 'library.json', built)
            report['library_sha256'] = built['sha256']
    write_once(out / 'gates.json', report)
    write_once(out / 'psth.json', psth)
    if threshold is not None:
        write_once(out / 'thresholds.json', dict(unit='ADC steps', sigmas=detect.SIGMAS, channels=threshold))
    print('blocks passed: %d of %d' % (sum(b['passed'] for b in blocks), len(blocks)))
    for name in ('evoked', 'habituation'):
        if name in gates:
            for site, r in gates[name]['sites'].items():
                print('%-11s %s  AUC %s  p %s  %s' % (name, site, None if r['auc'] is None else round(r['auc'], 3),
                      None if r['p'] is None else round(r['p'], 4), 'pass' if r['passed'] else 'fail'))
    if 'stability' in gates:
        print('stability   ' + '  '.join('%s %s' % (s, 'pass' if r['passed'] else 'fail')
                                         for s, r in gates['stability']['sites'].items()))
    print('usable sites: %s' % gates.get('usable_sites', []))
    written = (out / 'library.json').is_file()
    print('library written: %s%s' % (written, '' if written else ' (%s)' % (
        'preview' if preview and gates['library_allowed'] else report.get('library_error') or gates.get('reason'))))
    if 'reliability' in gates:
        print('reliability rho: %s (stage 0 threshold: %s)' % (gates['reliability']['rho'], gates['reliability']['rho_min']))
    print('training allowed: %s%s' % (gates['training_allowed'] and written,
                                      '' if gates['training_allowed'] else ' (' + gates['reason'] + ')'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--rho-min', type=float, help='reliability the library must reach, measured by stage 0')
    p.add_argument('--preview', action='store_true')
    args = p.parse_args()
    main(args.run, args.out or args.run / ('preview' if args.preview else 'analysis'), args.rho_min, args.preview)
