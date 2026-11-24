"""On-site, read-only: what the closed blocks of a running or finished session show so far.

    python quicklook.py RUN_DIR

Reads only the JSON written by run_session.py; it never opens the device or a
recording, so it is safe to run in a second terminal during the session. The
numbers come from the device's own spike detection; the registered gates are
recounted from raw samples by build_library.py afterwards.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from oddball2 import analysis, design


def mean(values):
    return '   n/a' if not values else '%6.1f' % float(np.mean(values))


def gate_lines(gate):
    for site, r in gate['sites'].items():
        if r['auc'] is None:
            print('   %s vs sham  no trials' % site)
        else:
            print('   %s vs sham  AUC %.3f  p %.4f  mean %.1f vs %.1f  %s' % (
                site, r['auc'], r['p'], r['mean_positive'], r['mean_negative'], 'pass' if r['passed'] else 'fail'))


def show(run):
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    plan = read(run / 'plan.json')
    if (run / 'warmup.json').is_file():
        warm = read(run / 'warmup.json')
        print('warmup: readout channels %s, readable sites %s, reference %.1f spikes/s'
              % ({s: len(v) for s, v in warm['sites'].items()}, warm['readable'], warm['reference_hz']))
    for amplitude in plan['ladder_mv']:
        for stage, verdict in (('screen', 'confirm'), ('confirm', 'proceed')):
            path = run / ('%s-%d.json' % (stage, amplitude))
            if path.is_file():
                result = read(path)
                print('%s %d mV (actual %.2f): %s %s' % (stage, amplitude, result['actual_mv'],
                      'run confirmation' if stage == 'screen' else 'run main blocks', result[verdict]))
                gate_lines(result['gate'])
    rows = []
    print('block              ok  missed  background/reference   response on own readout (sham: H W F)')
    for block in plan['blocks']:
        path = run / (block['name'] + '-qc.json')
        if not path.is_file():
            continue
        qc = read(path)
        trials = qc['trials']
        by_site = ' '.join('%s %s' % (s, mean([t['r'][s] for t in trials if t['site'] == s and t['real'] and t['valid']]))
                           for s in design.SITES)
        sham = ' '.join(mean([t['r'][s] for t in trials if not t['real']]).strip() for s in design.SITES)
        ratios = ' '.join('%.2f' % r for r in qc.get('activity', {}).get('ratios', []))
        print('%-18s %-3s %6d  %-22s %s (%s)' % (block['name'], 'yes' if qc['passed'] else 'NO',
                                                len(qc.get('missed_onsets', [])), ratios, by_site, sham))
        for error in qc['errors'][:5]:
            print('     ! ' + error)
        if block['stage'] == 'main' and qc['passed']:
            rows += analysis.table(block, qc)
    if rows:
        print('main blocks so far, mean response by recent-use bin (low 0-1, mid 2-4, high 5-8):')
        for site in design.SITES:
            mine = [r for r in rows if r['real'] and r['valid'] and r['site'] == site]
            print('   %s  ' % site + '  '.join('%s %s (n=%d)' % (b, mean([r['r'][site] for r in mine if r['bin'] == b]),
                  sum(r['bin'] == b for r in mine)) for b in design.BIN_NAMES))
    if (run / 'dose.jsonl').is_file():
        lines = (run / 'dose.jsonl').read_text(encoding='utf-8').splitlines()
        if lines:
            print('site pulses booked: %d of %d' % (json.loads(lines[-1])['total'], design.MAX_REAL_PULSES))
    if (run / 'end.json').is_file():
        end = read(run / 'end.json')
        print('ended: %s %s' % (end.get('outcome'), end.get('stop_reason', '')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    show(p.parse_args().run)
