"""On-site, read-only: what the closed blocks of a running or finished session show so far.

    python quicklook.py RUN_DIR

Reads only the JSON written by run_session.py; it never opens the device or a
recording, so it is safe to run in a second terminal during the session.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from oddball import analysis, design


def mean(values):
    return '   n/a' if not values else '%6.1f' % float(np.mean(values))


def show(run):
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    plan = read(run / 'plan.json')
    for path in sorted(run.glob('screen-*.json')):
        result = read(path)
        print('screen %d mV (actual %.2f): proceed %s, readout channels %d'
              % (result['amplitude_mv'], result['actual_mv'], result['proceed'], len(result['roi'])))
        for site, r in result['gate'].get('sites', {}).items():
            print('   %s vs sham  AUC %.3f  p %.4f  mean %.1f vs %.1f  %s' % (
                site, r['auc'], r['p'], r['mean_positive'], r['mean_negative'], 'pass' if r['passed'] else 'fail'))
    rows = []
    print('block              ok  missed  background/reference   response by site (sham)')
    for block in plan['blocks']:
        path = run / (block['name'] + '-qc.json')
        if not path.is_file():
            continue
        qc = read(path)
        trials = qc['trials']
        by_site = ' '.join('%s %s' % (s, mean([t['response'] for t in trials if t['site'] == s and t['real'] and t['valid']]))
                           for s in design.SITES)
        ratios = ' '.join('%.2f' % r for r in qc.get('activity', {}).get('ratios', []))
        print('%-18s %-3s %6d  %-22s %s (%s)' % (block['name'], 'yes' if qc['passed'] else 'NO', len(qc.get('missed_onsets', [])),
              ratios, by_site, mean([t['response'] for t in trials if not t['real']]).strip()))
        for error in qc['errors'][:5]:
            print('     ! ' + error)
        if block['stage'] == 'main' and qc['passed']:
            rows += analysis.table(block, qc)
    if rows:
        print('main blocks so far, mean response by recent-use bin (low 0-1, mid 2-4, high 5-8):')
        for site in design.SITES:
            mine = [r for r in rows if r['real'] and r['valid'] and r['site'] == site]
            print('   %s  ' % site + '  '.join('%s %s (n=%d)' % (b, mean([r['response'] for r in mine if r['bin'] == b]),
                  sum(r['bin'] == b for r in mine)) for b in design.BIN_NAMES))
    if (run / 'dose.jsonl').is_file():
        lines = (run / 'dose.jsonl').read_text(encoding='utf-8').splitlines()
        if lines:
            print('real pulses booked: %d' % json.loads(lines[-1])['total'])
    if (run / 'end.json').is_file():
        end = read(run / 'end.json')
        print('ended: %s %s' % (end.get('outcome'), end.get('stop_reason', '')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    show(p.parse_args().run)
