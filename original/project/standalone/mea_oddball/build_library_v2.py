"""Offline oddball v2 library builder:
Applies library-fit non-negative channel weighting on log1p([5, 50) ms) counts, runs two-tier
gates on the held-out round, and writes:
- `evoked_library.json` whenever G2 passes (unlocks Track 2 Homeostatic Aversion)
- `habituation_library.json` whenever G2 + G3 pass (unlocks Track 1 Exploration & Dual-Drive)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from oddball import analysis as v1_analysis
from oddball import analysis_v2 as analysis
from oddball import design_v2 as design
from oddball import library_v2 as library
from oddball.session import write_once


def score_rows_with_library_weights(rows_with_channels):
    """Fit non-negative per-site channel weights strictly on library rounds, then score ALL rows.

    Each input row must include `channel_counts` (dict or 1D list aligned per site ROI).
    Returns `(scored_rows, channel_weights_by_site)`.
    """
    lib_rows, _, complete = v1_analysis.split_rounds(rows_with_channels)
    if len(complete) < 2:
        return [dict(r) for r in rows_with_channels], {}

    weights = {}
    for site in design.SITES:
        real_mat = [r['channel_counts'][site] for r in lib_rows if r['real'] and r['site'] == site and r.get('valid', True)]
        sham_mat = [r['channel_counts'][site] for r in lib_rows if not r['real'] and r.get('valid', True)]
        if real_mat and sham_mat:
            weights[site] = analysis.fit_channel_weights(np.asarray(real_mat), np.asarray(sham_mat))

    scored = []
    for r in rows_with_channels:
        copy = {k: v for k, v in r.items() if k != 'channel_counts'}
        if r['real'] and r['site'] in weights:
            val = float(analysis.weighted_response(np.asarray([r['channel_counts'][r['site']]]), weights[r['site']])[0])
            copy['response'] = val
        elif not r['real'] and weights:
            # Average across sites using each site's frozen weights for sham comparison
            vals = [float(analysis.weighted_response(np.asarray([r['channel_counts'][s]]), w)[0])
                    for s, w in weights.items() if s in r['channel_counts']]
            copy['response'] = float(np.mean(vals)) if vals else float(r.get('response', 0.0))
        scored.append(copy)
    return scored, {s: w.tolist() for s, w in weights.items()}


def build_from_rows(rows, out_dir, seed, *, rho_min=0.20, channel_weights=None):
    """Run two-tier gates and export `gates_v2.json` plus any qualified library files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gates = analysis.offline_gates_v2(rows, seed, rho_min=rho_min)
    lib_rows, _, _ = v1_analysis.split_rounds(rows)

    exported = {}
    if gates['evoked_training_allowed']:
        evk = library.build_evoked_library(lib_rows, gates['usable_evoked_sites'], channel_weights=channel_weights)
        write_once(out / 'evoked_library.json', evk)
        exported['evoked_library_sha256'] = evk['sha256']

    if gates['habituation_training_allowed']:
        hab = library.build_habituation_library(lib_rows, gates['usable_habituation_sites'], channel_weights=channel_weights)
        write_once(out / 'habituation_library.json', hab)
        exported['habituation_library_sha256'] = hab['sha256']

    report = dict(schema='mea.oddball.gates.v2', seed=int(seed), rho_min=float(rho_min),
                  gates=gates, exported=exported)
    write_once(out / 'gates_v2.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rows_json', type=Path, help='JSON file containing extracted v2 trial rows')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026112301)
    p.add_argument('--rho-min', type=float, default=0.20)
    args = p.parse_args()
    raw_rows = json.loads(args.rows_json.read_text(encoding='utf-8'))
    if raw_rows and 'channel_counts' in raw_rows[0]:
        rows, weights = score_rows_with_library_weights(raw_rows)
    else:
        rows, weights = raw_rows, None
    report = build_from_rows(rows, args.out, args.seed, rho_min=args.rho_min, channel_weights=weights)
    print(json.dumps(dict(
        evoked_allowed=report['gates']['evoked_training_allowed'],
        habituation_allowed=report['gates']['habituation_training_allowed'],
        rho_lib=report['gates']['rho_lib'],
        exported=report['exported'],
    ), ensure_ascii=False))


if __name__ == '__main__':
    main()
