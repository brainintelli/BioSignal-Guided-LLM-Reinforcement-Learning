"""Dual v2 response library builder:
1) `evoked_library.json` (`mea.oddball.evoked_library.v2`): built whenever G2 passes -> Track 2 (Homeo)
2) `habituation_library.json` (`mea.oddball.habituation_library.v2`): built whenever G2+G3 pass -> Track 1/Dual

All standardization parameters and pools come strictly from library rounds (never the held-out round).
"""
import numpy as np

from . import design_v2 as design

EVOKED_SCHEMA = 'mea.oddball.evoked_library.v2'
HABITUATION_SCHEMA = 'mea.oddball.habituation_library.v2'
MIN_ENTRIES = 10


def build_evoked_library(library_rows, usable_sites, channel_weights=None):
    """Build real-vs-sham standardized response pools for Track 2 Homeostatic Aversion."""
    usable = [s for s in design.SITES if s in usable_sites]
    if len(usable) < 2:
        raise ValueError('At least 2 G2-qualified sites are required for evoked library')
    sham_raw = [float(r['response']) for r in library_rows if not r['real'] and r.get('valid', True)]
    if len(sham_raw) < MIN_ENTRIES:
        raise ValueError(f'Fewer than {MIN_ENTRIES} valid sham trials in library rounds')

    real_by_site = {}
    for s in usable:
        vals = [float(r['response']) for r in library_rows if r['real'] and r.get('valid', True) and r['site'] == s]
        if len(vals) < MIN_ENTRIES:
            raise ValueError(f'Site {s} has fewer than {MIN_ENTRIES} valid real trials')
        real_by_site[s] = vals

    pooled_real = [v for vals in real_by_site.values() for v in vals]
    center = 0.5 * (float(np.mean(pooled_real)) + float(np.mean(sham_raw)))
    scale = float(np.std(pooled_real + sham_raw))
    if scale <= 1e-12:
        raise ValueError('Evoked library responses do not vary')

    sites = {s: dict(real=[(v - center) / scale for v in vals]) for s, vals in real_by_site.items()}
    sham = [(v - center) / scale for v in sham_raw]
    stats = dict(
        center=center,
        scale=scale,
        sham_mean=float(np.mean(sham)),
        sham_n=len(sham),
        site_means={s: float(np.mean(sites[s]['real'])) for s in usable},
        site_n={s: len(sites[s]['real']) for s in usable},
    )
    body = dict(
        schema=EVOKED_SCHEMA,
        usable_sites=usable,
        sites=sites,
        sham=sham,
        stats=stats,
        channel_weights={k: [float(x) for x in v] for k, v in (channel_weights or {}).items()},
    )
    return dict(body, sha256=design.digest(body))


def build_habituation_library(library_rows, usable_sites, channel_weights=None):
    """Build per-site (low/mid/high) standardized habituation pools for Track 1 & Dual-Drive."""
    usable = [s for s in design.SITES if s in usable_sites]
    if len(usable) < 2:
        raise ValueError('At least 2 G3-qualified sites are required for habituation library')
    sites, stats = {}, {}
    for s in usable:
        mine = [r for r in library_rows if r['real'] and r.get('valid', True) and r['site'] == s]
        by_bin = {b: [float(r['response']) for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
        short = [b for b, v in by_bin.items() if len(v) < MIN_ENTRIES]
        if short:
            raise ValueError(f'Site {s} has fewer than {MIN_ENTRIES} responses in bins {short}')
        center = float(np.mean([np.mean(v) for v in by_bin.values()]))
        scale = float(np.std([float(r['response']) for r in mine]))
        if scale <= 1e-12:
            raise ValueError(f'Site {s} habituation responses do not vary')
        sites[s] = {b: [(v - center) / scale for v in vals] for b, vals in by_bin.items()}
        stats[s] = dict(
            center=center,
            scale=scale,
            n={b: len(v) for b, v in by_bin.items()},
            bin_means={b: float(np.mean(v)) for b, v in sites[s].items()},
        )
    body = dict(
        schema=HABITUATION_SCHEMA,
        history=design.HISTORY,
        bins=[list(e) for e in design.BIN_EDGES],
        usable_sites=usable,
        sites=sites,
        stats=stats,
        channel_weights={k: [float(x) for x in v] for k, v in (channel_weights or {}).items()},
    )
    return dict(body, sha256=design.digest(body))


def check_library(lib, expected_schema):
    body = {k: v for k, v in lib.items() if k != 'sha256'}
    if lib.get('schema') != expected_schema or lib.get('sha256') != design.digest(body):
        raise ValueError(f'Library schema or SHA-256 mismatch (expected {expected_schema})')
    return lib
