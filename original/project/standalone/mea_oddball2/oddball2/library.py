"""Response library: recorded 5-50 ms responses indexed by site and recent-use bin.

The game side maps a map region to a site, counts how often that site occurred
in the last eight decisions, and draws one recorded response from the matching
bin. The value is larger when the organoid had not been driven at that site
recently, which is what makes it usable as an action-dependent novelty signal.
Only sites that passed every response gate are stored; a region whose site is
missing gets no signal.
"""
import random

import numpy as np

from . import analysis, design

SCHEMA = 'mea.oddball.library.v2'
MIN_ENTRIES = 10


def build(rows, sites, reliability=None):
    """Standardize per site so that equal use of the three bins averages to zero."""
    if len(sites) < analysis.MIN_SITES or not set(sites) <= set(design.SITES):
        raise ValueError('At least two registered sites required')
    stored, stats = {}, {}
    for site in sites:
        mine = [r for r in rows if r['real'] and r['valid'] and r['site'] == site]
        by_bin = {b: [float(r['r'][site]) for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
        short = [b for b, v in by_bin.items() if len(v) < MIN_ENTRIES]
        if short:
            raise ValueError('Site %s has fewer than %d responses in bins %s' % (site, MIN_ENTRIES, short))
        fit = analysis.scaling(rows, site)
        if fit is None:
            raise ValueError('Site %s responses do not vary' % site)
        center, scale = fit
        stored[site] = {b: [(v - center) / scale for v in values] for b, values in by_bin.items()}
        stats[site] = dict(center=center, scale=scale, n={b: len(v) for b, v in by_bin.items()},
                           bin_means={b: float(np.mean(v)) for b, v in stored[site].items()})
    body = dict(schema=SCHEMA, history=design.HISTORY, bins=[list(e) for e in design.BIN_EDGES],
                window_frames=list(analysis.RESPONSE), sites=stored, stats=stats, reliability=reliability)
    return dict(body, sha256=design.digest(body))


def check(library):
    body = {k: v for k, v in library.items() if k != 'sha256'}
    if library.get('schema') != SCHEMA or library.get('sha256') != design.digest(body):
        raise ValueError('Response library changed or has another format')
    return library


def game_site(rx, ry):
    """Neighbouring map regions always use different sites."""
    return design.SITES[(rx + ry) % len(design.SITES)]


class Context:
    """Rolling record of the sites used by the last HISTORY decisions of one episode."""

    def __init__(self):
        self.recent = []

    def push(self, site):
        """Count of `site` before this decision, then record the decision."""
        if site not in design.SITES:
            raise ValueError('Unknown site')
        count = self.recent.count(site)
        self.recent = (self.recent + [site])[-design.HISTORY:]
        return count


class Sampler:
    def __init__(self, library, seed):
        self.sites = check(library)['sites']
        self.rng = random.Random(seed)

    def draw(self, site, count):
        """One recorded response, or None when the site did not pass the gates."""
        if site not in design.SITES:
            raise ValueError('Unknown site')
        if site not in self.sites:
            return None
        return self.rng.choice(self.sites[site][design.bin_of(count)])
