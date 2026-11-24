"""Closed-file checks, the 5-50 ms response readout and the registered gates.

Works on an opened `mea_stim_v2.recording.H5Recording` (or anything with the
same attributes). Nothing here contacts a device.

A site has several stimulation channels (centre first) and its own readout
channels: active, 100-1000 um from its centre, not near any site. Every trial
gets one count per site, so a sham trial serves as the control of each site.
The counts here come from the device's own spike detection, which is reliable
outside [-1.0, +2.5) ms around a pulse; `detect` recounts from raw samples.
"""
import math

import numpy as np

from . import design

RESPONSE = (100, 1000)     # [5, 50) ms after the marker, in frames
ACTIVE = (1000, 36000)     # [50 ms, 1.8 s); background firing per trial
PSTH = (-2000, 6000, 100)  # 5 ms bins from -100 to +300 ms
ONSET_FRAMES = 40          # an output artifact must begin within 2 ms of its marker
GUARD_FRAMES = 8
LOCAL_FRAMES = 200         # local median reference before the guard
LATE_FRAMES = (20000, 20200)
EXCURSION_UV = 1000.
MIN_RATE_HZ = .05
NEAR_UM = 100.             # readout channels keep this distance from every site centre
FAR_UM = 1000.             # and lie within this distance of their own site centre
MIN_SITE_CHANNELS = 20
MIN_CHANNELS = 50
MIN_SITES = 2
MISSED_ONSET_LIMIT = .05
RAIL_TRIAL_LIMIT = .05
SEGMENT_TRIALS = 15        # 30 s of stimulation
ACTIVITY_LIMITS = (.25, 4.)
MIN_PRE_S = 5.
MIN_POST_S = 5.
PERMUTATIONS = 10000
SCREEN = dict(auc_min=.65, p_max=.05)
EVOKED = dict(auc_min=.70, p_max=.01)
HABITUATION = dict(auc_min=.65, p_max=.05)
STABLE_FRACTION = .5       # held-out evoked increment against the first round
IDEAL = dict(low=1., mid=0., high=-1.)


def auc(positive, negative):
    """P(positive > negative) + half of ties."""
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if not len(a) or not len(b):
        return None
    return float(_auc_from_ranks(_ranks(np.concatenate([a, b]))[None, :], len(a))[0])


def _ranks(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, ranks)
    return sums[inverse] / counts[inverse]


def _auc_from_ranks(ranks, n):
    m = ranks.shape[1] - n
    return (ranks[:, :n].sum(axis=1) - n * (n + 1) / 2) / (n * m)


def permutation_p(positive, negative, seed, permutations=PERMUTATIONS):
    """One-sided probability of an AUC at least this large under exchanged labels."""
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if not len(a) or not len(b):
        return None
    ranks = _ranks(np.concatenate([a, b]))
    observed = _auc_from_ranks(ranks[None, :], len(a))[0]
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(ranks, (permutations, 1)), axis=1)
    hits = int(np.count_nonzero(_auc_from_ranks(shuffled, len(a)) >= observed - 1e-12))
    return (hits + 1) / (permutations + 1)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def compare(positive, negative, seed):
    a = [float(v) for v in positive]
    b = [float(v) for v in negative]
    return dict(n_positive=len(a), n_negative=len(b), auc=auc(a, b), p=permutation_p(a, b, seed),
                mean_positive=float(np.mean(a)) if a else None, mean_negative=float(np.mean(b)) if b else None)


def _spike_frames(rec, channels):
    keep = np.isin(rec.spike_channels, np.asarray(channels, dtype=int))
    return np.sort(np.rint(rec.spike_times[keep] * rec.fs).astype(np.int64))


def _count(spikes, anchors, window):
    anchors = np.asarray(anchors, dtype=np.int64)
    return np.searchsorted(spikes, anchors + window[1]) - np.searchsorted(spikes, anchors + window[0])


def union(roi):
    return sorted(set().union(*roi['sites'].values()))


def readable(roi):
    return [s for s in design.SITES if len(roi['sites'][s]) >= MIN_SITE_CHANNELS]


def stimulation_channels(site_channels):
    return [c for s in design.SITES for c in site_channels[s]]


def select_roi(rec, site_channels, bad_channels, skip_seconds=4.):
    """Readout channels of each site from a quiet recording made with the session route."""
    xy = rec.channel_xy_um
    driven = stimulation_channels(site_channels)
    if xy is None or not set(driven) <= set(xy) or len(set(driven)) != len(driven):
        raise ValueError('Distinct routed stimulation channels with physical coordinates are required')
    seconds = rec.duration - skip_seconds
    if seconds < 30:
        raise ValueError('Quiet reference too short')
    counts = np.bincount(rec.spike_channels[rec.spike_times >= skip_seconds], minlength=1024)
    centres = {s: np.asarray(xy[site_channels[s][0]], dtype=float) for s in design.SITES}
    sites, near = {s: [] for s in design.SITES}, []
    for channel in sorted(set(rec.channels.tolist()) - set(driven) - set(bad_channels)):
        if counts[channel] / seconds < MIN_RATE_HZ:
            continue
        here = np.asarray(xy[channel], dtype=float)
        distance = {s: float(np.hypot(*(here - centres[s]))) for s in design.SITES}
        if min(distance.values()) < NEAR_UM:
            near.append(channel)
            continue
        for s in design.SITES:
            if distance[s] < FAR_UM:
                sites[s].append(channel)
    roi = dict(sites=sites)
    everything = union(roi)
    if len(everything) < MIN_CHANNELS:
        raise ValueError('Fewer than %d active readout channels around the sites' % MIN_CHANNELS)
    if len(readable(roi)) < MIN_SITES:
        raise ValueError('Fewer than %d sites have %d active readout channels within %d-%d um'
                         % (MIN_SITES, MIN_SITE_CHANNELS, NEAR_UM, FAR_UM))
    return dict(roi, readable=readable(roi), excluded_near_sites=near,
                channel_hz={str(c): float(counts[c] / seconds) for c in everything},
                reference_hz=float(counts[everything].sum() / seconds), reference_seconds=float(seconds),
                min_rate_hz=MIN_RATE_HZ, near_um=NEAR_UM, far_um=FAR_UM)


def check_events(rec, block, session_id):
    """Marker frames relative to the file start, plus every way they differ from the plan."""
    errors = []
    events = rec.events
    n = len(block['trials'])
    if len(events) != n:
        return None, ['event_count %d expected %d' % (len(events), n)]
    for i, event in enumerate(events):
        wanted = design.event_properties(block, i)
        message = event['message']
        if (event['eventtype'] != 1 or event['id'] != design.event_id(block, i) or message.get('session') != session_id
                or any(str(message.get(k)) != str(v) for k, v in wanted.items())):
            errors.append('event %d differs from the plan' % i)
    frames = [e['frame'] - rec.first_frame for e in events]
    for i, (a, b) in enumerate(zip(frames, frames[1:])):
        gap = b - a
        if (i + 1) % design.CHUNK:
            if gap != design.TRIAL_FRAMES:
                errors.append('trial %d interval %d frames' % (i + 1, gap))
        elif not design.GAP_FRAMES[0] <= gap <= design.GAP_FRAMES[1]:
            errors.append('chunk gap before trial %d is %d frames' % (i + 1, gap))
    if frames[0] < MIN_PRE_S * rec.fs or rec.frame_count - frames[-1] < MIN_POST_S * rec.fs:
        errors.append('recording does not cover the registered margins')
    return frames, errors


def site_onsets(rec, frames, block, site_channels):
    """Per trial: which sites show an output artifact starting at the marker.

    A site counts as driven when any of its stimulation channels reaches an ADC
    rail right after the marker; neighbouring electrodes of one site saturate
    together, so single electrodes are not told apart.
    """
    channels = stimulation_channels(site_channels)
    owner = np.asarray([k for k, s in enumerate(design.SITES) for _ in site_channels[s]])
    uv = rec.metadata['lsb_uv']
    sham_block = block['stage'] == 'rehearsal'
    lead = LOCAL_FRAMES + GUARD_FRAMES
    of_site = lambda flags: [s for k, s in enumerate(design.SITES) if flags[owner == k].any()]
    rows = []
    for i, frame in enumerate(frames):
        site = None if sham_block else block['trials'][i]
        raw = rec.sample(channels, (frame - lead) / rec.fs, (lead + ONSET_FRAMES) / rec.fs).astype(float)
        late = rec.sample(channels, (frame + LATE_FRAMES[0]) / rec.fs, (LATE_FRAMES[1] - LATE_FRAMES[0]) / rec.fs)
        rail = (raw == 0) | (raw == 1023)
        guard_rail = rail[:, LOCAL_FRAMES:lead].any(axis=1)
        after_rail = rail[:, lead:].any(axis=1)
        median = np.median(raw[:, :LOCAL_FRAMES], axis=1)
        peak = np.abs(raw[:, lead:] - median[:, None]).max(axis=1) * uv
        onset = after_rail & ~guard_rail
        row = dict(trial=i, site=site or design.SHAM, rail_onset=of_site(onset), excursion=of_site(peak >= EXCURSION_UV),
                   rail_before=of_site(guard_rail), rail_late=of_site(((late == 0) | (late == 1023)).any(axis=1)),
                   onset_channels={s: int(onset[owner == k].sum()) for k, s in enumerate(design.SITES)},
                   peak_uv={s: float(peak[owner == k].max()) for k, s in enumerate(design.SITES)})
        if site is None:
            row['intended'] = True
            row['unexpected'] = bool(row['rail_onset'] or row['excursion'])
        else:
            row['intended'] = site in row['rail_onset']
            row['unexpected'] = any(s != site for s in row['rail_onset'])
        rows.append(row)
    return rows


def responses(rec, frames, roi):
    """Per trial: spikes 5-50 ms after the marker on each site's readout channels, and the background rate."""
    counts = {s: _count(_spike_frames(rec, roi['sites'][s]), frames, RESPONSE) if roi['sites'][s]
              else np.zeros(len(frames), dtype=int) for s in design.SITES}
    rate = _count(_spike_frames(rec, union(roi)), frames, ACTIVE) / ((ACTIVE[1] - ACTIVE[0]) / rec.fs)
    return [dict(r={s: int(counts[s][i]) for s in design.SITES}, rate_hz=float(rate[i])) for i in range(len(frames))]


def psth(rec, frames, channels):
    """Spikes per trial in 5 ms bins; rows follow `frames`."""
    spikes = _spike_frames(rec, channels)
    edges = np.arange(PSTH[0], PSTH[1] + 1, PSTH[2])
    anchors = np.asarray(frames, dtype=np.int64)[:, None] + edges[None, :]
    return np.diff(np.searchsorted(spikes, anchors), axis=1)


def roi_rails(rec, frames, channels):
    """trials x channels: readout channel at an ADC rail inside the response window."""
    out = np.zeros((len(frames), len(channels)), dtype=bool)
    for i, frame in enumerate(frames):
        raw = rec.sample(channels, (frame + RESPONSE[0]) / rec.fs, (RESPONSE[1] - RESPONSE[0]) / rec.fs)
        out[i] = ((raw == 0) | (raw == 1023)).any(axis=1)
    return out


def prune_roi(roi, channels, rails, trial_sites):
    """Drop a readout channel from a site when it is saturated after more than 5% of that site's pulses."""
    rails = np.asarray(rails, dtype=bool)
    trial_sites = np.asarray(trial_sites, dtype=object)
    column = {c: i for i, c in enumerate(channels)}
    kept, dropped = {}, {}
    for site in design.SITES:
        mine = rails[trial_sites == site]
        fraction = {c: float(mine[:, column[c]].mean()) if len(mine) else 0. for c in roi['sites'][site]}
        kept[site] = [c for c in roi['sites'][site] if fraction[c] <= RAIL_TRIAL_LIMIT]
        dropped[site] = {str(c): f for c, f in fraction.items() if f > RAIL_TRIAL_LIMIT}
    return dict(sites=kept), dropped


def activity(rows, reference_hz):
    """Firing rate in 30 s segments against the quiet reference recorded before any output."""
    if not math.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError('Positive reference rate required')
    rates = [float(np.mean([r['rate_hz'] for r in rows[a:a + SEGMENT_TRIALS]]))
             for a in range(0, len(rows) - SEGMENT_TRIALS + 1, SEGMENT_TRIALS)]
    ratios = [r / reference_hz for r in rates]
    outlier = [not ACTIVITY_LIMITS[0] <= r <= ACTIVITY_LIMITS[1] for r in ratios]
    return dict(reference_hz=reference_hz, segment_hz=rates, ratios=ratios, outlier=outlier,
                stop=any(a and b for a, b in zip(outlier, outlier[1:])))


def block_qc(rec, block, session_id, site_channels, roi, reference_hz):
    """Everything the session needs from one closed block file. `passed` gates further output."""
    out = dict(block=block['name'], index=block['index'], stage=block['stage'], errors=[], trials=[],
               source=dict(rec.source), duration_s=float(rec.duration),
               first_frame=rec.first_frame, last_frame=rec.last_frame)
    frames, errors = check_events(rec, block, session_id)
    out['errors'] += errors
    if frames is None or errors:
        out['passed'] = False
        return out
    onsets = site_onsets(rec, frames, block, site_channels)
    values = responses(rec, frames, roi)
    contexts = design.recent_counts(block['trials'])
    real = 0
    missed = []
    for i, (frame, onset, value, count) in enumerate(zip(frames, onsets, values, contexts)):
        site = block['trials'][i]
        is_real = site is not None and block['stage'] != 'rehearsal'
        real += is_real
        if is_real and not onset['intended']:
            missed.append(i)
        if onset['unexpected']:
            out['errors'].append('trial %d: output artifact on an unregistered site' % i)
        if onset['rail_before'] or onset['rail_late']:
            out['errors'].append('trial %d: site channel still at an ADC rail' % i)
        out['trials'].append(dict(value, trial=i, frame=frame, site=site or design.SHAM, real=bool(is_real),
                                  count=count, bin=None if count is None else design.bin_of(count),
                                  valid=bool(onset['intended']), onset=onset))
    out['missed_onsets'] = missed
    if len(missed) > MISSED_ONSET_LIMIT * max(real, 1):
        out['errors'].append('%d of %d registered pulses show no onset artifact' % (len(missed), real))
    out['activity'] = activity(out['trials'], reference_hz)
    if out['activity']['stop']:
        out['errors'].append('firing rate outside 25%-400% of reference in two consecutive 30 s segments')
    out['passed'] = not out['errors']
    return out


def table(block, qc):
    """Trial rows of one passed block, as used by the gates and the library."""
    keys = ('trial', 'frame', 'site', 'real', 'count', 'bin', 'valid', 'r', 'rate_hz')
    return [dict({k: t[k] for k in keys}, block=block['name'], stage=block['stage'], kind=block['kind'],
                 round=block.get('round'), frequent=block.get('frequent')) for t in qc['trials']]


def _real(rows, site):
    return [r for r in rows if r['real'] and r['valid'] and r['site'] == site]


def site_gate(rows, seed, auc_min, p_max, sites=design.SITES):
    """Is each site told apart from sham on its own readout channels? `sites` are the readable ones."""
    sham = [r for r in rows if not r['real']]
    out = {}
    for k, site in enumerate(design.SITES):
        result = compare([r['r'][site] for r in _real(rows, site)], [r['r'][site] for r in sham], seed + k)
        result['passed'] = bool(site in sites and result['auc'] is not None and result['auc'] >= auc_min
                                and result['p'] <= p_max)
        out[site] = result
    passed = [s for s in design.SITES if out[s]['passed']]
    return dict(sites=out, passed_sites=passed, qualified=len(passed) >= MIN_SITES, auc_min=auc_min, p_max=p_max)


def habituation_gate(rows, seed, sites=design.SITES):
    """Responses when the site was rare in the last 8 trials against when it was frequent."""
    out = {}
    for k, site in enumerate(design.SITES):
        mine = _real(rows, site)
        by_bin = {b: [r['r'][site] for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
        result = compare(by_bin['low'], by_bin['high'], seed + 10 + k)
        means = {b: float(np.mean(v)) if v else None for b, v in by_bin.items()}
        ordered = all(m is not None for m in means.values()) and means['low'] > means['mid'] > means['high']
        uniform = [r for r in mine if r['kind'] == 'uniform']
        result.update(bin_means=means, bin_n={b: len(v) for b, v in by_bin.items()}, ordered=bool(ordered),
                      uniform_spearman=spearman([r['count'] for r in uniform], [r['r'][site] for r in uniform]),
                      passed=bool(site in sites and ordered and result['auc'] is not None
                                  and result['auc'] >= HABITUATION['auc_min'] and result['p'] <= HABITUATION['p_max']))
        out[site] = result
    passed = [s for s in design.SITES if out[s]['passed']]
    return dict(HABITUATION, sites=out, passed_sites=passed, passed=len(passed) >= MIN_SITES)


def increment(rows, site):
    """Mean response to the site's pulses above the sham trials of the same rows."""
    real = [r['r'][site] for r in _real(rows, site)]
    sham = [r['r'][site] for r in rows if not r['real']]
    return float(np.mean(real) - np.mean(sham)) if real and sham else None


def stability_gate(first, held):
    """The evoked increment must not have run down between the first and the held-out round."""
    out = {}
    for site in design.SITES:
        a, b = increment(first, site), increment(held, site)
        out[site] = dict(first_round=a, held_out_round=b,
                         passed=bool(a is not None and b is not None and a > 0 and b >= STABLE_FRACTION * a))
    return dict(sites=out, passed_sites=[s for s in design.SITES if out[s]['passed']], fraction=STABLE_FRACTION)


def split_rounds(rows):
    """Complete rounds only. The last complete round is held out; earlier ones build the library."""
    blocks = {}
    for r in rows:
        blocks.setdefault(r['round'], set()).add(r['block'])
    complete = sorted(n for n, names in blocks.items() if len(names) == len(design.SITES) + 1)
    if len(complete) < 2:
        return [], [], complete
    held = complete[-1]
    return ([r for r in rows if r['round'] in complete[:-1]], [r for r in rows if r['round'] == held], complete)


def scaling(library_rows, site):
    """Centre and scale of a site's responses, as the library stores them."""
    mine = _real(library_rows, site)
    by_bin = [[r['r'][site] for r in mine if r['bin'] == b] for b in design.BIN_NAMES]
    if not all(by_bin):
        return None
    scale = float(np.std([r['r'][site] for r in mine]))
    return (float(np.mean([np.mean(v) for v in by_bin])), scale) if scale > 0 else None


def reliability(library_rows, held, sites):
    """Correlation of the standardized held-out responses with the ideal value of their bin.

    Centre and scale come from the library rounds only. Every site and bin
    carries the same total weight, because the game decides how often each bin
    is used, not the oddball blocks.
    """
    z, ideal, weight = [], [], []
    for site in sites:
        fit = scaling(library_rows, site)
        mine = _real(held, site)
        n = {b: sum(r['bin'] == b for r in mine) for b in design.BIN_NAMES}
        if fit is None or not all(n.values()):
            return None
        for r in mine:
            z.append((r['r'][site] - fit[0]) / fit[1])
            ideal.append(IDEAL[r['bin']])
            weight.append(1. / n[r['bin']])
    if not z:
        return None
    z, ideal, weight = np.asarray(z), np.asarray(ideal), np.asarray(weight) / np.sum(weight)
    dz, di = z - np.sum(weight * z), ideal - np.sum(weight * ideal)
    spread = math.sqrt(np.sum(weight * dz * dz) * np.sum(weight * di * di))
    return float(np.sum(weight * dz * di) / spread) if spread > 0 else None


def offline_gates(rows, seed, sites=design.SITES, rho_min=None):
    """The registered gates on the held-out round. `sites` are the readable ones.

    A library needs two sites that pass every response gate. Training also needs
    the library's reliability to reach `rho_min`, which stage 0 measures.
    """
    library_rows, held, complete = split_rounds(rows)
    out = dict(complete_rounds=complete, library_rounds=complete[:-1] if len(complete) >= 2 else [],
               held_out_round=complete[-1] if len(complete) >= 2 else None, library_allowed=False,
               training_allowed=False)
    if not held:
        out['reason'] = 'fewer than two complete rounds'
        return out
    out['evoked'] = dict(site_gate(held, seed, sites=sites, **EVOKED))
    out['evoked']['passed'] = out['evoked']['qualified']
    out['habituation'] = habituation_gate(held, seed, sites)
    out['library_direction'] = habituation_gate(library_rows, seed + 100, sites)
    out['stability'] = stability_gate([r for r in rows if r['round'] == complete[0]], held)
    ordered = lambda site: (out['library_direction']['sites'][site]['auc'] or 0) > .5
    usable = [s for s in design.SITES if s in out['evoked']['passed_sites'] and s in out['habituation']['passed_sites']
              and s in out['stability']['passed_sites'] and ordered(s)]
    out['usable_sites'] = usable
    out['library_allowed'] = len(usable) >= MIN_SITES
    if not out['library_allowed']:
        out['reason'] = ('fewer than two sites pass the evoked, habituation and stability gates on the held-out round '
                         'with the same direction in the library rounds')
        return out
    rho = reliability(library_rows, held, usable)
    passed = None if rho_min is None or rho is None else bool(rho >= rho_min)
    out['reliability'] = dict(rho=rho, rho_min=rho_min, passed=passed)
    out['training_allowed'] = passed is True
    if not out['training_allowed']:
        out['reason'] = ('reliability threshold from stage 0 not supplied yet' if rho_min is None
                         else 'library reliability below the stage 0 threshold')
    return out
