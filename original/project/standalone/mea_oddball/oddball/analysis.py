"""Closed-file checks and early evoked-response readout for the oddball session.

Works on an opened `mea_stim_v2.recording.H5Recording` (or anything with the
same attributes). Nothing here contacts a device. Responses are spike counts on
readout channels in a short window after each marker, which is the window the
earlier sessions never measured.
"""
import math

import numpy as np

from . import design

EARLY = (200, 6000)        # [10, 300) ms after the marker, in frames
EARLY_LATE = (400, 6000)   # [20, 300) ms; reported next to EARLY as an artifact check
BEFORE = (-6000, -200)     # same length immediately before the marker
ARTIFACT = (0, 200)        # [0, 10) ms; the pulse itself is detected as spikes on most channels
ACTIVE = (200, 36000)      # [10 ms, 1.8 s); overall firing per trial without the artifact
PSTH = (-6000, 12000, 200) # 10 ms bins from -300 to +600 ms
ONSET_FRAMES = 40          # an output artifact must begin within 2 ms of its marker
GUARD_FRAMES = 8
LOCAL_FRAMES = 200         # local median reference before the guard
LATE_FRAMES = (20000, 20200)
EXCURSION_UV = 1000.
RAIL_WINDOW = (200, 1200)  # [10, 60) ms; readout channels still saturated here are unusable
MIN_RATE_HZ = .05
EXCLUDE_UM = 200.
MIN_CHANNELS = 50
MISSED_ONSET_LIMIT = .05
RAIL_TRIAL_LIMIT = .05
SEGMENT_TRIALS = 15        # 30 s of stimulation
ACTIVITY_LIMITS = (.25, 4.)
MIN_PRE_S = 5.
MIN_POST_S = 5.
PERMUTATIONS = 10000


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


def select_roi(rec, site_channels, bad_channels, skip_seconds=4.):
    """Readout channels: spontaneously active, away from every stimulation site."""
    xy = rec.channel_xy_um
    if xy is None or not set(site_channels.values()) <= set(xy):
        raise ValueError('Physical coordinates of the routed channels are required')
    seconds = rec.duration - skip_seconds
    if seconds < 30:
        raise ValueError('Quiet reference too short')
    late = rec.spike_times >= skip_seconds
    counts = np.bincount(rec.spike_channels[late], minlength=1024)
    sites = np.asarray([xy[c] for c in site_channels.values()], dtype=float)
    roi, near = [], []
    for channel in sorted(set(rec.channels.tolist()) - set(site_channels.values()) - set(bad_channels)):
        if counts[channel] / seconds < MIN_RATE_HZ:
            continue
        distance = float(np.sqrt(np.square(sites - np.asarray(xy[channel], dtype=float)).sum(axis=1)).min())
        (roi if distance >= EXCLUDE_UM else near).append(channel)
    if len(roi) < MIN_CHANNELS:
        raise ValueError('Fewer than %d active readout channels away from the sites' % MIN_CHANNELS)
    return dict(roi=roi, excluded_near_sites=near, reference_hz=float(counts[roi].sum() / seconds),
                reference_seconds=float(seconds), min_rate_hz=MIN_RATE_HZ, exclude_um=EXCLUDE_UM)


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
    """Per trial: which site channels show an output artifact starting at the marker."""
    order = list(design.SITES)
    channels = [site_channels[s] for s in order]
    uv = rec.metadata['lsb_uv']
    sham_block = block['stage'] == 'rehearsal'
    lead = LOCAL_FRAMES + GUARD_FRAMES
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
        row = dict(trial=i, site=site or design.SHAM,
                   rail_onset=[s for s, g, a in zip(order, guard_rail, after_rail) if a and not g],
                   excursion=[s for s, p in zip(order, peak) if p >= EXCURSION_UV],
                   rail_before=[s for s, g in zip(order, guard_rail) if g],
                   rail_late=[s for s, r in zip(order, ((late == 0) | (late == 1023)).any(axis=1)) if r],
                   peak_uv=[float(p) for p in peak])
        if site is None:
            row['intended'] = True
            row['unexpected'] = bool(row['rail_onset'] or row['excursion'])
        else:
            row['intended'] = site in row['rail_onset']
            row['unexpected'] = any(s != site for s in row['rail_onset'])
        rows.append(row)
    return rows


def responses(rec, frames, roi):
    spikes = _spike_frames(rec, roi)
    before = _count(spikes, frames, BEFORE)
    after = _count(spikes, frames, EARLY)
    later = _count(spikes, frames, EARLY_LATE)
    artifact = _count(spikes, frames, ARTIFACT)
    rate = _count(spikes, frames, ACTIVE) / ((ACTIVE[1] - ACTIVE[0]) / rec.fs)
    return [dict(before=int(b), after=int(a), response=int(a - b), after_20ms=int(l), artifact=int(x), rate_hz=float(r))
            for b, a, l, x, r in zip(before, after, later, artifact, rate)]


def psth(rec, frames, roi):
    """Spikes per trial in 10 ms bins; rows follow `frames`."""
    spikes = _spike_frames(rec, roi)
    edges = np.arange(PSTH[0], PSTH[1] + 1, PSTH[2])
    anchors = np.asarray(frames, dtype=np.int64)[:, None] + edges[None, :]
    return np.diff(np.searchsorted(spikes, anchors), axis=1)


def roi_rails(rec, frames, roi):
    """trials x channels: readout channel at an ADC rail 10-60 ms after the marker."""
    out = np.zeros((len(frames), len(roi)), dtype=bool)
    for i, frame in enumerate(frames):
        raw = rec.sample(roi, (frame + RAIL_WINDOW[0]) / rec.fs, (RAIL_WINDOW[1] - RAIL_WINDOW[0]) / rec.fs)
        out[i] = ((raw == 0) | (raw == 1023)).any(axis=1)
    return out


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
    """Flat trial rows of one passed block, as used by the gates and the library."""
    keys = ('trial', 'frame', 'site', 'real', 'count', 'bin', 'valid', 'before', 'after', 'response', 'after_20ms',
            'artifact')
    return [dict({k: t[k] for k in keys}, block=block['name'], stage=block['stage'], kind=block['kind'],
                 round=block.get('round'), frequent=block.get('frequent')) for t in qc['trials']]


def prune_roi(roi, rails, real):
    """Drop readout channels saturated after more than 5% of the real pulses."""
    rails = np.asarray(rails, dtype=bool)[np.asarray(real, dtype=bool)]
    fraction = rails.mean(axis=0) if len(rails) else np.zeros(len(roi))
    kept = [c for c, f in zip(roi, fraction) if f <= RAIL_TRIAL_LIMIT]
    return kept, {int(c): float(f) for c, f in zip(roi, fraction) if f > RAIL_TRIAL_LIMIT}


def screen_gate(rows, seed, auc_min=.70, p_max=.05):
    """On-site check after the screening files: is each site told apart from sham?"""
    sham = [r['response'] for r in rows if not r['real']]
    sites = {}
    for k, site in enumerate(design.SITES):
        real = [r['response'] for r in rows if r['real'] and r['site'] == site and r['valid']]
        result = compare(real, sham, seed + k)
        result['passed'] = bool(result['auc'] is not None and result['auc'] >= auc_min and result['p'] <= p_max)
        sites[site] = result
    passed = [s for s in design.SITES if sites[s]['passed']]
    return dict(sites=sites, passed_sites=passed, qualified=len(passed) >= 2, auc_min=auc_min, p_max=p_max)


def evoked_gate(rows, seed, auc_min=.70, p_max=.01):
    """Gate 1: real pulses at a site against sham trials of the same rounds."""
    gate = screen_gate(rows, seed, auc_min, p_max)
    return dict(gate, passed=gate['qualified'])


def habituation_gate(rows, seed, auc_min=.65, p_max=.05):
    """Gate 2: responses when the site was rare in the last 8 trials against when it was frequent."""
    sites = {}
    for k, site in enumerate(design.SITES):
        mine = [r for r in rows if r['real'] and r['site'] == site and r['valid']]
        by_bin = {b: [r['response'] for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
        result = compare(by_bin['low'] + by_bin['mid'], by_bin['high'], seed + 10 + k)
        means = {b: float(np.mean(v)) if v else None for b, v in by_bin.items()}
        ordered = all(means[b] is not None for b in design.BIN_NAMES) and means['high'] < min(means['low'], means['mid'])
        uniform = [r for r in mine if r['kind'] == 'uniform']
        result.update(bin_means=means, bin_n={b: len(v) for b, v in by_bin.items()}, ordered=bool(ordered),
                      uniform_spearman=spearman([r['count'] for r in uniform], [r['response'] for r in uniform]),
                      passed=bool(ordered and result['auc'] is not None and result['auc'] >= auc_min
                                  and result['p'] <= p_max))
        sites[site] = result
    passed = [s for s in design.SITES if sites[s]['passed']]
    return dict(sites=sites, passed_sites=passed, passed=len(passed) >= 2, auc_min=auc_min, p_max=p_max)


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


def offline_gates(rows, seed):
    library, held, complete = split_rounds(rows)
    out = dict(complete_rounds=complete, library_rounds=complete[:-1] if len(complete) >= 2 else [],
               held_out_round=complete[-1] if len(complete) >= 2 else None, training_allowed=False)
    if not held:
        out['reason'] = 'fewer than two complete rounds'
        return out
    out['evoked'] = evoked_gate(held, seed)
    out['habituation'] = habituation_gate(held, seed)
    out['library_direction'] = habituation_gate(library, seed + 100)
    usable = [s for s in design.SITES if s in out['evoked']['passed_sites'] and s in out['habituation']['passed_sites']
              and (out['library_direction']['sites'][s]['auc'] or 0) > .5]
    out['usable_sites'] = usable
    out['training_allowed'] = len(usable) >= 2
    if not out['training_allowed']:
        out['reason'] = 'fewer than two sites pass both gates on the held-out round with the same direction in the library rounds'
    return out
