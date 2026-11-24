"""Oddball v2 offline electrophysiology analysis:
- Multi-electrode cluster site selection (5 electrodes/site within 40 um, >=20 local active neighbors)
- Blank-then-causal 4th-order Butterworth bandpass (300-3000 Hz, pure NumPy, zero scipy needed)
- Short-latency [5, 50) ms readout without baseline subtraction
- log1p burst tail compression + library-fit non-negative channel weighting
- Two-tier offline gates on the held-out round (G2 Evoked -> Track 2; G3 Habituation + rho_lib -> Track 1/Dual)
"""
import math

import numpy as np

from . import analysis as v1_analysis
from . import design_v2 as design

# Blanking & readout windows (in frames at 20 kHz; 1 ms = 20 frames)
PRE_BLANK_FRAMES = (-5, 50)          # [-0.25, +2.50) ms linear interpolation before causal filtering
RAIL_EXTEND_FRAMES = (5, 60)         # [-0.25, +3.00] ms around any ADC rail sample
POST_BLANK_FRAMES = (-20, 100)       # [-1.0, +5.0) ms unified mask after filtering
RESPONSE_WINDOW = (100, 1000)        # [5, 50) ms primary evoked/habituation readout window
REFRACTORY_FRAMES = 20               # 1.0 ms spike refractory period
THRESHOLD_SIGMA = 5.0                # negative-peak threshold (-5.0 * MAD_sigma)
ADC_RAIL_LO = 2
ADC_RAIL_HI = 1021
MIN_RATE_HZ = 0.05
MIN_TOTAL_ACTIVE = 50

# Gate thresholds (frozen before observing data)
SCREEN_AUC_MIN = 0.65
SCREEN_P_MAX = 0.05
CONFIRM_AUC_MIN = 0.70
CONFIRM_P_MAX = 0.01
EVOKED_AUC_MIN = 0.70
EVOKED_P_MAX = 0.01
HABIT_AUC_MIN = 0.65
HABIT_P_MAX = 0.05
STABILITY_MIN_RATIO = 0.50

IDEAL_BIN_VALUES = {'low': 1.0, 'mid': 0.0, 'high': -1.0}


# ---------------------------------------------------------------------------
# Pure-NumPy causal 4th-order Butterworth bandpass (300-3000 Hz at fs=20 kHz)
# ---------------------------------------------------------------------------

def butter_bandpass_sos(low_hz=300.0, high_hz=3000.0, fs=20000.0, order=4):
    """Compute exact digital second-order sections (SOS) for a causal Butterworth bandpass filter.

    Uses analog prototype -> bandpass frequency transformation -> bilinear transform -> real biquads.
    Produces an `(order, 6)` SOS matrix identical to `scipy.signal.butter(..., output='sos')`.
    """
    if not (0.0 < low_hz < high_hz < fs / 2.0):
        raise ValueError('Cutoff frequencies must satisfy 0 < low_hz < high_hz < fs/2')
    # Pre-warp digital frequencies for bilinear transform (T = 2)
    w1 = 2.0 * fs * math.tan(math.pi * low_hz / fs)
    w2 = 2.0 * fs * math.tan(math.pi * high_hz / fs)
    bw = w2 - w1
    w0 = math.sqrt(w1 * w2)

    # Analog lowpass Butterworth prototype poles on left half-plane
    k = np.arange(1, order + 1)
    proto_poles = np.exp(1j * math.pi * (2 * k + order - 1) / (2 * order))

    # Lowpass -> Bandpass: s_bp^2 - bw*p*s_bp + w0^2 = 0
    bp_poles = []
    for p in proto_poles:
        disc = np.sqrt((bw * p) ** 2 - 4.0 * (w0 ** 2))
        bp_poles.append(0.5 * (bw * p + disc))
        bp_poles.append(0.5 * (bw * p - disc))
    bp_poles = np.asarray(bp_poles, dtype=np.complex128)
    bp_zeros = np.zeros(order, dtype=np.complex128)  # order zeros at s=0, order zeros at s=inf

    # Bilinear transform: z = (2*fs + s) / (2*fs - s)
    two_fs = 2.0 * fs
    z_poles = (two_fs + bp_poles) / (two_fs - bp_poles)
    z_zeros = np.concatenate([
        (two_fs + bp_zeros) / (two_fs - bp_zeros),          # -> +1 (DC)
        -np.ones(order, dtype=np.complex128),               # s=inf -> -1 (Nyquist)
    ])

    # Analog gain at center frequency s = j*w0 is 1 -> match digital gain at z = exp(j*theta0)
    theta0 = 2.0 * math.atan(w0 / two_fs)
    z0 = np.exp(1j * theta0)
    raw_gain = np.abs(np.prod(z0 - z_zeros) / np.prod(z0 - z_poles))
    total_gain = 1.0 / float(raw_gain)

    # Pair conjugate poles into real second-order denominators; pair +1/-1 zeros into (1 - z^-2)
    used = np.zeros(len(z_poles), dtype=bool)
    pole_pairs = []
    for i in range(len(z_poles)):
        if used[i]:
            continue
        used[i] = True
        p = z_poles[i]
        # Find closest complex conjugate
        best_j = None
        best_dist = float('inf')
        for j in range(i + 1, len(z_poles)):
            if not used[j]:
                d = abs(z_poles[j] - np.conj(p))
                if d < best_dist:
                    best_dist = d
                    best_j = j
        used[best_j] = True
        pole_pairs.append((p, z_poles[best_j]))

    # Sort pole pairs by closeness to unit circle for numerical stability
    pole_pairs.sort(key=lambda pair: max(abs(pair[0]), abs(pair[1])))
    sos = np.zeros((len(pole_pairs), 6), dtype=np.float64)
    sec_gain = total_gain ** (1.0 / len(pole_pairs))
    for idx, (p1, p2) in enumerate(pole_pairs):
        a = np.real(np.poly([p1, p2]))
        # Each section gets one zero at +1 and one zero at -1 => (1 - z^-2)
        b = sec_gain * np.array([1.0, 0.0, -1.0], dtype=np.float64)
        sos[idx, :3] = b
        sos[idx, 3:] = a
    return sos


def sosfilt_causal(sos, x):
    """Causal Direct-Form-II Transposed SOS filter along the last axis of `x` (shape (..., T))."""
    y = np.asarray(x, dtype=np.float64)
    single = y.ndim == 1
    if single:
        y = y[None, :]
    n_ch, n_t = y.shape
    out = y.copy()
    for sec in sos:
        b0, b1, b2, a0, a1, a2 = sec
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        z1 = np.zeros(n_ch, dtype=np.float64)
        z2 = np.zeros(n_ch, dtype=np.float64)
        nxt = np.empty_like(out)
        for t in range(n_t):
            xt = out[:, t]
            yt = b0 * xt + z1
            z1 = b1 * xt - a1 * yt + z2
            z2 = b2 * xt - a2 * yt
            nxt[:, t] = yt
        out = nxt
    return out[0] if single else out


def blank_raw_waveform(raw_adc, anchor_idx, *, pre_blank=PRE_BLANK_FRAMES, rail_extend=RAIL_EXTEND_FRAMES):
    """Linearly interpolate over fixed pulse window and dynamic ADC rail excursions BEFORE filtering.

    `raw_adc`: shape (n_channels, n_frames), float/int ADC codes.
    `anchor_idx`: index of marker frame inside `raw_adc`.
    Returns `(blanked_adc, post_mask)` where `post_mask` is bool `(n_channels, n_frames)` of samples
    to suppress after filtering (unified `[-1, +5) ms` plus `+2.5 ms` after any rail recovery).
    """
    arr = np.asarray(raw_adc, dtype=np.float64).copy()
    n_ch, n_t = arr.shape
    post_mask = np.zeros((n_ch, n_t), dtype=bool)
    p0 = max(0, anchor_idx + POST_BLANK_FRAMES[0])
    p1 = min(n_t, anchor_idx + POST_BLANK_FRAMES[1])
    post_mask[:, p0:p1] = True

    b0 = max(0, anchor_idx + pre_blank[0])
    b1 = min(n_t, anchor_idx + pre_blank[1])
    for c in range(n_ch):
        seg_mask = np.zeros(n_t, dtype=bool)
        seg_mask[b0:b1] = True
        rails = (arr[c] <= ADC_RAIL_LO) | (arr[c] >= ADC_RAIL_HI)
        if rails.any():
            idx = np.flatnonzero(rails)
            for k in idx:
                r0 = max(0, int(k) - rail_extend[0])
                r1 = min(n_t, int(k) + rail_extend[1] + 1)
                seg_mask[r0:r1] = True
                post_mask[c, r0:min(n_t, r1 + 50)] = True  # extra +2.5 ms guard after rail recovery
        # Interpolate each contiguous masked span using quiet medians immediately outside span
        edges = np.diff(np.concatenate([[0], seg_mask.view(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            left = np.median(arr[c, max(0, s - 10):s]) if s > 0 else arr[c, min(e, n_t - 1)]
            right = np.median(arr[c, e:min(n_t, e + 10)]) if e < n_t else arr[c, max(0, s - 1)]
            arr[c, s:e] = np.linspace(left, right, e - s, endpoint=False)
    return arr, post_mask


def estimate_channel_sigma(warmup_raw_adc, sos=None, fs=design.RATE):
    """Robust MAD noise scale per channel computed strictly from quiet warmup waveforms."""
    if sos is None:
        sos = butter_bandpass_sos(300.0, 3000.0, fs=fs, order=4)
    arr = np.asarray(warmup_raw_adc, dtype=np.float64)
    arr = arr - np.median(arr, axis=-1, keepdims=True)
    filt = sosfilt_causal(sos, arr)
    # Discard initial 20 ms filter transient
    trim = min(int(0.02 * fs), filt.shape[-1] // 4)
    core = filt[:, trim:]
    return np.median(np.abs(core), axis=-1) / 0.6745


def detect_trial_spikes_causal(raw_adc, anchor_idx, sigma_per_ch, *, sos=None, fs=design.RATE,
                               window=RESPONSE_WINDOW):
    """Blank -> causal bandpass -> unified [-1,+5)ms mask -> count negative spikes in `window`.

    Returns integer array of shape `(n_channels,)` with spike counts in `window`, plus bool array
    `saturated_after_5ms` of shape `(n_channels,)` flagging channels still saturated past +5 ms.
    """
    if sos is None:
        sos = butter_bandpass_sos(300.0, 3000.0, fs=fs, order=4)
    raw = np.asarray(raw_adc, dtype=np.float64)
    n_ch, n_t = raw.shape
    w0 = max(0, anchor_idx + window[0])
    w1 = min(n_t, anchor_idx + window[1])
    rails_late = (raw[:, w0:w1] <= ADC_RAIL_LO) | (raw[:, w0:w1] >= ADC_RAIL_HI)
    saturated = rails_late.any(axis=1)

    blanked, post_mask = blank_raw_waveform(raw, anchor_idx)
    pre_med = np.median(blanked[:, :max(1, anchor_idx - 20)], axis=1, keepdims=True)
    filt = sosfilt_causal(sos, blanked - pre_med)
    filt[post_mask] = 0.0

    counts = np.zeros(n_ch, dtype=np.int64)
    thresh = -THRESHOLD_SIGMA * np.asarray(sigma_per_ch, dtype=np.float64)
    for c in range(n_ch):
        if saturated[c] or thresh[c] >= 0:
            continue
        seg = filt[c, w0:w1]
        if len(seg) < 3:
            continue
        # Local negative minima below threshold with 1.0 ms refractory period
        below = (seg < thresh[c])
        if not below.any():
            continue
        is_min = np.zeros(len(seg), dtype=bool)
        is_min[1:-1] = (seg[1:-1] <= seg[:-2]) & (seg[1:-1] < seg[2:]) & below[1:-1]
        peaks = np.flatnonzero(is_min)
        last = -REFRACTORY_FRAMES - 1
        n_sp = 0
        for p in peaks:
            if p - last >= REFRACTORY_FRAMES:
                n_sp += 1
                last = int(p)
        counts[c] = n_sp
    return counts, saturated


# ---------------------------------------------------------------------------
# Multi-electrode site cluster selection & per-site local ROI
# ---------------------------------------------------------------------------

def select_site_clusters(channel_xy_um, channel_rates_hz, bad_channels=()):
    """Select 3 stimulation sites (H, W, F) with:
    1) >= 20 active readout channels (>=0.05 Hz) within 100-500 um of each site center
    2) Pairwise center spacing >= 500 um
    3) 3..5 clustered stimulation channels within 40 um of each site center
    """
    bad = set(bad_channels)
    all_ch = sorted(c for c in channel_xy_um if c not in bad)
    active_ch = [c for c in all_ch if float(channel_rates_hz.get(c, 0.0)) >= MIN_RATE_HZ]
    if len(active_ch) < MIN_TOTAL_ACTIVE:
        raise ValueError(f'Fewer than {MIN_TOTAL_ACTIVE} active channels on chip ({len(active_ch)})')

    all_xy = np.asarray([channel_xy_um[c] for c in all_ch], dtype=float)
    act_xy = np.asarray([channel_xy_um[c] for c in active_ch], dtype=float)

    candidates = []
    for c in active_ch:
        cx, cy = channel_xy_um[c]
        d_act = np.hypot(act_xy[:, 0] - cx, act_xy[:, 1] - cy)
        dense_neighbors = int(np.count_nonzero((d_act >= design.LOCAL_ROI_INNER_UM) & (d_act <= design.LOCAL_ROI_DENSE_UM)))
        if dense_neighbors < design.MIN_LOCAL_CHANNELS:
            continue
        d_all = np.hypot(all_xy[:, 0] - cx, all_xy[:, 1] - cy)
        near_idx = np.flatnonzero(d_all <= design.CLUSTER_RADIUS_UM)
        near_sorted = sorted(near_idx, key=lambda idx: (d_all[idx], -float(channel_rates_hz.get(all_ch[idx], 0.0)), all_ch[idx]))
        cluster = [all_ch[idx] for idx in near_sorted[:design.CLUSTER_TARGET]]
        if len(cluster) < design.CLUSTER_MIN:
            continue
        candidates.append(dict(
            center_channel=int(c),
            center_xy=[float(cx), float(cy)],
            cluster_channels=[int(x) for x in cluster],
            dense_neighbors=dense_neighbors,
            rate_hz=float(channel_rates_hz[c]),
        ))

    candidates.sort(key=lambda r: (-r['dense_neighbors'], -r['rate_hz'], r['center_channel']))
    if len(candidates) < len(design.SITES):
        raise ValueError(f'Only {len(candidates)} candidate sites meet local density & cluster criteria')

    # Greedy-exact search over top 40 dense candidates for max minimum pairwise spacing >= 500 um
    pool = candidates[:40]
    best = None
    for i in range(len(pool)):
        xi = np.asarray(pool[i]['center_xy'])
        for j in range(i + 1, len(pool)):
            xj = np.asarray(pool[j]['center_xy'])
            dij = float(np.hypot(*(xi - xj)))
            if dij < design.SITE_MIN_SPACING_UM:
                continue
            for k in range(j + 1, len(pool)):
                xk = np.asarray(pool[k]['center_xy'])
                dik = float(np.hypot(*(xi - xk)))
                djk = float(np.hypot(*(xj - xk)))
                min_d = min(dij, dik, djk)
                if min_d < design.SITE_MIN_SPACING_UM:
                    continue
                # Ensure clusters are strictly disjoint
                used = set(pool[i]['cluster_channels']) | set(pool[j]['cluster_channels']) | set(pool[k]['cluster_channels'])
                if len(used) < (len(pool[i]['cluster_channels']) + len(pool[j]['cluster_channels']) + len(pool[k]['cluster_channels'])):
                    continue
                score = (
                    pool[i]['dense_neighbors'] + pool[j]['dense_neighbors'] + pool[k]['dense_neighbors'],
                    min_d,
                    pool[i]['rate_hz'] + pool[j]['rate_hz'] + pool[k]['rate_hz'],
                )
                if best is None or score > best[0]:
                    best = (score, (pool[i], pool[j], pool[k]), min_d)
    if best is None:
        raise ValueError('Cannot place 3 disjoint multi-electrode clusters >= 500 um apart with >= 20 local neighbors')
    chosen = {site: info for site, info in zip(design.SITES, best[1])}
    return dict(sites=chosen, min_spacing_um=float(best[2]), total_active_channels=len(active_ch))


def select_site_rois(channel_xy_um, channel_rates_hz, site_clusters, bad_channels=()):
    """Per-site local readout ROI: active non-stimulation channels within 100-1000 um of each site."""
    stim_all = {ch for info in site_clusters.values() for ch in info['cluster_channels']}
    excluded = stim_all | set(bad_channels)
    rois = {}
    for site, info in site_clusters.items():
        cx, cy = info['center_xy']
        roi = []
        for ch, xy in sorted(channel_xy_um.items()):
            if ch in excluded or float(channel_rates_hz.get(ch, 0.0)) < MIN_RATE_HZ:
                continue
            dist = math.hypot(xy[0] - cx, xy[1] - cy)
            if design.LOCAL_ROI_INNER_UM <= dist <= design.LOCAL_ROI_OUTER_UM:
                roi.append(int(ch))
        if len(roi) < design.MIN_LOCAL_CHANNELS:
            raise ValueError(f'Site {site} has fewer than {design.MIN_LOCAL_CHANNELS} local ROI channels ({len(roi)})')
        rois[site] = roi
    return rois


# ---------------------------------------------------------------------------
# Anti-burst log1p transform, library-fit channel weighting, & two-tier gates
# ---------------------------------------------------------------------------

def log1p_counts(channel_counts):
    return np.log1p(np.maximum(0.0, np.asarray(channel_counts, dtype=np.float64)))


def fit_channel_weights(real_counts, sham_counts, eps=0.25):
    """Fit non-negative evoked-SNR weights STRICTLY on library/screening rounds.

    `real_counts`: shape `(n_real, n_ch)`, `sham_counts`: shape `(n_sham, n_ch)`.
    """
    r = log1p_counts(real_counts)
    s = log1p_counts(sham_counts)
    if r.ndim != 2 or s.ndim != 2 or r.shape[1] != s.shape[1]:
        raise ValueError('real_counts and sham_counts must be 2D with matching channel counts')
    snr = (r.mean(axis=0) - s.mean(axis=0)) / (s.std(axis=0) + eps)
    w = np.maximum(0.0, snr)
    total = float(w.sum())
    if total <= 1e-12:
        w = np.full(r.shape[1], 1.0 / r.shape[1], dtype=np.float64)
    else:
        w = w / total
    return w


def weighted_response(channel_counts, weights):
    """Scalar trial response = sum_c w_c * log1p(spikes_c)."""
    x = log1p_counts(channel_counts)
    w = np.asarray(weights, dtype=np.float64)
    return x @ w


def compute_rho_lib(held_rows, library_stats, usable_sites):
    """Held-out reliability rho_lib = Corr(standardized_response, ideal_bin_value).

    Standardization parameters (center, scale) come strictly from library rounds.
    """
    xs, ys = [], []
    for row in held_rows:
        if not row.get('real') or not row.get('valid', True):
            continue
        site = row['site']
        if site not in usable_sites or site not in library_stats:
            continue
        st = library_stats[site]
        if st['scale'] <= 0:
            continue
        z = (float(row['response']) - st['center']) / st['scale']
        xs.append(z)
        ys.append(IDEAL_BIN_VALUES[row['bin']])
    if len(xs) < 6 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def screen_gate_v2(rows, seed, auc_min=SCREEN_AUC_MIN, p_max=SCREEN_P_MAX):
    return v1_analysis.screen_gate(rows, seed, auc_min=auc_min, p_max=p_max)


def confirm_gate_v2(rows, seed, auc_min=CONFIRM_AUC_MIN, p_max=CONFIRM_P_MAX):
    return v1_analysis.screen_gate(rows, seed, auc_min=auc_min, p_max=p_max)


def _strictly_decreasing_habituation(rows, seed, auc_min=HABIT_AUC_MIN, p_max=HABIT_P_MAX):
    """G3 habituation check: low vs high AUC >= 0.65, p <= 0.05, and mean(low) > mean(mid) > mean(high)."""
    sites = {}
    for k, site in enumerate(design.SITES):
        mine = [r for r in rows if r['real'] and r['site'] == site and r.get('valid', True)]
        by_bin = {b: [float(r['response']) for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
        res = v1_analysis.compare(by_bin['low'], by_bin['high'], seed + 10 + k)
        means = {b: float(np.mean(v)) if v else None for b, v in by_bin.items()}
        monotonic = (
            all(means[b] is not None for b in design.BIN_NAMES)
            and means['low'] > means['mid'] > means['high']
        )
        res.update(
            bin_means=means,
            bin_n={b: len(v) for b, v in by_bin.items()},
            monotonic=bool(monotonic),
            passed=bool(monotonic and res['auc'] is not None and res['auc'] >= auc_min and res['p'] <= p_max),
        )
        sites[site] = res
    passed = [s for s in design.SITES if sites[s]['passed']]
    return dict(sites=sites, passed_sites=passed, passed=len(passed) >= 2, auc_min=auc_min, p_max=p_max)


def offline_gates_v2(rows, seed, *, rho_min=0.20):
    """Two-tier v2 offline gates evaluated strictly on the held-out last complete round.

    - Tier 1 (`evoked_training_allowed`): Gate G2 (evoked AUC >= 0.70, p <= 0.01, stability >= 0.5x R1)
      passes on >= 2 sites -> unlocks Track 2 (Homeostatic Aversion: fear of thirst/hunger/death).
    - Tier 2 (`habituation_training_allowed`): Gate G2 + Gate G3 (monotonic habituation AUC >= 0.65,
      p <= 0.05, same direction in library rounds, and held-out rho_lib >= rho_min) -> unlocks
      Track 1 (Exploration Novelty) and Dual-Drive.
    """
    lib_rows, held_rows, complete = v1_analysis.split_rounds(rows)
    out = dict(
        complete_rounds=complete,
        library_rounds=complete[:-1] if len(complete) >= 2 else [],
        held_out_round=complete[-1] if len(complete) >= 2 else None,
        evoked_training_allowed=False,
        habituation_training_allowed=False,
        rho_min=float(rho_min),
        rho_lib=None,
    )
    if not held_rows:
        out['reason'] = 'fewer than two complete rounds'
        return out

    evoked = v1_analysis.evoked_gate(held_rows, seed, auc_min=EVOKED_AUC_MIN, p_max=EVOKED_P_MAX)
    r1_rows = [r for r in lib_rows if r['round'] == complete[0]]
    stability = {}
    stable_evoked = []
    for s in evoked['passed_sites']:
        r1_real = [float(r['response']) for r in r1_rows if r['real'] and r['site'] == s and r.get('valid', True)]
        held_real = [float(r['response']) for r in held_rows if r['real'] and r['site'] == s and r.get('valid', True)]
        r1_m = float(np.mean(r1_real)) if r1_real else 0.0
        held_m = float(np.mean(held_real)) if held_real else 0.0
        ok = (r1_m <= 0.0 and held_m >= r1_m) or (r1_m > 0.0 and held_m >= STABILITY_MIN_RATIO * r1_m)
        stability[s] = dict(round1_mean=r1_m, held_mean=held_m, passed=bool(ok))
        if ok:
            stable_evoked.append(s)

    out['evoked'] = evoked
    out['stability'] = stability
    out['usable_evoked_sites'] = stable_evoked
    out['evoked_training_allowed'] = len(stable_evoked) >= 2

    hab_held = _strictly_decreasing_habituation(held_rows, seed + 50)
    hab_lib = _strictly_decreasing_habituation(lib_rows, seed + 150)
    out['habituation'] = hab_held
    out['library_habituation'] = hab_lib

    usable_hab = [
        s for s in stable_evoked
        if s in hab_held['passed_sites'] and (hab_lib['sites'][s]['auc'] or 0.0) > 0.5
    ]
    out['usable_habituation_sites'] = usable_hab

    # Fit standardization strictly on library rounds, then compute held-out rho_lib
    lib_stats = {}
    for s in usable_hab:
        mine = [r for r in lib_rows if r['real'] and r['site'] == s and r.get('valid', True)]
        by_bin = {b: [float(r['response']) for r in mine if r['bin'] == b] for b in design.BIN_NAMES}
        if all(by_bin.values()):
            center = float(np.mean([np.mean(v) for v in by_bin.values()]))
            scale = float(np.std([float(r['response']) for r in mine]))
            lib_stats[s] = dict(center=center, scale=scale)
    rho_lib = compute_rho_lib(held_rows, lib_stats, usable_hab)
    out['rho_lib'] = rho_lib
    out['habituation_training_allowed'] = bool(
        len(usable_hab) >= 2 and rho_lib is not None and rho_lib >= float(rho_min)
    )
    if not out['evoked_training_allowed']:
        out['reason'] = 'fewer than two stable sites passed G2 evoked gate on held-out round'
    elif not out['habituation_training_allowed']:
        out['reason'] = 'G2 passed (Track 2 unlocked), but G3 habituation/rho_lib did not pass'
    else:
        out['reason'] = 'both G2 and G3 passed (Track 1, Track 2, and Dual-Drive unlocked)'
    return out
