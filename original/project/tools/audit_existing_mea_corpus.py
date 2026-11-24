"""Phase 0 read-only audit of existing local MEA recordings (zero new acquisition).

Part 0A: Narrow-window [5, 50) ms blank-then-causal-filter offline evoked check
         on historical 2026-11-19 recordings (28080 crossover, 28080 site-rhythm,
         22021 crossover, 22554 H/W/F single-pulse) that were previously only
         inspected on wide [2, 7) s or [12, 13) s windows.

Part 0B: High-resolution (200 ms) spontaneous multi-channel neural dynamics
         extraction across all 5 active chips on disk (22021, 28080, 22554,
         27701, 28073), Gate-S biological dynamics verification (heavy-tailed
         bursts, temporal persistence / Hurst H, cross-channel spatial synchrony,
         nonlinearity vs IAAFT), and export of SHA-256 locked real / shuffled (C)
         / white (G1) / IAAFT (G2) libraries for Phase 2-3 training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / '.tools' / 'mea_baseline_deps', ROOT / 'standalone' / 'mea_oddball'):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from oddball import analysis_v2 as av2

FS = 20000.0
BIN_MS = 200
BIN_FRAMES = int(round(FS * BIN_MS / 1000.0))  # 4000 frames
MIN_ACTIVE_RATE_HZ = 0.05
MIN_ACTIVE_CHANNELS = 50
RAIL_LO = 2
RAIL_HI = 1021

# Gate-S frozen biological structure thresholds (locked prior to viewing Phase 0 results)
GATE_S_MIN_CHANNELS = 50
GATE_S_MIN_DURATION_S = 60.0
GATE_S_MIN_FANO = 3.0
GATE_S_MIN_HURST = 0.55
GATE_S_P_MAX = 0.01


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return sha256_bytes(raw)


def _scalar(group, key):
    arr = np.asarray(group[key][()])
    val = arr.reshape(-1)[0]
    if isinstance(val, bytes):
        return val.decode('utf-8').rstrip('\x00')
    return val.item() if hasattr(val, 'item') else val


def open_maxone_store(f: h5py.File):
    d = f['data_store/data0000'] if 'data_store' in f else f['recordings/rec0000/well000']
    groups = d['groups']
    if 'routed' in groups:
        g = groups['routed']
    else:
        gname = next(iter(groups.keys()))
        g = groups[gname]
    return d, g


def sosfilt_causal_fast(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized causal DF-II transposed SOS filter over last axis of x (..., T)."""
    out = np.asarray(x, dtype=np.float64).copy()
    orig_shape = out.shape
    flat = out.reshape(-1, orig_shape[-1])
    n_series, n_t = flat.shape
    for sec in sos:
        b0, b1, b2, a0, a1, a2 = sec
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        z1 = np.zeros(n_series, dtype=np.float64)
        z2 = np.zeros(n_series, dtype=np.float64)
        nxt = np.empty_like(flat)
        for t in range(n_t):
            xt = flat[:, t]
            yt = b0 * xt + z1
            z1 = b1 * xt - a1 * yt + z2
            z2 = b2 * xt - a2 * yt
            nxt[:, t] = yt
        flat = nxt
    return flat.reshape(orig_shape)


def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Tie handling
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def perm_test_auc(pos: np.ndarray, neg: np.ndarray, seed: int, n_perm: int = 5000) -> dict:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    obs = mann_whitney_auc(pos, neg)
    if math.isnan(obs):
        return dict(auc=None, p=None, n_pos=len(pos), n_neg=len(neg),
                    pos_mean=None, neg_mean=None)
    pooled = np.concatenate([pos, neg])
    n1 = len(pos)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if mann_whitney_auc(perm[:n1], perm[n1:]) >= obs - 1e-12:
            ge += 1
    return dict(
        auc=obs,
        p=float((1 + ge) / (1 + n_perm)),
        n_pos=int(n1),
        n_neg=int(len(neg)),
        pos_mean=float(pos.mean()),
        neg_mean=float(neg.mean()),
        pos_median=float(np.median(pos)),
        neg_median=float(np.median(neg)),
    )


# ---------------------------------------------------------------------------
# Part 0A: Narrow-window [5, 50) ms causal check on 2026-11-19 stimulated H5s
# ---------------------------------------------------------------------------

def _calibrate_warmup_sigma(warmup_h5: Path, sos: np.ndarray, exclude_channels: set[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Identify active channels (>=0.05 Hz in warmup t >= 4s) and causal noise sigma_c."""
    with h5py.File(warmup_h5, 'r') as f:
        d, g = open_maxone_store(f)
        fs = float(_scalar(d, 'settings/sampling'))
        mapping = d['settings/mapping'][:]
        mapped_ch = sorted({int(row['channel']) for row in mapping} - exclude_channels)
        frames = g['frame_nos']
        f0 = int(frames[0])
        n_frames = len(frames)
        dur = n_frames / fs
        spk = d['spikes'][:]
        rel_t = (spk['frameno'].astype(np.int64) - f0) / fs
        late = rel_t >= 4.0
        counts = np.bincount(spk['channel'][late].astype(np.int64), minlength=1024)
        active = np.array([c for c in mapped_ch if counts[c] / max(dur - 4.0, 1.0) >= MIN_ACTIVE_RATE_HZ], dtype=int)
        if len(active) == 0:
            active = np.array(mapped_ch, dtype=int)

        raw = g['raw']
        # Sample 16 non-overlapping 0.25 s (5000-frame) chunks after t=4s -> 4.0 s total, vectorized!
        chunk_len = int(0.25 * fs)
        start_min = int(4.0 * fs)
        start_max = max(start_min + 1, n_frames - chunk_len)
        starts = np.linspace(start_min, start_max - 1, 16, dtype=int)
        chunks = np.empty((len(active), len(starts), chunk_len), dtype=np.float64)
        for ci, s0 in enumerate(starts):
            block = np.asarray(raw[:, s0:s0 + chunk_len], dtype=np.float64)[active]
            block -= np.median(block, axis=1, keepdims=True)
            chunks[:, ci, :] = block
        filt = sosfilt_causal_fast(sos, chunks)[:, :, 400:]  # drop first 20 ms settle
        sigma_ch = np.median(np.abs(filt), axis=(1, 2)) / 0.6745
        lsb_uv = float(_scalar(d, 'settings/lsb')) * 1e6
        meta = dict(
            chip_id=str(_scalar(f, 'wellplate/id')),
            fs=fs,
            duration_s=dur,
            n_mapped=len(mapped_ch),
            n_active=int(len(active)),
            sigma_uv_median=float(np.median(sigma_ch) * lsb_uv),
            lsb_uv=lsb_uv,
        )
    return active, sigma_ch, meta


def _detect_window_spikes_causal(raw_win: np.ndarray, anchor_idx: int, sigma_ch: np.ndarray, sos: np.ndarray):
    """Apply identical blank-then-causal-filter + -5*sigma detection to real & sham anchors."""
    # raw_win: (n_ch, n_t) centered at anchor_idx (covering [-1200, +1600) = [-60, +80) ms)
    blanked, post_mask = av2.blank_raw_waveform(raw_win, anchor_idx)
    blanked -= np.median(blanked[:, :max(1, anchor_idx - 20)], axis=1, keepdims=True)
    filt = sosfilt_causal_fast(sos, blanked)
    # Count spikes in three windows: pre [-35, -5) ms, ring [+5, +10) ms, primary [+5, +50) ms
    pre_s, pre_e = anchor_idx - 700, anchor_idx - 100
    ring_s, ring_e = anchor_idx + 100, anchor_idx + 200
    pri_s, pri_e = anchor_idx + 100, anchor_idx + 1000
    # Detect negative threshold crossings with 1 ms (20 frame) refractory per channel
    thresh = -av2.THRESHOLD_SIGMA * sigma_ch[:, None]
    valid = ~post_mask
    below = (filt < thresh) & valid
    n_ch = filt.shape[0]
    pre_c = np.zeros(n_ch, dtype=np.int32)
    ring_c = np.zeros(n_ch, dtype=np.int32)
    pri_c = np.zeros(n_ch, dtype=np.int32)
    rail_past_5ms = 0
    for ci in range(n_ch):
        if post_mask[ci, pri_s:pri_e].any():
            rail_past_5ms += 1
        idx = np.where(below[ci])[0]
        if len(idx) == 0:
            continue
        last = -10000
        for t in idx:
            if t - last >= av2.REFRACTORY_FRAMES:
                last = int(t)
                if pre_s <= t < pre_e:
                    pre_c[ci] += 1
                if ring_s <= t < ring_e:
                    ring_c[ci] += 1
                if pri_s <= t < pri_e:
                    pri_c[ci] += 1
    valid_pri_frames = valid[:, pri_s:pri_e].sum()
    valid_pri_ch_ms = float(valid_pri_frames) / (FS / 1000.0)
    return dict(
        pre_total=int(pre_c.sum()),
        ring_total=int(ring_c.sum()),
        pri_total=int(pri_c.sum()),
        pri_rate_per_ch_s=float(pri_c.sum() / max(valid_pri_ch_ms / 1000.0, 1e-9)),
        pri_log1p=float(np.log1p(pri_c).sum()),
        rail_channels_past_5ms=int(rail_past_5ms),
        valid_pri_frac=float(valid_pri_frames / max(n_ch * (pri_e - pri_s), 1)),
    )


def audit_crossover_session_narrow(session_dir: Path, label: str, seed: int) -> dict:
    """Audit a 2026-11-19 crossover/site-rhythm session on [5, 50) ms causal window."""
    t0 = time.monotonic()
    sos = av2.butter_bandpass_sos(300.0, 3000.0, FS, order=4)
    warmup_h5 = session_dir / 'warmup' / 'warmup.raw.h5'
    plan_path = session_dir / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))

    # Determine stimulated channels from route/session metadata if available
    stim_channels: set[int] = set()
    for cand in (session_dir.parent / 'session-route.json', session_dir.parent / 'route.json', session_dir / 'route.json'):
        if cand.exists():
            rt = json.loads(cand.read_text(encoding='utf-8'))
            for k in ('channels', 'site_channels'):
                if isinstance(rt.get(k), dict):
                    stim_channels.update(int(v) for v in rt[k].values())
            for item in rt.get('sites', {}).values() if isinstance(rt.get('sites'), dict) else []:
                if isinstance(item, dict) and 'channel' in item:
                    stim_channels.add(int(item['channel']))

    active, sigma_ch, wmeta = _calibrate_warmup_sigma(warmup_h5, sos, stim_channels)

    real_pulses = []
    sham_pulses = []
    real_trials = []
    sham_trials = []
    trial_dirs = sorted(session_dir.glob('trial_*'))
    trial_h5s = [td / f'{td.name}.raw.h5' for td in trial_dirs if (td / f'{td.name}.raw.h5').exists()]

    pre_pad = 1200   # -60 ms
    post_pad = 1600  # +80 ms

    for h5_path in trial_h5s:
        with h5py.File(h5_path, 'r') as f:
            d, g = open_maxone_store(f)
            frames = g['frame_nos']
            f_start = int(frames[0])
            n_frames = len(frames)
            raw = g['raw']
            evs = d['events'][:]
            parsed = []
            for e in evs:
                msg = e['eventmessage']
                if isinstance(msg, bytes):
                    msg = msg.decode('utf-8').rstrip('\x00')
                info = json.loads(msg)
                parsed.append((int(e['frameno']), info))
            if not parsed:
                continue
            sites = [info.get('site', '') for _, info in parsed]
            is_sham = all(s in ('quiet', 'sham', 'Z', '') for s in sites)
            if is_sham:
                # Place 3 virtual anchors matching the 3 pulse slots (+0, +200, +400 ms)
                base_f = parsed[0][0]
                anchors = [(base_f + off, 'Z') for off in (0, 4000, 8000)]
            else:
                anchors = [(fn, info.get('site', 'real')) for fn, info in parsed]

            trial_pri_sum = 0.0
            trial_log_sum = 0.0
            n_valid_anchors = 0
            for fn, st_name in anchors:
                rel = fn - f_start
                if rel - pre_pad < 0 or rel + post_pad > n_frames:
                    continue
                win = np.asarray(raw[:, rel - pre_pad : rel + post_pad], dtype=np.float64)[active]
                res = _detect_window_spikes_causal(win, pre_pad, sigma_ch, sos)
                res['site'] = st_name
                res['trial'] = h5_path.stem
                if is_sham:
                    sham_pulses.append(res)
                else:
                    real_pulses.append(res)
                trial_pri_sum += res['pri_total']
                trial_log_sum += res['pri_log1p']
                n_valid_anchors += 1
            if n_valid_anchors == len(anchors) and n_valid_anchors > 0:
                t_rec = dict(trial=h5_path.stem, pri_sum=trial_pri_sum, log_sum=trial_log_sum, sites=sites)
                if is_sham:
                    sham_trials.append(t_rec)
                else:
                    real_trials.append(t_rec)

    pulse_count_test = perm_test_auc(
        [r['pri_total'] for r in real_pulses],
        [r['pri_total'] for r in sham_pulses],
        seed=seed,
    )
    pulse_log_test = perm_test_auc(
        [r['pri_log1p'] for r in real_pulses],
        [r['pri_log1p'] for r in sham_pulses],
        seed=seed + 1,
    )
    trial_count_test = perm_test_auc(
        [r['pri_sum'] for r in real_trials],
        [r['pri_sum'] for r in sham_trials],
        seed=seed + 2,
    )
    # Ringing sanity check: [+5, +10) ms mean real vs sham
    ring_real = float(np.mean([r['ring_total'] for r in real_pulses])) if real_pulses else None
    ring_sham = float(np.mean([r['ring_total'] for r in sham_pulses])) if sham_pulses else None
    pre_real = float(np.mean([r['pre_total'] for r in real_pulses])) if real_pulses else None
    pre_sham = float(np.mean([r['pre_total'] for r in sham_pulses])) if sham_pulses else None

    pulse_rate_test = perm_test_auc(
        [r['pri_rate_per_ch_s'] for r in real_pulses],
        [r['pri_rate_per_ch_s'] for r in sham_pulses],
        seed=seed + 3,
    )
    valid_frac_real = float(np.mean([r['valid_pri_frac'] for r in real_pulses])) if real_pulses else None
    valid_frac_sham = float(np.mean([r['valid_pri_frac'] for r in sham_pulses])) if sham_pulses else None

    g2_passed = bool(
        pulse_rate_test['auc'] is not None
        and pulse_rate_test['auc'] >= 0.70
        and pulse_rate_test['p'] <= 0.01
    )
    return dict(
        label=label,
        session_dir=str(session_dir),
        warmup_meta=wmeta,
        excluded_stim_channels=sorted(stim_channels),
        n_trial_files=len(trial_h5s),
        n_real_pulses=len(real_pulses),
        n_sham_pulses=len(sham_pulses),
        n_real_trials=len(real_trials),
        n_sham_trials=len(sham_trials),
        pulse_5_50ms_count=pulse_count_test,
        pulse_5_50ms_rate=pulse_rate_test,
        pulse_5_50ms_log1p=pulse_log_test,
        trial_sum_5_50ms_count=trial_count_test,
        ringing_check_5_10ms=dict(real_mean=ring_real, sham_mean=ring_sham),
        pre_check_m35_m5ms=dict(real_mean=pre_real, sham_mean=pre_sham),
        valid_pri_frac=dict(real_mean=valid_frac_real, sham_mean=valid_frac_sham),
        max_rail_ch_past_5ms=max([r['rail_channels_past_5ms'] for r in real_pulses] or [0]),
        g2_passed=g2_passed,
        seconds=round(time.monotonic() - t0, 2),
    )


def audit_22554_single_pulse_narrow(base_dir: Path) -> dict:
    """Descriptive [5, 50) ms causal check on 22554 11/19 single-pulse H/W/F files (no sham)."""
    t0 = time.monotonic()
    sos = av2.butter_bandpass_sos(300.0, 3000.0, FS, order=4)
    baseline_h5 = base_dir / 'records/session/paired/baseline/baseline.raw.h5'
    site_plan = json.loads((base_dir / 'records/session/site-plan.json').read_text(encoding='utf-8'))
    stim_ch = set()
    for s in ('H', 'W', 'F'):
        if s in site_plan and isinstance(site_plan[s], dict) and 'channel' in site_plan[s]:
            stim_ch.add(int(site_plan[s]['channel']))
    active, sigma_ch, wmeta = _calibrate_warmup_sigma(baseline_h5, sos, stim_ch)

    sites_out = {}
    pre_pad, post_pad = 1200, 1600
    for s in ('H', 'W', 'F'):
        matches = list((base_dir / 'records/session' / s).rglob('stimulation.raw.h5'))
        if not matches:
            continue
        with h5py.File(matches[0], 'r') as f:
            d, g = open_maxone_store(f)
            frames = g['frame_nos']
            f_start = int(frames[0])
            n_frames = len(frames)
            raw = g['raw']
            evs = d['events'][:]
            if len(evs) == 0:
                continue
            fn = int(evs[0]['frameno'])
            rel = fn - f_start
            # Evaluate pulse + 20 pre-pulse quiet baseline windows spaced by 1 s before pulse
            win = np.asarray(raw[:, rel - pre_pad : rel + post_pad], dtype=np.float64)[active]
            pulse_res = _detect_window_spikes_causal(win, pre_pad, sigma_ch, sos)
            bg_counts = []
            for k in range(1, 21):
                bf = rel - k * int(FS)
                if bf - pre_pad >= 0 and bf + post_pad <= n_frames:
                    bwin = np.asarray(raw[:, bf - pre_pad : bf + post_pad], dtype=np.float64)[active]
                    bres = _detect_window_spikes_causal(bwin, pre_pad, sigma_ch, sos)
                    bg_counts.append(bres['pri_total'])
            bg = np.asarray(bg_counts, dtype=np.float64)
            z = float((pulse_res['pri_total'] - bg.mean()) / max(bg.std(), 1e-6)) if len(bg) else None
            sites_out[s] = dict(
                file=str(matches[0]),
                pulse_5_50ms=pulse_res['pri_total'],
                pulse_5_10ms_ring=pulse_res['ring_total'],
                rail_channels_past_5ms=pulse_res['rail_channels_past_5ms'],
                bg_20win_mean=float(bg.mean()) if len(bg) else None,
                bg_20win_std=float(bg.std()) if len(bg) else None,
                bg_20win_p95=float(np.percentile(bg, 95)) if len(bg) else None,
                z_vs_pre_bg=z,
            )
    return dict(
        label='22554_20261119_single_pulse_descriptive',
        note='Single pulse per site with no sham control; descriptive only, cannot pass G2.',
        warmup_meta=wmeta,
        sites=sites_out,
        g2_passed=False,
        seconds=round(time.monotonic() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Part 0B: 200 ms multi-channel spontaneous dynamics & Gate-S across 5 chips
# ---------------------------------------------------------------------------

def dfa_hurst(series: np.ndarray) -> float:
    """Detrended Fluctuation Analysis (order-1 DFA) Hurst exponent H on 1D series."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < 64 or float(x.std()) < 1e-12:
        return 0.5
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.geomspace(4, max(8, n // 6), 14).astype(int))
    scales = scales[scales >= 4]
    flucts = []
    valid_s = []
    for s in scales:
        n_seg = n // s
        if n_seg < 4:
            continue
        seg = y[:n_seg * s].reshape(n_seg, s)
        t = np.arange(s, dtype=np.float64)
        t_c = t - t.mean()
        denom = float((t_c ** 2).sum())
        slope = (seg @ t_c) / denom
        intercept = seg.mean(axis=1) - slope * t.mean()
        trend = intercept[:, None] + slope[:, None] * t[None, :]
        rms = np.sqrt(np.mean((seg - trend) ** 2, axis=1))
        f_val = float(np.sqrt(np.mean(rms ** 2)))
        if f_val > 1e-12:
            valid_s.append(float(s))
            flucts.append(f_val)
    if len(valid_s) < 3:
        return 0.5
    coef = np.polyfit(np.log(valid_s), np.log(flucts), 1)
    return float(coef[0])


def acf_summary(series: np.ndarray, max_lag: int = 25) -> dict:
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var < 1e-12 or len(x) <= max_lag:
        return dict(lag1=0.0, lag3=0.0, lag5=0.0, tau_steps=0.0, tau_s=0.0)
    ac = [1.0]
    for lag in range(1, max_lag + 1):
        ac.append(float(np.dot(x[:-lag], x[lag:]) / var))
    tau = float(max_lag)
    for lag in range(1, max_lag + 1):
        if ac[lag] < math.exp(-1.0):
            # linear interpolation between lag-1 and lag
            prev = ac[lag - 1]
            curr = ac[lag]
            frac = (prev - math.exp(-1.0)) / max(prev - curr, 1e-12)
            tau = (lag - 1) + float(np.clip(frac, 0.0, 1.0))
            break
    return dict(lag1=ac[1], lag3=ac[3], lag5=ac[5], tau_steps=tau, tau_s=tau * (BIN_MS / 1000.0))


def iaaft_surrogate(series: np.ndarray, seed: int, max_iter: int = 100) -> np.ndarray:
    """Iterative Amplitude Adjusted Fourier Transform surrogate (preserves spectrum + exact histogram)."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < 8 or float(x.std()) < 1e-12:
        return x.copy()
    rng = np.random.default_rng(seed)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    surr = rng.permutation(x)
    prev_ranks = None
    for _ in range(max_iter):
        fft_s = np.fft.rfft(surr)
        phases = np.angle(fft_s)
        matched = np.fft.irfft(target_amp * np.exp(1j * phases), n=n)
        ranks = np.argsort(np.argsort(matched, kind='stable'), kind='stable')
        surr = sorted_x[ranks]
        if prev_ranks is not None and np.array_equal(ranks, prev_ranks):
            break
        prev_ranks = ranks
    return surr


def extract_spontaneous_segment(h5_path: Path, *, max_seconds: float | None = None, skip_start_s: float = 2.0) -> dict:
    """Extract clean 200 ms binned spike matrix and physical channel coordinates from one spontaneous H5."""
    with h5py.File(h5_path, 'r') as f:
        d, g = open_maxone_store(f)
        chip_id = str(_scalar(f, 'wellplate/id'))
        fs = float(_scalar(d, 'settings/sampling'))
        frames = g['frame_nos']
        f_start = int(frames[0])
        n_frames = len(frames)
        # Check frame continuity on sampled/chunked blocks
        gaps = 0
        prev_last = None
        for off in range(0, n_frames, 1_000_000):
            blk = np.asarray(frames[off:off + 1_000_000], dtype=np.uint64)
            if prev_last is not None and int(blk[0]) != prev_last + 1:
                gaps += 1
            gaps += int(np.count_nonzero(blk[1:] != blk[:-1] + np.uint64(1)))
            prev_last = int(blk[-1])

        mapping = d['settings/mapping'][:]
        ch_to_xy = {int(row['channel']): (float(row['x']), float(row['y'])) for row in mapping}
        mapped_channels = sorted(ch_to_xy.keys())

        # Quick rail saturation probe across 6 sampled 0.2 s raw windows
        raw = g['raw']
        raw_channels = (
            np.asarray(g['channels'][:], dtype=int)
            if 'channels' in g
            else np.arange(raw.shape[0], dtype=int)
        )
        probe_len = int(0.2 * fs)
        s_start = int(skip_start_s * fs)
        s_end = n_frames - probe_len
        if max_seconds is not None:
            s_end = min(s_end, s_start + int(max_seconds * fs))
        rail_bad = set()
        if s_end > s_start:
            probe_pts = np.linspace(s_start, s_end, 6, dtype=int)
            rail_hits = np.zeros(raw.shape[0], dtype=np.int64)
            total_probe = len(probe_pts) * probe_len
            for pt in probe_pts:
                blk = np.asarray(raw[:, pt:pt + probe_len])
                rail_hits += ((blk <= RAIL_LO) | (blk >= RAIL_HI)).sum(axis=1)
            for row_i, phys_c in enumerate(raw_channels.tolist()):
                if rail_hits[row_i] / max(total_probe, 1) > 0.01:
                    rail_bad.add(int(phys_c))

        usable_end = n_frames
        if max_seconds is not None:
            usable_end = min(usable_end, s_start + int(max_seconds * fs))
        n_bins = max(0, (usable_end - s_start) // BIN_FRAMES)
        dur_s = n_bins * (BIN_MS / 1000.0)

        spk = d['spikes'][:]
        rel_f = spk['frameno'].astype(np.int64) - (f_start + s_start)
        spk_ch = spk['channel'].astype(np.int64)
        in_range = (rel_f >= 0) & (rel_f < n_bins * BIN_FRAMES)
        rel_f = rel_f[in_range]
        spk_ch = spk_ch[in_range]

        ch_counts = np.bincount(spk_ch, minlength=1024)
        active = np.array([
            c for c in mapped_channels
            if c not in rail_bad and (ch_counts[c] / max(dur_s, 1.0)) >= MIN_ACTIVE_RATE_HZ
        ], dtype=int)
        ch_lookup = np.full(1024, -1, dtype=np.int32)
        ch_lookup[active] = np.arange(len(active), dtype=np.int32)

        mapped_idx = ch_lookup[spk_ch]
        keep = mapped_idx >= 0
        bins_idx = (rel_f[keep] // BIN_FRAMES).astype(np.int32)
        ch_idx = mapped_idx[keep]

        # Dense binned matrix: shape (n_bins, n_active), uint16
        mat = np.zeros((n_bins, len(active)), dtype=np.uint16)
        np.add.at(mat, (bins_idx, ch_idx), 1)
        xy = np.array([ch_to_xy[int(c)] for c in active], dtype=np.float64)

    return dict(
        path=str(h5_path),
        chip_id=chip_id,
        fs=fs,
        frame_gaps=int(gaps),
        duration_s=float(dur_s),
        n_bins=int(n_bins),
        n_mapped=len(mapped_channels),
        n_rail_excluded=len(rail_bad),
        n_active=len(active),
        n_spikes_kept=int(mat.sum()),
        active_channels=active,
        xy=xy,
        mat=mat,
    )


def build_chip_dynamics_and_gate_s(chip_name: str, h5_paths: list[Path], out_dir: Path, seed: int) -> dict:
    """Build 200 ms dynamics library + run Gate-S biology verification + export C/G1/G2 surrogates."""
    t0 = time.monotonic()
    # Use the longest clean segment as primary continuous recording (preserves unbroken temporal correlation!)
    segs = [extract_spontaneous_segment(p) for p in h5_paths if p.exists()]
    segs.sort(key=lambda s: (s['frame_gaps'] == 0, s['n_active'] >= GATE_S_MIN_CHANNELS, s['duration_s']), reverse=True)
    primary = segs[0]
    mat = primary['mat'].astype(np.float64)  # (T, C)
    xy = primary['xy']                        # (C, 2)
    T, C = mat.shape

    total_counts = mat.sum(axis=1)  # spikes per 200 ms bin across all active channels
    mean_c = float(total_counts.mean())
    var_c = float(total_counts.var())
    fano = float(var_c / max(mean_c, 1e-12))
    p50 = float(np.median(total_counts))
    p95 = float(np.percentile(total_counts, 95))
    p99 = float(np.percentile(total_counts, 99))
    burst_thresh = max(p95, mean_c + 2.0 * math.sqrt(max(var_c, 1e-12)))
    burst_rate_frac = float(np.mean(total_counts >= burst_thresh))

    # 1. Burst intensity z-score via log1p robust normalization
    log_c = np.log1p(total_counts)
    med_l = float(np.median(log_c))
    mad_l = max(1.4826 * float(np.median(np.abs(log_c - med_l))), float(log_c.std()) * 0.25, 1e-6)
    burst_z = np.clip((log_c - med_l) / mad_l, -4.0, 4.0)

    # 2. Spatial entropy per 200 ms bin across active channels
    denom_h = math.log(max(C, 2))
    spatial_entropy = np.zeros(T, dtype=np.float64)
    nonzero_t = total_counts > 0
    if nonzero_t.any():
        p_mat = mat[nonzero_t] / total_counts[nonzero_t, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -np.sum(np.where(p_mat > 0, p_mat * np.log(p_mat), 0.0), axis=1)
        spatial_entropy[nonzero_t] = np.clip(ent / denom_h, 0.0, 1.0)

    # 3. 4-Quadrant spatial manifold vector u_t in R^4 mapped to (N, E, S, W) around active centroid
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
    # Assign each active channel to nearest cardinal sector (N: dy<0 & |dy|>=|dx|, E: dx>=0 & |dx|>|dy|, S: dy>=0 & |dy|>=|dx|, W: dx<0 & |dx|>|dy|)
    q_masks = [
        (dy < 0) & (np.abs(dy) >= np.abs(dx)),   # N (0, -1)
        (dx >= 0) & (np.abs(dx) > np.abs(dy)),   # E (1, 0)
        (dy >= 0) & (np.abs(dy) >= np.abs(dx)),  # S (0, 1)
        (dx < 0) & (np.abs(dx) > np.abs(dy)),    # W (-1, 0)
    ]
    q_counts = [int(m.sum()) for m in q_masks]
    q_series = np.zeros((T, 4), dtype=np.float64)
    for qi, m in enumerate(q_masks):
        if m.any():
            raw_q = np.log1p(mat[:, m].mean(axis=1))
            q_series[:, qi] = (raw_q - raw_q.mean()) / max(float(raw_q.std()), 1e-6)
    # Zero-mean across 4 directions per step, then unit-norm (zero directional bias on average!)
    u_centered = q_series - q_series.mean(axis=1, keepdims=True)
    u_norms = np.linalg.norm(u_centered, axis=1, keepdims=True)
    u_4dir = np.where(u_norms > 1e-8, u_centered / np.maximum(u_norms, 1e-8), 0.0)

    # Gate-S statistical tests
    rng = np.random.default_rng(seed)
    n_perm = 400
    acf_real = acf_summary(burst_z)
    h_real = dfa_hurst(burst_z)

    shuf_lag1 = np.empty(n_perm, dtype=np.float64)
    shuf_hurst = np.empty(n_perm, dtype=np.float64)
    for pi in range(n_perm):
        perm_b = rng.permutation(burst_z)
        shuf_lag1[pi] = acf_summary(perm_b, max_lag=5)['lag1']
        shuf_hurst[pi] = dfa_hurst(perm_b)
    p_lag1 = float((1 + (shuf_lag1 >= acf_real['lag1']).sum()) / (1 + n_perm))
    p_hurst = float((1 + (shuf_hurst >= h_real).sum()) / (1 + n_perm))

    # Cross-channel spatial synchrony (top-100 most active channels vs circular time-shift null)
    top_k = min(100, C)
    top_idx = np.argsort(mat.sum(axis=0))[-top_k:]
    sub = mat[:, top_idx]
    sub_std = sub.std(axis=0)
    keep_s = sub_std > 1e-8
    sub = sub[:, keep_s]
    if sub.shape[1] >= 4:
        sub_z = (sub - sub.mean(axis=0)) / sub.std(axis=0)
        corr_mat = (sub_z.T @ sub_z) / T
        tri = np.triu_indices(sub.shape[1], k=1)
        mean_sync = float(corr_mat[tri].mean())
        null_sync = np.empty(200, dtype=np.float64)
        for pi in range(200):
            shifts = rng.integers(5, max(6, T - 5), size=sub.shape[1])
            shifted = np.column_stack([np.roll(sub_z[:, j], int(shifts[j])) for j in range(sub.shape[1])])
            cm = (shifted.T @ shifted) / T
            null_sync[pi] = float(cm[tri].mean())
        p_sync = float((1 + (null_sync >= mean_sync).sum()) / (1 + 200))
    else:
        mean_sync, p_sync = 0.0, 1.0

    # Generate locked controls: C (time-shuffled), G1 (matched-variance white), G2 (IAAFT surrogate)
    perm_idx = rng.permutation(T)
    c_burst = burst_z[perm_idx]
    c_entropy = spatial_entropy[perm_idx]
    c_u = u_4dir[perm_idx]

    g1_burst = rng.normal(float(burst_z.mean()), max(float(burst_z.std()), 1e-6), size=T)
    g1_entropy = np.clip(rng.normal(float(spatial_entropy.mean()), max(float(spatial_entropy.std()), 1e-6), size=T), 0.0, 1.0)
    g1_raw_u = rng.normal(0.0, 1.0, size=(T, 4))
    g1_raw_u -= g1_raw_u.mean(axis=1, keepdims=True)
    g1_u = g1_raw_u / np.maximum(np.linalg.norm(g1_raw_u, axis=1, keepdims=True), 1e-8)

    g2_burst = iaaft_surrogate(burst_z, seed=seed + 101)
    g2_entropy = iaaft_surrogate(spatial_entropy, seed=seed + 102)
    g2_q = np.column_stack([iaaft_surrogate(q_series[:, qi], seed=seed + 103 + qi) for qi in range(4)])
    g2_qc = g2_q - g2_q.mean(axis=1, keepdims=True)
    g2_u = g2_qc / np.maximum(np.linalg.norm(g2_qc, axis=1, keepdims=True), 1e-8)

    # Nonlinearity check: real vs IAAFT burst-run clustering (mean consecutive burst run length) and quadrant synchrony
    def _mean_burst_run(arr):
        high = arr >= np.percentile(arr, 90)
        padded = np.concatenate(([False], high, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        stops = np.where(diff == -1)[0]
        return float(np.mean(stops - starts)) if len(starts) else 0.0

    def _skewness(arr):
        z = (arr - arr.mean()) / max(float(arr.std()), 1e-8)
        return float(np.mean(z ** 3))

    def _third_order_asymmetry(arr):
        """Time-reversal asymmetry E[(x_{t+1} - x_t)^3] / std^3 — vanishes for linear Gaussian processes."""
        dx_s = np.diff(arr)
        return float(np.mean(dx_s ** 3) / max(float(dx_s.std()) ** 3, 1e-8))

    asym_real = _third_order_asymmetry(burst_z)
    asym_iaaft = _third_order_asymmetry(g2_burst)

    gate_checks = dict(
        enough_channels=bool(C >= GATE_S_MIN_CHANNELS),
        enough_duration=bool(primary['duration_s'] >= GATE_S_MIN_DURATION_S),
        zero_frame_gaps=bool(primary['frame_gaps'] == 0),
        heavy_tailed_bursts=bool(fano >= GATE_S_MIN_FANO),
        temporal_persistence=bool(h_real >= GATE_S_MIN_HURST and p_hurst <= GATE_S_P_MAX and p_lag1 <= GATE_S_P_MAX),
        spatial_synchrony=bool(mean_sync > 0.01 and p_sync <= GATE_S_P_MAX),
    )
    gate_s_passed = all(gate_checks.values())

    # Save frozen dynamics library (.npz + manifest .json)
    lib_dir = out_dir / 'libraries' / chip_name
    lib_dir.mkdir(parents=True, exist_ok=True)
    npz_path = lib_dir / f'{chip_name}_dynamics_200ms.npz'
    np.savez_compressed(
        npz_path,
        real_burst_z=burst_z.astype(np.float32),
        real_entropy=spatial_entropy.astype(np.float32),
        real_u4=u_4dir.astype(np.float32),
        shuffled_burst_z=c_burst.astype(np.float32),
        shuffled_entropy=c_entropy.astype(np.float32),
        shuffled_u4=c_u.astype(np.float32),
        white_burst_z=g1_burst.astype(np.float32),
        white_entropy=g1_entropy.astype(np.float32),
        white_u4=g1_u.astype(np.float32),
        iaaft_burst_z=g2_burst.astype(np.float32),
        iaaft_entropy=g2_entropy.astype(np.float32),
        iaaft_u4=g2_u.astype(np.float32),
        active_channels=primary['active_channels'].astype(np.int32),
    )
    npz_sha = sha256_bytes(npz_path.read_bytes())

    summary = dict(
        schema='wetware.spontaneous.dynamics.v1',
        chip=chip_name,
        source_h5=primary['path'],
        verified_chip_id=primary['chip_id'],
        bin_ms=BIN_MS,
        duration_s=primary['duration_s'],
        n_bins=int(T),
        n_active_channels=int(C),
        n_rail_excluded=primary['n_rail_excluded'],
        frame_gaps=primary['frame_gaps'],
        n_spikes_kept=primary['n_spikes_kept'],
        quadrant_channel_counts=dict(N=q_counts[0], E=q_counts[1], S=q_counts[2], W=q_counts[3]),
        burst_stats=dict(
            mean_spikes_per_200ms=mean_c,
            var_spikes_per_200ms=var_c,
            fano_factor=fano,
            p50=p50,
            p95=p95,
            p99=p99,
            burst_frac=burst_rate_frac,
            skewness_burst_z=_skewness(burst_z),
            mean_burst_run_real=_mean_burst_run(burst_z),
            mean_burst_run_iaaft=_mean_burst_run(g2_burst),
            time_reversal_asym_real=asym_real,
            time_reversal_asym_iaaft=asym_iaaft,
        ),
        temporal_stats=dict(
            acf=acf_real,
            lag1_shuf_mean=float(shuf_lag1.mean()),
            p_lag1=p_lag1,
            hurst_dfa=h_real,
            hurst_shuf_mean=float(shuf_hurst.mean()),
            p_hurst=p_hurst,
            iaaft_lag1=acf_summary(g2_burst, max_lag=5)['lag1'],
            iaaft_hurst=dfa_hurst(g2_burst),
        ),
        spatial_stats=dict(
            mean_pairwise_corr_top100=mean_sync,
            p_spatial_sync=p_sync,
        ),
        gate_s_checks=gate_checks,
        gate_s_passed=gate_s_passed,
        library_npz=str(npz_path),
        library_npz_sha256=npz_sha,
        seconds=round(time.monotonic() - t0, 2),
    )
    summary['manifest_sha256'] = sha256_json(summary)
    (lib_dir / 'manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def run_phase0(out_dir: Path, skip_narrow: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'output'
    t_all = time.monotonic()

    # Part 0A: Narrow-window [5, 50) ms check on historical stimulated H5s
    part_0a = []
    if not skip_narrow:
        sessions = [
            ('28080_20261119_crossover', out / 'mea_predictability_28080_20261119T065953Z/cpp-crossover-v1', 2026112301),
            ('28080_20261119_site_rhythm', out / 'mea_predictability_28080_20261119T065953Z/site-rhythm-v1/records', 2026112302),
            ('22021_20261119_crossover', out / 'mea_predictability_22021_20261119T042416Z/cpp-crossover-v1', 2026112303),
        ]
        for label, sdir, seed in sessions:
            if (sdir / 'warmup' / 'warmup.raw.h5').exists():
                print(f'[0A] Auditing narrow [5,50)ms: {label} ...', flush=True)
                res = audit_crossover_session_narrow(sdir, label, seed)
                part_0a.append(res)
                print(f"     done in {res['seconds']}s | pulse AUC={res['pulse_5_50ms_count']['auc']} p={res['pulse_5_50ms_count']['p']} | G2={res['g2_passed']}", flush=True)
        d22554 = out / '22554_full_20261119_1224'
        if (d22554 / 'records/session/paired/baseline/baseline.raw.h5').exists():
            print('[0A] Auditing narrow [5,50)ms descriptive: 22554_20261119 ...', flush=True)
            res2 = audit_22554_single_pulse_narrow(d22554)
            part_0a.append(res2)
            print(f"     done in {res2['seconds']}s", flush=True)

    # Part 0B: 5-chip 200 ms spontaneous dynamics & Gate-S
    chip_sources = {
        '22021': [
            out / 'mea_live_intake_20261117_main/live_22021_20261117_032840/baseline_600s/baseline.raw.h5',
            out / 'mea_predictability_22021_20261119T042416Z/baseline/quiet.raw.h5',
        ],
        '28080': [
            out / 'mea_full_28080_20261117_1406/backup/recording_resume2/baseline.raw.h5',
            out / 'mea_predictability_28080_20261119T065953Z/baseline/quiet.raw.h5',
        ],
        '22554': [
            out / 'mea_22554_baseline_20261116/source/22554_baseline.raw.h5',
            out / '22554_full_20261119_1224/records/session/paired/baseline/baseline.raw.h5',
        ],
        '27701': [
            out / 'mea_predictability_27701_20261119T072625Z/cpp-crossover-v1/warmup/warmup.raw.h5',
            out / 'mea_predictability_27701_20261119T072625Z/baseline/quiet.raw.h5',
        ],
        '28073': [
            out / 'pair28073-20261118T070238Z/baseline-final-route.raw.h5',
            out / 'pair28073-20261118T070238Z/readback-run/connected-baseline.raw.h5',
        ],
    }
    part_0b = []
    for idx, (chip, paths) in enumerate(chip_sources.items()):
        print(f'[0B] Building 200ms dynamics + Gate-S for chip {chip} ...', flush=True)
        res = build_chip_dynamics_and_gate_s(chip, paths, out_dir, seed=2026112400 + idx * 100)
        part_0b.append(res)
        print(
            f"     done in {res['seconds']}s | ch={res['n_active_channels']} dur={res['duration_s']:.0f}s "
            f"Fano={res['burst_stats']['fano_factor']:.2f} H={res['temporal_stats']['hurst_dfa']:.3f} "
            f"(p={res['temporal_stats']['p_hurst']:.4f}) sync={res['spatial_stats']['mean_pairwise_corr_top100']:.4f} "
            f"(p={res['spatial_stats']['p_spatial_sync']:.4f}) | Gate-S={res['gate_s_passed']}",
            flush=True,
        )

    passed_chips = [r['chip'] for r in part_0b if r['gate_s_passed']]
    any_g2 = any(r.get('g2_passed') for r in part_0a)
    report = dict(
        schema='wetware.phase0.audit.v1',
        total_seconds=round(time.monotonic() - t_all, 2),
        any_historical_g2_passed=any_g2,
        gate_s_passed_chips=passed_chips,
        gate_s_ready_for_phase2_3=len(passed_chips) >= 3,
        part_0a_narrow_evoked=part_0a,
        part_0b_spontaneous_dynamics=part_0b,
    )
    report['report_sha256'] = sha256_json(report)
    (out_dir / 'phase0_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Write concise human-readable Markdown report
    lines = [
        '# Phase 0 现有 MEA 数据只读体检与自发动力学建库报告',
        '',
        f'- 总耗时：`{report["total_seconds"]} s`',
        f'- 历史窄窗 `[5, 50) ms` 是否有芯片意外通过 G2：**`{any_g2}`**',
        f'- 通过 `Gate-S` 生物动力学门控的独立芯片（**{len(passed_chips)}/{len(part_0b)}**）：**`{", ".join(passed_chips) or "无"}`**',
        '',
        '## Part 0A：历史 11/19 刺激记录 `[5, 50) ms` 先屏蔽后因果滤波复查',
        '',
        '| 会话 | 活跃通道 | 真脉冲 / 假窗 | 原始计数 AUC (p) | 有效时长归一化速率 AUC (p) | 未屏蔽比例 真/假 | 振铃 `[5,10)ms` 真/假 | G2 |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in part_0a:
        if 'pulse_5_50ms_count' in r:
            pc = r['pulse_5_50ms_count']
            pr = r['pulse_5_50ms_rate']
            vf = r['valid_pri_frac']
            rg = r['ringing_check_5_10ms']
            lines.append(
                f"| `{r['label']}` | {r['warmup_meta']['n_active']} | {r['n_real_pulses']} / {r['n_sham_pulses']} | "
                f"{pc['auc']:.3f} ({pc['p']:.4f}) | {pr['auc']:.3f} ({pr['p']:.4f}) | "
                f"{vf['real_mean']:.3f} / {vf['sham_mean']:.3f} | {rg['real_mean']:.2f} / {rg['sham_mean']:.2f} | {'PASS' if r['g2_passed'] else 'FAIL'} |"
            )
        else:
            for s, sv in r.get('sites', {}).items():
                lines.append(
                    f"| `22554_{s}` (单脉冲无假刺激) | {r['warmup_meta']['n_active']} | 1 / 20前窗 | "
                    f"脉冲={sv['pulse_5_50ms']}, 前窗均值={sv['bg_20win_mean']:.2f}±{sv['bg_20win_std']:.2f} (z={sv['z_vs_pre_bg']:.2f}) | — | — | 振铃={sv['pulse_5_10ms_ring']} | N/A |"
                )

    lines += [
        '',
        '## Part 0B：5 块活跃芯片 200 ms 自发群体动力学与 `Gate-S` 体检',
        '',
        '| 芯片 | 时长 (s) | 活跃通道 | Fano 因子 | Lag-1 ACF (p) | DFA Hurst H (p) | Top100 空间同步 (p) | 时间反转非对称 (真 / IAAFT) | `Gate-S` |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in part_0b:
        bs = r['burst_stats']
        ts = r['temporal_stats']
        ss = r['spatial_stats']
        lines.append(
            f"| `{r['chip']}` | {r['duration_s']:.0f} | {r['n_active_channels']} | {bs['fano_factor']:.2f} | "
            f"{ts['acf']['lag1']:.3f} ({ts['p_lag1']:.4f}) | {ts['hurst_dfa']:.3f} ({ts['p_hurst']:.4f}) | "
            f"{ss['mean_pairwise_corr_top100']:.4f} ({ss['p_spatial_sync']:.4f}) | "
            f"{bs['time_reversal_asym_real']:+.3f} / {bs['time_reversal_asym_iaaft']:+.3f} | "
            f"{'**PASS**' if r['gate_s_passed'] else 'FAIL'} |"
        )

    lines += [
        '',
        '## 冻结动力学库与对照文件（已落盘）',
        '',
    ]
    for r in part_0b:
        lines.append(f"- 芯片 `{r['chip']}`：`{r['library_npz']}`（SHA-256 `{r['library_npz_sha256'][:16]}...`，`Gate-S={'PASS' if r['gate_s_passed'] else 'FAIL'}`）")

    (out_dir / 'PHASE0_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'output/wetware_existing_data_phase0_20261123')
    p.add_argument('--skip-narrow', action='store_true')
    args = p.parse_args()
    run_phase0(args.out, skip_narrow=args.skip_narrow)


if __name__ == '__main__':
    main()
