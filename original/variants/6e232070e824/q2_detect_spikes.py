import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy import signal
from mea_stim_v2.recording import H5Recording
from oddball import analysis

t_start = time.time()
rec_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')

plan = json.loads((rec_dir / 'plan.json').read_text(encoding='utf-8'))
decision = json.loads((rec_dir / 'decision.json').read_text(encoding='utf-8'))
route = json.loads((rec_dir / 'route.json').read_text(encoding='utf-8'))
warmup_meta = json.loads((rec_dir / 'warmup.json').read_text(encoding='utf-8'))
binding = json.loads((rec_dir / 'binding.json').read_text(encoding='utf-8'))

roi = np.array(sorted(decision['roi']), dtype=int)
site_channels = {k: int(v) for k, v in route['channels'].items()}
cfg = SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])

# Causal 4th-order Butterworth bandpass filter 300-3000 Hz at 20 kHz
FS = 20000.0
SOS = signal.butter(4, [300.0, 3000.0], btype='bandpass', fs=FS, output='sos')

# Fixed blanking parameters (in frames relative to marker f0):
# Raw linear interpolation across [-5, +50) frames = [-0.25, +2.50) ms
RAW_MASK_PRE = -5
RAW_MASK_POST = 50
# Dynamic rail mask expansion: any sample <=2 or >=1021 expands by [-5, +60] frames = [-0.25, +3.00] ms
RAIL_LOW = 2
RAIL_HIGH = 1021
RAIL_PAD_PRE = 5
RAIL_PAD_POST = 60
# Post-filter spike blanking across [-20, +100) frames = [-1.0, +5.0) ms -> b = 5.0 ms!
POST_BLANK_PRE = -20
POST_BLANK_POST = 100   # b = +5.0 ms (100 frames)
FILT_SETTLE_FRAMES = 50 # 2.5 ms post-interpolation filter settle guard for dynamic rails
REFRACTORY_FRAMES = 20  # 1.0 ms refractory period
THRESH_MULT = 5.0       # 5.0 * sigma_noise (matching device spike_threshold = 5)

# Step 1: Identify all active channels (>= 0.05 Hz in warmup t >= 4.0s) and compute per-channel sigma_c from warmup ONLY!
with H5Recording(rec_dir / 'warmup/warmup.raw.h5', cfg) as wrec:
    dur_w = wrec.duration - 4.0
    late = wrec.spike_times >= 4.0
    w_counts = np.bincount(wrec.spike_channels[late], minlength=1024)
    active_all = np.array(sorted(
        c for c in wrec.channels.tolist()
        if c not in binding.get('bad_channels', []) and (w_counts[c] / dur_w) >= 0.05
    ), dtype=int)
    # Ensure all 233 ROI channels are in active_all (they were selected from warmup >= 0.05 Hz!)
    assert set(roi.tolist()) <= set(active_all.tolist()), 'ROI must be subset of active_all'
    ch_xy = wrec.channel_xy_um

    # Compute per-channel noise sigma from warmup [4.0s, 64.0s] in 10s chunks to keep memory small
    medians_per_chunk = []
    for t0 in np.arange(4.0, wrec.duration - 9.9, 10.0):
        raw_chunk = wrec.sample(active_all, float(t0), 10.0).astype(np.float32)
        raw_chunk -= np.median(raw_chunk, axis=1, keepdims=True)
        filt_chunk = signal.sosfilt(SOS, raw_chunk, axis=1)[:, 400:]  # drop first 20ms filter settle
        medians_per_chunk.append(np.median(np.abs(filt_chunk), axis=1) / 0.6745)
    sigma_ch = np.median(np.stack(medians_per_chunk, axis=0), axis=0)  # shape (n_active,)
    lsb_uv = wrec.metadata['lsb_uv']

print(f'Active channels (>=0.05 Hz in warmup): {len(active_all)} (ROI={len(roi)})')
print(f'Per-channel noise sigma (uV): median={np.median(sigma_ch)*lsb_uv:.2f}, IQR=[{np.percentile(sigma_ch,25)*lsb_uv:.2f}, {np.percentile(sigma_ch,75)*lsb_uv:.2f}], max={np.max(sigma_ch)*lsb_uv:.2f}')

roi_mask_in_active = np.isin(active_all, roi)
roi_indices_in_active = np.where(roi_mask_in_active)[0]

def blank_and_filter_trial(raw_win):
    """
    raw_win: float32 array of shape (n_ch, 12800), covering [-6400, +6400) frames around marker f0 (index 6400).
    Identical processing for real and sham trials!
    Returns:
      filt_300: float32 (n_ch, 12000) covering [-6000, +6000) frames ([-300, +300) ms)
      valid_mask_300: bool (n_ch, 12000), False where blanked
      extra_rail_after_5ms: bool (n_ch,) True if channel had rail mask extending past +5.0 ms
    """
    n_ch, n_tot = raw_win.shape
    work = raw_win.copy()
    center = 6400  # marker frame index in raw_win

    # 1. Detect ADC rail samples anywhere in [-6400, +6400) identically on real & sham
    rail = (work <= RAIL_LOW) | (work >= RAIL_HIGH)

    # Build raw interpolation mask: start with fixed [-5, +50) around center for ALL channels & ALL trials
    interp_mask = np.zeros((n_ch, n_tot), dtype=bool)
    interp_mask[:, center + RAW_MASK_PRE : center + RAW_MASK_POST] = True

    # Expand any rail hits by [-RAIL_PAD_PRE, +RAIL_PAD_POST]
    if np.any(rail):
        kernel = np.ones(RAIL_PAD_PRE + RAIL_PAD_POST + 1, dtype=int)
        ch_with_rail = np.where(rail.any(axis=1))[0]
        for ci in ch_with_rail:
            # Convolve with asymmetric pad: rail at t masks [t - RAIL_PAD_PRE, t + RAIL_PAD_POST]
            idx_r = np.where(rail[ci])[0]
            # Fast segment union
            starts = np.maximum(0, idx_r - RAIL_PAD_PRE)
            stops = np.minimum(n_tot, idx_r + RAIL_PAD_POST + 1)
            for s0, s1 in zip(starts, stops):
                interp_mask[ci, s0:s1] = True

    # Linearly interpolate each masked contiguous interval per channel BEFORE filtering!
    for ci in range(n_ch):
        m = interp_mask[ci]
        # Find contiguous True runs
        padded = np.concatenate(([False], m, [False]))
        diff = np.diff(padded.astype(np.int8))
        run_starts = np.where(diff == 1)[0]
        run_stops = np.where(diff == -1)[0]
        for s0, s1 in zip(run_starts, run_stops):
            left_a = max(0, s0 - 20)
            left_val = float(np.median(work[ci, left_a:s0])) if s0 > left_a else float(work[ci, min(s1, n_tot - 1)])
            right_b = min(n_tot, s1 + 20)
            right_val = float(np.median(work[ci, s1:right_b])) if right_b > s1 else left_val
            work[ci, s0:s1] = np.linspace(left_val, right_val, s1 - s0, endpoint=False, dtype=np.float32)

    # Subtract pre-stimulus baseline median [-320, -50) ms = [0, 5400)
    work -= np.median(work[:, :5400], axis=1, keepdims=True)

    # Causal 4th-order Butterworth bandpass filter 300-3000 Hz
    filt = signal.sosfilt(SOS, work, axis=1)

    # Post-filter valid mask: blank [-20, +100) frames around center ([-1.0, +5.0) ms) identically on ALL trials,
    # PLUS blank [s0, s1 + FILT_SETTLE_FRAMES) for any extended rail interval!
    valid = ~interp_mask
    valid[:, center + POST_BLANK_PRE : center + POST_BLANK_POST] = False
    if np.any(rail):
        for ci in ch_with_rail:
            m = interp_mask[ci]
            padded = np.concatenate(([False], m, [False]))
            diff = np.diff(padded.astype(np.int8))
            for s1 in np.where(diff == -1)[0]:
                valid[ci, s1 : min(n_tot, s1 + FILT_SETTLE_FRAMES)] = False

    # Slice [-300, +300) ms = [center - 6000, center + 6000) = [400, 12400)
    filt_300 = filt[:, 400:12400]
    valid_300 = valid[:, 400:12400]
    # Check if channel was still invalid/blanked past +5.0 ms (index 6000 + 100 = 6100 in 12000-len slice)
    extra_rail_after_5ms = ~valid_300[:, 6100:].all(axis=1)
    return filt_300, valid_300, extra_rail_after_5ms

def detect_negative_spikes(filt_300, valid_300, thresholds):
    """
    Detect local minima below -thresholds (thresholds > 0) where valid_300 is True,
    with 20-frame (1.0 ms) refractory dead-time per channel.
    Returns list of (ch_local_idx, rel_frame_in_-6000..+5999).
    """
    # Local minimum: y[t] < -thr and y[t] <= y[t-1] and y[t] < y[t+1]
    neg_thr = -thresholds[:, None]
    mid = filt_300[:, 1:-1]
    cand = (mid < neg_thr) & valid_300[:, 1:-1] & (mid <= filt_300[:, :-2]) & (mid < filt_300[:, 2:])
    ch_idxs, t_idxs_minus1 = np.where(cand)
    t_idxs = t_idxs_minus1 + 1  # in [1, 11998], relative to -6000

    out_ch = []
    out_rel_f = []
    if len(ch_idxs) == 0:
        return np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int16)

    # Enforce refractory period per channel (keep deepest trough within 20 frames)
    splits = np.searchsorted(ch_idxs, np.arange(len(thresholds) + 1))
    for ci in range(len(thresholds)):
        a, b_idx = splits[ci], splits[ci + 1]
        if a == b_idx:
            continue
        ts = t_idxs[a:b_idx]
        if len(ts) == 1:
            out_ch.append(ci)
            out_rel_f.append(int(ts[0]) - 6000)
            continue
        vals = filt_300[ci, ts]
        # Greedy refractory: walk left-to-right; within 20 frames of accepted spike, replace if deeper
        kept_t = int(ts[0])
        kept_v = float(vals[0])
        for t_cur, v_cur in zip(ts[1:], vals[1:]):
            t_cur = int(t_cur)
            v_cur = float(v_cur)
            if t_cur - kept_t < REFRACTORY_FRAMES:
                if v_cur < kept_v:
                    kept_t, kept_v = t_cur, v_cur
            else:
                out_ch.append(ci)
                out_rel_f.append(kept_t - 6000)
                kept_t, kept_v = t_cur, v_cur
        out_ch.append(ci)
        out_rel_f.append(kept_t - 6000)
    return np.array(out_ch, dtype=np.int16), np.array(out_rel_f, dtype=np.int16)

# Process all 12 main blocks!
thresholds = THRESH_MULT * sigma_ch
main_blocks = [b for b in plan['blocks'] if b['stage'] == 'main']

trial_meta = []
all_off_trial_id = []
all_off_ch_idx = []
all_off_rel_frame = []

all_on_trial_id = []
all_on_ch_idx = []
all_on_rel_frame = []

extra_rail_counts = {s: np.zeros(len(active_all), dtype=int) for s in ('H', 'W', 'F', 'Z')}

ch_to_local = np.full(1024, -1, dtype=int)
for idx_c, ch_num in enumerate(active_all):
    ch_to_local[ch_num] = idx_c

global_tid = 0
for b in main_blocks:
    bname = b['name']
    t_b0 = time.time()
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    h5_path = rec_dir / bname / f'{bname}.raw.h5'
    with H5Recording(h5_path, cfg) as rec:
        # Prepare online spikes for active_all
        keep_on = np.isin(rec.spike_channels, active_all)
        on_frames = np.rint(rec.spike_times[keep_on] * rec.fs).astype(np.int64)
        on_local_ch = ch_to_local[rec.spike_channels[keep_on]]
        order_on = np.argsort(on_frames)
        on_frames = on_frames[order_on]
        on_local_ch = on_local_ch[order_on]

        for t in qc['trials']:
            f0 = int(t['frame'])
            site = t['site']
            raw_win = rec.sample(active_all, (f0 - 6400) / rec.fs, 12800 / rec.fs).astype(np.float32)
            filt_300, valid_300, extra_rail = blank_and_filter_trial(raw_win)
            extra_rail_counts[site] += extra_rail.astype(int)

            off_c, off_f = detect_negative_spikes(filt_300, valid_300, thresholds)
            if len(off_c):
                all_off_trial_id.append(np.full(len(off_c), global_tid, dtype=np.int16))
                all_off_ch_idx.append(off_c)
                all_off_rel_frame.append(off_f)

            # Online spikes in [-6000, +6000) around f0
            lo_on = np.searchsorted(on_frames, f0 - 6000)
            hi_on = np.searchsorted(on_frames, f0 + 6000)
            if hi_on > lo_on:
                rel_on = (on_frames[lo_on:hi_on] - f0).astype(np.int16)
                ch_on = on_local_ch[lo_on:hi_on].astype(np.int16)
                all_on_trial_id.append(np.full(len(rel_on), global_tid, dtype=np.int16))
                all_on_ch_idx.append(ch_on)
                all_on_rel_frame.append(rel_on)

            trial_meta.append({
                'tid': global_tid,
                'block': bname,
                'round': b['round'],
                'kind': b['kind'],
                'frequent': b.get('frequent'),
                'trial': t['trial'],
                'frame': f0,
                'site': site,
                'real': bool(t['real']),
                'count': t['count'],
                'bin': t['bin'],
                'valid': bool(t['valid']),
            })
            global_tid += 1
    print(f'Processed {bname} in {time.time() - t_b0:.1f}s')

off_tid = np.concatenate(all_off_trial_id)
off_ch = np.concatenate(all_off_ch_idx)
off_rf = np.concatenate(all_off_rel_frame)

on_tid = np.concatenate(all_on_trial_id)
on_ch = np.concatenate(all_on_ch_idx)
on_rf = np.concatenate(all_on_rel_frame)

np.savez_compressed(
    out_dir / 'q2_detected_spikes.npz',
    active_all=active_all,
    roi=roi,
    roi_mask_in_active=roi_mask_in_active,
    sigma_ch=sigma_ch,
    lsb_uv=lsb_uv,
    off_tid=off_tid,
    off_ch=off_ch,
    off_rf=off_rf,
    on_tid=on_tid,
    on_ch=on_ch,
    on_rf=on_rf,
)
(out_dir / 'q2_trials_meta.json').write_text(json.dumps(trial_meta, indent=2), encoding='utf-8')

# Self-check on sham (Z) trials on ROI channels!
off_is_roi = roi_mask_in_active[off_ch]
# Apply same clean mask [-20, +100) frames ([-1.0, +5.0) ms) to online spikes for apples-to-apples comparison on sham!
on_is_roi_clean = roi_mask_in_active[on_ch] & ((on_rf < POST_BLANK_PRE) | (on_rf >= POST_BLANK_POST))
on_is_roi_raw = roi_mask_in_active[on_ch]

sham_tids = np.array([m['tid'] for m in trial_meta if m['site'] == 'Z'], dtype=int)
off_per_trial = np.bincount(off_tid[off_is_roi], minlength=1200)
on_clean_per_trial = np.bincount(on_tid[on_is_roi_clean], minlength=1200)
on_raw_per_trial = np.bincount(on_tid[on_is_roi_raw], minlength=1200)

sham_off = off_per_trial[sham_tids]
sham_on_clean = on_clean_per_trial[sham_tids]
sham_on_raw = on_raw_per_trial[sham_tids]

pearson_clean = float(np.corrcoef(sham_off, sham_on_clean)[0, 1])
spearman_clean = float(analysis.spearman(sham_off, sham_on_clean))
pearson_raw = float(np.corrcoef(sham_off, sham_on_raw)[0, 1])
spearman_raw = float(analysis.spearman(sham_off, sham_on_raw))

# Per-channel correlation across the 233 ROI channels on sham trials!
sham_off_mask = off_is_roi & np.isin(off_tid, sham_tids)
sham_on_mask = on_is_roi_clean & np.isin(on_tid, sham_tids)
ch_off_counts = np.bincount(off_ch[sham_off_mask], minlength=len(active_all))[roi_mask_in_active]
ch_on_counts = np.bincount(on_ch[sham_on_mask], minlength=len(active_all))[roi_mask_in_active]
pearson_by_ch = float(np.corrcoef(ch_off_counts, ch_on_counts)[0, 1])
spearman_by_ch = float(analysis.spearman(ch_off_counts, ch_on_counts))

# Artifact check right after blanking window: [+5.0, +10.0) ms ([+100, +200) frames) on ROI channels!
early_5_10_mask = off_is_roi & (off_rf >= 100) & (off_rf < 200)
early_5_10_trial = np.bincount(off_tid[early_5_10_mask], minlength=1200)
pre_5ms_mask = off_is_roi & (off_rf >= -120) & (off_rf < -20)  # [-6, -1) ms (5 ms window right before blanking!)
pre_5ms_trial = np.bincount(off_tid[pre_5ms_mask], minlength=1200)

print('\n=== Q2 SHAM SELF-CHECK (120 sham trials, 233 ROI channels, [-300, +300) ms) ===')
print(f'Sham per-trial mean: offline={sham_off.mean():.2f} (SD={sham_off.std():.2f}), online_masked={sham_on_clean.mean():.2f} (SD={sham_on_clean.std():.2f}), online_unmasked={sham_on_raw.mean():.2f}')
print(f'Per-trial correlation (offline vs online_masked): Pearson r = {pearson_clean:.4f}, Spearman rho = {spearman_clean:.4f}')
print(f'Per-trial correlation (offline vs online_unmasked): Pearson r = {pearson_raw:.4f}, Spearman rho = {spearman_raw:.4f}')
print(f'Per-channel correlation across 233 ROI channels on sham: Pearson r = {pearson_by_ch:.4f}, Spearman rho = {spearman_by_ch:.4f}')

print('\n=== Q2 ARTIFACT RINGING CHECK AROUND BLANKING [-1.0, +5.0) ms ON ROI CHANNELS ===')
for s in ('H', 'W', 'F', 'Z'):
    idx_s = [m['tid'] for m in trial_meta if m['site'] == s]
    print(f'  Site {s}: [-6,-1)ms mean={pre_5ms_trial[idx_s].mean():.3f}, [+5,+10)ms mean={early_5_10_trial[idx_s].mean():.3f}, extra_rail_after_5ms_roi_total={int(extra_rail_counts[s][roi_mask_in_active].sum())}')

print('\n=== NEAR-SITE (<200 um) & SITE CHANNELS RAIL EXTENSION PAST +5.0 ms ===')
non_roi_idx = np.where(~roi_mask_in_active)[0]
for ci in non_roi_idx:
    ch_id = int(active_all[ci])
    xy = ch_xy[ch_id]
    dists = {sk: float(np.hypot(xy[0] - ch_xy[sc][0], xy[1] - ch_xy[sc][1])) for sk, sc in site_channels.items()}
    print(f'  ch={ch_id:4d} dists_um={ {k: round(v,1) for k,v in dists.items()} } extra_rail_past_5ms={ {s: int(extra_rail_counts[s][ci]) for s in ("H","W","F","Z")} }')

q2_summary = {
    'n_active_all': int(len(active_all)),
    'n_roi': int(len(roi)),
    'blanking_raw_ms': [-0.25, 2.50],
    'blanking_post_filter_ms': [-1.0, 5.0],
    'b_ms': 5.0,
    'filter': 'Causal 4th-order Butterworth bandpass 300-3000 Hz (fs=20 kHz)',
    'threshold_rule': '5.0 * median(|filt_warmup|) / 0.6745 per channel (warmup 4.0-64.0s only)',
    'refractory_ms': 1.0,
    'sigma_uv_median': float(np.median(sigma_ch) * lsb_uv),
    'sigma_uv_iqr': [float(np.percentile(sigma_ch, 25) * lsb_uv), float(np.percentile(sigma_ch, 75) * lsb_uv)],
    'sham_self_check': {
        'n_sham_trials': int(len(sham_tids)),
        'offline_mean': float(sham_off.mean()),
        'offline_sd': float(sham_off.std()),
        'online_masked_mean': float(sham_on_clean.mean()),
        'online_masked_sd': float(sham_on_clean.std()),
        'online_unmasked_mean': float(sham_on_raw.mean()),
        'pearson_r_masked': pearson_clean,
        'spearman_rho_masked': spearman_clean,
        'pearson_r_unmasked': pearson_raw,
        'spearman_rho_unmasked': spearman_raw,
        'per_channel_pearson_r': pearson_by_ch,
        'per_channel_spearman_rho': spearman_by_ch,
    },
    'ringing_check_5ms_windows': {
        s: {
            'pre_minus6_to_minus1_ms_mean': float(pre_5ms_trial[[m['tid'] for m in trial_meta if m['site'] == s]].mean()),
            'post_plus5_to_plus10_ms_mean': float(early_5_10_trial[[m['tid'] for m in trial_meta if m['site'] == s]].mean()),
            'roi_extra_rail_past_5ms': int(extra_rail_counts[s][roi_mask_in_active].sum()),
        }
        for s in ('H', 'W', 'F', 'Z')
    },
    'elapsed_s': round(time.time() - t_start, 1),
}
(out_dir / 'q2_summary.json').write_text(json.dumps(q2_summary, indent=2), encoding='utf-8')
print(f'\nQ2 complete in {q2_summary["elapsed_s"]}s.')
