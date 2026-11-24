import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.recording import H5Recording
from oddball import analysis

SEED = 2026112301
ALPHA_BONF = 0.05 / 12.0  # 0.0041666667

rec_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')

plan = json.loads((rec_dir / 'plan.json').read_text(encoding='utf-8'))
decision = json.loads((rec_dir / 'decision.json').read_text(encoding='utf-8'))
route = json.loads((rec_dir / 'route.json').read_text(encoding='utf-8'))
binding = json.loads((rec_dir / 'binding.json').read_text(encoding='utf-8'))
q4_spec = json.loads((out_dir / 'q4_frozen_spec.json').read_text(encoding='utf-8'))
trial_meta = json.loads((out_dir / 'q2_trials_meta.json').read_text(encoding='utf-8'))

npz = np.load(out_dir / 'q2_detected_spikes.npz')
active_all = npz['active_all']
roi = npz['roi']
roi_mask_in_active = npz['roi_mask_in_active']
off_tid, off_ch, off_rf = npz['off_tid'], npz['off_ch'], npz['off_rf']
on_tid, on_ch, on_rf = npz['on_tid'], npz['on_ch'], npz['on_rf']

n_trials = len(trial_meta)
assert n_trials == 1200

# ==============================================================================
# QUESTION 3: Evoked responses in [5, 30), [30, 50), [50, 100), [100, 300) ms
# Both offline-detected spikes and online-detected spikes (avoiding [-2, +5) ms)
# ==============================================================================
Q3_WINDOWS = [
    ('[5,30)', 100, 600),
    ('[30,50)', 600, 1000),
    ('[50,100)', 1000, 2000),
    ('[100,300)', 2000, 6000),
]

def run_q3_for_source(tid_arr, ch_arr, rf_arr, source_name):
    in_roi = roi_mask_in_active[ch_arr]
    t_roi = tid_arr[in_roi]
    rf_roi = rf_arr[in_roi]

    # Pre-stimulus [-300, -20) ms = [-6000, -400) frames for "quiet" trial definition
    pre_mask = (rf_roi >= -6000) & (rf_roi < -400)
    pre_counts = np.bincount(t_roi[pre_mask], minlength=n_trials)
    pre_median = float(np.median(pre_counts))
    quiet_trial_mask = pre_counts <= pre_median

    sites_idx = {s: np.array([m['tid'] for m in trial_meta if m['site'] == s], dtype=int) for s in ('H', 'W', 'F', 'Z')}
    rounds_idx = {
        r: {s: np.array([m['tid'] for m in trial_meta if m['site'] == s and m['round'] == r], dtype=int) for s in ('H', 'W', 'F', 'Z')}
        for r in (1, 2, 3)
    }

    results_rows = []
    passed_bonf = []
    passed_judgment_a = []
    prop_nonzero_by_win = {}
    win_trial_counts = {}

    for w_label, f_lo, f_hi in Q3_WINDOWS:
        w_mask = (rf_roi >= f_lo) & (rf_roi < f_hi)
        counts = np.bincount(t_roi[w_mask], minlength=n_trials)
        win_trial_counts[w_label] = counts

        prop_nonzero_by_win[w_label] = {
            s: float(np.mean(counts[sites_idx[s]] >= 1)) for s in ('H', 'W', 'F', 'Z')
        }

        sham_all = counts[sites_idx['Z']]
        sham_quiet = counts[sites_idx['Z'][quiet_trial_mask[sites_idx['Z']]]]

        for site in ('H', 'W', 'F'):
            real_all = counts[sites_idx[site]]
            comp_all = analysis.compare(real_all, sham_all, SEED)

            # Per-round AUC (site in round r vs sham in round r; also vs all 120 sham)
            round_aucs = {}
            round_aucs_vs_all_sham = {}
            for r in (1, 2, 3):
                r_real = counts[rounds_idx[r][site]]
                r_sham = counts[rounds_idx[r]['Z']]
                round_aucs[f'r{r}'] = analysis.auc(r_real, r_sham)
                round_aucs_vs_all_sham[f'r{r}'] = analysis.auc(r_real, sham_all)

            # Quiet subset comparison
            real_quiet = counts[sites_idx[site][quiet_trial_mask[sites_idx[site]]]]
            comp_quiet = analysis.compare(real_quiet, sham_quiet, SEED)
            quiet_round_aucs = {}
            for r in (1, 2, 3):
                rq_idx = rounds_idx[r][site][quiet_trial_mask[rounds_idx[r][site]]]
                sq_idx = rounds_idx[r]['Z'][quiet_trial_mask[rounds_idx[r]['Z']]]
                quiet_round_aucs[f'r{r}'] = analysis.auc(counts[rq_idx], counts[sq_idx])

            hit_bonf = bool(comp_all['p'] <= ALPHA_BONF)
            hit_judg = bool(
                comp_all['auc'] >= 0.60
                and comp_all['p'] <= ALPHA_BONF
                and all(v is not None and v > 0.5 for v in round_aucs.values())
            )
            if hit_bonf:
                passed_bonf.append((site, w_label))
            if hit_judg:
                passed_judgment_a.append((site, w_label))

            row = {
                'source': source_name,
                'site': site,
                'window_ms': w_label,
                'real_mean': comp_all['mean_positive'],
                'sham_mean': comp_all['mean_negative'],
                'real_sd': float(np.std(real_all)),
                'sham_sd': float(np.std(sham_all)),
                'auc': comp_all['auc'],
                'p': comp_all['p'],
                'round_aucs': round_aucs,
                'round_aucs_vs_all_sham': round_aucs_vs_all_sham,
                'prop_ge1_real': float(np.mean(real_all >= 1)),
                'prop_ge1_sham': float(np.mean(sham_all >= 1)),
                'quiet_subset': {
                    'pre_median_cutoff': pre_median,
                    'n_real_quiet': comp_quiet['n_positive'],
                    'n_sham_quiet': comp_quiet['n_negative'],
                    'real_mean': comp_quiet['mean_positive'],
                    'sham_mean': comp_quiet['mean_negative'],
                    'auc': comp_quiet['auc'],
                    'p': comp_quiet['p'],
                    'round_aucs': quiet_round_aucs,
                },
                'pass_bonferroni_00417': hit_bonf,
                'pass_judgment_criteria': hit_judg,
            }
            results_rows.append(row)

    return {
        'source': source_name,
        'pre_300_20_median': pre_median,
        'n_quiet_total': int(np.sum(quiet_trial_mask)),
        'prop_nonzero_by_window': prop_nonzero_by_win,
        'comparisons': results_rows,
        'passed_bonferroni': passed_bonf,
        'passed_judgment_criteria': passed_judgment_a,
    }, win_trial_counts

q3_offline, off_win_counts = run_q3_for_source(off_tid, off_ch, off_rf, 'offline_raw_detected')
# For online spikes, mask [-2.0, +5.0) ms ([-40, +100) frames) identically on real & sham
clean_on_mask = (on_rf < -40) | (on_rf >= 100)
q3_online, on_win_counts = run_q3_for_source(on_tid[clean_on_mask], on_ch[clean_on_mask], on_rf[clean_on_mask], 'online_h5_clean')

print('=== QUESTION 3: OFFLINE DETECTED SPIKES ===')
print(f'Pre [-300,-20)ms median = {q3_offline["pre_300_20_median"]} (quiet trials = {q3_offline["n_quiet_total"]}/1200)')
for r in q3_offline['comparisons']:
    ra = r['round_aucs']
    q = r['quiet_subset']
    print(f"{r['site']} {r['window_ms']:10s} | real={r['real_mean']:6.2f} sham={r['sham_mean']:6.2f} | AUC={r['auc']:.3f} p={r['p']:.4f} | R1/R2/R3={ra['r1']:.3f}/{ra['r2']:.3f}/{ra['r3']:.3f} | p>=1:{r['prop_ge1_real']:.3f} vs {r['prop_ge1_sham']:.3f} | quiet AUC={q['auc']:.3f} (p={q['p']:.4f})")

print('\n=== QUESTION 3: ONLINE DETECTED SPIKES (CLEAN >=5ms) ===')
print(f'Pre [-300,-20)ms median = {q3_online["pre_300_20_median"]} (quiet trials = {q3_online["n_quiet_total"]}/1200)')
for r in q3_online['comparisons']:
    ra = r['round_aucs']
    q = r['quiet_subset']
    print(f"{r['site']} {r['window_ms']:10s} | real={r['real_mean']:6.2f} sham={r['sham_mean']:6.2f} | AUC={r['auc']:.3f} p={r['p']:.4f} | R1/R2/R3={ra['r1']:.3f}/{ra['r2']:.3f}/{ra['r3']:.3f} | p>=1:{r['prop_ge1_real']:.3f} vs {r['prop_ge1_sham']:.3f} | quiet AUC={q['auc']:.3f} (p={q['p']:.4f})")

# ==============================================================================
# QUESTION 4: Does stimulation alter bursts? (Using frozen spec from warmup!)
# Thresholds: primary=20, half=10, double=40 spikes per 20 ms bin on 233 ROI ch
# Windows: [5, 300) ms, [300, 1000) ms, and latency to next burst onset in [5, 1800) ms
# ==============================================================================
cfg = SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])
main_blocks = [b for b in plan['blocks'] if b['stage'] == 'main']

# Build per-trial 20 ms bin counts from [-22, -2) ms (pre-bin) and [5, 1805) ms (90 post-bins of 20 ms = 400 frames each!)
# Mask [-40, +100) frames ([-2.0, +5.0) ms) around EVERY marker in the file so even adjacent markers cannot pollute!
post_bin_edges_frames = 100 + np.arange(0, 91) * 400  # [100, 500, 900, ..., 36100] -> [5, 25, ..., 1805] ms (90 bins)
post_bin_starts_ms = 5.0 + np.arange(0, 90) * 20.0    # 5, 25, ..., 1785 ms

trial_20ms_pre = np.zeros(n_trials, dtype=np.int32)
trial_20ms_post = np.zeros((n_trials, 90), dtype=np.int32)

tid_cursor = 0
for b in main_blocks:
    bname = b['name']
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    with H5Recording(rec_dir / bname / f'{bname}.raw.h5', cfg) as rec:
        keep = np.isin(rec.spike_channels, roi)
        sp_f = np.sort(np.rint(rec.spike_times[keep] * rec.fs).astype(np.int64))
        markers = np.array([int(t['frame']) for t in qc['trials']], dtype=np.int64)
        # Mask [-40, +100) around every marker in the block identically!
        bad_sp = np.zeros(len(sp_f), dtype=bool)
        for mf in markers:
            lo_m = np.searchsorted(sp_f, mf - 40)
            hi_m = np.searchsorted(sp_f, mf + 100)
            bad_sp[lo_m:hi_m] = True
        sp_clean = sp_f[~bad_sp]

        for i_t, t in enumerate(qc['trials']):
            f0 = int(t['frame'])
            # Pre-bin [-22, -2) ms = [-440, -40) frames
            trial_20ms_pre[tid_cursor] = int(np.searchsorted(sp_clean, f0 - 40) - np.searchsorted(sp_clean, f0 - 440))
            # 90 post-bins [5, 1805) ms
            anchors = f0 + post_bin_edges_frames
            c90 = np.diff(np.searchsorted(sp_clean, anchors))
            # Check gap to next trial (at chunk boundary every 10th trial gap is >= 1.8s = 36000 frames -> 1800 ms is ALWAYS before next marker!)
            trial_20ms_post[tid_cursor] = c90
            tid_cursor += 1

q4_results = {}
for label_thr, thr in [('primary_20', 20), ('half_10', 10), ('double_40', 40)]:
    # Concatenate pre-bin (col 0) and 90 post-bins (cols 1..90)
    full_bins = np.concatenate([trial_20ms_pre[:, None], trial_20ms_post], axis=1)
    above = full_bins >= thr
    # Onset at post-bin k (k in 0..89, corresponding to full_bins col k+1): above[:, k+1] & (~above[:, k])
    onset_mat = above[:, 1:] & (~above[:, :-1])  # shape (1200, 90)

    # Window 1: [5, 300) ms -> bin starts in [5, 300) ms -> k = 0..14 (starts 5..285 ms; note bin 14 is [285,305), let's also check 5..280 k=0..13; k=0..14 has start < 300 ms!)
    w1_mask = post_bin_starts_ms < 300.0   # 15 bins: starts 5..285 ms
    w2_mask = (post_bin_starts_ms >= 300.0) & (post_bin_starts_ms < 1000.0)  # 35 bins: starts 305..985 ms

    has_onset_w1 = onset_mat[:, w1_mask].any(axis=1).astype(float)
    has_onset_w2 = onset_mat[:, w2_mask].any(axis=1).astype(float)

    # First burst onset latency in [5, 1805) ms:
    any_onset_1800 = onset_mat.any(axis=1)
    first_bin_idx = np.argmax(onset_mat, axis=1)
    latency_ms = np.where(any_onset_1800, post_bin_starts_ms[first_bin_idx], 2000.0)  # 2000.0 = right-censored at >1800ms

    site_stats = {}
    z_idx = np.array([m['tid'] for m in trial_meta if m['site'] == 'Z'], dtype=int)
    for s in ('H', 'W', 'F', 'Z'):
        s_idx = np.array([m['tid'] for m in trial_meta if m['site'] == s], dtype=int)
        uncens = latency_ms[s_idx][any_onset_1800[s_idx]]
        entry = {
            'n_trials': int(len(s_idx)),
            'prob_onset_5_300ms': float(has_onset_w1[s_idx].mean()),
            'count_onset_5_300ms': int(has_onset_w1[s_idx].sum()),
            'prob_onset_300_1000ms': float(has_onset_w2[s_idx].mean()),
            'count_onset_300_1000ms': int(has_onset_w2[s_idx].sum()),
            'prob_onset_within_1800ms': float(any_onset_1800[s_idx].mean()),
            'uncensored_latency_median_ms': float(np.median(uncens)) if len(uncens) else None,
            'uncensored_latency_iqr_ms': [float(np.percentile(uncens, 25)), float(np.percentile(uncens, 75))] if len(uncens) else None,
            'uncensored_latency_mean_ms': float(np.mean(uncens)) if len(uncens) else None,
            'all_trials_censored_median_ms': float(np.median(latency_ms[s_idx])),
        }
        if s != 'Z':
            entry['vs_sham_5_300ms'] = analysis.compare(has_onset_w1[s_idx], has_onset_w1[z_idx], SEED)
            entry['vs_sham_300_1000ms'] = analysis.compare(has_onset_w2[s_idx], has_onset_w2[z_idx], SEED)
            # Test both shortened latency (-latency_ms > -sham_latency_ms) and lengthened latency
            entry['vs_sham_shorter_latency'] = analysis.compare(-latency_ms[s_idx], -latency_ms[z_idx], SEED)
            entry['vs_sham_longer_latency'] = analysis.compare(latency_ms[s_idx], latency_ms[z_idx], SEED)
        site_stats[s] = entry
    q4_results[label_thr] = {'threshold_spikes_per_20ms': thr, 'sites': site_stats}

print('\n=== QUESTION 4: BURST ONSET PROBABILITY & LATENCY ===')
for label_thr, res in q4_results.items():
    print(f"\nThreshold {label_thr} (>= {res['threshold_spikes_per_20ms']} spikes / 20 ms):")
    zs = res['sites']['Z']
    print(f"  Sham Z: P([5,300))={zs['prob_onset_5_300ms']:.3f} ({zs['count_onset_5_300ms']}/120), P([300,1000))={zs['prob_onset_300_1000ms']:.3f} ({zs['count_onset_300_1000ms']}/120), P(<1800)={zs['prob_onset_within_1800ms']:.3f}, uncens_med={zs['uncensored_latency_median_ms']} ms")
    for s in ('H', 'W', 'F'):
        st = res['sites'][s]
        c1 = st['vs_sham_5_300ms']
        c2 = st['vs_sham_300_1000ms']
        cl = st['vs_sham_shorter_latency']
        print(f"  Site {s}: P([5,300))={st['prob_onset_5_300ms']:.3f} (AUC={c1['auc']:.3f}, p={c1['p']:.4f}) | P([300,1000))={st['prob_onset_300_1000ms']:.3f} (AUC={c2['auc']:.3f}, p={c2['p']:.4f}) | P(<1800)={st['prob_onset_within_1800ms']:.3f}, uncens_med={st['uncensored_latency_median_ms']} ms (shorter_lat AUC={cl['auc']:.3f}, p={cl['p']:.4f})")

# ==============================================================================
# QUESTION 5: Spatial distance profile across ALL 243 active channels (>=0.05 Hz)
# Distance bands from the STIMULATED site: <200, 200-500, 500-1000, >1000 um
# First check exact rail/saturation recovery on <200 um channels in raw data!
# ==============================================================================
site_channels = {k: int(v) for k, v in route['channels'].items()}
with H5Recording(rec_dir / 'warmup/warmup.raw.h5', cfg) as wrec:
    ch_xy = wrec.channel_xy_um

# Inspect <200 um channels in detail on one block per site (main_r1_H, main_r1_W, main_r1_F) to measure exact saturation end time!
near_200_saturation_report = {}
for s_site, bname in [('H', 'main_r1_H'), ('W', 'main_r1_W'), ('F', 'main_r1_F')]:
    sc = site_channels[s_site]
    sx, sy = ch_xy[sc]
    near_list = [int(c) for c in active_all if np.hypot(ch_xy[int(c)][0] - sx, ch_xy[int(c)][1] - sy) < 200.0]
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    s_trials = [t for t in qc['trials'] if t['site'] == s_site]
    ch_rail_end_ms = {c: [] for c in near_list}
    with H5Recording(rec_dir / bname / f'{bname}.raw.h5', cfg) as rec:
        for t in s_trials:
            f0 = int(t['frame'])
            raw_n = rec.sample(near_list, f0 / rec.fs, 0.300).astype(np.float32)  # [0, +300) ms = 6000 frames
            rail = (raw_n <= 2) | (raw_n >= 1021)
            for idx_n, c in enumerate(near_list):
                r_idx = np.where(rail[idx_n])[0]
                ch_rail_end_ms[c].append(float(r_idx[-1] / 20.0) if len(r_idx) else 0.0)
    near_200_saturation_report[s_site] = {
        int(c): {
            'dist_um': round(float(np.hypot(ch_xy[c][0] - sx, ch_xy[c][1] - sy)), 1),
            'frac_rail_gt_5ms': float(np.mean(np.array(ch_rail_end_ms[c]) > 5.0)),
            'median_rail_end_ms': round(float(np.median(ch_rail_end_ms[c])), 2),
            'p95_rail_end_ms': round(float(np.percentile(ch_rail_end_ms[c], 95)), 2),
            'max_rail_end_ms': round(float(np.max(ch_rail_end_ms[c])), 2),
        }
        for c in near_list
    }

print('\n=== QUESTION 5: <200 um CHANNEL SATURATION DURATION CHECK ===')
for s_site, dmap in near_200_saturation_report.items():
    print(f'When stimulating {s_site}:')
    for c, info in dmap.items():
        print(f'  ch={c:4d} (dist={info["dist_um"]:5.1f} um): frac_rail>5ms={info["frac_rail_gt_5ms"]:.2f}, med_end={info["median_rail_end_ms"]:6.2f} ms, p95_end={info["p95_rail_end_ms"]:6.2f} ms, max_end={info["max_rail_end_ms"]:6.2f} ms')

# Compute per-trial, per-channel spike counts in Q3 windows for all 243 active channels (offline & online clean)
DIST_BINS = [
    ('<200', 0.0, 200.0),
    ('200-500', 200.0, 500.0),
    ('500-1000', 500.0, 1000.0),
    ('>1000', 1000.0, 1e9),
]

def compute_q5_spatial(tid_arr, ch_arr, rf_arr):
    # Matrix per window: shape (n_trials, n_active)
    sham_tids = np.array([m['tid'] for m in trial_meta if m['site'] == 'Z'], dtype=int)
    out_q5 = {}
    for w_label, f_lo, f_hi in Q3_WINDOWS:
        m = (rf_arr >= f_lo) & (rf_arr < f_hi)
        flat_idx = tid_arr[m].astype(np.int64) * len(active_all) + ch_arr[m].astype(np.int64)
        mat = np.bincount(flat_idx, minlength=n_trials * len(active_all)).reshape(n_trials, len(active_all))
        sham_ch_mean = mat[sham_tids].mean(axis=0)  # per-channel sham mean (shape n_active,)

        out_q5[w_label] = {}
        # Per-site AND pooled across sites
        pooled_diffs = {b[0]: [] for b in DIST_BINS}
        pooled_clean_lt200 = []
        for s_site in ('H', 'W', 'F'):
            s_tids = np.array([m_tr['tid'] for m_tr in trial_meta if m_tr['site'] == s_site], dtype=int)
            real_ch_mean = mat[s_tids].mean(axis=0)
            diff_ch = real_ch_mean - sham_ch_mean  # real mean minus sham mean per channel!

            sc = site_channels[s_site]
            sx, sy = ch_xy[sc]
            dists = np.array([np.hypot(ch_xy[int(c)][0] - sx, ch_xy[int(c)][1] - sy) for c in active_all])

            out_q5[w_label][s_site] = {}
            for b_label, d_lo, d_hi in DIST_BINS:
                bm = (dists >= d_lo) & (dists < d_hi)
                vals = diff_ch[bm]
                pooled_diffs[b_label].extend(vals.tolist())
                entry_b = {
                    'n_channels': int(bm.sum()),
                    'mean_diff_per_ch': float(np.mean(vals)) if len(vals) else None,
                    'sd_diff_per_ch': float(np.std(vals)) if len(vals) else None,
                    'real_mean_per_ch': float(np.mean(real_ch_mean[bm])) if len(vals) else None,
                    'sham_mean_per_ch': float(np.mean(sham_ch_mean[bm])) if len(vals) else None,
                }
                if b_label == '<200':
                    # Also report unsaturated subset (150-200 um, i.e., dist > 30 um where 0% rail past 5ms)
                    bm_unsat = bm & (dists > 30.0)
                    vals_u = diff_ch[bm_unsat]
                    pooled_clean_lt200.extend(vals_u.tolist())
                    entry_b['unsaturated_gt30um_n'] = int(bm_unsat.sum())
                    entry_b['unsaturated_gt30um_mean_diff'] = float(np.mean(vals_u)) if len(vals_u) else None
                out_q5[w_label][s_site][b_label] = entry_b

        out_q5[w_label]['pooled_across_sites'] = {
            b_label: {
                'n_site_channel_pairs': len(pooled_diffs[b_label]),
                'mean_diff_per_ch': float(np.mean(pooled_diffs[b_label])) if pooled_diffs[b_label] else None,
                'sd_diff_per_ch': float(np.std(pooled_diffs[b_label])) if pooled_diffs[b_label] else None,
            }
            for b_label, _, _ in DIST_BINS
        }
        out_q5[w_label]['pooled_across_sites']['<200_unsaturated_gt30um'] = {
            'n_site_channel_pairs': len(pooled_clean_lt200),
            'mean_diff_per_ch': float(np.mean(pooled_clean_lt200)) if pooled_clean_lt200 else None,
        }
    return out_q5

q5_offline = compute_q5_spatial(off_tid, off_ch, off_rf)
q5_online = compute_q5_spatial(on_tid[clean_on_mask], on_ch[clean_on_mask], on_rf[clean_on_mask])

print('\n=== QUESTION 5: PER-CHANNEL (REAL MEAN - SHAM MEAN) BY DISTANCE BAND (OFFLINE DETECTED) ===')
for w_label, _, _ in Q3_WINDOWS:
    print(f'\nWindow {w_label}:')
    for s_site in ('H', 'W', 'F'):
        parts = []
        for b_label, _, _ in DIST_BINS:
            info = q5_offline[w_label][s_site][b_label]
            val_s = 'n/a' if info['mean_diff_per_ch'] is None else f"{info['mean_diff_per_ch']:+.4f}(n={info['n_channels']})"
            parts.append(f'{b_label}:{val_s}')
        u_info = q5_offline[w_label][s_site]['<200']
        u_str = 'n/a' if u_info['unsaturated_gt30um_mean_diff'] is None else f"{u_info['unsaturated_gt30um_mean_diff']:+.4f}(n={u_info['unsaturated_gt30um_n']})"
        print(f'  Site {s_site}: ' + '  '.join(parts) + f'  [<200 unsat 160-180um: {u_str}]')
    p_parts = [f"{b_label}:{q5_offline[w_label]['pooled_across_sites'][b_label]['mean_diff_per_ch']:+.4f}(n={q5_offline[w_label]['pooled_across_sites'][b_label]['n_site_channel_pairs']})" for b_label, _, _ in DIST_BINS]
    print('  Pooled: ' + '  '.join(p_parts))

# ==============================================================================
# QUESTION 6: Habituation — ONLY if at least one (site, window) in Q3 passes p <= 0.05/12!
# ==============================================================================
any_q3_passed = len(q3_offline['passed_bonferroni']) > 0 or len(q3_online['passed_bonferroni']) > 0
if not any_q3_passed:
    q6_result = {
        'tested': False,
        'status': '未检验，因为没有诱发反应',
        'passed_q3_offline': q3_offline['passed_bonferroni'],
        'passed_q3_online': q3_online['passed_bonferroni'],
    }
else:
    q6_items = []
    for src_label, q3_obj, win_cnts in [('offline', q3_offline, off_win_counts), ('online', q3_online, on_win_counts)]:
        for site, w_label in q3_obj['passed_bonferroni']:
            cnts = win_cnts[w_label]
            low_mid = [cnts[m['tid']] for m in trial_meta if m['site'] == site and m['valid'] and m['count'] is not None and 0 <= m['count'] <= 4]
            high = [cnts[m['tid']] for m in trial_meta if m['site'] == site and m['valid'] and m['count'] is not None and 5 <= m['count'] <= 8]
            comp_hab = analysis.compare(low_mid, high, SEED)
            unif = [m for m in trial_meta if m['site'] == site and m['valid'] and m['kind'] == 'uniform']
            rho_u = analysis.spearman([m['count'] for m in unif], [cnts[m['tid']] for m in unif])
            q6_items.append({
                'source': src_label,
                'site': site,
                'window_ms': w_label,
                'low_mid_0_4_vs_high_5_8': comp_hab,
                'uniform_spearman_rho': rho_u,
            })
    q6_result = {'tested': True, 'items': q6_items}

print('\n=== QUESTION 6 ===')
print(json.dumps(q6_result, ensure_ascii=False, indent=2))

all_q3_q6 = {
    'alpha_bonferroni': ALPHA_BONF,
    'seed': SEED,
    'q3_offline': q3_offline,
    'q3_online': q3_online,
    'q4_frozen_spec': q4_spec,
    'q4_results': q4_results,
    'q5_near_200_saturation': near_200_saturation_report,
    'q5_offline': q5_offline,
    'q5_online': q5_online,
    'q6': q6_result,
}
(out_dir / 'q3_to_q6_results.json').write_text(json.dumps(all_q3_q6, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nSaved q3_to_q6_results.json successfully.')
