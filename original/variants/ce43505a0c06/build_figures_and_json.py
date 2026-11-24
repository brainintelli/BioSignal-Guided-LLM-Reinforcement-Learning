import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from mea_stim_v2.recording import H5Recording

rec_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')

q0 = json.loads((out_dir / 'q0_integrity.json').read_text(encoding='utf-8'))
q1_p1 = json.loads((out_dir / 'q1_part1.json').read_text(encoding='utf-8'))
q1_p2 = json.loads((out_dir / 'q1_part2.json').read_text(encoding='utf-8'))
q2_sum = json.loads((out_dir / 'q2_summary.json').read_text(encoding='utf-8'))
q3_q6 = json.loads((out_dir / 'q3_to_q6_results.json').read_text(encoding='utf-8'))
trial_meta = json.loads((out_dir / 'q2_trials_meta.json').read_text(encoding='utf-8'))
route = json.loads((rec_dir / 'route.json').read_text(encoding='utf-8'))
binding = json.loads((rec_dir / 'binding.json').read_text(encoding='utf-8'))

npz = np.load(out_dir / 'q2_detected_spikes.npz')
active_all = npz['active_all']
roi = npz['roi']
roi_mask_in_active = npz['roi_mask_in_active']
off_tid, off_ch, off_rf = npz['off_tid'], npz['off_ch'], npz['off_rf']
on_tid, on_ch, on_rf = npz['on_tid'], npz['on_ch'], npz['on_rf']

cfg = SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])
site_channels = {k: int(v) for k, v in route['channels'].items()}
stim_ch_set = set(site_channels.values())

with H5Recording(rec_dir / 'warmup/warmup.raw.h5', cfg) as wrec:
    ch_xy = wrec.channel_xy_um
    # Also reload warmup 20ms counts for Fig 4A
    keep_w = np.isin(wrec.spike_channels, roi)
    wt = wrec.spike_times[keep_w]
    wt_clean = wt[wt >= 4.0]
    warmup_20ms_counts, _ = np.histogram(wt_clean, bins=np.arange(4.0, wrec.duration + 1e-9, 0.020))

# Refine Q5 non-stim recording channels (<200 um excluding the 3 DAC-driven 0-um channels, and separating saturated <=25um vs unsaturated 160-180um)
Q3_WINDOWS = [('[5,30)', 100, 600), ('[30,50)', 600, 1000), ('[50,100)', 1000, 2000), ('[100,300)', 2000, 6000)]
DIST_BINS = [('<200_nonstim_all', 0.1, 200.0), ('<200_unsat_30_200', 30.0, 200.0), ('200-500', 200.0, 500.0), ('500-1000', 500.0, 1000.0), ('>1000', 1000.0, 1e9)]

def compute_clean_q5(tid_arr, ch_arr, rf_arr):
    n_trials = 1200
    sham_tids = np.array([m['tid'] for m in trial_meta if m['site'] == 'Z'], dtype=int)
    non_stim_mask = np.array([int(c) not in stim_ch_set for c in active_all], dtype=bool)
    res = {}
    for w_label, f_lo, f_hi in Q3_WINDOWS:
        m = (rf_arr >= f_lo) & (rf_arr < f_hi)
        flat_idx = tid_arr[m].astype(np.int64) * len(active_all) + ch_arr[m].astype(np.int64)
        mat = np.bincount(flat_idx, minlength=n_trials * len(active_all)).reshape(n_trials, len(active_all))
        sham_ch_mean = mat[sham_tids].mean(axis=0)
        res[w_label] = {'by_site': {}, 'pooled': {}}
        pooled = {b[0]: [] for b in DIST_BINS}
        for s_site in ('H', 'W', 'F'):
            s_tids = np.array([tr['tid'] for tr in trial_meta if tr['site'] == s_site], dtype=int)
            diff_ch = mat[s_tids].mean(axis=0) - sham_ch_mean
            sx, sy = ch_xy[site_channels[s_site]]
            dists = np.array([np.hypot(ch_xy[int(c)][0] - sx, ch_xy[int(c)][1] - sy) for c in active_all])
            res[w_label]['by_site'][s_site] = {}
            for b_lbl, d0, d1 in DIST_BINS:
                bm = non_stim_mask & (dists >= d0) & (dists < d1)
                vals = diff_ch[bm]
                pooled[b_lbl].extend(vals.tolist())
                res[w_label]['by_site'][s_site][b_lbl] = {
                    'n_channels': int(bm.sum()),
                    'mean_diff': float(np.mean(vals)) if len(vals) else None,
                    'sd_diff': float(np.std(vals)) if len(vals) else None,
                }
        for b_lbl, _, _ in DIST_BINS:
            vals = pooled[b_lbl]
            res[w_label]['pooled'][b_lbl] = {
                'n_pairs': len(vals),
                'mean_diff': float(np.mean(vals)) if vals else None,
                'sd_diff': float(np.std(vals)) if vals else None,
            }
    return res

q5_clean_offline = compute_clean_q5(off_tid, off_ch, off_rf)
clean_on = (on_rf < -40) | (on_rf >= 100)
q5_clean_online = compute_clean_q5(on_tid[clean_on], on_ch[clean_on], on_rf[clean_on])

print('=== CLEAN Q5 (EXCLUDING 3 DAC-DRIVEN STIM ELECTRODES 804,96,737) OFFLINE ===')
for w_lbl, _, _ in Q3_WINDOWS:
    p = q5_clean_offline[w_lbl]['pooled']
    print(w_lbl, {k: (None if v['mean_diff'] is None else round(v['mean_diff'], 4), v['n_pairs']) for k, v in p.items()})

print('\n=== CLEAN Q5 (EXCLUDING 3 DAC-DRIVEN STIM ELECTRODES 804,96,737) ONLINE ===')
for w_lbl, _, _ in Q3_WINDOWS:
    p = q5_clean_online[w_lbl]['pooled']
    print(w_lbl, {k: (None if v['mean_diff'] is None else round(v['mean_diff'], 4), v['n_pairs']) for k, v in p.items()})

# ==============================================================================
# FIGURE 1: Q1 Online Pre-Marker Spike Timing & Filter Pre-Ringing Mechanism
# ==============================================================================
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

# Panel A: 1 ms PSTH (-50 to +100 ms)
ax = axes[0, 0]
t_1ms = np.array(q1_p1['psth_1ms_bins_ms']) + 0.5
colors = {'H': '#d62728', 'W': '#1f77b4', 'F': '#2ca02c', 'Z': '#555555'}
for s in ('H', 'W', 'F', 'Z'):
    y = np.maximum(np.array(q1_p1['psth_1ms_per_trial'][s]), 1e-3)
    ax.plot(t_1ms, y, label=f'Site {s}' if s != 'Z' else 'Sham Z', color=colors[s], lw=1.4 if s != 'Z' else 1.8)
ax.axvline(0, color='k', ls='--', lw=0.9)
ax.axvspan(-1.0, 2.5, color='orange', alpha=0.22, label='Online polluted [-1.0, +2.5) ms')
ax.set_yscale('log')
ax.set_xlim(-50, 100)
ax.set_xlabel('Time relative to marker (ms)')
ax.set_ylabel('Online ROI spikes / trial / 1 ms (log)')
ax.set_title('A. Online Spike 1-ms PSTH (-50 to +100 ms, 233 ROI ch)')
ax.legend(frameon=False, fontsize=8.5)

# Panel B: 1-frame (50 us) histogram (-1.5 to +2.5 ms) showing discrete Gibbs sidelobes!
ax = axes[0, 1]
f_offs = np.arange(-30, 50)
i0, i1 = -30 + 1000, 50 + 1000
for s in ('H', 'W', 'F', 'Z'):
    cnt_f = np.array(q1_p1['hist_1frame_counts'][s][i0:i1]) / q1_p1['trial_counts'][s]
    ax.plot(f_offs * 0.05, np.maximum(cnt_f, 1e-4), marker='.', ms=3.5, lw=1.1, color=colors[s], label=s)
ax.axvline(0.0, color='k', ls='--', lw=0.9, label='Marker (frame 0)')
ax.axvline(0.15, color='purple', ls=':', lw=1.1, label='Raw pulse start (+0.15 ms)')
ax.set_yscale('log')
ax.set_xlabel('Time relative to marker (ms, 50 us/frame)')
ax.set_ylabel('Online spikes / trial / frame (log)')
ax.set_title('B. Single-Frame (50 us) Pre-Ringing Sidelobes (~3.1 kHz)')
ax.legend(frameon=False, fontsize=8)

# Panel C: Raw waveform on stimulated channel vs 6 ROI channels in main_r1_uniform (Site H vs Sham Z)
ax = axes[1, 0]
tr_h = q1_p2['saved_traces_r1_uniform']['H']
tr_z = q1_p2['saved_traces_r1_uniform']['Z']
t_tr = (np.arange(1600) - 400) * 0.05  # -20 to +60 ms
lsb_uv = q1_p2['lsb_uv']
for k_ch in range(3, 9):  # 6 ROI channels
    raw_h = np.array(tr_h['raw_lsb'][k_ch])
    raw_h = (raw_h - np.median(raw_h[:300])) * lsb_uv
    ax.plot(t_tr, raw_h + (k_ch - 3) * 180, color='#d62728', lw=0.9, alpha=0.85, label='Real H (ROI ch)' if k_ch == 3 else None)
    raw_z = np.array(tr_z['raw_lsb'][k_ch])
    raw_z = (raw_z - np.median(raw_z[:300])) * lsb_uv
    ax.plot(t_tr, raw_z + (k_ch - 3) * 180, color='#555555', lw=0.7, alpha=0.7, label='Sham Z (ROI ch)' if k_ch == 3 else None)
ax.axvline(0.0, color='k', ls='--', lw=0.9)
ax.set_xlim(-5, 15)
ax.set_xlabel('Time relative to marker (ms)')
ax.set_ylabel('Raw voltage + offset (uV)')
ax.set_title('C. Raw Waveforms of 6 Readout Channels (-5 to +15 ms)')
ax.legend(frameon=False, fontsize=8.5)

# Panel D: Symmetric FIR pre-ringing vs Causal IIR on real ROI trace
ax = axes[1, 1]
raw_1ch = np.array(tr_h['raw_lsb'][3], dtype=float)
raw_1ch -= np.median(raw_1ch[:300])
fir_b = signal.firwin(41, [300, 3000], pass_zero=False, fs=20000.0)
filt_sym = np.convolve(raw_1ch, fir_b, mode='same') * lsb_uv
sos_c = signal.butter(4, [300, 3000], btype='bandpass', fs=20000.0, output='sos')
filt_cau = signal.sosfilt(sos_c, raw_1ch) * lsb_uv
ax.plot(t_tr, raw_1ch * lsb_uv, color='black', lw=1.2, label='Raw ADC (centered, uV)')
ax.plot(t_tr, filt_sym, color='#d62728', lw=1.2, label='Delay-comp. symmetric FIR (pre-rings < 0 ms)')
ax.plot(t_tr, filt_cau, color='#1f77b4', lw=1.2, ls='--', label='Causal IIR (zero change < +0.15 ms)')
ax.axvline(0.0, color='k', ls=':', lw=0.9)
ax.set_xlim(-2.5, 4.0)
ax.set_xlabel('Time relative to marker (ms)')
ax.set_ylabel('Voltage (uV)')
ax.set_title('D. Why Symmetric Bandpass Filter Fires Before Marker')
ax.legend(frameon=False, fontsize=8)

fig.savefig(out_dir / 'fig1_q1_artifact_and_preringing.png', dpi=180)
plt.close(fig)

# ==============================================================================
# FIGURE 2: Q2 Pre-Filter Blanking Demo & Sham Self-Check Correlation
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

# Panel A: Pre-filter interpolation + causal bandpass demo on real H trace
ax = axes[0]
work_demo = raw_1ch.copy()
# Center in tr_h is index 400; mask [-5, +50) -> [395, 450)
v_left = np.median(work_demo[375:395])
v_right = np.median(work_demo[450:470])
work_demo[395:450] = np.linspace(v_left, v_right, 55, endpoint=False)
filt_blanked = signal.sosfilt(sos_c, work_demo) * lsb_uv
ax.plot(t_tr, filt_cau, color='#bbbbbb', lw=1.0, label='Unblanked causal filter (rings)')
ax.plot(t_tr, filt_blanked, color='#1f77b4', lw=1.3, label='Blanked [-0.25,+2.5) ms + causal filter')
ax.axvspan(-1.0, 5.0, color='gold', alpha=0.28, label='Post-filter mask [-1, +5) ms')
ax.set_xlim(-4, 15)
ax.set_ylim(-120, 120)
ax.set_xlabel('Time relative to marker (ms)')
ax.set_ylabel('Filtered voltage (uV)')
ax.set_title('A. Pre-Filter Blanking Eliminates Ringing')
ax.legend(frameon=False, fontsize=8)

# Panel B: Sham trial-by-trial correlation (offline vs online masked)
ax = axes[1]
sham_tids = np.array([m['tid'] for m in trial_meta if m['site'] == 'Z'], dtype=int)
off_roi_m = roi_mask_in_active[off_ch]
on_roi_m = roi_mask_in_active[on_ch] & ((on_rf < -20) | (on_rf >= 100))
sham_off_cnt = np.bincount(off_tid[off_roi_m], minlength=1200)[sham_tids]
sham_on_cnt = np.bincount(on_tid[on_roi_m], minlength=1200)[sham_tids]
ax.scatter(sham_on_cnt, sham_off_cnt, s=22, alpha=0.7, color='#1f77b4', edgecolors='none')
xg = np.linspace(0, sham_on_cnt.max() * 1.02, 100)
slope, intercept = np.polyfit(sham_on_cnt, sham_off_cnt, 1)
ax.plot(xg, slope * xg + intercept, color='#d62728', lw=1.3,
        label=f'r = {q2_sum["sham_self_check"]["pearson_r_masked"]:.4f}\nrho = {q2_sum["sham_self_check"]["spearman_rho_masked"]:.4f}')
ax.set_xlabel('Online detected spikes / sham trial ([-300,+300) ms)')
ax.set_ylabel('Offline detected spikes / sham trial ([-300,+300) ms)')
ax.set_title('B. Sham Self-Check (n = 120 Sham Trials)')
ax.legend(frameon=False, fontsize=9)

# Panel C: Ringing check [-6,-1) ms vs [+5,+10) ms across sites
ax = axes[2]
sites_lbl = ['H', 'W', 'F', 'Z']
x_pos = np.arange(len(sites_lbl))
pre_vals = [q2_sum['ringing_check_5ms_windows'][s]['pre_minus6_to_minus1_ms_mean'] for s in sites_lbl]
post_vals = [q2_sum['ringing_check_5ms_windows'][s]['post_plus5_to_plus10_ms_mean'] for s in sites_lbl]
ax.bar(x_pos - 0.17, pre_vals, width=0.34, color='#888888', label='Pre [-6, -1) ms')
ax.bar(x_pos + 0.17, post_vals, width=0.34, color='#2ca02c', label='Post [+5, +10) ms (after b=5ms)')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Site {s}' if s != 'Z' else 'Sham Z' for s in sites_lbl])
ax.set_ylabel('Mean offline spikes / trial (5 ms window)')
ax.set_title('C. Post-Blanking [+5, +10) ms Ringing Check')
ax.legend(frameon=False, fontsize=8.5)

fig.savefig(out_dir / 'fig2_q2_blanking_and_sham_selfcheck.png', dpi=180)
plt.close(fig)

# ==============================================================================
# FIGURE 3: Q3 Evoked Response AUCs & Clean Offline PSTH (-100 to +300 ms)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

# Panel A: AUC summary across 12 (site x window) comparisons (Offline All, Offline Quiet, Online Clean)
ax = axes[0]
labels_12 = [f"{r['site']} {r['window_ms']}" for r in q3_q6['q3_offline']['comparisons']]
y_idx = np.arange(len(labels_12))
auc_off_all = [r['auc'] for r in q3_q6['q3_offline']['comparisons']]
auc_off_q = [r['quiet_subset']['auc'] for r in q3_q6['q3_offline']['comparisons']]
auc_on_all = [r['auc'] for r in q3_q6['q3_online']['comparisons']]

ax.axvline(0.50, color='k', ls='--', lw=1.0, label='Chance (AUC = 0.50)')
ax.axvline(0.60, color='#d62728', ls=':', lw=1.2, label='Judgment threshold (AUC = 0.60)')
ax.scatter(auc_off_all, y_idx - 0.18, color='#1f77b4', s=38, label='Offline (all trials)')
ax.scatter(auc_off_q, y_idx, color='#2ca02c', marker='^', s=38, label='Offline (quiet subset)')
ax.scatter(auc_on_all, y_idx + 0.18, color='#ff7f0e', marker='s', s=34, label='Online clean (>=5 ms)')
ax.set_yticks(y_idx)
ax.set_yticklabels(labels_12)
ax.invert_yaxis()
ax.set_xlim(0.34, 0.66)
ax.set_xlabel('AUC (Real vs 120 Sham Trials)')
ax.set_title('A. Evoked Response AUCs Across All 12 Tests (All Null)')
ax.legend(frameon=False, fontsize=8.5, loc='lower right')

# Panel B: Clean offline PSTH (-100 to +300 ms in 5 ms bins) for H, W, F, Z
ax = axes[1]
edges_5ms = np.arange(-2000, 6000 + 1, 100)  # 5 ms = 100 frames
t_5ms = (edges_5ms[:-1] + 50) / 20.0
for s in ('H', 'W', 'F', 'Z'):
    s_tids = np.array([m['tid'] for m in trial_meta if m['site'] == s], dtype=int)
    m_s = off_roi_m & np.isin(off_tid, s_tids)
    cnt_5, _ = np.histogram(off_rf[m_s], bins=edges_5ms)
    rate_5 = cnt_5 / len(s_tids)
    # Mask bin [0, 5) ms (blanked) as NaN so plot cleanly breaks over blanked zone
    rate_plot = rate_5.astype(float)
    rate_plot[(edges_5ms[:-1] >= -20) & (edges_5ms[:-1] < 100)] = np.nan
    ax.plot(t_5ms, rate_plot, color=colors[s], lw=1.3 if s != 'Z' else 1.8, label=f'Site {s}' if s != 'Z' else 'Sham Z')
ax.axvspan(-1.0, 5.0, color='gold', alpha=0.3, label='Blanked [-1, +5) ms')
ax.set_xlim(-100, 300)
ax.set_xlabel('Time relative to marker (ms)')
ax.set_ylabel('Offline ROI spikes / trial / 5 ms')
ax.set_title('B. Clean Offline PSTH (-100 to +300 ms, 233 ROI ch)')
ax.legend(frameon=False, fontsize=8.5)

fig.savefig(out_dir / 'fig3_q3_evoked_auc_and_psth.png', dpi=180)
plt.close(fig)

# ==============================================================================
# FIGURE 4: Q4 Burst Thresholds / Latency & Q5 Spatial Distance Profile
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)

# Panel A: Warmup 20ms spike count histogram with frozen thresholds 10, 20, 40
ax = axes[0]
ax.hist(warmup_20ms_counts, bins=np.arange(0, 145, 2), color='#6baed6', edgecolor='none')
ax.axvline(10, color='#2ca02c', ls='--', lw=1.3, label='Half (10)')
ax.axvline(20, color='#d62728', ls='-', lw=1.6, label='Primary frozen (20, p95)')
ax.axvline(40, color='#9467bd', ls='--', lw=1.3, label='Double (40)')
ax.set_yscale('log')
ax.set_xlabel('Spikes per 20 ms bin (233 ROI ch, warmup)')
ax.set_ylabel('Bin count (log scale)')
ax.set_title('A. Warmup-Only Burst Threshold Definition')
ax.legend(frameon=False, fontsize=8.5)

# Panel B: Burst onset probability in [5,300) and [300,1000) ms across thresholds
ax = axes[1]
q4_p20 = q3_q6['q4_results']['primary_20']['sites']
x_s = np.arange(4)
w1_p = [q4_p20[s]['prob_onset_5_300ms'] for s in ('H', 'W', 'F', 'Z')]
w2_p = [q4_p20[s]['prob_onset_300_1000ms'] for s in ('H', 'W', 'F', 'Z')]
ax.bar(x_s - 0.18, w1_p, width=0.35, color='#1f77b4', label='[5, 300) ms')
ax.bar(x_s + 0.18, w2_p, width=0.35, color='#ff7f0e', label='[300, 1000) ms')
ax.set_xticks(x_s)
ax.set_xticklabels(['Site H', 'Site W', 'Site F', 'Sham Z'])
ax.set_ylim(0, 0.50)
ax.set_ylabel('Probability of >=1 burst onset (T=20)')
ax.set_title('B. Post-Stimulus Burst Onset Probability (Null)')
ax.legend(frameon=False, fontsize=8.5)

# Panel C: Q5 per-channel (real - sham) across distance bands (non-stim active channels)
ax = axes[2]
band_keys = ['<200_unsat_30_200', '200-500', '500-1000', '>1000']
band_labels = ['<200 um\n(unsat 160-180)', '200-500\num', '500-1000\num', '>1000\num']
xb = np.arange(len(band_keys))
w_colors = {'[5,30)': '#d62728', '[30,50)': '#ff7f0e', '[50,100)': '#2ca02c', '[100,300)': '#1f77b4'}
for wi, (w_lbl, _, _) in enumerate(Q3_WINDOWS):
    ys = [q5_clean_offline[w_lbl]['pooled'][bk]['mean_diff'] for bk in band_keys]
    ax.plot(xb + (wi - 1.5) * 0.06, ys, marker='o', lw=1.2, color=w_colors[w_lbl], label=f'{w_lbl} ms')
ax.axhline(0.0, color='k', ls='--', lw=0.9)
ax.set_xticks(xb)
ax.set_xticklabels(band_labels, fontsize=8.5)
ax.set_ylabel('Pooled (Real mean - Sham mean) spikes / ch')
ax.set_title('C. Spatial Profile by Distance to Stim Site (Q5)')
ax.legend(frameon=False, fontsize=8)

fig.savefig(out_dir / 'fig4_q4_bursts_and_q5_spatial.png', dpi=180)
plt.close(fig)

# ==============================================================================
# UNIFIED results.json
# ==============================================================================
unified = {
    'session_id': 'oddball22021-20261123',
    'chip_id': '22021',
    'organoid_id': '22021',
    'judgment': {
        'category': 'A',
        'label': 'A 没有诱发反应',
        'one_line_basis': '离线与在线去伪迹检测下，3个位点×4个时间窗共12项检验无一满足 AUC>=0.60 且 p<=0.05/12（离线最大 AUC=0.512、最小 p=0.3478；在线最大 AUC=0.515、最小 p=0.2942），安静试验子集、距离分档（>=160 um）与爆发概率/延迟均无刺激效应。',
    },
    'q0_integrity': q0,
    'q1_artifact_and_online_timing': {
        'raw_pulse_onset_frames': {'H': [2, 3], 'W': [3], 'F': [1, 2, 3]},
        'raw_pulse_onset_ms': [0.05, 0.15],
        'raw_roi_recovery_ms': 2.0,
        'online_polluted_interval_ms': [-1.0, 2.5],
        'online_conservative_exclude_ms': [-2.0, 5.0],
        'mechanism_verified': 'Raw ADC waveforms on all 233 readout channels remain at baseline noise prior to +0.10..+0.15 ms (frame +2..+3). Pre-marker online spikes in [-1, 0) ms (H=145.3, W=37.9, F=31.9 per trial) fall at discrete frames -19, -13..-12, -6..-5 (~0.325 ms = 3.08 kHz spacing), matching Gibbs pre-ringing sidelobes of a delay-compensated linear-phase bandpass filter inside mxwserver. Slight mean elevation in [-30, -10) ms (5.9-8.6 vs sham 2.5) disappears completely when excluding ongoing spontaneous bursts already active in [-200, -30) ms (non-burst means: H=0.42, W=0.86, F=0.52 vs Z=0.54).',
        'psth_1ms_per_trial': q1_p1['psth_1ms_per_trial'],
        'psth_1ms_bins_ms': q1_p1['psth_1ms_bins_ms'],
        'window_stats': q1_p2['window_stats'],
    },
    'q2_offline_spike_detection': q2_sum,
    'q3_evoked_responses': {
        'alpha_bonferroni': q3_q6['alpha_bonferroni'],
        'offline': q3_q6['q3_offline'],
        'online_clean': q3_q6['q3_online'],
    },
    'q4_burst_analysis': {
        'frozen_spec': q3_q6['q4_frozen_spec'],
        'results_by_threshold': q3_q6['q4_results'],
    },
    'q5_spatial_profile': {
        'near_200_saturation_check': q3_q6['q5_near_200_saturation'],
        'note_on_lt200_um': 'The 3 stimulated electrodes themselves (0 um: ch 804, 96, 737) are electrically connected to DAC output buffers during stimulation blocks and carry continuous 16-22 LSB DAC buffer noise even on sham trials (unusable). Around site H, 3 immediate neighbor electrodes at 17.5-24.7 um (ch 520, 524, 796) saturate at the ADC rail until 45.6-63.6 ms on 100% of H pulses (unreadable in [5, 30) and [30, 50) ms). Unsaturated <200 um channels (ch 879 at 24.7 um from F, and ch 720, 816, 613 at 161.3-178.5 um) are readable and show near-zero real-minus-sham differences, matching 200-500, 500-1000, and >1000 um.',
        'non_stim_channels_offline': q5_clean_offline,
        'non_stim_channels_online': q5_clean_online,
        'all_active_including_dac_electrodes_offline': q3_q6['q5_offline'],
    },
    'q6_habituation': q3_q6['q6'],
}

(out_dir / 'results.json').write_text(json.dumps(unified, ensure_ascii=False, indent=2), encoding='utf-8')
print('Generated figures fig1..fig4 and unified results.json successfully.')
