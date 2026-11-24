import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.recording import H5Recording

rec_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')

q1_p1 = json.loads((out_dir / 'q1_part1.json').read_text(encoding='utf-8'))
hist_1f = q1_p1['hist_1frame_counts']

print('=== ONLINE SPIKE 1-FRAME HISTOGRAM FROM -25 TO +80 FRAMES (-1.25 ms to +4.00 ms) ===')
for f_off in range(-25, 80):
    idx = f_off + 1000
    h, w, f, z = hist_1f['H'][idx], hist_1f['W'][idx], hist_1f['F'][idx], hist_1f['Z'][idx]
    if (h + w + f) > 12 or -22 <= f_off <= 35:
        print(f'frame {f_off:+4d} ({f_off*0.05:+6.2f} ms): H={h:5d} W={w:5d} F={f:5d} Z={z:3d}')

# Check 0.5 ms bins from +1.0 ms to +15.0 ms across all 12 blocks
print('\n=== ONLINE SPIKES IN 0.5-MS BINS FROM -3.0 TO +12.0 MS (per trial mean) ===')
tc = q1_p1['trial_counts']
for half_ms in range(-6, 24):
    f_start = half_ms * 10
    f_end = f_start + 10
    i0, i1 = f_start + 1000, f_end + 1000
    vals = {s: sum(hist_1f[s][i0:i1]) / tc[s] for s in ('H', 'W', 'F', 'Z')}
    print(f'{half_ms*0.5:+5.1f}..{(half_ms+1)*0.5:+5.1f} ms: H={vals["H"]:7.3f} W={vals["W"]:7.3f} F={vals["F"]:7.3f} Z={vals["Z"]:7.3f}')

# Verify whether high spike counts in [-30, -10) ms are ongoing spontaneous bursts from [-200, -30) ms!
plan = json.loads((rec_dir / 'plan.json').read_text(encoding='utf-8'))
decision = json.loads((rec_dir / 'decision.json').read_text(encoding='utf-8'))
warmup_meta = json.loads((rec_dir / 'warmup.json').read_text(encoding='utf-8'))
binding = json.loads((rec_dir / 'binding.json').read_text(encoding='utf-8'))
roi = np.array(sorted(decision['roi']), dtype=int)
near_chs = np.array(sorted(warmup_meta['excluded_near_sites']), dtype=int)
cfg = SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])

burst_check_rows = []
for b in [b for b in plan['blocks'] if b['stage'] == 'main']:
    bname = b['name']
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    with H5Recording(rec_dir / bname / f'{bname}.raw.h5', cfg) as rec:
        keep = np.isin(rec.spike_channels, roi)
        sp_frames = np.sort(np.rint(rec.spike_times[keep] * rec.fs).astype(np.int64))
        for t in qc['trials']:
            f0 = int(t['frame'])
            c_far_pre = int(np.searchsorted(sp_frames, f0 - 600) - np.searchsorted(sp_frames, f0 - 4000))  # [-200, -30) ms
            c_30_10 = int(np.searchsorted(sp_frames, f0 - 200) - np.searchsorted(sp_frames, f0 - 600))     # [-30, -10) ms
            c_10_50 = int(np.searchsorted(sp_frames, f0 + 1000) - np.searchsorted(sp_frames, f0 + 200))    # [+10, +50) ms
            burst_check_rows.append((t['site'], bname, t['trial'], c_far_pre, c_30_10, c_10_50))

print('\n=== CHECK TRIALS WITH >20 SPIKES IN [-30,-10) ms: WERE THEY ALREADY BURSTING IN [-200,-30) ms? ===')
high_pre = [r for r in burst_check_rows if r[4] > 20]
print(f'Total trials with >20 spikes in [-30,-10) ms: {len(high_pre)} / {len(burst_check_rows)}')
for r in sorted(high_pre, key=lambda x: -x[4])[:12]:
    print(f'  site={r[0]} {r[1]} tr={r[2]:2d}: [-200,-30)ms={r[3]:4d} spikes, [-30,-10)ms={r[4]:4d} spikes, [+10,+50)ms={r[5]:4d} spikes')

# Excluding trials that were ALREADY bursting in [-200, -30) ms (c_far_pre > 10), what is mean in [-30, -10) ms?!
print('\n=== [-30,-10) ms MEAN WHEN EXCLUDING ONGOING PRE-BURSTS ([-200,-30) ms > 10 spikes) ===')
for s in ('H', 'W', 'F', 'Z'):
    all_v = [r[4] for r in burst_check_rows if r[0] == s]
    quiet_v = [r[4] for r in burst_check_rows if r[0] == s and r[3] <= 10]
    print(f'  {s}: all_mean={np.mean(all_v):.3f} (n={len(all_v)}) -> non_burst_mean={np.mean(quiet_v):.3f} (n={len(quiet_v)})')

# Check single-channel WORST artifact on ROI (233 ch) AND near_sites (<200 um) in main_r1_uniform!
print(f'\nNear-site channels (<200 um): count={len(near_chs)}, channels={near_chs.tolist()}')
with H5Recording(rec_dir / 'main_r1_uniform/main_r1_uniform.raw.h5', cfg) as rec:
    qc_u = json.loads((rec_dir / 'main_r1_uniform-qc.json').read_text(encoding='utf-8'))
    roi_max_step = {s: np.zeros(1200) for s in ('H', 'W', 'F', 'Z')}
    near_rail_frac = {s: np.zeros(1200) for s in ('H', 'W', 'F', 'Z')}
    near_max_step = {s: np.zeros(1200) for s in ('H', 'W', 'F', 'Z')}
    for t in qc_u['trials']:
        s = t['site']
        f0 = int(t['frame'])
        r_roi = rec.sample(roi, (f0 - 200) / rec.fs, 1200 / rec.fs).astype(np.float32)
        step_roi = np.max(np.abs(np.diff(r_roi, axis=1, prepend=r_roi[:, :1])), axis=0)
        roi_max_step[s] = np.maximum(roi_max_step[s], step_roi)
        if len(near_chs):
            r_near = rec.sample(near_chs, (f0 - 200) / rec.fs, 1200 / rec.fs).astype(np.float32)
            rail_n = ((r_near == 0) | (r_near == 1023)).mean(axis=0)
            step_n = np.max(np.abs(np.diff(r_near, axis=1, prepend=r_near[:, :1])), axis=0)
            near_rail_frac[s] = np.maximum(near_rail_frac[s], rail_n)
            near_max_step[s] = np.maximum(near_max_step[s], step_n)

print('\n=== ABSOLUTE MAX SINGLE-FRAME STEP (LSB) ACROSS ALL 233 ROI CHANNELS & ALL TRIALS IN main_r1_uniform ===')
for ms in range(0, 15):
    i0, i1 = 200 + ms * 20, 200 + (ms + 1) * 20
    print(f'+{ms:2d}..+{ms+1:2d} ms: ' + '  '.join(f'{s}:roi_max_step={np.max(roi_max_step[s][i0:i1]):5.1f}' for s in ('H', 'W', 'F', 'Z'))
          + ' | near_rail_max=' + '/'.join(f'{np.max(near_rail_frac[s][i0:i1]):.2f}' for s in ('H', 'W', 'F')))

# QUESTION 4 PRE-REGISTRATION FROM WARMUP ONLY!
# Open ONLY warmup.raw.h5 (zero stimulation!) and inspect 20 ms binned counts on the 233 ROI channels!
with H5Recording(rec_dir / 'warmup/warmup.raw.h5', cfg) as wrec:
    print(f'\n=== WARMUP RECORDING (duration={wrec.duration:.2f} s) 20-MS BIN DISTRIBUTION ON 233 ROI CHANNELS ===')
    keep = np.isin(wrec.spike_channels, roi)
    # Skip first 4.0s settle just like select_roi does (or check both full and >=4s)
    wt = wrec.spike_times[keep]
    wt_clean = wt[wt >= 4.0]
    edges_20ms = np.arange(4.0, wrec.duration + 1e-9, 0.020)
    counts_20ms, _ = np.histogram(wt_clean, bins=edges_20ms)
    print(f'Total 20ms bins (4s..{wrec.duration:.1f}s): {len(counts_20ms)}')
    print(f'Mean={counts_20ms.mean():.3f}, SD={counts_20ms.std():.3f}, Median={np.median(counts_20ms):.1f}')
    for p in (50, 75, 90, 95, 97, 98, 99, 99.5, 100):
        print(f'  p{p}: {np.percentile(counts_20ms, p):.1f}')
    for thr in (10, 15, 20, 25, 30, 40, 50):
        above = counts_20ms >= thr
        # Burst onsets: bin >= thr preceded by bin < thr (or with >=100 ms quiet gap)
        onsets_1bin = int(np.sum(above[1:] & (~above[:-1]))) + int(above[0])
        # Also burst episodes merging gaps <= 60 ms (3 bins)
        idx_ab = np.where(above)[0]
        episodes = 0
        if len(idx_ab):
            episodes = 1 + int(np.sum(np.diff(idx_ab) > 3))
        print(f'  thr>={thr:2d}: bins_above={above.sum():4d} ({above.mean()*100:5.2f}%), raw_onsets={onsets_1bin:3d}, merged_episodes(gap>60ms)={episodes:3d} ({episodes/((wrec.duration-4)/60):.1f}/min)')
