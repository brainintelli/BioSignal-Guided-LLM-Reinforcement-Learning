import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from scipy import signal
from mea_stim_v2.recording import H5Recording

rec_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')

plan = json.loads((rec_dir / 'plan.json').read_text(encoding='utf-8'))
decision = json.loads((rec_dir / 'decision.json').read_text(encoding='utf-8'))
route = json.loads((rec_dir / 'route.json').read_text(encoding='utf-8'))
binding = json.loads((rec_dir / 'binding.json').read_text(encoding='utf-8'))

roi = np.array(sorted(decision['roi']), dtype=int)
site_channels = {k: int(v) for k, v in route['channels'].items()}
cfg = SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])
main_blocks = [b for b in plan['blocks'] if b['stage'] == 'main']

# 1. Check per-trial spike counts in key windows across all 1200 trials:
#    [-50,-30), [-30,-10), [-10,-2), [-2,0), [0,2), [2,10), [10,20), [20,30), [30,50), [50,100) ms
win_defs = [
    ('[-50,-30)', -1000, -600),
    ('[-30,-10)', -600, -200),
    ('[-10,-2)', -200, -40),
    ('[-2,-1)', -40, -20),
    ('[-1,0)', -20, 0),
    ('[0,2)', 0, 40),
    ('[2,5)', 40, 100),
    ('[5,10)', 100, 200),
    ('[10,20)', 200, 400),
    ('[20,30)', 400, 600),
    ('[30,50)', 600, 1000),
    ('[50,100)', 1000, 2000),
]

per_trial_win = {s: {w[0]: [] for w in win_defs} for s in ('H', 'W', 'F', 'Z')}
# Also 1-ms trial-incidence (fraction of trials with >=1 spike in each 1-ms bin, -50..+100 ms)
bins_1ms_edges = np.arange(-1000, 2000 + 1, 20)
psth_trial_any = {s: np.zeros(150, dtype=np.int64) for s in ('H', 'W', 'F', 'Z')}

for b in main_blocks:
    bname = b['name']
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    h5_path = rec_dir / bname / f'{bname}.raw.h5'
    with H5Recording(h5_path, cfg) as rec:
        keep = np.isin(rec.spike_channels, roi)
        sp_frames = np.sort(np.rint(rec.spike_times[keep] * rec.fs).astype(np.int64))
        for t_info in qc['trials']:
            s = t_info['site']
            f0 = int(t_info['frame'])
            for wname, w0, w1 in win_defs:
                cnt = int(np.searchsorted(sp_frames, f0 + w1) - np.searchsorted(sp_frames, f0 + w0))
                per_trial_win[s][wname].append(cnt)
            lo = np.searchsorted(sp_frames, f0 - 1000)
            hi = np.searchsorted(sp_frames, f0 + 2000)
            rel = sp_frames[lo:hi] - f0
            if len(rel):
                c1, _ = np.histogram(rel, bins=bins_1ms_edges)
                psth_trial_any[s] += (c1 > 0).astype(np.int64)

print('=== WINDOW SUMMARY: mean | median | p90 | max | frac(>=1) (per 10ms normalized rate in parens) ===')
for wname, w0, w1 in win_defs:
    dur_10ms = (w1 - w0) / 200.0
    print(f'\nWindow {wname} ({(w1-w0)/20.0:.1f} ms):')
    for s in ('H', 'W', 'F', 'Z'):
        arr = np.array(per_trial_win[s][wname], dtype=float)
        print(f'  {s}: mean={arr.mean():7.3f} ({arr.mean()/dur_10ms:7.3f}/10ms)  med={np.median(arr):5.1f}  p90={np.percentile(arr,90):6.1f}  max={arr.max():6.0f}  frac>=1={(arr>=1).mean():.3f}')

# 2. Raw waveform inspection across ALL 4 blocks of Round 1 (400 trials: 100 H, 100 W, 100 F, 40 Z + uniform)!
#    Measure on all 233 ROI channels from -10 ms (-200 frames) to +50 ms (+1000 frames):
#    (a) Exact frame when raw deviates from pre-marker [-10, -2) ms baseline!
#    (b) Fraction of ROI channels at rail (0 or 1023) vs frame
#    (c) |raw(t) - pre_median| percentiles (50%, 95%, 99%, 99.9%, max) in LSB and uV vs time!
#    (d) |diff(raw)| (frame-to-frame step, sensitive to high-freq artifact vs slow drift) vs time!
raw_dev_p99 = {s: [] for s in ('H', 'W', 'F', 'Z')}
raw_dev_max = {s: [] for s in ('H', 'W', 'F', 'Z')}
raw_diff_p99 = {s: [] for s in ('H', 'W', 'F', 'Z')}
raw_rail_cnt = {s: [] for s in ('H', 'W', 'F', 'Z')}

# Also save example raw traces from main_r1_uniform (has H, W, F, Z in the same block!)
saved_traces = {}

r1_blocks = [b for b in main_blocks if b['round'] == 1]
lsb_uv = None
for b in r1_blocks:
    bname = b['name']
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    h5_path = rec_dir / bname / f'{bname}.raw.h5'
    with H5Recording(h5_path, cfg) as rec:
        lsb_uv = rec.metadata['lsb_uv']
        for t_info in qc['trials']:
            s = t_info['site']
            f0 = int(t_info['frame'])
            # Read [-10 ms, +50 ms] = [-200, +1000] = 1200 frames on all 233 ROI channels
            raw = rec.sample(roi, (f0 - 200) / rec.fs, 1200 / rec.fs).astype(np.float32)
            base = np.median(raw[:, :160], axis=1, keepdims=True)  # [-10, -2) ms
            dev = np.abs(raw - base)
            dstep = np.abs(np.diff(raw, axis=1, prepend=raw[:, :1]))
            rail = (raw == 0) | (raw == 1023)
            raw_dev_p99[s].append(np.percentile(dev, 99, axis=0))
            raw_dev_max[s].append(np.max(dev, axis=0))
            raw_diff_p99[s].append(np.percentile(dstep, 99, axis=0))
            raw_rail_cnt[s].append(np.sum(rail, axis=0))

            if bname == 'main_r1_uniform' and s not in saved_traces:
                # Save 6 representative ROI channels (nearest to sites + medium + far) + all 3 site channels
                all_ch = list(site_channels.values()) + roi[:6].tolist()
                tr_raw = rec.sample(all_ch, (f0 - 400) / rec.fs, 1600 / rec.fs).astype(float)  # [-20, +60] ms
                saved_traces[s] = {
                    'trial': t_info['trial'],
                    'frame': f0,
                    'channels': all_ch,
                    'site_channels': site_channels,
                    'raw_lsb': tr_raw.tolist(),
                }

print(f'\nLSB_uV = {lsb_uv:.4f} uV')
print('=== RAW WAVEFORM ARTIFACT ONSET AROUND MARKER (frames -15 to +25, mean across R1 trials of max_dev_lsb and rail_channels) ===')
for f_rel in range(-15, 25):
    idx = f_rel + 200
    parts = []
    for s in ('H', 'W', 'F', 'Z'):
        m_max = np.mean([x[idx] for x in raw_dev_max[s]])
        m_rail = np.mean([x[idx] for x in raw_rail_cnt[s]])
        parts.append(f'{s}:max={m_max:6.1f}(rail={m_rail:5.1f})')
    print(f'frame {f_rel:+3d} ({f_rel*0.05:+6.2f} ms): ' + '  '.join(parts))

print('\n=== RAW WAVEFORM ARTIFACT RECOVERY AFTER MARKER (1 ms windows from 0 to +40 ms: mean p99_dev_lsb, p99_step_lsb, mean_rail_ch) ===')
for ms in range(0, 40):
    s_idx = 200 + ms * 20
    e_idx = s_idx + 20
    parts = []
    for s in ('H', 'W', 'F', 'Z'):
        p99_d = np.mean([np.max(x[s_idx:e_idx]) for x in raw_dev_p99[s]])
        p99_s = np.mean([np.max(x[s_idx:e_idx]) for x in raw_diff_p99[s]])
        m_rail = np.mean([np.max(x[s_idx:e_idx]) for x in raw_rail_cnt[s]])
        parts.append(f'{s}:dev99={p99_d:5.1f},step99={p99_s:4.1f},rail={m_rail:4.2f}')
    print(f'+{ms:2d}..+{ms+1:2d} ms: ' + ' | '.join(parts))

# 3. Demonstrate WHY mxwserver online detector fires at frames -19, -13..-12, -6..-5 BEFORE frame +1!
#    Test zero-phase / symmetric FIR bandpass filter (300-3000 Hz) on an actual real trial raw trace from main_r1_H!
with H5Recording(rec_dir / 'main_r1_H/main_r1_H.raw.h5', cfg) as rec, h5py.File(rec_dir / 'main_r1_H/main_r1_H.raw.h5', 'r') as hf:
    qc_h = json.loads((rec_dir / 'main_r1_H-qc.json').read_text(encoding='utf-8'))
    t0 = next(t for t in qc_h['trials'] if t['site'] == 'H')
    f0 = int(t0['frame'])
    # Pick channel 964 or any ROI channel that had an online spike in [-20, 0) frames in t0
    sp = hf['data_store/data0000/spikes'][:]
    sp_rel = sp['frameno'].astype(np.int64) - int(rec.first_frame)
    mask_pre = (sp_rel >= f0 - 25) & (sp_rel < f0) & np.isin(sp['channel'], roi)
    pre_chs = sp['channel'][mask_pre]
    pre_frs = sp_rel[mask_pre] - f0
    print(f'\nTrial {t0["trial"]} (site H) online pre-marker spikes in [-25, 0) frames: count={len(pre_chs)}')
    if len(pre_chs):
        ch_demo = int(pre_chs[0])
        fr_demo = int(pre_frs[0])
        raw_demo = rec.sample([ch_demo], (f0 - 400) / rec.fs, 1200 / rec.fs)[0].astype(float)
        # Design symmetric linear-phase FIR bandpass 300-3000 Hz (say 41 taps -> half-len 20 frames = 1.0 ms!)
        fir_b = signal.firwin(41, [300, 3000], pass_zero=False, fs=20000.0)
        # Delay-compensated output (same center alignment as zero-phase / group-delay subtracted online FIR)
        filt_sym = np.convolve(raw_demo - np.median(raw_demo[:300]), fir_b, mode='same')
        # Causal IIR Butterworth 300-3000 Hz for comparison
        sos_causal = signal.butter(3, [300, 3000], btype='bandpass', fs=20000.0, output='sos')
        filt_causal = signal.sosfilt(sos_causal, raw_demo - np.median(raw_demo[:300]))
        noise_rms = np.std(filt_sym[:300])
        print(f'Demo ch={ch_demo}, online spike at rel_frame={fr_demo}: pre_noise_rms={noise_rms:.3f} LSB')
        for k in range(-22, 15):
            idx = 400 + k
            print(f'  frame {k:+3d}: raw={raw_demo[idx]:6.1f}  symFIR/rms={filt_sym[idx]/noise_rms:+7.2f}  causalIIR/rms={filt_causal[idx]/noise_rms:+7.2f}')

q1_part2 = {
    'lsb_uv': lsb_uv,
    'psth_trial_any_1ms': {s: (psth_trial_any[s] / len(per_trial_win[s]['[-1,0)'])).tolist() for s in ('H', 'W', 'F', 'Z')},
    'window_stats': {
        wname: {
            s: {
                'mean': float(np.mean(per_trial_win[s][wname])),
                'median': float(np.median(per_trial_win[s][wname])),
                'p90': float(np.percentile(per_trial_win[s][wname], 90)),
                'max': int(np.max(per_trial_win[s][wname])),
                'frac_ge_1': float(np.mean(np.array(per_trial_win[s][wname]) >= 1)),
            }
            for s in ('H', 'W', 'F', 'Z')
        }
        for wname, _, _ in win_defs
    },
    'saved_traces_r1_uniform': saved_traces,
}
(out_dir / 'q1_part2.json').write_text(json.dumps(q1_part2, indent=2), encoding='utf-8')
