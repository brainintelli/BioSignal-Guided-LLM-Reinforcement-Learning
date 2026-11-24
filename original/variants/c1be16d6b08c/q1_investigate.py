import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
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

# 1. Inspect raw H5 spikes dtype and attributes on first main block
first_h5 = rec_dir / main_blocks[0]['name'] / (main_blocks[0]['name'] + '.raw.h5')
with h5py.File(first_h5, 'r') as f:
    d = f['data_store/data0000']
    sp_ds = d['spikes']
    ev_ds = d['events']
    print('spikes dtype:', sp_ds.dtype, 'shape:', sp_ds.shape)
    print('events dtype:', ev_ds.dtype, 'shape:', ev_ds.shape)
    print('settings keys:', list(d['settings'].keys()))
    for k in d['settings'].keys():
        if k != 'mapping':
            try:
                print('  setting', k, ':', d['settings'][k][()])
            except Exception as e:
                print('  setting', k, 'err:', e)

# 2. Across all 12 main blocks:
#    - 1 ms binned PSTH from -50 to +100 ms (150 bins, each 20 frames) on ROI channels for H, W, F, Z
#    - Exact 1-frame histogram from -1000 to +2000 frames (-50 to +100 ms) on ROI channels for H, W, F, Z
#    - Also check site-channel exact first deviation & first rail frame across ALL 1080 real trials!
bins_1ms_edges = np.arange(-1000, 2000 + 1, 20)  # 150 bins: [-50, -49), ..., [99, 100) ms
psth_1ms = {s: np.zeros(150, dtype=np.int64) for s in ('H', 'W', 'F', 'Z')}
hist_1frame = {s: np.zeros(3000, dtype=np.int64) for s in ('H', 'W', 'F', 'Z')}
trial_counts = {s: 0 for s in ('H', 'W', 'F', 'Z')}

site_onset_frames = {s: [] for s in ('H', 'W', 'F')}
site_dev_frames = {s: [] for s in ('H', 'W', 'F')}

# Also check spontaneous spike-triggered raw waveform in warmup (or pre-marker) to verify spike frameno alignment
spont_peak_offsets = []
premarker_examples = []  # store details of pre-marker online spikes [-30, 0) ms

for b in main_blocks:
    bname = b['name']
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    h5_path = rec_dir / bname / f'{bname}.raw.h5'
    with H5Recording(h5_path, cfg) as rec, h5py.File(h5_path, 'r') as hf:
        sp_raw = hf['data_store/data0000/spikes'][:]
        # Filter to ROI channels and valid frame coverage
        in_roi = np.isin(sp_raw['channel'], roi)
        sp_roi = sp_raw[in_roi]
        # Relative frame to file start
        sp_rel_frames = sp_roi['frameno'].astype(np.int64) - int(rec.first_frame)
        order = np.argsort(sp_rel_frames)
        sp_rel_frames = sp_rel_frames[order]
        sp_roi_sorted = sp_roi[order]

        for t_info in qc['trials']:
            site = t_info['site']
            f0 = int(t_info['frame'])  # already relative to rec.first_frame! Let's verify!
            trial_counts[site] += 1

            lo = np.searchsorted(sp_rel_frames, f0 - 1000, side='left')
            hi = np.searchsorted(sp_rel_frames, f0 + 2000, side='left')
            rel = sp_rel_frames[lo:hi] - f0  # in [-1000, 1999]
            if len(rel):
                c1, _ = np.histogram(rel, bins=bins_1ms_edges)
                psth_1ms[site] += c1
                np.add.at(hist_1frame[site], rel + 1000, 1)

            # Check site channel raw onset on all real trials
            if site in ('H', 'W', 'F'):
                sch = site_channels[site]
                # Read [-5 ms, +5 ms] = [-100, +100] frames around f0
                w_raw = rec.sample([sch], (f0 - 100) / rec.fs, 200 / rec.fs)[0].astype(float)
                base_med = np.median(w_raw[:80])  # [-100, -20) frames
                dev = np.abs(w_raw - base_med)
                rail = (w_raw == 0) | (w_raw == 1023)
                rail_idx = np.where(rail)[0] - 100
                dev_idx = np.where(dev > 20)[0] - 100  # >20 LSB (~126 uV) deviation
                site_onset_frames[site].append(int(rail_idx[0]) if len(rail_idx) else None)
                site_dev_frames[site].append(int(dev_idx[0]) if len(dev_idx) else None)

        # On first main block, inspect spontaneous spikes vs pre-marker spikes in raw waveform!
        if bname == 'main_r1_H':
            # Check f0 convention vs rec.events[0]['frame'] - rec.first_frame
            ev0_rel = rec.events[0]['frame'] - rec.first_frame
            print('Verify frame convention: qc frame[0]=', qc['trials'][0]['frame'], 'ev0_rel=', ev0_rel)

            # Spontaneous spikes in [-6000, -2000) before markers
            for t_info in qc['trials'][:30]:
                f0 = int(t_info['frame'])
                lo_s = np.searchsorted(sp_rel_frames, f0 - 6000)
                hi_s = np.searchsorted(sp_rel_frames, f0 - 2000)
                for idx in range(lo_s, min(hi_s, lo_s + 5)):
                    sf = int(sp_rel_frames[idx])
                    sch = int(sp_roi_sorted['channel'][idx])
                    snip = rec.sample([sch], (sf - 60) / rec.fs, 120 / rec.fs)[0].astype(float)
                    med = np.median(snip)
                    # Find minimum (negative spike) or extremum relative to sf (index 60)
                    min_idx = int(np.argmin(snip - med)) - 60
                    spont_peak_offsets.append(min_idx)

            # Pre-marker spikes in [-600, 0) frames ([-30, 0) ms) on real trials
            for t_info in qc['trials'][:20]:
                if t_info['site'] == 'Z':
                    continue
                f0 = int(t_info['frame'])
                lo_p = np.searchsorted(sp_rel_frames, f0 - 600)
                hi_p = np.searchsorted(sp_rel_frames, f0)
                if hi_p > lo_p and len(premarker_examples) < 15:
                    for idx in range(lo_p, min(hi_p, lo_p + 3)):
                        sf = int(sp_rel_frames[idx])
                        sch = int(sp_roi_sorted['channel'][idx])
                        amp = float(sp_roi_sorted['amplitude'][idx]) if 'amplitude' in sp_roi_sorted.dtype.names else None
                        # Read raw on this readout channel from f0-800 to f0+1600 (-40 ms to +80 ms)
                        raw_tr = rec.sample([sch], (f0 - 800) / rec.fs, 2400 / rec.fs)[0].astype(float)
                        base_m = float(np.median(raw_tr[:400]))  # [-40, -20) ms
                        base_sd = float(np.std(raw_tr[:400]))
                        max_pre_dev = float(np.max(np.abs(raw_tr[:800] - base_m)))  # [-40, 0) ms!
                        at_sf_dev = float(raw_tr[800 + (sf - f0)] - base_m)
                        post_min = float(np.min(raw_tr[800:860]))
                        post_max = float(np.max(raw_tr[800:860]))
                        premarker_examples.append({
                            'trial': t_info['trial'],
                            'site': t_info['site'],
                            'channel': sch,
                            'spike_rel_frame': sf - f0,
                            'spike_rel_ms': (sf - f0) / 20.0,
                            'h5_amplitude': amp,
                            'raw_base_median': base_m,
                            'raw_base_sd_lsb': round(base_sd, 3),
                            'raw_max_abs_dev_before_marker_lsb': round(max_pre_dev, 3),
                            'raw_dev_at_spike_frame_lsb': round(at_sf_dev, 3),
                            'raw_post_0_3ms_min_max': [post_min, post_max],
                        })

print('\n=== TRIAL COUNTS ===', trial_counts)
for s in ('H', 'W', 'F'):
    onsets = [x for x in site_onset_frames[s] if x is not None]
    devs = [x for x in site_dev_frames[s] if x is not None]
    print(f'Site {s}: rail_onset frame min={min(onsets)}, max={max(onsets)}, unique={sorted(set(onsets))}')
    print(f'Site {s}: >20LSB dev frame min={min(devs)}, max={max(devs)}, unique={sorted(set(devs))}')

print('\n=== SPONTANEOUS SPIKE RAW MIN OFFSET (frames relative to spikes.frameno) ===')
vals, cnts = np.unique(spont_peak_offsets, return_counts=True)
top = sorted(zip(cnts, vals), reverse=True)[:10]
print('Top offsets (count, frame_offset):', top)

print('\n=== PRE-MARKER ONLINE SPIKE EXAMPLES (raw check!) ===')
for ex in premarker_examples[:10]:
    print(ex)

print('\n=== 1-MS PSTH (spikes per trial across 233 ROI channels) -35 to +55 ms ===')
for ms in range(-35, 55):
    b_idx = ms + 50
    row_str = f'{ms:+4d}..{ms+1:+4d} ms: ' + '  '.join(
        f"{s}={psth_1ms[s][b_idx] / trial_counts[s]:6.3f}" for s in ('H', 'W', 'F', 'Z')
    )
    print(row_str)

print('\n=== EXACT FRAME OFFSETS WITH NONZERO EXCESS SPIKES BEFORE MARKER (-600..+20 frames) ===')
for f_off in range(-600, 20):
    idx = f_off + 1000
    tot_real = hist_1frame['H'][idx] + hist_1frame['W'][idx] + hist_1frame['F'][idx]
    if tot_real > 5 or f_off >= -25:
        print(f'frame {f_off:+5d} ({f_off/20.0:+6.2f} ms): H={hist_1frame["H"][idx]:4d} W={hist_1frame["W"][idx]:4d} F={hist_1frame["F"][idx]:4d} Z={hist_1frame["Z"][idx]:4d}')

# Save Q1 part 1 data
q1_part1 = {
    'trial_counts': trial_counts,
    'psth_1ms_bins_ms': list(range(-50, 100)),
    'psth_1ms_per_trial': {s: (psth_1ms[s] / trial_counts[s]).tolist() for s in ('H', 'W', 'F', 'Z')},
    'psth_1ms_counts': {s: psth_1ms[s].tolist() for s in ('H', 'W', 'F', 'Z')},
    'hist_1frame_offset_start': -1000,
    'hist_1frame_counts': {s: hist_1frame[s].tolist() for s in ('H', 'W', 'F', 'Z')},
    'site_rail_onset_unique_frames': {s: sorted(set(x for x in site_onset_frames[s] if x is not None)) for s in ('H', 'W', 'F')},
    'site_dev_onset_unique_frames': {s: sorted(set(x for x in site_dev_frames[s] if x is not None)) for s in ('H', 'W', 'F')},
    'spontaneous_raw_trough_offset_top': [[int(v), int(c)] for c, v in top],
    'premarker_examples': premarker_examples,
}
(out_dir / 'q1_part1.json').write_text(json.dumps(q1_part1, indent=2), encoding='utf-8')
