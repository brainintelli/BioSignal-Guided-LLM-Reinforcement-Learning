import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.recording import H5Recording

rec_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out_dir = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')

route = json.loads((rec_dir / 'route.json').read_text(encoding='utf-8'))
binding = json.loads((rec_dir / 'binding.json').read_text(encoding='utf-8'))
trial_meta = json.loads((out_dir / 'q2_trials_meta.json').read_text(encoding='utf-8'))
npz = np.load(out_dir / 'q2_detected_spikes.npz')
active_all = npz['active_all']
off_tid, off_ch, off_rf = npz['off_tid'], npz['off_ch'], npz['off_rf']
on_tid, on_ch, on_rf = npz['on_tid'], npz['on_ch'], npz['on_rf']

cfg = SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])
site_channels = {k: int(v) for k, v in route['channels'].items()}
with H5Recording(rec_dir / 'warmup/warmup.raw.h5', cfg) as wrec:
    ch_xy = wrec.channel_xy_um

# Print per-channel real_mean and sham_mean for EVERY individual channel with dist < 200 um to H, W, or F!
sham_tids = np.array([m['tid'] for m in trial_meta if m['site'] == 'Z'], dtype=int)
wins = [('[5,30)', 100, 600), ('[30,50)', 600, 1000), ('[50,100)', 1000, 2000), ('[100,300)', 2000, 6000)]

print('=== INDIVIDUAL <200 um CHANNELS: OFFLINE vs ONLINE (real_mean / sham_mean / diff) ===')
for s_site in ('H', 'W', 'F'):
    sc = site_channels[s_site]
    sx, sy = ch_xy[sc]
    s_tids = np.array([m['tid'] for m in trial_meta if m['site'] == s_site], dtype=int)
    print(f'\n--- Stimulated Site {s_site} (site_ch={sc}) ---')
    for ci, ch in enumerate(active_all):
        ch = int(ch)
        d_um = float(np.hypot(ch_xy[ch][0] - sx, ch_xy[ch][1] - sy))
        if d_um < 200.0:
            row_off = []
            row_on = []
            for w_lbl, flo, fhi in wins:
                moff = (off_ch == ci) & (off_rf >= flo) & (off_rf < fhi)
                c_off = np.bincount(off_tid[moff], minlength=1200)
                row_off.append(f"{w_lbl}:{c_off[s_tids].mean():.3f}-{c_off[sham_tids].mean():.3f}={c_off[s_tids].mean()-c_off[sham_tids].mean():+.3f}")

                mon = (on_ch == ci) & (on_rf >= flo) & (on_rf < fhi)
                c_on = np.bincount(on_tid[mon], minlength=1200)
                row_on.append(f"{w_lbl}:{c_on[s_tids].mean():.3f}-{c_on[sham_tids].mean():.3f}={c_on[s_tids].mean()-c_on[sham_tids].mean():+.3f}")
            print(f'  ch={ch:4d} dist={d_um:5.1f} um:')
            print('    OFFLINE:', ' | '.join(row_off))
            print('    ONLINE :', ' | '.join(row_on))

# Inspect raw waveform of ch 737 (F, 0um), ch 879 (F, 24.7um), ch 96 (W, 0um), ch 520 (H, 17.5um) on 5 trials!
for s_site, bname, ch_list in [('F', 'main_r1_F', [737, 879, 613]), ('W', 'main_r1_W', [96]), ('H', 'main_r1_H', [804, 520, 720, 816])]:
    qc = json.loads((rec_dir / f'{bname}-qc.json').read_text(encoding='utf-8'))
    st = [t for t in qc['trials'] if t['site'] == s_site][:5]
    with H5Recording(rec_dir / bname / f'{bname}.raw.h5', cfg) as rec:
        print(f'\n=== RAW WAVEFORM SUMMARY FOR {s_site} ({bname}, first 5 {s_site} trials) ===')
        for ch in ch_list:
            devs_by_win = {w[0]: [] for w in wins}
            steps_by_win = {w[0]: [] for w in wins}
            for t in st:
                f0 = int(t['frame'])
                raw = rec.sample([ch], (f0 - 1000) / rec.fs, 7000 / rec.fs)[0].astype(float)  # [-50, +300) ms
                base = np.median(raw[:800])
                for w_lbl, flo, fhi in wins:
                    seg = raw[1000 + flo : 1000 + fhi]
                    devs_by_win[w_lbl].append(float(np.max(np.abs(seg - base))))
                    steps_by_win[w_lbl].append(float(np.max(np.abs(np.diff(seg)))))
            print(f'  ch={ch:4d}: ' + ' | '.join(f"{w[0]} max_dev={np.mean(devs_by_win[w[0]]):.1f}LSB, max_step={np.mean(steps_by_win[w[0]]):.1f}LSB" for w in wins))
