"""Offline qualification on the closed fresh file; records explicit selection rules."""
import os
os.environ['HDF5_PLUGIN_PATH'] = '/pub/soft/MaxLab/so'
from pathlib import Path
import hashlib
import json
import re
import shutil
import h5py
import numpy as np

base = Path(__file__).resolve().parent
root = base / 'qualification_120s_v2'
capture = json.loads((root / 'capture.json').read_text())
assert capture['status'] == 'completed' and not capture['cleanup_errors']
route = Path('/home/researcher/configs/261117/11h16m30s.cfg')
with h5py.File(root / 'qualification.raw.h5', 'r') as f:
    d = f['data_store/data0000']
    m = d['settings/mapping'][:]
    spikes = d['spikes'][:]
    frames = d['groups/routed/frame_nos'][:]
    fs = float(d['settings/sampling'][0])
    uv = float(d['settings/lsb'][0]) * 1e6
    duration = len(frames) / fs
    chip_id = f['wellplate/id'][0].decode()
    pairs = sorted((int(a), int(b)) for a, b in re.findall(r'(\d+)\((\d+)\)', route.read_text().split()[0]))
    recorded_pairs = sorted((int(x['channel']), int(x['electrode'])) for x in m)
    raw = d['groups/routed/raw']
    # Four distributed 1 s windows, all channels read together to avoid repeated decompression.
    sample = np.concatenate([raw[:, int(t*fs):int((t+1)*fs)] for t in (0, 40, 80, 110)], axis=1)
    means = sample.mean(axis=1)
    std = sample.std(axis=1)
    rails = ((sample == 0) | (sample == 1023)).mean(axis=1)
    valid = np.isin(spikes['channel'], m['channel']) & (spikes['frameno'] >= frames[0]) & (spikes['frameno'] <= frames[-1])
    candidates = []
    for row in m:
        ch = int(row['channel'])
        sp = spikes[(spikes['channel'] == ch) & valid]
        if len(sp) < 100:
            continue
        amplitude = np.abs(sp['amplitude'])
        times = (sp['frameno'].astype(float) - float(frames[0])) / fs
        bin_counts = np.histogram(times, bins=[0,40,80,duration])[0]
        refractory_fraction = float(np.mean(np.diff(np.sort(times)) < .001))
        median_uv = float(np.median(amplitude) * uv)
        snr = float(np.median(amplitude) / max(std[ch], 1e-12))
        ok = (median_uv >= 50 and snr >= 3 and rails[ch] == 0 and
              min(bin_counts) >= 10 and refractory_fraction < .02)
        candidates.append({'channel': ch, 'electrode': int(row['electrode']), 'count':len(sp),
            'spike_rate_hz':len(sp)/duration, 'median_amplitude_uv':median_uv,
            'p90_amplitude_uv':float(np.percentile(amplitude,90)*uv),
            'sample_raw_std_uv':float(std[ch]*uv), 'snr_median_over_raw_std':snr,
            'sample_rail_fraction':float(rails[ch]), 'sample_mean_adc':float(means[ch]),
            'counts_40s':bin_counts.tolist(), 'refractory_under_1ms_fraction':refractory_fraction,
            'eligible':bool(ok)})
    candidates.sort(key=lambda x:(x['eligible'],x['snr_median_over_raw_std'],x['count']), reverse=True)
    eligible = [r for r in candidates if r['eligible']]
    errors = []
    if chip_id != '22554': errors.append('chip_id_mismatch')
    if fs != 20000 or not 119 <= duration <= 123: errors.append('unexpected_duration_or_sample_rate')
    if np.any(np.diff(frames) != 1): errors.append('noncontiguous_frames')
    if pairs != recorded_pairs: errors.append('route_not_matching_real_H5')
    if len(d['events']) != 0: errors.append('unexpected_events_in_unstimulated_capture')
    if not eligible: errors.append('no_eligible_stimulation_site')
    # Invalid detector records are retained and excluded; raw signal and selected channel remain authoritative.
    report = {'status':'pass_for_first_probe' if not errors else 'failed', 'errors':errors,
        'chip_id':chip_id, 'device_model':f['wellplate/version'][0].decode(),
        'duration_seconds':duration, 'sample_rate_hz':fs, 'frames':len(frames),
        'frame_gap_count':int(np.count_nonzero(np.diff(frames)!=1)),
        'mapped_channels':len(m), 'spike_count':len(spikes),
        'invalid_spike_records_excluded':int(np.count_nonzero(~valid)),
        'routed_spike_count':int(np.count_nonzero(valid)),
        'active_mapped_channels':len(np.unique(spikes['channel'][valid])),
        'raw_sample_windows_seconds':[[0,1],[40,41],[80,81],[110,111]],
        'lsb_uv':uv, 'selected_electrode':eligible[0]['electrode'] if eligible else None,
        'candidates':candidates, 'qualified_route_source':str(route),
        'route_sha256':hashlib.sha256(route.read_bytes()).hexdigest(),
        'selection_rule':'At least100spikes, median>=50uV, median/rawStd>=3, no sampled rails, >=10events/40s, <2% ISI<1ms; choose highest median/rawStd',
        'limits':['Signal-quality first probe only, not evidence of biological efficacy.',
                  'Raw saturation/noise checked in four distributed one-second windows, not every raw sample.']}
(base / 'qualification_qc.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='candidates'}, indent=2))
print('TOP_CANDIDATES',json.dumps(candidates[:8]))
if errors:
    raise SystemExit(2)
shutil.copyfile(route, base / 'qualified_route.cfg')
profile = json.loads((base / 'BrainStimulation/profiles/maxone_first_probe.template.json').read_text())
profile['electrodes_selection'] = {'recording':[e for c,e in recorded_pairs],
                                   'stimulation':[report['selected_electrode']]}
profile['assay_path'] = str(base / 'qualified_route.cfg')
(base / 'first_probe.json').write_text(json.dumps(profile,indent=2))
