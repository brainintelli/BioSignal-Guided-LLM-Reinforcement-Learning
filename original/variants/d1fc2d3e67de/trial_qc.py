"""Frozen recording gates for this one 22021 photo-parameter trial."""
import os
os.environ.setdefault('HDF5_PLUGIN_PATH', '/pub/soft/MaxLab/so')
import json
import re
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/home/researcher/mea-acquisition/live_22021_20261117_032840/baseline_600s/baseline.raw.h5')


def mapping(text):
    return sorted((int(a), int(b)) for a, b in re.findall(r'(\d+)\((\d+)\)', text.split()[0]))


def reference_rates(channels):
    with h5py.File(REFERENCE, 'r') as f:
        d = f['data_store/data0000']
        g = next(iter(d['groups'].values()))
        frames = g['frame_nos']
        s = d['spikes'][:]
        fs = float(d['settings/sampling'][0])
        t = (s['frameno'].astype(np.int64) - int(frames[0])) / fs
        valid = np.isin(s['channel'], channels) & (t >= 0) & (t < 540)
        return np.histogram(t[valid], bins=np.arange(0, 541, 60))[0] / 60


def inspect(path, expected_seconds, *, connected=False):
    expected = mapping((ROOT / 'qualified_route.cfg').read_text())
    errors = []
    with h5py.File(path, 'r') as f:
        d = f['data_store/data0000']
        m = sorted((int(x['channel']), int(x['electrode'])) for x in d['settings/mapping'][:])
        fs = float(d['settings/sampling'][0])
        g = d['groups/routed']
        frames = g['frame_nos'][:]
        raw = g['raw']
        channels = g['channels'][:] if 'channels' in g else None
        # This owned group explicitly requested channel IDs 0..1023. Do not
        # infer arbitrary compact row order when a dataset has a different shape.
        if raw.shape[0] != 1024:
            raise RuntimeError('Owned raw group does not contain 1024 explicit rows')
        if channels is not None and not np.array_equal(channels.reshape(-1), np.arange(1024)):
            raise RuntimeError('Unexpected raw row/channel order')
        duration = len(frames) / fs
        selected = dict((e, ch) for ch, e in m).get(7587)
        if selected is None:
            raise RuntimeError('Frozen candidate missing')
        independent = [ch for ch, e in expected if e != 7587]
        times = [float(t) for t in range(1, int(duration)-1, 60)]
        if len(times) == 1:
            times = [1, 20, 40, max(1, int(duration)-2)]
        sample = np.concatenate([raw[:, int(t*fs):int((t+1)*fs)] for t in times], axis=1)
        rails = ((sample == 0) | (sample == 1023)).mean(axis=1)
        s = d['spikes'][:]
        t = (s['frameno'].astype(np.int64) - int(frames[0])) / fs
        valid = np.isin(s['channel'], independent) & (t >= 0) & (t < duration)
        bins = int((duration + .01) // 60)
        rates = np.histogram(t[valid], bins=np.arange(bins+1)*60)[0] / 60 if bins else np.array([])
        ref = reference_rates(independent)
        outside = (abs(rates / np.mean(ref) - 1) > .5) & ((rates < min(ref)) | (rates > max(ref)))
        selected_valid = (s['channel'] == selected) & (t >= 0) & (t < duration)
        st = np.sort(t[selected_valid])
        selected_count = int(np.count_nonzero(selected_valid))
        active = len(np.unique(s['channel'][valid]))
        events = d['events'][:]
        ev = [{'frame':int(x['frameno']), 'type':int(x['eventtype']), 'id':int(x['eventid']),
               'message':x['eventmessage'].decode(errors='replace')} for x in events]
        chip = f['wellplate/id'][0].decode()
        model = f['wellplate/version'][0].decode()
        if chip != '22021' or model != 'MaxOne Single Well MEA': errors.append('identity')
        if m != expected: errors.append('actual_h5_mapping')
        if fs != 20000 or abs(duration-expected_seconds) > 3: errors.append('duration_or_sampling')
        if np.any(np.diff(frames) != 1): errors.append('frame_gaps')
        if np.mean(rails[independent]) > .001 or max(rails[independent]) > .01: errors.append('independent_saturation')
        if rails[selected] > .01: errors.append('selected_saturation')
        if active < len(independent)*.5 or np.count_nonzero(valid) == 0: errors.append('insufficient_independent_activity')
        if connected:
            if np.any(outside[:-1] & outside[1:]): errors.append('two_consecutive_minutes_drift')
            if len(ev)!=1 or ev[0]['id']!=7001 or ev[0]['type']!=1: errors.append('event_marker_missing_or_unexpected')
        else:
            if selected_count < 10: errors.append('candidate_not_currently_active')
            if len(st)>1 and np.mean(np.diff(st)<.001) >= .02: errors.append('candidate_short_isi_fraction')
            if len(ev): errors.append('unexpected_identity_events')
        report = {'status':'passed' if not errors else 'failed','errors':errors,'source':str(path),
            'chip_id':chip,'model':model,'duration_s':duration,'mapped_channels':len(m),
            'selected_channel':selected,'selected_electrode':7587,'selected_count':selected_count,
            'independent_channels':len(independent),'active_independent_channels':active,
            'mapped_independent_spikes':int(np.count_nonzero(valid)),
            'sampled_raw_seconds':times,'sampled_independent_mean_rail_fraction':float(np.mean(rails[independent])),
            'sampled_independent_max_rail_fraction':float(max(rails[independent])),
            'sampled_selected_rail_fraction':float(rails[selected]),
            'minute_rates_hz':rates.tolist(),'reference_minute_rates_hz':ref.tolist(),
            'minute_outside_reference_and_50pct':outside.tolist(),'events':ev,
            'limitations':['Sampled saturation gates are recording-quality checks, not biological safety proof.',
                          'Stimulus electrode excluded from independent recording gates after connection.']}
    return report


if __name__ == '__main__':
    result = inspect(ROOT/'identity_60s/qualification.raw.h5',60)
    (ROOT/'identity_qc.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 2)
