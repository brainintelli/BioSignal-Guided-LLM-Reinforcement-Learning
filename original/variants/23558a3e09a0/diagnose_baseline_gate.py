"""Read only closed H5 files after the no-output qualification failure."""
import os
os.environ.setdefault('HDF5_PLUGIN_PATH','/pub/soft/MaxLab/so')
import json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'progress.json').read_text())
assert not p['stimulus_attempted'] and not p['stimulation_sent']
with h5py.File(p['baseline_file'],'r') as f:
    d=f['data_store/data0000'];g=d['groups/routed'];raw=g['raw']
    fs=int(d['settings/sampling'][0]);frames=g['frame_nos'][:]
    mapping={int(x['channel']):int(x['electrode']) for x in d['settings/mapping'][:]}
    independent=[ch for ch,e in mapping.items() if e!=7587]
    sample_times=list(range(1,int(len(frames)/fs)-1,60))
    per_window=[]
    for start in sample_times:
        x=raw[:,start*fs:(start+1)*fs]
        fraction=((x==0)|(x==1023)).mean(axis=1)
        per_window.append(fraction)
    values=np.asarray(per_window)
    bad=[ch for ch in independent if values[:,ch].mean()>.01]
    summaries=[]
    # Scan every raw sample of the failing channels; retain per-second counts.
    counts={ch:[] for ch in bad};first={ch:None for ch in bad};last={ch:None for ch in bad}
    for start in range(0,raw.shape[1],fs*5):
        stop=min(start+fs*5,raw.shape[1])
        x=raw[bad,start:stop] if bad else np.empty((0,stop-start))
        for i,ch in enumerate(bad):
            rail=(x[i]==0)|(x[i]==1023);locations=np.flatnonzero(rail)
            if len(locations):
                if first[ch] is None:first[ch]=int(start+locations[0])
                last[ch]=int(start+locations[-1])
            for offset in range(0,len(rail),fs):counts[ch].append(int(np.count_nonzero(rail[offset:offset+fs])))
    for ch in bad:
        summaries.append({'channel':ch,'electrode':mapping[ch],
            'sampled_mean_rail_fraction':float(values[:,ch].mean()),
            'sampled_window_rail_fractions':values[:,ch].tolist(),
            'full_record_rail_samples':sum(counts[ch]),'full_record_rail_fraction':sum(counts[ch])/len(frames),
            'first_rail_s':first[ch]/fs if first[ch] is not None else None,
            'last_rail_s':last[ch]/fs if last[ch] is not None else None,
            'rail_samples_per_second':counts[ch]})
    diag={'status':'qualification_failed_no_stimulation','source':p['baseline_file'],
          'sampling_hz':fs,'duration_s':len(frames)/fs,'frame_gaps':int(np.count_nonzero(np.diff(frames)!=1)),
          'sample_times_s':sample_times,'failing_channels':summaries,
          'original_gate_unchanged':True,'stimulation_sent':False,'model_training_performed':False}
(ROOT/'gate_diagnosis.json').write_text(json.dumps(diag,indent=2))
print(json.dumps({**diag,'failing_channels':[{k:v for k,v in x.items() if k!='rail_samples_per_second'} for x in summaries]},indent=2))
