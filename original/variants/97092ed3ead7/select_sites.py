"""Rank candidates on a current, closed quiet baseline without contacting MaxLab."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording


def candidates(rec, chip, excluded):
    if rec.metadata['chip_id']!=chip or rec.duration<60 or rec.fs!=20000:
        raise ValueError('At least one minute of current-chip baseline required')
    if any(e['time_s']<60 and e['message'].get('mode')!='baseline_no_nonzero_output' for e in rec.events):
        raise ValueError('Unidentified event in the selection interval')
    channels=sorted(set(rec.channels.tolist())-set(excluded))
    counts=[]
    for i in range(3):
        mask=(rec.spike_times>=i*20)&(rec.spike_times<(i+1)*20)
        freq=np.bincount(rec.spike_channels[mask],minlength=1024)
        counts.append(freq)
    scores={c:int(min(x[c] for x in counts)) for c in channels}
    ranked=sorted((c for c in channels if scores[c]>0),key=lambda c:(-scores[c],c))[:64]
    if len(ranked)<3:raise ValueError('Too few consistently active channels')
    ref=rec.sample(ranked,3.)
    ok=np.ones(len(ranked),dtype=bool)
    for t in (23.,43.,57.):
        ok &= stability(ref,rec.sample(ranked,t))['channel_pass']
    mapping=dict(rec.mapping)
    rows=[dict(channel=c,electrode=mapping[c],min_spikes_per_20s=scores[c])
          for c,valid in zip(ranked,ok) if valid]
    if len(rows)<3:raise ValueError('Too few stable, independently routable candidates for a routing attempt')
    return dict(chip=chip,source=rec.source,candidates=rows,suggested=dict(zip('HWF',rows[:3])),
                verified_stimulation_routes=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('closed_quiet_h5',type=Path)
    p.add_argument('--chip',required=True)
    p.add_argument('--exclude',nargs='*',type=int,default=[])
    args=p.parse_args()
    with H5Recording(args.closed_quiet_h5,SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')) as rec:
        print(json.dumps(candidates(rec,args.chip,args.exclude),indent=2))
