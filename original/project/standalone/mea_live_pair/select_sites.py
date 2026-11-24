"""Rank candidates on a current, closed quiet baseline without contacting MaxLab."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mea_stim_v2.qc import stability
from mea_stim_v2.recording import H5Recording


def coordinates(rec):
    xy=rec.channel_xy_um
    if xy is None or set(xy)!=set(rec.channels.tolist()):
        raise ValueError('Physical H5 mapping coordinates in micrometres required')
    values=np.asarray(list(xy.values()),dtype=float)
    if values.shape!=(len(xy),2) or not np.isfinite(values).all() or len(set(xy.values()))!=len(xy):
        raise ValueError('Incomplete, nonfinite or overlapping physical coordinates')
    return xy,(values.min(axis=0)+values.max(axis=0))/2


def spatial_choice(rows,mid):
    """Search all qualified triangles; break exact ties reproducibly by activity and channel."""
    rows=sorted(rows,key=lambda r:(-r['min_spikes_per_20s'],r['channel']))
    if len(rows)<3:
        raise ValueError('At least three stable active electrodes required')
    xy=np.asarray([[r['x_um'],r['y_um']] for r in rows],dtype=float)
    region=(xy[:,0]>=mid[0]).astype(int)+2*(xy[:,1]>=mid[1]).astype(int)
    names=('upper_left','upper_right','lower_left','lower_right')
    dist2=np.square(xy[:,None]-xy[None,:]).sum(axis=2)
    activity=np.asarray([r['min_spikes_per_20s'] for r in rows])
    left,right=np.triu_indices(len(rows),1)
    best=None
    for i in range(len(rows)-2):
        first=left.searchsorted(i+1)
        j,k=left[first:],right[first:]
        coverage=1+(region[j]!=region[i]).astype(int)+((region[k]!=region[i])&(region[k]!=region[j]))
        shortest=np.minimum(np.minimum(dist2[i,j],dist2[i,k]),dist2[j,k])
        # Only materialize triples for one first vertex at a time.
        ix=np.flatnonzero(coverage==coverage.max())
        ix=ix[shortest[ix]==shortest[ix].max()]
        ix=ix[activity[k[ix]]==activity[k[ix]].max()]
        sums=activity[j[ix]]+activity[k[ix]]
        n=ix[sums.argmax()]
        score=(int(coverage[n]),float(shortest[n]),int(activity[k[n]]),int(activity[i]+sums.max()))
        if best is None or score>best:
            best=score
            indices=(i,int(j[n]),int(k[n]))
    sites={s:dict(rows[v],region=names[region[v]]) for s,v in zip('HWF',indices)}
    h,w,f=indices
    distances={name:float(np.sqrt(dist2[a,b])) for name,a,b in (('H-W',h,w),('H-F',h,f),('W-F',w,f))}
    return dict(suggested=sites,distances_um=distances,minimum_distance_um=min(distances.values()),
                region_count=best[0],quadrant_midpoint_um=list(map(float,mid)))


def candidates(rec, chip, excluded):
    if rec.metadata['chip_id']!=chip or rec.duration<60 or rec.fs!=20000:
        raise ValueError('At least one minute of current-chip baseline required')
    if any(e['time_s']<60 and e['message'].get('mode')!='baseline_no_nonzero_output' for e in rec.events):
        raise ValueError('Unidentified event in the selection interval')
    xy,mid=coordinates(rec)
    channels=sorted(set(rec.channels.tolist())-set(excluded))
    counts=[]
    for i in range(3):
        mask=(rec.spike_times>=i*20)&(rec.spike_times<(i+1)*20)
        freq=np.bincount(rec.spike_channels[mask],minlength=1024)
        counts.append(freq)
    scores={c:int(min(x[c] for x in counts)) for c in channels}
    ranked=sorted((c for c in channels if scores[c]>0),key=lambda c:(-scores[c],c))
    if len(ranked)<3:raise ValueError('Too few consistently active channels')
    mapping=dict(rec.mapping)
    rows=[]
    for offset in range(0,len(ranked),64):
        batch=ranked[offset:offset+64]
        ref=rec.sample(batch,3.)
        ok=np.ones(len(batch),dtype=bool)
        for t in (23.,43.,57.):
            ok &= stability(ref,rec.sample(batch,t))['channel_pass']
        rows.extend(dict(channel=c,electrode=mapping[c],min_spikes_per_20s=scores[c],x_um=xy[c][0],y_um=xy[c][1])
                    for c,valid in zip(batch,ok) if valid)
    return dict(chip=chip,source=rec.source,candidates=rows,**spatial_choice(rows,mid),
                verified_stimulation_routes=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('closed_quiet_h5',type=Path)
    p.add_argument('--chip',required=True)
    p.add_argument('--exclude',nargs='*',type=int,default=[])
    args=p.parse_args()
    with H5Recording(args.closed_quiet_h5,SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')) as rec:
        print(json.dumps(candidates(rec,args.chip,args.exclude),indent=2))
