"""Admission from a closed MaxOne segment; marker ACKs alone cannot earn reward."""
import numpy as np
from mea_state2_r2.features import digest, extract
from mea_stim_v2.qc import stability
from .protocol import RATE, window


def read_window(rec, start, roi, bid, origin):
    if start < rec.first_frame or start+RATE-1 > rec.last_frame:
        raise ValueError('Incomplete state2 window')
    # The four registered features need timestamps and channel IDs only.
    frames = np.rint(rec.spike_times*RATE).astype(np.int64)+rec.first_frame
    keep = (frames>=start)&(frames<start+RATE)&np.isin(rec.spike_channels,roi)
    events=sorted((int(f),int(c),0.) for f,c in zip(frames[keep],rec.spike_channels[keep]))
    return window(bid,start,extract(events,start,start+RATE,roi,RATE),origin)


def checked_metadata(rec,b,mapping):
    expected=dict(chip_id=b['chip_id'],fs=20000.,gain=512.,hpf=300.,frame_gaps=0,
                  threshold_value=b['threshold_value'],outside_frame_coverage=0)
    if any(rec.metadata[k]!=v for k,v in expected.items()) or sorted(rec.mapping)!=mapping:
        raise ValueError('Measured chip, route, frame or detector configuration differs')


def admit(rec,req,event_id,roi,site_channels):
    if set(site_channels)!=set('HWF') or set(roi)&set(site_channels.values()) or len(set(roi))<50:
        raise ValueError('Independent fixed ROI required')
    n=req['pulses']
    ev=rec.events
    if n not in (0,3) or len(ev)!=max(1,n):
        raise ValueError('Missing, duplicate or extra event')
    anchor=ev[0]['frame']
    for i,e in enumerate(ev):
        if (e['eventtype']!=1 or e['id']!=event_id+i or e['message'].get('tx')!=req['id']
                or int(e['message']['idx'])!=i or e['frame']!=anchor+200*i):
            raise ValueError('Command/event alignment failed')
    t=(anchor-rec.first_frame)/RATE
    last=t+(408/RATE if n else 0)
    before=rec.sample(roi,t-3.)
    recovery=[stability(before,rec.sample(roi,s))['fraction_channels_passed'] for s in (last+2,last+10,t+12)]
    if min(recovery)<.95:
        raise ValueError('Artifact/noise recovery failed')
    # A threshold crossing must start afresh at each onset, so a lingering rail
    # from pulse 1 cannot qualify pulses 2 and 3. Quiet checks all three sites.
    checked=[site_channels[req['site']]] if n else list(site_channels.values())
    med=np.median(rec.sample(checked,t-3.),axis=1,keepdims=True)
    trace=rec.sample(checked,t-.003,.604)
    high=np.abs(trace.astype(float)-med)*rec.metadata['lsb_uv']>=1000
    if n:
        for i in range(n):
            at=60+200*i
            if high[:,at-8:at].any() or not high[:,at:at+40].any():
                raise ValueError('Distinct pulse onset absent from current route readback')
        if np.any((trace[:,10060:]==0)|(trace[:,10060:]==1023)):
            raise ValueError('Persistent output rail')
    elif high.any() or np.any((trace==0)|(trace==1023)):
        raise ValueError('Output during quiet interval')
    w=read_window(rec,anchor+240000,roi,req['binding'],req['id'])
    r=dict(status='accepted',request=req,frames=[e['frame'] for e in ev] if n else [],
           observed='output' if n else 'quiet',window=w,anchor=anchor,recovery=recovery,
           file=rec.source,coverage=[rec.first_frame,rec.last_frame+1])
    if n and req['phase']=='qualification':
        r['pre_window']=read_window(rec,anchor-3*RATE,roi,req['binding'],req['id'])
    return dict(r,id=digest(r))
