"""Closed-file engineering gates. These are not tissue-safety certification."""
from pathlib import Path
import re
import h5py
import numpy as np


def parse_mapping(text):
    head=text.split()[0] if text.split() else ''
    pairs=sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',head))
    if (len(pairs)<2 or len({a for a,b in pairs})!=len(pairs)
        or len({b for a,b in pairs})!=len(pairs)
        or any(not 0<=a<1024 or not 0<=b<26400 for a,b in pairs)):
        raise ValueError('Invalid or duplicate routing mapping')
    return pairs


def check_events(events,phase):
    if phase=='identity': return ['unexpected_identity_events'] if events else []
    if phase=='baseline':
        return [] if len(events)==1 and events[0]['id']==7001 and events[0]['type']==1 else ['baseline_event']
    if phase!='post': raise ValueError('Unknown audit phase')
    if len(events)!=15 or [(x['type'],x['id']) for x in events]!=[(1,x) for x in range(2,17)]:
        return ['pulse_event_ids']
    expected=[b*4000+p*200 for b in range(5) for p in range(3)]
    return [] if [x['frame']-events[0]['frame'] for x in events]==expected else ['pulse_frame_intervals']


def drift_rejected(rates):
    rates=np.asarray(rates,dtype=float)
    if len(rates)!=10 or not np.all(np.isfinite(rates)): return True
    reference=float(np.median(rates[:5]))
    if reference<=0:return True
    outside=abs(rates[5:]/reference-1)>.5
    return bool(np.any(outside[1:] & outside[:-1]))


def inspect_h5(path,chip_id,expected_mapping,expected_seconds,*,selected=None,phase='identity',heartbeat=None):
    errors=[]
    with h5py.File(path,'r') as f:
        d=f['data_store/data0000']; g=d['groups/routed']
        chip=f['wellplate/id'][0].decode();model=f['wellplate/version'][0].decode()
        mapping=sorted((int(x['channel']),int(x['electrode'])) for x in d['settings/mapping'][:])
        if chip!=chip_id or model!='MaxOne Single Well MEA':errors.append('identity')
        if mapping!=expected_mapping:errors.append('mapping')
        if len({x[0] for x in mapping})!=len(mapping) or len({x[1] for x in mapping})!=len(mapping):
            raise ValueError('Duplicate H5 mapping')
        fs=float(d['settings/sampling'][0]); frames=g['frame_nos']; raw=g['raw'];n=len(frames)
        if fs!=20000 or n==0:raise ValueError('Missing frames or non-20kHz recording')
        if raw.shape!=(1024,n):raise ValueError('Raw shape does not match full explicit channel group')
        if raw.dtype.kind not in 'ui':raise ValueError('Raw ADC must use integer samples')
        if 'channels' not in g or not np.array_equal(g['channels'][:].reshape(-1),np.arange(1024)):
            raise ValueError('Raw row/channel order is not explicit 0..1023')
        duration=n/fs
        if abs(duration-expected_seconds)>3:errors.append('duration')
        channels=np.array([a for a,b in expected_mapping],dtype=int)
        if not len(channels) or np.any(channels<0) or np.any(channels>1023):raise ValueError('Invalid channels')
        rails=np.zeros(len(channels),dtype='i8'); gaps=0; previous=None; rail_seconds=[]
        for lo in range(0,n,20000):
            if heartbeat:heartbeat()
            hi=min(lo+20000,n); block=raw[channels,lo:hi]
            if np.any(block<0) or np.any(block>1023):raise ValueError('ADC sample outside 10-bit range')
            counts=((block==0)|(block>=1023)).sum(axis=1);rails+=counts
            if counts.any():rail_seconds.append(dict(start_s=lo/fs,end_s=hi/fs,max_channel_fraction=float(max(counts)/(hi-lo))))
            frame=frames[lo:hi].astype('i8')
            gaps+=int(np.count_nonzero(np.diff(frame)!=1))
            if previous is not None and frame[0]!=previous+1:gaps+=1
            previous=int(frame[-1])
        if gaps:errors.append('frame_gaps')
        start=int(frames[0]);last=int(frames[-1]);sp=d['spikes'][:]
        st=(sp['frameno'].astype('i8')-start)/fs
        valid=(st>=0)&(st<duration)
        selected_ch=None if selected is None or phase=='identity' else selected['channel']
        independent=[int(ch) for ch in channels if ch!=selected_ch]
        independent_indices=[i for i,ch in enumerate(channels) if ch!=selected_ch]
        if not independent:raise ValueError('No independent recording channels')
        valid_ind=valid & np.isin(sp['channel'],independent)
        active=len(np.unique(sp['channel'][valid_ind]));fraction=rails/n
        if phase!='post':
            if float(fraction[independent_indices].mean())>.001 or float(fraction.max())>.01:
                errors.append('saturation')
            if any(x['max_channel_fraction']>.01 for x in rail_seconds):errors.append('rail_excursion')
            if active<len(independent)*.5 or not np.count_nonzero(valid_ind):errors.append('activity')
        candidates=[]
        if phase=='identity':
            for i,(ch,electrode) in enumerate(expected_mapping):
                times=np.sort(st[valid & (sp['channel']==ch)])
                if len(times)>=10 and fraction[i]<=.01 and float(np.mean(np.diff(times)<.001))<.02:
                    candidates.append((-len(times),electrode,ch))
            candidates.sort()
            if not candidates:errors.append('candidate')
        events=[dict(frame=int(x['frameno']),type=int(x['eventtype']),id=int(x['eventid'])) for x in d['events'][:]]
        errors.extend(check_events(events,phase))
        if any(x['frame']<start or x['frame']>last for x in events):errors.append('event_outside_raw')
        rates=np.histogram(st[valid_ind],bins=np.arange(0,601,60))[0]/60
        if phase=='baseline' and drift_rejected(rates):errors.append('baseline_drift')
        if phase=='post' and events and (last-events[-1]['frame']-8+1)/fs<1800:
            errors.append('incomplete_30min_after_last_pulse')
        report=dict(status='passed' if not errors else 'failed',errors=errors,source=str(Path(path)),
                    chip_id=chip,model=model,duration_s=duration,mapping=[list(x) for x in mapping],
                    frame_gaps=gaps,raw_scan='all_samples_of_mapped_channels',
                    max_rail_fraction=float(fraction.max()),rail_seconds=rail_seconds,
                    active_independent_channels=active,independent_channels=len(independent),
                    spike_count=int(np.count_nonzero(valid_ind)),
                    rate_hz_per_channel=float(np.count_nonzero(valid_ind)/duration/len(independent)),
                    active_channel_fraction=active/len(independent),
                    events=events,first_frame=start,last_frame=last,
                    minute_rates_hz=rates.tolist() if phase=='baseline' else None,
                    r2_mode='record_only',training_admitted=False,
                    limitation='Engineering recording gate only; event timing does not measure physical voltage or tissue safety.')
        if candidates:report['candidate']=dict(channel=candidates[0][2],electrode=candidates[0][1])
        if phase=='identity' and selected is not None:
            eligible=any(ch==selected['channel'] and e==selected['electrode'] for count,e,ch in candidates)
            if eligible:report['candidate']=dict(selected)
            else:report['errors'].append('frozen_candidate');report['status']='failed'
        return report
