"""Descriptive analysis only; never admits stimulated data to training."""
import os
os.environ.setdefault('HDF5_PLUGIN_PATH','/pub/soft/MaxLab/so')
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent


def open_data(path):
    f=h5py.File(path,'r')
    d=f['data_store/data0000'];g=d['groups/routed']
    frames=g['frame_nos'][:]
    fs=float(d['settings/sampling'][0])
    mapping=sorted((int(x['channel']),int(x['electrode'])) for x in d['settings/mapping'][:])
    s=d['spikes'][:]
    t=(s['frameno'].astype(np.int64)-int(frames[0]))/fs
    channels=[ch for ch,e in mapping if e!=7587]
    valid=np.isin(s['channel'],channels)&(t>=0)&(t<len(frames)/fs)
    ev=[{'frame':int(x['frameno']),'type':int(x['eventtype']),'id':int(x['eventid']),
         'message':x['eventmessage'].decode(errors='replace')} for x in d['events'][:]]
    info={'chip_id':f['wellplate/id'][0].decode(),'model':f['wellplate/version'][0].decode(),
          'duration_s':len(frames)/fs,'first_frame':int(frames[0]),'last_frame':int(frames[-1]),
          'frame_gap_count':int(np.count_nonzero(np.diff(frames)!=1)),
          'mapped_channels':len(mapping),'independent_channels':len(channels),
          'total_spike_records':len(s),'excluded_unmapped_or_outside_or_stimsite':int(np.count_nonzero(~valid)),
          'events':ev}
    return f,d,g,frames,fs,mapping,s,t,valid,float(d['settings/lsb'][0])*1e6,info


def metrics(s,t,valid,lo,hi,n,uv):
    mask=valid&(t>=lo)&(t<hi)
    count=int(np.count_nonzero(mask))
    return {'start_s':lo,'end_s':hi,'duration_s':hi-lo,'spike_count':count,
            'population_rate_hz':count/(hi-lo),'rate_hz_per_channel':count/(hi-lo)/n,
            'active_channels':len(np.unique(s['channel'][mask])),
            'active_channel_fraction':len(np.unique(s['channel'][mask]))/n,
            'mean_abs_amplitude_uv':float(np.mean(abs(s['amplitude'][mask]))*uv) if count else None}


def main():
    progress=json.loads((ROOT/'progress.json').read_text())
    if progress['status']!='completed' or not progress['stimulation_sent']:
        raise RuntimeError('Completed stimulated trial required; do not invent an effect for a failed gate')
    b=open_data(progress['baseline_file']);p=open_data(progress['main_file'])
    errors=[]
    try:
        bf,bd,bg,bframes,bfs,bmap,bs,bt,bvalid,buv,binfo=b
        pf,pd,pg,pframes,pfs,pmap,ps,pt,pvalid,puv,pinfo=p
        for info in [binfo,pinfo]:
            if info['chip_id']!='22021' or info['frame_gap_count']!=0:errors.append('identity_or_frames')
        if bmap!=pmap or bfs!=pfs or buv!=puv:errors.append('mapping_or_acquisition_settings_changed')
        events=sorted([x for x in pinfo['events'] if x['type']==1 and 2<=x['id']<=16],key=lambda x:x['id'])
        if len(events)!=15 or [x['id'] for x in events]!=list(range(2,17)):
            raise RuntimeError('Expected 15 original H5 stimulus events; diagnostic reconstruction is separate')
        eframes=np.array([x['frame'] for x in events],dtype=np.int64)
        expected=np.array([b*4000+j*200 for b in range(5) for j in range(3)],dtype=np.int64)
        if not np.array_equal(eframes-eframes[0],expected):errors.append('event_onset_timing')
        first=(eframes[0]-int(pframes[0]))/pfs
        last=(eframes[-1]-int(pframes[0]))/pfs
        n=binfo['independent_channels']
        windows={
            'connected_baseline_600s':metrics(bs,bt,bvalid,0,min(600,binfo['duration_s']),n,buv),
            'main_pre_60s':metrics(ps,pt,pvalid,max(0,first-60),first,n,puv),
            'post_60_to_660s':metrics(ps,pt,pvalid,last+60,last+660,n,puv),
            'final_recovery_600s':metrics(ps,pt,pvalid,pinfo['duration_s']-600,pinfo['duration_s'],n,puv)}
        if pinfo['duration_s']-last<1799:errors.append('incomplete_30_minute_post_recording')
        # Examine waveform clipping around the whole train. Dense pulses can
        # overlap; do not manufacture 15 independent artifact or recovery estimates.
        raw=pg['raw'];channels=[ch for ch,e in pmap if e!=7587];stim=dict((e,ch) for ch,e in pmap)[7587]
        start=int(first*pfs);end=min(raw.shape[1],int((last+2)*pfs))
        near=raw[:,start:end]
        rails=(near==0)|(near==1023)
        any_rail=np.any(rails[channels],axis=0)
        positions=np.flatnonzero(any_rail)
        stim_positions=np.flatnonzero(rails[stim])
        late=[]
        for sec in [last+60,last+600,max(0,pinfo['duration_s']-2)]:
            sample=raw[:,int(sec*pfs):int((sec+1)*pfs)]
            rail=((sample==0)|(sample==1023)).mean(axis=1)
            late.append({'start_s':sec,'independent_mean_rail_fraction':float(np.mean(rail[channels])),
                         'independent_max_rail_fraction':float(max(rail[channels])),
                         'stimulus_site_rail_fraction':float(rail[stim])})
        rows=[]
        baseline_offset=(int(bframes[0])-int(eframes[0]))/bfs
        for label,s,t,v,uv,info,offset in [('baseline',bs,bt,bvalid,buv,binfo,baseline_offset),('main',ps,pt,pvalid,puv,pinfo,-first)]:
            for lo in np.arange(0,info['duration_s']-59,60):
                m=metrics(s,t,v,float(lo),float(lo+60),n,uv)
                rows.append({'source':label,'relative_to_first_stimulus_s':float(lo+offset),**m})
        with (ROOT/'minute_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        before=windows['connected_baseline_600s']['population_rate_hz']
        after=windows['post_60_to_660s']['population_rate_hz']
        result={'status':'descriptive_recording_audit_passed' if not errors else 'audit_failed',
                'errors':errors,'baseline':binfo,'main':pinfo,'windows':windows,
                'stimulus':{'first_onset_in_main_s':first,'last_onset_in_main_s':last,
                            'event_relative_ms':((eframes-eframes[0])/pfs*1000).tolist(),
                            'expected_events':15,'found_events':len(events),'target_mv':600,
                            'nominal_mv':599.362101,'physical_output_voltage_measured':False},
                'whole_train_artifact':{'window_end_after_last_s':2,
                    'independent_channels_hitting_rails':int(np.count_nonzero(np.any(rails[channels],axis=1))),
                    'last_sampled_independent_rail_after_first_ms':float(positions[-1]/pfs*1000) if len(positions) else None,
                    'last_sampled_stimsite_rail_after_first_ms':float(stim_positions[-1]/pfs*1000) if len(stim_positions) else None,
                    'late_samples':late,'rail_end_is_not_full_artifact_recovery':True},
                'population_rate_change_percent':(after/before-1)*100 if before else None,
                'r2_reward_enabled':False,'alpha':1,'realtime_training':False,
                'admitted_to_training':False,'model_effect_evaluated':False,
                'limitations':['One organoid, no randomized sham; no causal stimulation or model-improvement claim.',
                    'Dense train artifacts overlap; no claim of 15 independent evoked responses.',
                    'R2 telemetry only; original unstimulated admission path must not be used.',
                    'Raw clipping is not physical electrode voltage or tissue safety measurement.']}
        (ROOT/'descriptive_analysis.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'status':result['status'],'errors':errors,'events':len(events),
                          'rate_change_percent':result['population_rate_change_percent'],'windows':windows},indent=2))
    finally:b[0].close();p[0].close()


if __name__=='__main__':main()
