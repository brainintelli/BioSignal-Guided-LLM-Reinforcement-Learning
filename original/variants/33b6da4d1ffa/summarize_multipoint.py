"""Read closed recordings only; never imports a device adapter or changes QC."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
    sys.path.insert(0,str(root/'program_eventfix'))
    sys.path.insert(0,str(root/'multipoint_program'))
    from multipoint import binding
    from mea_stim_v2.recording import H5Recording
    from mea_stim_v2.qc import stability
    run=root/'multipoint_run';state=json.loads((run/'state.json').read_text())
    if 'finished_wall' not in state:raise RuntimeError('Run still owns recording; wait for completion')
    plan=json.loads((root/'multipoint_plan.json').read_text());cfg=binding(plan,plan['points'][0]);roi=plan['roi']
    out=root/'multipoint_report';out.mkdir(exist_ok=False)
    result={'scope':'posthoc_descriptive_capability_only','r2_enabled':False,'training_started':False,
      'qc_decision_unchanged':True,'status':state['status'],'errors':state['errors'],
      'cleanup_errors':state['cleanup_errors'],'reserved_true_pulses':state['reserved_true_pulses'],
      'session_id':state['session_id'],'started_wall':state['started_wall'],
      'finished_wall':state['finished_wall'],'hard_deadline_wall':state['hard_deadline_wall'],
      'training_eligibility':'quarantine_failed_recovery_QC_do_not_use_as_positive_reward_or_qualified_state',
      'records':[],'posthoc_stability':{},'roi':roi}
    arrays={}
    train=run/'point_01/train/train.raw.h5'
    with H5Recording(train,cfg) as r:
        first=r.events[0]['time_s'];end=r.events[-1]['time_s']+.0004
        before=r.sample(roi,first-3,1)
        result['posthoc_stability']['pre_vs_pre']=stability(r.sample(roi,first-10,1),before)
        trace=r.sample([plan['points'][0]['channel']],first-.05,1.5)[0]
        center=float(np.median(r.sample([plan['points'][0]['channel']],first-3,1)))
        arrays['trace_t']=np.arange(len(trace))/r.fs-.05
        arrays['trace_uv']=(trace.astype(float)-center)*r.metadata['lsb_uv']
        arrays['event_t']=np.array([e['time_s']-first for e in r.events])
        for key in ['train']:
            bins=np.arange(0,int(r.duration)+1,dtype=float)
            counts=np.histogram(r.spike_times[np.isin(r.spike_channels,roi)],bins)[0]
            arrays[key+'_rate_t']=(bins[:-1]+.5)-first;arrays[key+'_rate_hz']=counts
        result['train_first_event_s']=first;result['train_final_pulse_end_s']=end
        result['confirmed_events']=len(r.events)
    recovery=run/'point_01/recovery/recovery.raw.h5'
    if recovery.exists():
        with H5Recording(recovery,cfg) as r:
            result['recovery_metadata']=r.metadata
            bins=np.arange(0,int(r.duration)+1,dtype=float)
            arrays['recovery_rate_t']=bins[:-1]+.5
            arrays['recovery_rate_hz']=np.histogram(r.spike_times[np.isin(r.spike_channels,roi)],bins)[0]
            for t in [2,60,300,r.duration-3]:
                result['posthoc_stability']['recovery_'+str(round(t,2))+'s']=stability(before,r.sample(roi,t,1))
            result['recovery_30s_rates']=[float(np.count_nonzero(np.isin(r.spike_channels,roi)&(r.spike_times>=t)&(r.spike_times<t+30))/30) for t in range(0,int(r.duration)-29,30)]
    for record in state['records']:
        path=Path(record['path']);pre=path.stat()
        digest=sha(path);post=path.stat()
        if (pre.st_size,pre.st_mtime_ns)!=(post.st_size,post.st_mtime_ns):raise RuntimeError('Source changed')
        result['records'].append(dict(record,size_bytes=pre.st_size,sha256=digest))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'traces.npz',**arrays)
    print(json.dumps({k:result[k] for k in ['status','reserved_true_pulses','confirmed_events','training_eligibility']}))
if __name__=='__main__':main()
