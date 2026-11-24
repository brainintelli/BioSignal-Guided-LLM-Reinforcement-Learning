"""Read-only exploratory before/after spike counts; no device or training access."""
import csv,json
from pathlib import Path
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
src=r/'backup/recording_resume2';out=r/'single_site_response';out.mkdir(exist_ok=True)
fs=20000
def read(name):
    with h5py.File(src/name,'r') as f:
        d=f['data_store/data0000'];fr=d['groups/routed/frame_nos']
        mapping=sorted((int(x['channel']),int(x['electrode'])) for x in d['settings/mapping'][:])
        return dict(first=int(fr[0]),end=int(fr[-1])+1,spikes=d['spikes'][:],events=d['events'][:],mapping=mapping)
b=read('baseline.raw.h5');p=read('stimulus_post.raw.h5')
assert b['mapping']==p['mapping'] and len(p['events'])==15
assert list(p['events']['eventid'])==list(range(2,17))
channels=np.array([c for c,e in b['mapping'] if c!=640]);n=len(channels)
assert n==804
t0=int(p['events'][0]['frameno']);tail=int(p['events'][-1]['frameno'])+8
for d in [b,p]:
    s=d['spikes'];ok=np.isin(s['channel'],channels)&(s['frameno']>=d['first'])&(s['frameno']<d['end'])
    d['selected']=s[ok]

def metric(d,lo,hi,label):
    assert d['first']<=lo<hi<=d['end']
    s=d['selected'];s=s[(s['frameno']>=lo)&(s['frameno']<hi)]
    counts=np.array([(s['channel']==c).sum() for c in channels]);duration=(hi-lo)/fs
    return dict(period=label,seconds=duration,start_relative_first_stim_s=(lo-t0)/fs,end_relative_first_stim_s=(hi-t0)/fs,
                spike_count=int(len(s)),population_spikes_per_s=len(s)/duration,rate_hz_per_channel=len(s)/duration/n,
                active_channels_at_least_one=int((counts>0).sum()),counts=counts.tolist())

windows=[(b,b['end']-600*fs,b['end'],'baseline_600s'),
         (b,b['end']-600*fs,b['end']-300*fs,'baseline_first_300s'),
         (b,b['end']-300*fs,b['end'],'baseline_last_300s'),
         (p,t0-60*fs,t0,'immediate_pre_60s'),
         (p,tail+10*fs,tail+70*fs,'post_10_to_70s'),
         (p,tail+10*fs,tail+610*fs,'post_10_to_610s'),
         (p,tail+610*fs,tail+1210*fs,'post_610_to_1210s'),
         (p,tail+1210*fs,min(p['end'],tail+1800*fs),'post_1210_to_1800s')]
rows=[metric(*w) for w in windows]
baseline=rows[0]['rate_hz_per_channel'];pre=rows[3]['rate_hz_per_channel']
for q in rows:
    q['change_vs_baseline_pct']=100*(q['rate_hz_per_channel']/baseline-1)
    q['change_vs_immediate_pre_pct']=100*(q['rate_hz_per_channel']/pre-1)
sensitivity=[metric(p,tail+sec*fs,p['end'],f'post_exclude_first_{sec}s') for sec in (2,10,30,60)]
for q in sensitivity:
    q['change_vs_immediate_pre_pct']=100*(q['rate_hz_per_channel']/pre-1)
    q['change_vs_baseline_pct']=100*(q['rate_hz_per_channel']/baseline-1)
result=dict(status='exploratory_descriptive_only',chip_id='28080',stim_electrode=8858,excluded_stim_channel=640,
    independent_channels=n,stimulus_pulses=15,first_pulse_frame=t0,last_pulse_tail_frame=tail,
    baseline_end_to_stim_s=(t0-b['end'])/fs,post_coverage_s=(p['end']-tail)/fs,
    windows=rows,artifact_exclusion_sensitivity=sensitivity,alpha=1,r2_mode='record_only',training_performed=False,
    limitations=['One organoid, one single-site exposure, no sham/randomization/biological replicates.',
      'Baseline is separated in time; immediate pre-stim recording is only 60 seconds.',
      'Excludes stimulation channel and initial10seconds; spike summaries are not a complete evoked-response artifact analysis.',
      'Channels and time bins are not independent biological replicates; no causal effect or training improvement claim.'])
(out/'metrics.json').write_text(json.dumps(result,indent=2))
flat=[{k:v for k,v in q.items() if k!='counts'} for q in rows]
with (out/'windows.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True,gridspec_kw={'width_ratios':[1,2.7]})
for d,ax,label in [(b,axes[0],'Earlier baseline'),(p,axes[1],'Single-site stimulus and follow-up')]:
    lo=d['end']-600*fs if d is b else t0-60*fs
    edges=np.arange(lo,d['end']+1,10*fs,dtype=np.int64)
    # Plot only complete10s bins; a final partial bin has different variance.
    count,_=np.histogram(d['selected']['frameno'],bins=edges)
    rate=count/(np.diff(edges)/fs)/n;x=((edges[:-1]+edges[1:])/2-t0)/fs/60
    if d is p:rate=rate.astype(float);rate[(edges[:-1]<tail+10*fs)&(edges[1:]>t0)]=np.nan
    ax.plot(x,rate,color='#2878B5',linewidth=1);ax.axhline(baseline,color='#999999',linestyle='--',label='Earlier baseline mean')
    ax.axhline(pre,color='#E38C2B',linestyle=':',label='Immediate pre-stim mean')
    ax.set(title=label,xlabel='Minutes from first stimulus');ax.grid(alpha=.2)
axes[0].set_ylabel('Detected spikes / second / channel (804 channels)')
axes[0].set_ylim(0,.30)
axes[1].axvline(0,color='#D44C47',label='15-pulse stimulus');axes[1].legend(fontsize=8)
fig.suptitle('Chip 28080: descriptive before/after comparison; no sham control')
fig.tight_layout();fig.savefig(out/'spike_rate_comparison.png',dpi=180);plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k not in ('windows','artifact_exclusion_sensitivity')},indent=2))
print(json.dumps(flat,indent=2))
