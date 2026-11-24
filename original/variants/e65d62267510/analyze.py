"""Read closed files only. No MaxLab import, no device command, no gate writes."""
import os
os.environ['HDF5_PLUGIN_PATH']='/pub/soft/MaxLab/so'
from pathlib import Path
import json,hashlib,re,sys,csv
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_math import trial_response,paired_summary,rate_bins,completion_issues,covered_rate
from qc_math import recovery_gate
from protocol import make_control_schedule

B=Path(__file__).resolve().parent;stage=sys.argv[1]
assert stage in ('pilot','control')
state=json.loads((B/'state.json').read_text())
assert (stage=='pilot' and state['status'] in ('awaiting_pilot_qc','completed','stopped')) or (stage=='control' and state['status'] in ('completed','stopped'))
baseline=json.loads((B/'baseline_qc.json').read_text());roi=np.array(baseline['roi_channels'],dtype=int)
source=B/stage/(stage+'.raw.h5');out=B/'analysis';out.mkdir(exist_ok=True)
schedule=make_control_schedule() if stage=='control' else [{'event_id':8201,'arm':'sham','pair':0},{'event_id':8202,'arm':'true','pair':0}]
issues=completion_issues(state,baseline,stage)
receipt=json.loads((B/'gate_baseline.json').read_text())
if receipt['qc_sha256']!=hashlib.sha256((B/'baseline_qc.json').read_bytes()).hexdigest():issues.append('baseline_receipt_hash_mismatch')
with h5py.File(source,'r') as f:
 d=f['data_store/data0000'];mapping=d['settings/mapping'][:];frames=d['groups/routed/frame_nos'][:];ev=d['events'][:];sp=d['spikes'][:]
 raw=d['groups/routed/raw'];fs=float(d['settings/sampling'][0]);uv=float(d['settings/lsb'][0])*1e6;duration=len(frames)/fs
 tt=(sp['frameno'].astype(float)-int(frames[0]))/fs
 good=np.isin(sp['channel'],mapping['channel'])&(tt>=0)&(tt<duration);s=sp[good];tt=tt[good];rt=tt[np.isin(s['channel'],roi)]
 et=(ev['frameno'].astype(float)-int(frames[0]))/fs
 expected=sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',(B/'qualified_route.cfg').read_text().split()[0]))
 if sorted((int(x['channel']),int(x['electrode'])) for x in mapping)!=expected:issues.append('mapping_mismatch')
 if f['wellplate/id'][0].decode()!='22554' or fs!=20000:issues.append('identity_sampling_mismatch')
 if np.any(np.diff(frames)!=1):issues.append('frame_gaps')
 messages=[json.loads(x['eventmessage'].decode()) for x in ev]
 if ev['eventid'].tolist()!=[x['event_id'] for x in schedule] or ev['eventtype'].tolist()!=[1]*len(schedule):issues.append('event_count_or_identity')
 if [x.get('arm') for x in messages]!=[x['arm'] for x in schedule]:issues.append('event_arm_labels')
 if len(ev)!=len(schedule):raise RuntimeError('Cannot align events; no trial analysis')
 if np.any(np.diff(ev['frameno'].astype(np.int64))!=900000):issues.append('event_spacing')
 if duration-et[-1]<599:issues.append('less_than_599s_final_recording')
 def sample(t,seconds=1):return raw[:,int(round(t*fs)):int(round((t+seconds)*fs))]
 trials=[];trace_t=np.arange(-1000,12000)/fs;traces=[];recovery=[]
 for i,(row,t) in enumerate(zip(schedule,et)):
  before=sample(t-3);post=np.stack([sample(t+2),sample(t+10)])
  rv=recovery_gate(before[roi].std(axis=1),post[:,roi].std(axis=2),np.mean((post[:,roi]==0)|(post[:,roi]==1023),axis=2))
  if not rv['passed']:issues.append('trial_%s_ROI_recovery_failed'%(i+1))
  ix=int(round(t*fs));a=raw[:,ix-1000:ix+12000];y=a[213].astype(float);center=np.median(y[trace_t<-.005]);v=(y-center)*uv
  rails=((y==0)|(y==1023))&(trace_t>=0);last=float(trace_t[rails][-1]) if rails.any() else None
  cross=np.flatnonzero((np.abs(v)>=1000)&(trace_t>=0));onset=float(trace_t[cross[0]]) if len(cross) else None
  if row['arm']=='true' and (onset is None or onset>.01 or (last is not None and last>.5)):issues.append('trial_%s_true_artifact_not_valid'%(i+1))
  if row['arm']=='sham' and (rails.any() or onset is not None):issues.append('trial_%s_sham_artifact'%(i+1))
  response=trial_response(tt,s['channel'],roi,t)
  trials.append({'trial':i+1,'event_id':int(ev[i]['eventid']),'event_seconds':float(t),'arm':row['arm'],'pair':row['pair'],**response})
  traces.append(v);recovery.append({'trial':i+1,'arm':row['arm'],'roi_recovery':rv,'selected_last_rail_s':last,'selected_first_1000uv_crossing_s':onset})
 final=recovery_gate(sample(et[0]-10)[roi].std(axis=1),sample(duration-3)[roi].std(axis=1)[None,:],np.mean((sample(duration-3)[roi]==0)|(sample(duration-3)[roi]==1023),axis=1)[None,:])
 if not final['passed']:issues.append('final_ROI_recovery')
 pre_window=covered_rate(rt,et[0]-50,et[0]-5,duration);post_window=covered_rate(rt,et[-1]+60,et[-1]+600,duration)
 pre_rate=pre_window['hz'];post_rate=post_window['hz']
 analysis_status='qualified_exploratory' if not issues else ('incomplete_diagnostic_only' if 'stage_not_recorded_as_fully_closed' in issues else 'qc_failed_diagnostic_only')
 result={'stage':stage,'status':analysis_status,'issues':issues,
  'source_h5':str(source),'h5_bytes':source.stat().st_size,'duration_seconds':duration,'frames':len(frames),'frame_gaps':int(np.count_nonzero(np.diff(frames)!=1)),
  'event_ids':ev['eventid'].tolist(),'event_times_seconds':et.tolist(),'event_messages':messages,'roi_count':len(roi),
  'native_spikes':len(sp),'excluded_invalid_spikes':int((~good).sum()),'trials':trials,'artifact_recovery':recovery,'final_recovery':final,
  'descriptive_pre_roi_rate_hz_45s':pre_rate,'descriptive_late_roi_rate_hz_540s':post_rate,'r2_enabled':False,'training_enabled':False,
  'descriptive_pre_window':pre_window,'descriptive_late_window':post_window,
  'physical_voltage_measured':False,'nominal_phase_amplitude_mv':298.1901,'phase_duration_us':200,
  'limits':['One organoid; repeated trials are not independent biological replicates.','0-2s excluded from neural efficacy analysis.','Rate increase alone is not a validated improvement, learning, fear, or thirst metric.']}
 if stage=='control':result['paired_statistics']=paired_summary(trials)
 np.savez_compressed(out/(stage+'_data.npz'),spike_seconds=tt,spike_channels=s['channel'],roi_channels=roi,event_seconds=et,trace_seconds=trace_t,stim_channel_traces_uv=np.array(traces))
 centers,rate=rate_bins(rt,duration,10)

h=hashlib.sha256()
with source.open('rb') as f:
 for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
result['h5_sha256']=h.hexdigest()
(out/(stage+'_summary.json')).write_text(json.dumps(result,indent=2))
with (out/(stage+'_trials.csv')).open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(trials[0]));w.writeheader();w.writerows(trials)
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
colors={'true':'#dc2626','sham':'#2563eb'}
axs[0,0].plot(centers/60,rate,color='#334155',lw=1.2)
for row,t in zip(schedule,et):axs[0,0].axvline(t/60,color=colors[row['arm']],alpha=.55,lw=1)
axs[0,0].set(xlabel='Minutes from recording start',ylabel='Fixed ROI spikes / second',title='A. Fixed 377-channel readout; red=true, blue=sham')
for row,v in zip(schedule,traces):axs[0,1].plot(trace_t*1000,v,color=colors[row['arm']],alpha=.6,lw=.7)
axs[0,1].set(xlim=(-10,600),xlabel='ms from recorded event',ylabel='Stim-electrode recording (uV)',title='B. Output artifact; electrode excluded from ROI')
if stage=='control':
 p=result['paired_statistics'];pairs=p['pairs']
 for row in pairs:axs[1,0].plot([0,1],[row['sham_delta_hz'],row['true_delta_hz']],'-o',alpha=.7,label='Pair %d'%row['pair'])
 axs[1,0].set(xticks=[0,1],xticklabels=['Sham','True'],ylabel='After - before ROI rate (Hz)',title='C. Prespecified response: +2..5s minus -5..-2s')
 axs[1,0].axhline(0,color='#64748b',ls=':',lw=1)
 axs[1,1].bar([x['pair'] for x in pairs],[x['difference_hz'] for x in pairs],color='#7c3aed')
 axs[1,1].axhline(0,color='#64748b',lw=1)
 axs[1,1].set(xlabel='Pair',ylabel='True delta - sham delta (Hz)',title='D. Mean = %.2f Hz; balanced-order p = %.2f'%(p['mean_true_minus_sham_hz'],p['balanced_order_randomization_two_sided_p']))
else:
 vals=[x['delta_hz'] for x in trials]
 axs[1,0].bar(['Sham','True'],vals,color=[colors[x['arm']] for x in schedule]);axs[1,0].axhline(0,color='#64748b',lw=1)
 axs[1,0].set(ylabel='After - before ROI rate (Hz)',title='C. Pilot only: one observation per arm, no efficacy claim')
 rv=np.array([x['roi_recovery']['fraction_channels_recovered_each_window'] for x in recovery])*100
 for i,label in enumerate(['2-3s','10-11s']):axs[1,1].bar(np.arange(2)+(i-.5)*.3,rv[:,i],.3,label=label)
 axs[1,1].axhline(95,color='#dc2626',ls='--',label='QC threshold');axs[1,1].legend(frameon=False)
 axs[1,1].set(xticks=[0,1],xticklabels=['Sham','True'],ylim=(0,105),ylabel='ROI channels with recovered raw SD (%)',title='D. Recovery in prespecified windows')
fig.suptitle('22554 | %s | %s | R2 OFF'%(stage.upper(),result['status']),fontsize=14)
fig.savefig(out/(stage+'_comparison.png'));fig.savefig(out/(stage+'_comparison.svg'));plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
details=baseline['channel_details'];roi_set=set(roi.tolist())
bx=np.array([x['sd_disconnected_uv'] for x in details]);by=np.array([x['sd_connected_uv'] for x in details]);keep=np.array([x['channel'] in roi_set for x in details])
axs[0].scatter(bx[~keep],by[~keep],s=6,color='#94a3b8',label='Excluded');axs[0].scatter(bx[keep],by[keep],s=8,color='#2563eb',label='Fixed ROI')
mx=max(np.percentile(bx,99.5),np.percentile(by,99.5))*1.2;axs[0].plot([0,mx],[0,mx],':',color='#64748b');axs[0].set(xlim=(0,mx),ylim=(0,mx),xlabel='Disconnected raw SD (uV)',ylabel='Connected raw SD (uV)',title='A. Connection baseline');axs[0].legend(frameon=False)
axs[1].bar(['Disconnected','Connected'],[baseline['population_rate_before_hz'],baseline['population_rate_connected_hz']],color=['#64748b','#2563eb']);axs[1].set(ylabel='ROI spikes / second',title='B. +%.2f%% after connection'%(100*(baseline['population_rate_ratio']-1)))
diag=baseline['stim_electrode_diagnostic'];axs[2].bar(['Disconnected','Connected'],[diag['disconnected_sd_uv'],diag['connected_sd_uv']],color=['#64748b','#dc2626']);axs[2].set(ylabel='Electrode 10900 raw SD (uV)',title='C. Stim electrode excluded from readout')
fig.suptitle('22554 | Zero-output baseline: %d / %d active candidates stable'%(len(roi),baseline['candidate_channels']))
fig.savefig(out/'baseline_comparison.png');fig.savefig(out/'baseline_comparison.svg');plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k not in ['trials','artifact_recovery','event_messages']},indent=2))
