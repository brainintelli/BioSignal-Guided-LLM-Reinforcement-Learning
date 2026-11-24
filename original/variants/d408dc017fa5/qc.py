"""Read-only analysis of closed stage files; never writes a hardware gate receipt."""
import os
os.environ['HDF5_PLUGIN_PATH']='/pub/soft/MaxLab/so'
from pathlib import Path
import hashlib,json,re,sys
import h5py,numpy as np
from protocol import SESSION_ID
from qc_math import baseline_gate,recovery_gate

BASE=Path(__file__).resolve().parent
mode=sys.argv[1]
assert mode in ('baseline','pilot')
state=json.loads((BASE/'state.json').read_text())
if state['status']!='awaiting_'+mode+'_qc':raise RuntimeError('Stage not closed/awaiting QC')
source=BASE/mode/(mode+'.raw.h5')
issues=[]
with h5py.File(source,'r') as f:
 d=f['data_store/data0000'];mapping=d['settings/mapping'][:];channels=mapping['channel'].astype(int)
 frames=d['groups/routed/frame_nos'][:];events=d['events'][:];sp=d['spikes'][:];raw=d['groups/routed/raw']
 fs=float(d['settings/sampling'][0]);uv=float(d['settings/lsb'][0])*1e6;duration=len(frames)/fs;f0=int(frames[0])
 chip=f['wellplate/id'][0].decode();event_times=(events['frameno'].astype(float)-f0)/fs
 good=np.isin(sp['channel'],channels)&(sp['frameno']>=frames[0])&(sp['frameno']<=frames[-1]);s=sp[good]
 times=(s['frameno'].astype(float)-f0)/fs
 expected=sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',(BASE/'qualified_route.cfg').read_text().split()[0]))
 actual=sorted((int(x['channel']),int(x['electrode'])) for x in mapping)
 if chip!='22554' or fs!=20000 or actual!=expected:issues.append('identity_sampling_or_mapping_mismatch')
 if np.any(np.diff(frames)!=1):issues.append('noncontiguous_frames')
 ids=[8001,8002] if mode=='baseline' else [8201,8202]
 if events['eventid'].tolist()!=ids or events['eventtype'].tolist()!=[1,1]:issues.append('event_markers_incorrect')
 messages=[json.loads(x['eventmessage'].decode()) for x in events]
 if len(events)!=2:
  result={'status':'failed','errors':issues,'stage':mode,'session_id':SESSION_ID,'roi_channels':[]}
  (BASE/(mode+'_qc.json')).write_text(json.dumps(result,indent=2));raise SystemExit(2)
 result={'stage':mode,'session_id':SESSION_ID,'chip_id':chip,'source_h5':str(source),'h5_bytes':source.stat().st_size,
  'duration_seconds':duration,'frames':len(frames),'frame_gaps':int(np.count_nonzero(np.diff(frames)!=1)),
  'mapping_channels':len(mapping),'event_ids':events['eventid'].tolist(),'event_times_seconds':event_times.tolist(),
  'event_messages':messages,'invalid_spikes_excluded':int((~good).sum()),'sample_rate_hz':fs,
  'gain':float(d['settings/gain'][0]),'hpf_hz':float(d['settings/hpf'][0]),'lsb_uv':uv,'r2_enabled':False}
 def sample(t,seconds=1):
  return raw[:,int(t*fs):int((t+seconds)*fs)]
 def rates(a,b):return np.array([np.count_nonzero((s['channel']==c)&(times>=a)&(times<b)) for c in channels])
 if mode=='baseline':
  if [x.get('phase') for x in messages]!=['disconnected','connected']:issues.append('baseline_phase_labels_incorrect')
  if not 119<=event_times[1]-event_times[0]<=122 or duration-event_times[1]<119:issues.append('baseline_duration_short')
  before=np.concatenate([sample(event_times[0]+t) for t in [20,60,100]],axis=1)[channels]
  after=np.concatenate([sample(event_times[1]+t) for t in [20,60,100]],axis=1)[channels]
  b0,b1=event_times[0]+10,event_times[1]-2;c0,c1=event_times[1]+10,duration-2
  counts_before=rates(b0,b1);counts_after=rates(c0,c1)
  excluded=[x['channel'] for x in json.loads((BASE/'previous_artifact_channels.json').read_text())]
  sd0=before.std(axis=1);sd1=after.std(axis=1);med0=np.median(before,axis=1);med1=np.median(after,axis=1)
  rail0=np.mean((before==0)|(before==1023),axis=1);rail1=np.mean((after==0)|(after==1023),axis=1)
  gate=baseline_gate(channels=channels,counts_before=counts_before,counts_after=counts_after,seconds_before=b1-b0,seconds_after=c1-c0,
   sd_before=sd0,sd_after=sd1,median_before=med0,median_after=med1,rails_before=rail0,rails_after=rail1,excluded=excluded)
  issues.extend(gate.pop('errors'));gate.pop('status');result.update(gate,excluded_channels=excluded,
   raw_sample_seconds_before=[float(event_times[0]+t) for t in [20,60,100]],raw_sample_seconds_connected=[float(event_times[1]+t) for t in [20,60,100]])
  idx=int(np.flatnonzero(channels==213)[0]);result['stim_electrode_diagnostic']={'channel':213,'electrode':10900,'excluded_from_ROI':True,
   'disconnected_sd_uv':float(sd0[idx]*uv),'connected_sd_uv':float(sd1[idx]*uv),
   'disconnected_rate_hz':float(counts_before[idx]/(b1-b0)),'connected_rate_hz':float(counts_after[idx]/(c1-c0))}
  result['channel_details']=[{'channel':int(c),'electrode':int(mapping[i]['electrode']),'count_disconnected':int(counts_before[i]),
   'count_connected':int(counts_after[i]),'sd_disconnected_uv':float(sd0[i]*uv),'sd_connected_uv':float(sd1[i]*uv),
   'median_disconnected_adc':float(med0[i]),'median_connected_adc':float(med1[i]),
   'rail_fraction_disconnected':float(rail0[i]),'rail_fraction_connected':float(rail1[i])} for i,c in enumerate(channels)]
 else:
  baseline=json.loads((BASE/'baseline_qc.json').read_text());roi=np.array(baseline['roi_channels'],dtype=int)
  result['roi_channels']=roi.tolist()
  if [x.get('arm') for x in messages]!=['sham','true']:issues.append('pilot_arm_labels_incorrect')
  if int(events[1]['frameno'])-int(events[0]['frameno'])!=900000:issues.append('pilot_spacing_not_45seconds')
  if duration-event_times[-1]<599:issues.append('pilot_recovery_short')
  pulse_results=[]
  for arm,t in zip(['sham','true'],event_times):
   before=sample(t-3);after=np.stack([sample(t+2),sample(t+10)])
   rv=recovery_gate(before[roi].std(axis=1),after[:,roi].std(axis=2),np.mean((after[:,roi]==0)|(after[:,roi]==1023),axis=2))
   near=sample(t,.6);selected=near[213].astype(float);center=float(np.median(before[213]));tt=np.arange(near.shape[1])/fs
   rails=(selected==0)|(selected==1023);last_rail=float(tt[rails][-1]) if rails.any() else None
   crossing=np.flatnonzero(np.abs(selected-center)*uv>=1000)
   onset=float(tt[crossing[0]]) if len(crossing) else None
   row={'arm':arm,'event_seconds':float(t),'recovery':rv,'selected_peak_abs_uv_first600ms':float(np.max(np.abs(selected-center))*uv),
    'selected_first_1000uv_crossing_seconds':onset,
    'selected_last_rail_seconds_first600ms':last_rail,'roi_saturated_channels_first600ms':int(np.any((near[roi]==0)|(near[roi]==1023),axis=1).sum())}
   pulse_results.append(row)
   if not rv['passed']:issues.append(arm+'_ROI_not_recovered_by_prespecified_windows')
   if arm=='sham' and (rails.any() or row['selected_peak_abs_uv_first600ms']>1000):issues.append('sham_has_large_output_artifact')
   if arm=='true' and row['selected_peak_abs_uv_first600ms']<500:issues.append('true_output_artifact_not_confirmed')
   if arm=='true' and (onset is None or onset>.01):issues.append('true_artifact_not_aligned_with_event_within_10ms')
   if arm=='true' and last_rail is not None and last_rail>.5:issues.append('selected_electrode_saturation_over_500ms')
  result['pulses']=pulse_results
  before=sample(event_times[0]-10)[roi];late=sample(duration-3)[roi]
  late_test=recovery_gate(before.std(axis=1),late.std(axis=1)[None,:],np.mean((late==0)|(late==1023),axis=1)[None,:])
  result['final_recovery']=late_test
  if not late_test['passed']:issues.append('final_ROI_recovery_failed')
 result.update(status='pass' if not issues else 'failed',errors=issues)
 out=BASE/(mode+'_qc.json');tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)
 summary={k:v for k,v in result.items() if k not in ['roi_channels','channel_details','sd_ratios','median_shifts_adc']}
 summary['roi_count']=len(result['roi_channels'])
 print(json.dumps(summary,indent=2))
raise SystemExit(0 if result['status']=='pass' else 2)
