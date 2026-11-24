"""Post-run diagnostic amendment: missing markers and long amplifier saturation.

Never alters the source H5 or substitutes reconstructed onsets for recorded events.
The frozen primary analysis failed. No short-latency biological effect is reported.
"""
import os
os.environ['HDF5_PLUGIN_PATH']='/pub/soft/MaxLab/so'
from pathlib import Path
import json,hashlib,re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent
out=base/'diagnostic_analysis'
out.mkdir(exist_ok=True)
run=json.loads(next((base/'results').glob('*/run.json')).read_text())
assert run['status']=='completed' and not run['cleanup_errors']
source=Path(run['h5_directory_on_server'])/'stimulation.raw.h5'
reconstruction=json.loads((base/'raw_onset_reconstruction.json').read_text())
onsets=np.array([p['raw_threshold_onset_second'] for p in reconstruction['pulses']])
qualification=json.loads((base/'qualification_qc.json').read_text())
qualified=next(r for r in qualification['candidates'] if r['electrode']==10900)
with h5py.File(source,'r') as f:
 d=f['data_store/data0000'];m=d['settings/mapping'][:];sp=d['spikes'][:];frames=d['groups/routed/frame_nos'][:]
 fs=float(d['settings/sampling'][0]);uv=float(d['settings/lsb'][0])*1e6;duration=len(frames)/fs;f0=int(frames[0])
 good=np.isin(sp['channel'],m['channel'])&(sp['frameno']>=frames[0])&(sp['frameno']<=frames[-1])
 s=sp[good];times=(s['frameno'].astype(float)-f0)/fs
 ch=int(m[m['electrode']==10900]['channel'][0]);selected_times=times[s['channel']==ch]
 raw=d['groups/routed/raw'];recorded_channels=m['channel'].astype(int)
 expected=sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',(base/'qualified_route.cfg').read_text().split()[0]))
 actual=sorted((int(r['channel']),int(r['electrode'])) for r in m)
 sections={}
 for name,(a,b) in {'pre_matched_540s':(onsets[0]-540,onsets[0]),
                    'early_post_1_to_60s':(onsets[-1]+1,onsets[-1]+60),
                    'late_post_60_to_600s':(onsets[-1]+60,onsets[-1]+600)}.items():
  keep=(times>=a)&(times<b)
  sections[name]={'start_s':float(a),'end_s':float(b),'duration_s':float(b-a),'mapped_spikes':int(keep.sum()),
   'population_rate_hz':float(keep.sum()/(b-a)),
   'selected_rate_hz':float(np.count_nonzero((selected_times>=a)&(selected_times<b))/(b-a)),
   'active_channels':len(np.unique(s['channel'][keep]))}
 tt=np.arange(-1000,30000)/fs
 traces=[];artifact=[]
 for i,onset in enumerate(onsets):
  index=round(onset*fs);a=raw[:,index-1000:index+30000]
  y=a[ch].astype(float);median=float(np.median(y[tt<-.005]));traces.append((y-median)*uv)
  selected_rails=((y==0)|(y==1023))&(tt>=0)
  all_rails=((a[recorded_channels]==0)|(a[recorded_channels]==1023))
  artifact.append({'pulse':i+1,'reconstructed_onset_s':float(onset),
   'selected_last_rail_ms':float(tt[selected_rails][-1]*1000) if selected_rails.any() else None,
   'mapped_channels_touching_rail_first500ms':int(np.any(all_rails[:,(tt>=0)&(tt<.5)],axis=1).sum()),
   'selected_raw_range_adc':[int(y.min()),int(y.max())]})
 noise=[]
 for label,t in [('pre_connected',onsets[0]-30),('post_60s',onsets[-1]+60),('post_300s',onsets[-1]+300),('post_end',duration-2)]:
  a=raw[:,int(t*fs):int((t+1)*fs)];y=a[ch].astype(float)
  noise.append({'period':label,'second':float(t),'selected_raw_sd_uv':float(y.std()*uv),
   'selected_mean_adc':float(y.mean()),'mapped_rail_fraction':float(np.mean((a[recorded_channels]==0)|(a[recorded_channels]==1023)))})
 result={'analysis_status':'diagnostic_only_not_efficacy_qualified','primary_analysis_passed':False,
  'primary_failure':'H5 recorded zero event markers; event registry was cleared after sequence construction.',
  'h5_source':str(source),'h5_bytes':source.stat().st_size,'chip_id':f['wellplate/id'][0].decode(),
  'sample_rate_hz':fs,'gain':float(d['settings/gain'][0]),'hpf_hz':float(d['settings/hpf'][0]),
  'duration_seconds':duration,'frames':len(frames),'frame_gaps':int(np.count_nonzero(np.diff(frames)!=1)),
  'mapped_channels':len(m),'mapping_matches_qualified_route':actual==expected,
  'recorded_event_count':len(d['events']),'reconstructed_artifact_count':len(onsets),
  'reconstructed_onsets_seconds':onsets.tolist(),'reconstructed_intervals_seconds':np.diff(onsets).tolist(),
  'reconstruction_method':reconstruction['method'],'stim_electrode':10900,'stim_channel':ch,
  'target_phase_amplitude_mv':300,'nominal_phase_amplitude_mv':run['actual_amplitude_schedule_mv'][0],
  'physical_voltage_measured':False,'native_spike_records':len(sp),'excluded_invalid_spike_records':int((~good).sum()),
  'mapped_spike_records':len(s),'sections':sections,'artifact':artifact,'raw_sd_windows':noise,
  'qualification_selected_raw_sd_uv':qualified['sample_raw_std_uv'],
  'late_population_rate_change_percent':100*(sections['late_post_60_to_600s']['population_rate_hz']/sections['pre_matched_540s']['population_rate_hz']-1),
  'r2_enabled':False,'realtime_training':False,'cleanup_errors':run['cleanup_errors'],
  'limits':['Post-hoc diagnostic amendment after primary marker-integrity failure.',
            'Reconstructed artifact onsets are not original event timestamps; source H5 unchanged.',
            '171–175ms saturation and a longer tail invalidate the prespecified 20–500ms neural-response window.',
            'Pre/post activity rates are descriptive within one session; no sham, independent replicate, or model training.',
            'Raw signal SD differs between qualification and connected baseline; it cannot be attributed to stimulation because it changed beforehand.']}
 np.savez_compressed(out/'diagnostic_data.npz',spike_seconds=times,spike_channels=s['channel'],
  spike_amplitude_uv=s['amplitude']*uv,mapping=m,raw_reconstructed_onsets_s=onsets,
  raw_time_seconds=tt,raw_selected_traces_uv=np.asarray(traces))
h=hashlib.sha256()
with source.open('rb') as f:
 for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
result['h5_sha256']=h.hexdigest()
(out/'diagnostic_summary.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
# Drop the last partial 10 s bin; dividing its 0.23 s exposure by 10 would
# create a false terminal activity collapse in this recording.
edges=np.arange(0,np.floor(duration/10)*10+.001,10);counts=np.histogram(times,bins=edges)[0]
axs[0,0].plot((edges[:-1]+edges[1:])/120,counts/np.diff(edges),color='#2563eb',lw=1.2)
axs[0,0].axvspan(onsets[0]/60,onsets[-1]/60,color='#dc2626',alpha=.2,label='Output-artifact period')
axs[0,0].set(xlabel='Minutes from recording start',ylabel='Mapped spikes / second',title='A. Whole recording: descriptive activity');axs[0,0].legend(frameon=False)
for i,y in enumerate(traces):axs[0,1].plot(tt*1000,y,lw=.7,alpha=.6,label=str(i+1))
axs[0,1].axvspan(20,500,color='gray',alpha=.17,label='Original window invalid')
axs[0,1].set(xlim=(-20,650),xlabel='ms from reconstructed artifact onset',ylabel='Centered recorded voltage (uV)',title='B. Electrode 10900: saturation and recovery tail')
axs[0,1].legend(frameon=False,ncol=3,fontsize=8)
values=[x['selected_last_rail_ms'] for x in artifact]
axs[1,0].bar(range(1,6),values,color='#dc2626',alpha=.75)
for i,v in enumerate(values):axs[1,0].text(i+1,v+2,f'{v:.2f}',ha='center')
axs[1,0].set(xticks=range(1,6),ylim=(0,210),xlabel='Output artifact number',ylabel='Last saturated sample after onset (ms)',title='C. Five artifacts; 10 s spacing')
labels=['Qualification','Connected pre','Post +60 s','Post +300 s','Post end']
stds=[qualified['sample_raw_std_uv']]+[x['selected_raw_sd_uv'] for x in noise]
axs[1,1].plot(range(5),stds,'o-',color='#7c3aed')
for i,v in enumerate(stds):axs[1,1].text(i,v+2,f'{v:.1f}',ha='center')
axs[1,1].set(xticks=range(5),xticklabels=labels,ylim=(0,max(stds)*1.25),ylabel='Selected raw signal SD (uV)',title='D. Signal changed before stimulation')
axs[1,1].tick_params(axis='x',rotation=15)
fig.suptitle('22554 physical probe | EVENT MARKERS MISSING | diagnostic results only',fontsize=14,color='#991b1b')
fig.savefig(out/'diagnostic_comparison.png');fig.savefig(out/'diagnostic_comparison.svg');plt.close(fig)
print(json.dumps({k:result[k] for k in ['analysis_status','duration_seconds','frame_gaps','recorded_event_count','reconstructed_artifact_count','reconstructed_intervals_seconds','sections','raw_sd_windows','late_population_rate_change_percent','artifact','h5_sha256']},indent=2))
