"""Audit supplied spike-only H5; never admits it for formal training."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
import h5py,numpy as np
r=Path(__file__).resolve().parent
source=Path(r'./local-recordings/data.raw(1).h5')
pin='8ba000037b053f47e70c9576808b9d6f2e3609bf6d85dd8e2ffa55496f465771'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    with (r/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
if sha(source)!=pin:raise ValueError('Source changed')
copy=r/'source.raw.h5'
if not copy.exists():
    with source.open('rb') as src,copy.open('xb') as dst:shutil.copyfileobj(src,dst)
assert sha(copy)==pin
with h5py.File(copy,'r') as f:
    d=f['data_store/data0000'];s=d['spikes'][:];ev=d['events'][:];m=d['settings/mapping'][:]
    rate=float(d['settings/sampling'][0]);lsb=float(d['settings/lsb'][0]);chip=f['wellplate/id'][0].decode()
    assert chip=='22021' and rate==20000 and np.isfinite(s['amplitude']).all()
    assert np.all(np.diff(s['frameno'])>=0) and np.all(np.diff(ev['frameno'])>=0)
    selected=json.loads(f['assay/inputs/electrodes'][0].decode())['electrodes']['0']['stimulation']
    stim=np.unique(m['channel'][np.isin(m['electrode'],selected)])
    cv,cn=np.unique(m['channel'],return_counts=True);evv,en=np.unique(m['electrode'],return_counts=True)
    ambiguous=m[np.isin(m['channel'],cv[cn>1])|np.isin(m['electrode'],evv[en>1])]
    # Both ambiguous aliases are stimulus channels. Exclude every such channel;
    # never choose one electrode identity or label the full mapping verified.
    assert np.isin(ambiguous['channel'],stim).all()
    mapping=sorted([int(x['channel']),int(x['electrode'])] for x in m if x['channel'] not in stim)
    assert len({x[0] for x in mapping})==len(mapping)==len({x[1] for x in mapping})
    channels=np.array([x[0] for x in mapping]);last=int(ev['frameno'].max());start=last+61*int(rate);end=start+int(rate)
    assert start>=int(s['frameno'].min()) and end<=int(s['frameno'].max())+1
    mapped=np.isin(s['channel'],m['channel']);nonstim=np.isin(s['channel'],channels)
    current=s[(s['frameno']>=start)&(s['frameno']<end)&nonstim]
    events=sorted([int(x['frameno']),int(x['channel']),float(x['amplitude'])*lsb*1e6] for x in current)
    n=len(events);features=[float(n),n/len(mapping),len(np.unique(current['channel']))/len(mapping),float(np.abs(current['amplitude'].astype('float64')*lsb*1e6).mean()) if n else 0.]
    qi=dict(raw_waveform_present=False,frame_counter_present=False,raw_quality_verified=False,frame_continuity_verified=False,
        coverage_basis='observed_spike_frame_bounds_only',training_admitted=False,artifact_free_verified=False)
    names=[]
    d.visititems(lambda n,o:names.append(n) if isinstance(o,h5py.Dataset) else None)
    assert not any(n.rsplit('/',1)[-1] in ('raw','frame_nos') for n in names)
    def ts(n):return datetime.fromtimestamp(int(d[n][0])/1000,timezone(timedelta(hours=8))).isoformat()
    marker_frames=np.array(ev['frameno']);idx=np.searchsorted(marker_frames,s['frameno'],side='right')-1
    dt=np.where(idx>=0,s['frameno']-marker_frames[np.maximum(idx,0)],-1)/rate
    bins=[0,.001,.005,.01,.02,.05,.1,.5,1,100000]
    offsets=np.histogram(dt[dt>=0],bins=bins)[0].tolist()
    audit=dict(source_original_path=str(source),source_sha256=pin,chip_id=chip,chip_user_confirmed=True,organoid_identity=None,
        start_beijing=ts('start_time'),stop_beijing=ts('stop_time'),wall_duration_s=(int(d['stop_time'][0])-int(d['start_time'][0]))/1000,
        sample_rate_hz=rate,lsb_volts=lsb,mapping_rows=len(m),mapped_channels=len(np.unique(m['channel'])),
        ambiguous_mapping_rows=ambiguous.tolist(),mapping_hardware_readback_verified=False,configured_stimulus_electrodes=selected,
        configured_stimulus_channels=stim.tolist(),retained_channels=len(mapping),spikes_total=len(s),spikes_unmapped=int((~mapped).sum()),
        spikes_on_stimulus_channels=int(np.isin(s['channel'],stim).sum()),stimulus_marker_count=len(ev),
        marker_train_ids=int(len(np.unique(ev['eventid']))),first_marker_frame=int(ev['frameno'].min()),last_marker_frame=last,
        observed_spike_span_s=float(np.ptp(s['frameno']))/rate,observed_spike_tail_after_last_marker_s=(int(s['frameno'].max())-last)/rate,
        marker_metadata=[json.loads(x) for x in sorted(set(e.decode() for e in ev['eventmessage']))],
        absolute_amplitude_uv_quantiles=dict(zip(['min','p50','p90','p99','max'],np.quantile(np.abs(s['amplitude'].astype('float64')*lsb*1e6),[0,.5,.9,.99,1]).tolist())),
        spike_delay_to_preceding_marker=dict(bin_edges_seconds=bins,counts=offsets,caveat='Includes neuralresponses and possibleartifacts; not a validatedblankingrule.'),
        quality=qi,formal_training_rejection_reasons=['No rawwaveform/frame continuity evidence.','Onlyabout2seconds remain after60sguard; insufficient384train+60gap+150holdout.','Differentchipfromprior28080; no matched prestimulusbaseline.'])
    pack=dict(schema='mea.external_spikes.actor_input.v1',source_sha256=pin,chip_id=chip,sample_rate_hz=rate,lsb_volts=lsb,
        mapping=mapping,quality=qi,window=dict(index=0,start_frame=start,end_frame=end,last_stimulus_marker_frame=last,
        guard_seconds=61,guard_note='60s previousguard plus1s conservative margin; markerisnotphysicalwaveformmeasurement.',
        events=events,feature_names=['spike_count','population_rate_hz','active_channel_fraction','mean_abs_amplitude_uv'],features=features),
        alpha=1.,r2_reward_enabled=False,realtime_training=False,optimizer_updates=0)
write('audit.json',audit);write('state2_provisional.json',pack)
write('source_manifest.json',dict(files={n:dict(bytes=(r/n).stat().st_size,sha256=sha(r/n)) for n in ['source.raw.h5','audit.json','state2_provisional.json']}))
print(json.dumps({k:audit[k] for k in ['chip_id','wall_duration_s','mapped_channels','retained_channels','spikes_total','spikes_unmapped','observed_spike_tail_after_last_marker_s','quality']}|dict(probe_window_features=features),indent=2))
