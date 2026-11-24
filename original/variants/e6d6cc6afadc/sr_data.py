"""28073 offline spike-view gate. No device, network, stimulation or GPU API."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

STATE_FEATURES=('spike_count','rate_hz_per_channel','active_channel_fraction','mean_abs_amplitude_uv')
SPLIT_COUNTS={'calibration':60,'train':129,'evaluation':97}
MANIFEST_SCHEMA='mea.28073.qc_manifest.v1'
DATASET_SCHEMA='mea.28073.offline_state.dataset.v1'


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,x):
    with Path(path).open('x',encoding='utf-8') as f:f.write(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def require(ok,message):
    if not ok:raise ValueError(message)
def finite(x):
    require(type(x) in (int,float) and math.isfinite(x),'finite numeric value required')
    return float(x)
def interval(x):
    a,b=x['start_frame'],x['end_frame']
    require(type(a) is int and type(b) is int and 0<=a<=b,'invalid hardware frame interval')
    return a,b
def overlaps(a,b):return a[0]<b[1] and b[0]<a[1]
def file_ref(base,ref):
    p=(base/ref['path']).resolve()
    require(p.is_relative_to(base.resolve()),'manifest paths must remain inside bundle')
    require(p.is_file() and sha(p)==ref['sha256'],'file SHA mismatch: '+str(ref['path']))
    if 'bytes' in ref:require(p.stat().st_size==ref['bytes'],'file byte count mismatch')
    return p


def validate_feature_scope(source,qc):
    if 'feature_channels' in source or 'approved_feature_channels' in qc:
        require(source.get('feature_channels')==qc.get('approved_feature_channels') and bool(source.get('feature_channels')),'feature channels differ from approved QC scope')


def validate_v2_scope(base,source,qc):
    for reference in qc.get('evidence_files',[]):file_ref(base,reference)
    if 'original_v2_receipt' not in qc:return
    original=read(file_ref(base,qc['original_v2_receipt']));scope=read(file_ref(base,qc['scope_v2_receipt']))
    for receipt in (original,scope):
        result=receipt['result'];actual=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        require(result['status']=='pass' and result['errors']==[] and receipt['result_sha256']==actual,'original v2 QC result not passed/intact')
        require(result['r2_enabled'] is False and result['training_enabled'] is False,'original acquisition R2/training not disabled')
    require(original['result']['stage']==source['source_id'],'original v2 stage/source differs')
    require(all(original[k]==scope[k] for k in ('session_id','config_sha256','binding_sha256')),'scope receipt from another v2 session/configuration')
    require(scope['result']['roi_channels']==qc['approved_feature_channels'],'feature channels exceed/differ from original approved ROI')
    require(set(source['feature_channels'])<=set(original['result']['roi_channels']),'feature channels exceed this source QC scope')
    if scope['result']['stage']=='pilot':
        require(scope['result']['holdout19']['passed'] is True and scope['result']['holdout19']['passed_count']>=18,'pilot frozen ROI baseline holdout not passed')
        if source['source_id']=='baseline':require(scope['baseline_source']==original['source'],'pilot scope uses another baseline')


def clean_windows(rows,stimulations,bad_intervals,sample_rate):
    require(sample_rate==20000,'registered 20 kHz source required')
    masks=[]
    for event in stimulations:
        a,b=interval(event);masks.append((a,b+2*sample_rate,'stimulation_0_to_2s'))
    for bad in bad_intervals:
        a,b=interval(bad);require(a<b and bool(bad.get('reason')),'QC exclusion reason required')
        masks.append((a,b,'qc_failed:'+bad['reason']))
    clean=[];excluded=[]
    for row in rows:
        reasons=sorted({why for a,b,why in masks if overlaps(interval(row),(a,b))})
        if reasons:excluded.append(row|{'excluded_reasons':reasons})
        else:clean.append(row)
    return clean,excluded


def select_partitions(rows,partitions,sample_rate):
    require(sample_rate==20000,'registered sample rate changed')
    require(partitions and set(p['split'] for p in partitions)=={'calibration','train','evaluation','holdout'},'four explicit splits required')
    ordered=sorted(partitions,key=lambda p:interval(p))
    rank={'calibration':0,'train':1,'evaluation':2,'holdout':3}
    for p in ordered:require(interval(p)[0]<interval(p)[1],'empty partition')
    for a,b in zip(ordered,ordered[1:]):
        require(interval(a)[1]<=interval(b)[0],'partition overlap across hardware frame clock')
        require(rank[a['split']]<=rank[b['split']],'partitions must be chronological')
    last_train=max(p['end_frame'] for p in ordered if p['split']=='train')
    first_eval=min(p['start_frame'] for p in ordered if p['split']=='evaluation')
    require(first_eval-last_train>=10*sample_rate,'train/evaluation embargo must be at least 10 seconds')
    last_eval=max(p['end_frame'] for p in ordered if p['split']=='evaluation')
    first_holdout=min(p['start_frame'] for p in ordered if p['split']=='holdout')
    require(first_holdout-last_eval>=10*sample_rate,'evaluation/holdout embargo must be at least 10 seconds')
    holdout_clean=[r for p in ordered if p['split']=='holdout' for r in rows if r['source_id']==p['source_id'] and
                   p['start_frame']<=r['start_frame'] and r['end_frame']<=p['end_frame']]
    require(bool(holdout_clean),'holdout requires at least one complete clean window; masked/short intervals do not qualify')
    selected=[]
    for split,count in SPLIT_COUNTS.items():
        candidates=[]
        for p in ordered:
            if p['split']==split:
                candidates.extend(r for r in rows if r['source_id']==p['source_id'] and p['start_frame']<=r['start_frame'] and r['end_frame']<=p['end_frame'])
        candidates.sort(key=lambda r:(r['start_frame'],r['end_frame'],r['source_id']))
        require(len(candidates)>=count,f'insufficient {split}: need {count}, have {len(candidates)}; fixed budget cannot be reduced')
        selected.extend(r|{'split':split} for r in candidates[:count])
    require(len({(r['source_id'],r['start_frame']) for r in selected})==286,'duplicate selected windows')
    for a,b in zip(selected,selected[1:]):require(a['end_frame']<=b['start_frame'],'selected window overlap')
    return [r|{'index':i} for i,r in enumerate(selected)]


def fit_state_profile(rows):
    calibration=[r for r in rows if r['split']=='calibration']
    require(len(calibration)==60,'exactly 60 calibration windows required')
    x=[[finite(r[n]) for n in STATE_FEATURES] for r in calibration]
    center=[statistics.median(col) for col in zip(*x)]
    scale=[1.4826*statistics.median(abs(row[j]-center[j]) for row in x) for j in range(4)]
    require(any(s>1e-12 for s in scale),'all calibration state features constant; no informative state2')
    return dict(fit_indices=list(range(60)),state_names=list(STATE_FEATURES),state_center=center,
                state_scale=[s if s>1e-12 else 1. for s in scale],state_zero_scale=[i for i,s in enumerate(scale) if s<=1e-12],
                normalization_clip=5.,experimental=True,biological_validated=False)


def normalize(row,profile):
    return [max(-5.,min(5.,(finite(row[n])-c)/s)) for n,c,s in zip(STATE_FEATURES,profile['state_center'],profile['state_scale'])]


def read_events(dataset,chunk_size=1000000):
    require(type(chunk_size) is int and chunk_size>0,'positive spike read chunk required')
    events=[]
    for offset in range(0,len(dataset),chunk_size):
        block=dataset[offset:offset+chunk_size]
        events.extend([offset+i,int(e['frameno']),int(e['channel']),float(e['amplitude'])] for i,e in enumerate(block))
    return events


def read_spikeview(path,source):
    """Verify retained frame/event structure; never claim waveform QC was redone."""
    import h5py
    import numpy as np
    with h5py.File(path,'r') as f:
        require(f.attrs.get('schema')=='mea.spikeview.v1','explicit spikeview schema required')
        provenance=json.loads(f.attrs['provenance_json'])
        require(provenance['source_id']==source['source_id'],'spikeview source identity mismatch')
        require(provenance['parent_raw']==source['parent_raw'],'parent raw identity mismatch')
        require(provenance['qc_receipt_sha256']==source['qc_receipt']['sha256'],'QC provenance mismatch')
        require(provenance['contains_raw_waveforms'] is False,'spikeview must not claim waveforms')
        require(list(f['data_store'])==['data0000'],'one finalized data store required')
        s=f['data_store/data0000'];raw=s['groups/routed']
        require('raw' not in raw,'spikeview unexpectedly contains raw waveform')
        require(str(f['wellplate/id'][0].decode())=='28073' and int(s['well_id'][0])==source['well_id'],'chip/well identity mismatch')
        require(raw['frame_nos'].dtype.kind in 'iu','integer hardware frames required')
        frames=raw['frame_nos'][:]
        require(frames.ndim==1 and len(frames)>0 and bool(np.all(frames[1:]==frames[:-1]+1)),'noncontiguous frame sequence')
        start,end=int(frames[0]),int(frames[-1])+1
        rate=float(s['settings/sampling'][0]);lsb=float(s['settings/lsb'][0])
        require(rate==20000 and math.isfinite(lsb) and lsb>0,'sampling/LSB invalid')
        mapping=[[int(v['channel']),int(v['electrode'])] for v in s['settings/mapping']]
        channels=[v[0] for v in mapping]
        require(len(channels)>0 and len(channels)==len(set(channels))==len({v[1] for v in mapping}),'nonunique channel mapping')
        require(all(0<=c<1024 and 0<=e<26400 for c,e in mapping),'mapping physical range invalid')
        require(raw['channels'].dtype.kind in 'iu' and raw['channels'].ndim==1,'integer physical channel vector required')
        raw_channels=list(map(int,raw['channels'][:]))
        require(len(raw_channels)>0 and len(raw_channels)==len(set(raw_channels)) and all(0<=c<1024 for c in raw_channels),'physical channel vector duplicate/out of range')
        require(set(channels)<=set(raw_channels),'mapped channels missing from physical channel vector')
        feature_channels=source.get('feature_channels',channels)
        require(isinstance(feature_channels,list) and bool(feature_channels) and all(type(c) is int for c in feature_channels) and
                len(feature_channels)==len(set(feature_channels)) and set(feature_channels)<=set(channels),'feature channels duplicate/empty/outside mapping')
        feature_set=set(feature_channels)
        parent_shape=provenance.get('parent_raw_shape',provenance['raw_shape'])
        require(parent_shape==provenance['raw_shape']==[len(raw_channels),len(frames)],'parent raw frame/channel shape mismatch')
        require(provenance['raw_interval']==[start,end],'parent raw interval mismatch')
        anchor=source.get('window_anchor_frame',start)
        require(type(anchor) is int and start<=anchor<end,'window anchor outside raw interval')
        if anchor!=start or 'anchor_event_id' in source:
            require(source.get('anchor_event_id')==8002 and 'events' in s,'baseline anchor must reference recorded event 8002')
            ev=s['events']
            require(ev.dtype.names is not None and {'frameno','eventid','eventtype','eventmessage'}<=set(ev.dtype.names),'anchor event schema unavailable')
            matches=[e for e in ev if int(e['eventid'])==8002]
            require(len(matches)==1 and int(matches[0]['frameno'])==anchor and int(matches[0]['eventtype'])==1,'baseline anchor differs from actual 8002 event frame')
            message=matches[0]['eventmessage'];message=message.decode().rstrip('\x00') if isinstance(message,bytes) else str(message)
            require(json.loads(message).get('phase')=='connected','baseline anchor event is not connected phase')
        for dataset,fields in [(s['spikes'],('frameno','channel')),(s['settings/mapping'],('channel','electrode'))]:
            for field in fields:require(dataset.dtype.fields[field][0].kind in 'iu','integer event/mapping identity required')
        events=read_events(s['spikes'])
        frame_hash=hashlib.sha256(frames.tobytes()).hexdigest()
    require(source['raw_interval']==[start,end],'manifest/raw interval differs')
    mapped=set(channels);seen=set();last=None;valid=[];out_of_scope=[]
    for e in events:
        i,frame,ch,amp=e
        require(math.isfinite(amp),'nonfinite spike amplitude')
        require((frame,ch) not in seen and (last is None or frame>=last),'duplicate/unordered spike identity')
        seen.add((frame,ch));last=frame
        if start<=frame<end and ch in mapped:valid.append(e)
        else:out_of_scope.append(e)
    require(len(out_of_scope)==source['declared_out_of_scope_events'],'undeclared source event exclusion')
    rows=[];cursor=0
    for lo in range(anchor,end-20000+1,20000):
        hi=lo+20000
        while cursor<len(valid) and valid[cursor][1]<lo:cursor+=1
        end_cursor=cursor
        while end_cursor<len(valid) and valid[end_cursor][1]<hi:end_cursor+=1
        ee=[e for e in valid[cursor:end_cursor] if e[2] in feature_set];cursor=end_cursor
        rows.append(dict(source_id=source['source_id'],start_frame=lo,end_frame=hi,event_indices=[e[0] for e in ee],events_sha256=digest(ee),
            spike_count=len(ee),rate_hz_per_channel=len(ee)/len(feature_channels),active_channel_fraction=len({e[2] for e in ee})/len(feature_channels),
            mean_abs_amplitude_uv=sum(abs(e[3]) for e in ee)/len(ee)*lsb*1e6 if ee else 0.))
    identity=dict(raw_interval=[start,end],sample_rate_hz=20000,channel_map_sha256=digest([m for m in mapping if m[0] in feature_set]),mapping=mapping,
                  lsb_volts=lsb,raw_frames_sha256=frame_hash,events_sha256=digest(events),events=len(events),
                  out_of_scope_events=len(out_of_scope),out_of_scope_events_sha256=digest(out_of_scope),contains_raw_waveforms=False,
                  raw_channels=raw_channels,parent_raw_shape=parent_shape,window_anchor_frame=anchor,feature_channels=feature_channels,
                  mapped_events_outside_feature_scope=sum(e[2] not in feature_set for e in valid))
    return rows,identity


def derive(manifest_path):
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent;m=read(manifest_path)
    require(m['schema']==MANIFEST_SCHEMA and m['chip_id']=='28073','28073 QC manifest required')
    require(m['provenance']=='real_acquisition' and m['finalized'] is True,'finalized real acquisition required')
    require(m['source_association']=='prerecorded_action_independent','offline association only')
    require(m['frame_clock']=='session_continuous_hardware_frame','continuous hardware frame domain required')
    require(m['biological_specimen_id'] is None and m['session_label'].startswith('temporary_'),'explicit temporary session identity required')
    require(m['qc_status']=='passed_for_declared_scope','QC approval for declared scope required')
    require(m['sources'] and len({s['source_id'] for s in m['sources']})==len(m['sources']),'unique nonempty sources required')
    all_rows=[];excluded=[];identities={};source_ids=set();stimulation_count=0;loaded=[];global_events=[];frozen_features=None
    for source in m['sources']:
        sid=source['source_id'];source_ids.add(sid)
        require(source['kind']=='spikeview' and source['qc_status']=='passed_for_declared_scope','source QC spikeview required')
        view=file_ref(base,source);qc_path=file_ref(base,source['qc_receipt']);qc=read(qc_path)
        require(qc['schema']=='mea.28073.source_qc.v1' and qc['passed_for_declared_scope'] is True and qc['finalized'] is True,'source QC receipt not passed/finalized')
        require(qc['parent_raw']==source['parent_raw'] and qc['source_id']==sid,'QC receipt source binding')
        require(qc['excluded_intervals']==source['qc_exclusions'] and qc['declared_out_of_scope_events']==source['declared_out_of_scope_events'],'QC exclusions must match receipt')
        validate_feature_scope(source,qc)
        validate_v2_scope(base,source,qc)
        if m.get('preparation_mode') in ('stimulated','baseline-only'):
            require(qc.get('scope_v2_receipt',{}).get('sha256')==m.get('scope_v2_receipt_sha256'),'source scope receipt differs from fixed v2 scope')
            if frozen_features is None:frozen_features=source['feature_channels']
            require(source['feature_channels']==frozen_features,'v2 sources must share one frozen approved feature ROI')
        stim=source['stimulation'];log=read(file_ref(base,stim['evidence']))
        require(log['schema']=='mea.28073.stimulation_frames.v1' and log['source_id']==sid and log['complete'] is True,'complete stimulation record required')
        require(log['events']==stim['events'] and log['alignment']==stim['alignment'],'stimulation event evidence differs')
        require(stim['alignment'] in ('recorded_hardware_frames','none_confirmed_by_acquisition_log'),'uncertain stimulus alignment cannot enter training')
        require(stim['alignment']=='recorded_hardware_frames' or not stim['events'],'unstimulated declaration contains events')
        if m.get('preparation_mode')=='baseline-only':require(not stim['events'] and m.get('includes_stimulation_recordings') is False,'baseline-only fallback must not claim/include stimulation')
        rows,identity=read_spikeview(view,source)
        for event in stim['events']:
            a,b=interval(event);require(source['raw_interval'][0]<=a<=b<=source['raw_interval'][1],'stimulation event outside recording')
        loaded.append((source,rows));global_events.extend(stim['events']);identities[sid]=identity;stimulation_count+=len(stim['events'])
    # One session frame domain: a stimulus tail also masks the following recording.
    for source,rows in loaded:
        clean,bad=clean_windows(rows,global_events,source['qc_exclusions'],20000)
        all_rows.extend(clean);excluded.extend(bad)
    require(all(p['source_id'] in source_ids for p in m['partitions']),'partition source missing')
    for p in m['partitions']:
        raw=identities[p['source_id']]['raw_interval']
        require(raw[0]<=p['start_frame']<p['end_frame']<=raw[1],'partition exceeds finalized recording')
    windows=select_partitions(all_rows,m['partitions'],20000);profile=fit_state_profile(windows)
    normalization=[normalize(w,profile) for w in windows]
    require(any(any(v!=0 for v in x) for x in normalization[60:189]),'all training state2 values zero')
    return dict(schema=DATASET_SCHEMA,manifest=m,manifest_sha256=sha(manifest_path),sources=identities,windows=windows,excluded_windows=excluded,
                profile=profile,normalization=normalization,scores=[dict(p=0.,u=0.) for _ in windows],
                source=dict(chip_id='28073',session_label=m['session_label'],biological_specimen_id=None,
                            source_full_qc_passed=False,training_scope_qc_passed=True,contains_raw_waveforms=False,
                            waveform_qc_basis='external_hashed_QC_receipt',stimulation_events=stimulation_count,
                            association='prerecorded_action_independent',recording_files=len(m['sources']),biological_replication_units=None))


def build(manifest,output):
    d=derive(manifest);write(output,d)
    return dict(dataset_sha256=sha(output),manifest_sha256=d['manifest_sha256'],selected_windows=len(d['windows']),
                excluded_windows=len(d['excluded_windows']),stimulation_events=d['source']['stimulation_events'],r2_enabled=False,aux_enabled=False)


class Data:
    def __init__(self,path,pin,manifest_path=None):
        require(sha(path)==pin,'dataset pin mismatch');self.pin=pin;self.data=read(path)
        if manifest_path is None:manifest_path=Path(path).parent/'input'/'manifest.json'
        rebuilt=derive(manifest_path)
        require(self.data==rebuilt,'dataset differs from independent source/manifest reconstruction')
    def snapshot(self,index,context):
        require(type(index) is int and 0<=index<286,'finite nonrecycled window index required')
        w=self.data['windows'][index];sid=w['source_id'];source=next(s for s in self.data['manifest']['sources'] if s['source_id']==sid)
        v=dict(dataset_sha256=self.pin,source_sha256=self.data['manifest_sha256'],index=index,context_id=context,
               source_id=sid,parent_raw_sha256=source['parent_raw']['sha256'],spikeview_sha256=source['sha256'],
               start_frame=w['start_frame'],end_frame=w['end_frame'],events_sha256=w['events_sha256'],
               channel_map_sha256=self.data['sources'][sid]['channel_map_sha256'],raw_features={n:w[n] for n in STATE_FEATURES},
               state_values=self.data['normalization'][index],score=dict(p=0.,u=0.),association='prerecorded_action_independent',
               coverage_basis='qc_scoped_spikeview_without_stimulus_0_to_2s',source_full_qc_passed=False,training_scope_qc_passed=True,
               session_label=self.data['source']['session_label'],biological_specimen_id=None,split=w['split'])
        return v|{'snapshot_id':digest(v)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');args=p.parse_args()
    print(canonical(build(args.manifest,args.output)))
