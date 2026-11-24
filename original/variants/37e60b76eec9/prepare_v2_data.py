"""Read-only v2 closed-file admission and explicit offline dataset preparation.

No SDK/device/network/GPU imports. Original QC decisions are verified, never
manufactured. Run on the acquisition host with the actual closed raw files.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from export_spikeview import export
from sr_data import clean_windows,derive,read,read_spikeview,require,sha,write,canonical

FS=20000
PROTOCOL_SHA='6dfa076bcdfde8c0ef7bc9f76d65fc46621f5605c3d1946968eb225947147580'
PULSE_DURATION_FRAMES=8  # Inspected v2 pulse_codes: delay 4 + delay 4; source hash pinned.


def v2_digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def part(split,sid,start,end):return dict(split=split,source_id=sid,start_frame=start,end_frame=end)


def event_pair(value):
    if type(value) is int and value>=0:return value
    if isinstance(value,str) and re.fullmatch(r'[0-9]+',value,flags=re.ASCII):return int(value,10)
    raise ValueError('Event pair must be a nonnegative integer or ASCII decimal string')


def baseline_partitions(sid,connected):
    return [part(s,sid,connected+a*FS,connected+b*FS) for s,a,b in
            [('calibration',120,180),('train',610,739),('evaluation',760,857),('holdout',880,940)]]


def stimulated_partitions(baseline_id,connected,source_rows):
    """Frozen source IDs: baseline calibration, pilot train, pair_01 eval, pair_02 holdout."""
    indexed={sid:sorted(rows,key=lambda r:r['start_frame']) for sid,rows in source_rows}
    require(len(indexed)==len(source_rows),'duplicate source row assignment')
    require(all(indexed.get(sid) for sid in ('pilot','pair_01','pair_02')),
            'insufficient required independent sources: pilot, pair_01 and pair_02 must all pass; explicit baseline-only mode is available')
    result=[part('calibration',baseline_id,connected+120*FS,connected+180*FS)]
    previous_end=result[0]['end_frame']
    for split,sid,count in [('train','pilot',129),('evaluation','pair_01',97),('holdout','pair_02',1)]:
        cutoff=previous_end+(10*FS if split!='train' else 0)
        chosen=[r for r in indexed[sid] if r['start_frame']>=cutoff][:count]
        require(len(chosen)==count,f'insufficient {split}: fixed {sid} needs {count} clean windows after the required embargo')
        result.append(part(split,sid,chosen[0]['start_frame'],chosen[-1]['end_frame']))
        previous_end=chosen[-1]['end_frame']
    return result


def normalize_events(stage,events):
    if stage=='baseline':
        require(len(events)==2 and [(e.get('id'),e.get('phase')) for e in events]==[(8001,'disconnected'),(8002,'connected')],
                'baseline needs actual disconnected/connected markers only')
        require(all(e.get('arm') is None for e in events),'baseline must not contain a pulse arm')
        return []
    require(stage=='pilot' or re.fullmatch(r'pair_\d\d',stage),'unsupported stimulation stage')
    require(len(events)==2 and sorted(e.get('arm','') for e in events)==['sham','true'],'exactly one true and one sham recorded event required')
    return [dict(start_frame=e['frame'],end_frame=e['frame']+PULSE_DURATION_FRAMES,event_id=e['id'],arm=e['arm'],pair=e['pair'],
                 event_frame=e['frame'],duration_basis='pinned_v2_hardware_sequence_delay_4_plus_4_samples') for e in events]


def validate_envelope(receipt,stage,registration,state):
    result=receipt.get('result',{})
    require(result.get('status')=='pass' and result.get('errors')==[],'v2 QC did not pass cleanly')
    require(receipt.get('result_sha256')==v2_digest(result),'v2 QC result digest mismatch')
    config=registration['config'];session=config['session_id']
    require(config['chip_id']=='28073' and config['device_model']=='MaxOne' and config['r2_enabled'] is False and config['training_enabled'] is False,'registered v2 identity/R2 mismatch')
    for key,expected in [('stage',stage),('session_id',session),('binding_sha256',registration['binding_sha256'])]:
        require(receipt.get(key)==expected and result.get(key)==expected,'v2 envelope binding mismatch: '+key)
    require(receipt['config_sha256']==v2_digest(config),'v2 config hash mismatch')
    require(result['source']==receipt['source'],'v2 raw source binding mismatch')
    require(result.get('r2_enabled') is False and result.get('training_enabled') is False,'v2 acquisition R2/training not disabled')
    if state is not None:
        require(state['session_id']==session,'run state session differs')
        matching=[r for r in state['records'] if Path(r['path']).resolve()==Path(receipt['source']['path']).resolve()]
        require(len(matching)==1 and bool(matching[0].get('closed_utc')),'source lacks a unique actual closed-record receipt')
    return result


def load_admission(run,stage,registration,state):
    receipt_path=run/'qc'/(stage+'.json');receipt=read(receipt_path)
    try:result=validate_envelope(receipt,stage,registration,state)
    except ValueError as error:raise ValueError(stage+': '+str(error)) from error
    request_path=Path(receipt['request_path']).resolve()
    require(request_path.is_relative_to((run/'qc').resolve()),'QC request outside this run')
    request=read(request_path);unsigned=dict(request);claimed=unsigned.pop('request_sha256',None)
    require(claimed==receipt['request_sha256']==v2_digest(unsigned),'original v2 request digest mismatch')
    for key in ('source','baseline_source','stage','session_id','config_sha256','binding_sha256'):
        require(request[key]==receipt[key],'request/receipt binding differs: '+key)
    require(request['config']==registration['config'],'request differs from original registration')
    source=receipt['source'];raw=Path(source['path']).resolve()
    require(raw.is_relative_to(run.resolve()),'actual raw source must stay in this run directory')
    before=raw.stat()
    require(before.st_size==source['size_bytes'] and before.st_mtime_ns==source['mtime_ns'],'closed raw file changed since QC')
    import h5py
    import numpy as np
    with h5py.File(raw,'r') as f:
        d=f['data_store/data0000'];frames=d['groups/routed/frame_nos'];m=result['metadata']
        require(m['chip_id']=='28073' and m['fs']==FS and m['frame_gaps']==0 and m['raw_rows']==1024 and m['raw_axis_layout']=='physical_channel_index','QC metadata not registered physical MaxOne layout')
        require(list(d['groups/routed/raw'].shape)==[1024,len(frames)],'actual raw physical shape differs')
        require([int(frames[0]),int(frames[-1]),len(frames)]==[m['first_frame'],m['last_frame'],m['frames']],'actual raw frame bounds differ from QC')
        require(f['wellplate/id'][0]==b'28073' and float(d['settings/sampling'][0])==FS,'actual H5 identity differs')
        raw_events=[]
        for row in d['events']:
            message=row['eventmessage'];message=message.decode().rstrip('\x00') if isinstance(message,bytes) else str(message)
            payload=json.loads(message);frame=int(row['frameno']);pair=event_pair(payload['pair']) if 'pair' in payload else None
            raw_events.append(dict(id=int(row['eventid']),eventtype=int(row['eventtype']),frame=frame,time_s=(frame-m['first_frame'])/FS,
                                   arm=payload.get('arm'),pair=pair,phase=payload.get('phase'),session=payload.get('session'),message=payload))
        require(raw_events==result['events'],'actual H5 events differ from original v2 QC receipt')
    parent=dict(sha256=sha(raw),bytes=before.st_size)
    after=raw.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'raw changed while hashing')
    return dict(stage=stage,receipt_path=receipt_path,receipt=receipt,result=result,request_path=request_path,raw=raw,parent_raw=parent,
                events=normalize_events(stage,result['events']))


def prepare(run,output,mode,session_label,protocol_source=None):
    run=Path(run).resolve();output=Path(output).resolve()
    require(mode in ('stimulated','baseline-only'),'explicit stimulated or baseline-only mode required')
    require(not output.exists(),'new input output directory required')
    require(session_label.startswith('temporary_'),'temporary session label required; biological ID remains null')
    registration=read(run/'registration.json');state=read(run/'state.json')
    if mode=='stimulated':
        for stage in ('baseline','pilot','pair_01','pair_02'):
            path=run/'qc'/(stage+'.json')
            require(path.is_file(),'fixed stimulated plan missing '+stage+' receipt; explicit baseline-only fallback remains separate')
            validate_envelope(read(path),stage,registration,state)
    baseline=load_admission(run,'baseline',registration,state)
    baseline_result=baseline['result'];connected=next(e['frame'] for e in baseline_result['events'] if e['id']==8002)
    require(abs(baseline_result['connected_time_s']-(connected-baseline_result['metadata']['first_frame'])/FS)<1e-9,'baseline connected frame/time disagreement')
    admissions=[baseline];skipped=[]
    if mode=='stimulated':
        require(protocol_source is not None and sha(protocol_source)==PROTOCOL_SHA,'pinned actual v2 protocol source required for eight-sample pulse mask')
        pilot=load_admission(run,'pilot',registration,state)
        require(pilot['receipt']['baseline_source']==baseline['receipt']['source'],'pilot did not verify this baseline')
        require(pilot['result']['holdout19']['passed'] is True and pilot['result']['holdout19']['passed_count']>=18,'frozen pilot ROI baseline holdout not passed')
        scope=pilot;features=pilot['result']['roi_channels'];admissions.append(pilot)
        for stage in ('pair_01','pair_02'):
            item=load_admission(run,stage,registration,state)
            require(item['result']['roi_channels']==features,'pair ROI differs from frozen pilot ROI')
            admissions.append(item)
    else:
        scope=baseline;features=baseline_result['roi_channels']
    require(features==sorted(set(features)) and len(features)>=50 and all(type(c) is int and 0<=c<1024 for c in features),'approved ROI malformed or too small')
    require(set(features)<=set(baseline_result['roi_channels']),'frozen ROI exceeds baseline-qualified candidates')
    admissions.sort(key=lambda a:a['result']['metadata']['first_frame'])
    for a,b in zip(admissions,admissions[1:]):
        require(a['result']['metadata']['last_frame']<b['result']['metadata']['first_frame'],'recorded hardware frame clock overlaps/resets')
    output.mkdir();evidence=output/'evidence';evidence.mkdir()
    def preserve(path,name):
        destination=evidence/name
        if not destination.exists():shutil.copyfile(path,destination)
        require(sha(destination)==sha(path),'original evidence copy hash mismatch')
        return dict(path=destination.relative_to(output).as_posix(),sha256=sha(destination),bytes=destination.stat().st_size)
    registration_ref=preserve(run/'registration.json','v2_registration.json')
    # Freeze the state snapshot actually used for closed-file admission, not a later concurrent update.
    state_path=evidence/'v2_state_snapshot.json';write(state_path,state)
    state_ref=dict(path=state_path.relative_to(output).as_posix(),sha256=sha(state_path),bytes=state_path.stat().st_size)
    scope_ref=preserve(scope['receipt_path'],'v2_scope_'+scope['stage']+'.json')
    source_rows=[];entries=[];global_events=[e for a in admissions for e in a['events']]
    for admission in admissions:
        stage=admission['stage'];result=admission['result'];source_id=stage
        original_ref=preserve(admission['receipt_path'],'v2_'+stage+'.json')
        request_ref=preserve(admission['request_path'],'v2_'+stage+'_request.json')
        require(read(output/original_ref['path'])==admission['receipt'],'v2 QC receipt changed during preparation')
        qc=dict(schema='mea.28073.source_qc.v1',source_id=source_id,parent_raw=admission['parent_raw'],finalized=True,
                passed_for_declared_scope=True,excluded_intervals=[],declared_out_of_scope_events=result['metadata']['invalid_spikes_excluded'],
                approved_feature_channels=features,original_v2_receipt=original_ref,scope_v2_receipt=scope_ref,
                evidence_files=[original_ref,scope_ref,request_ref,registration_ref,state_ref],
                scope_basis='passed_v2_frozen_pilot_roi_with_baseline_holdout_recheck' if mode=='stimulated' else 'passed_v2_baseline_candidate_roi_only',
                qualification='existing_v2_QC_decision_only; no_new_waveform_QC_claim',raw_physical_rows=1024)
        qcp=output/(stage+'.qc.json');write(qcp,qc)
        alignment='recorded_hardware_frames' if admission['events'] else 'none_confirmed_by_acquisition_log'
        stim=dict(schema='mea.28073.stimulation_frames.v1',source_id=source_id,complete=True,alignment=alignment,events=admission['events'],
                  original_v2_receipt=original_ref,hardware_event_basis='actual_H5_event_table_identical_to_v2_receipt',
                  sequence_protocol_sha256=PROTOCOL_SHA if admission['events'] else None,
                  includes_true_stimulation=any(e.get('arm')=='true' for e in admission['events']))
        stimp=output/(stage+'.stim.json');write(stimp,stim)
        view=output/(stage+'.spikeview.h5');export_receipt=export(admission['raw'],view,qcp)
        entry=dict(source_id=source_id,kind='spikeview',path=view.name,sha256=export_receipt['sha256'],bytes=view.stat().st_size,well_id=0,
                   raw_interval=export_receipt['raw_interval'],parent_raw=admission['parent_raw'],qc_status='passed_for_declared_scope',
                   qc_receipt=dict(path=qcp.name,sha256=sha(qcp)),qc_exclusions=[],declared_out_of_scope_events=qc['declared_out_of_scope_events'],feature_channels=features,
                   stimulation=dict(alignment=alignment,events=admission['events'],evidence=dict(path=stimp.name,sha256=sha(stimp))))
        if stage=='baseline':entry.update(window_anchor_frame=connected,anchor_event_id=8002)
        rows,identity=read_spikeview(view,entry);clean,excluded=clean_windows(rows,global_events,[],FS)
        source_rows.append((stage,clean));entries.append(entry)
        write(output/(stage+'.export.json'),export_receipt)
    partitions=baseline_partitions('baseline',connected) if mode=='baseline-only' else stimulated_partitions('baseline',connected,[(sid,rows) for sid,rows in source_rows if sid!='baseline'])
    manifest=dict(schema='mea.28073.qc_manifest.v1',chip_id='28073',session_label=session_label,biological_specimen_id=None,
                  provenance='real_acquisition',finalized=True,qc_status='passed_for_declared_scope',source_association='prerecorded_action_independent',
                  frame_clock='session_continuous_hardware_frame',sources=entries,partitions=partitions,preparation_mode=mode,
                  includes_stimulation_recordings=mode=='stimulated',partition_policy='root_locked_v2_baseline_pilot_pair01_pair02_20261117',
                  preparation_code_sha256=sha(__file__),scope_v2_receipt_sha256=scope_ref['sha256'])
    pending=output/'manifest.pending.json';write(pending,manifest)
    verified=derive(pending)
    selected_post_true={split:sum(any(e.get('arm')=='true' and e['end_frame']+2*FS<=w['start_frame']
                                      for admission in admissions if admission['stage']==w['source_id'] for e in admission['events'])
                                  for w in verified['windows'] if w['split']==split)
                        for split in ('calibration','train','evaluation')}
    if mode=='stimulated':
        require(selected_post_true['train']>0,'stimulated mode must include at least one clean train window after an actual true pulse and its 2-second mask')
    pending.rename(output/'manifest.json')
    report=dict(passed=True,mode=mode,manifest_sha256=sha(output/'manifest.json'),selected_windows=len(verified['windows']),
                feature_channels=len(features),scope_v2_receipt_sha256=scope_ref['sha256'],skipped_failed_sources=skipped,
                source_stimulation_events=sum(len(a['events']) for a in admissions),baseline_only_not_stimulated=mode=='baseline-only',
                selected_clean_windows_after_true_pulse_by_split=selected_post_true,
                biological_specimen_id=None,training_started=False)
    write(output/'preparation_receipt.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',required=True,choices=['stimulated','baseline-only']);p.add_argument('--session-label',required=True)
    p.add_argument('--protocol-source');a=p.parse_args()
    print(canonical(prepare(a.run_dir,a.output,a.mode,a.session_label,a.protocol_source)))
