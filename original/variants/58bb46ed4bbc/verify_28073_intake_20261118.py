import json,sys,hashlib,datetime,math
from pathlib import Path
out=Path(sys.argv[1])
def load(name): return json.loads((out/name).read_text(encoding='utf-8-sig'))
def save(name,obj):
    with (out/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
a=load('routeB-audit.json'); selection=load('routeB-site-candidates.json'); route=load('routeB-route-evidence.json'); b=load('routeB-binding.json'); reg=load('registration.json')
assert b['started_at']==reg['started_at'] and selection['suggested']==a['selected_qc']['suggested']
assert route['three_distinct'] and len(set(r['unit'] for r in route['readback'].values()))==3
assert hashlib.sha256((out/'routeB.cfg').read_bytes()).hexdigest()==route['route_sha256']==b['route_sha256']
electrodes={k:r['electrode'] for k,r in selection['suggested'].items()}
assert all(electrodes[k]==r['electrode'] and r['verified'] for k,r in route['readback'].items())
save('load-calibration-intake.template.json',dict(session_id=b['session_id'],chip_id=b['chip_id'],route_sha256=b['route_sha256'],electrodes=electrodes,profile=route['profile'],dac_lsb_mv=route['dac_lsb_mv'],measurements={k:{'unit':r['unit'],'polarity':None,'positive_peak_mv':None,'negative_peak_mv':None,'trace_file':None,'trace_sha256':None} for k,r in route['readback'].items()}))
save('preflight-status.json',dict(as_of=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),registration=reg,acquisition_subrun=b['session_id'],route_selection_reason='Initial mapped F had no stimulation-unit readback. A distinct passive mapping retained the active electrodes and was rebaselined under the same clock before any response trial.',chip_read_back=a['chip_id'],organoid_id_user_reported=b['organoid_id'],final_baseline_seconds=a['duration_s'],mapped_channels=a['mapped_channels'],active_channels_full=a['active_channels_full'],spikes_full=a['spikes_full'],frame_gaps=a['frame_gaps'],qualified_channels=a['passed_all_gates'],sites=selection['suggested'],distances_um=selection['distances_um'],router_readback=route['readback'],nonzero_output_attempts_in_this_workflow=0,completed_response_trials={s:0 for s in 'HWF'},qualification_status='pending_measured_load_waveforms',qualification_passed=None,post_confusion=None,pre_confusion=None,training_jobs_started=0,state2_training_source='measured_MEA',r2_policy={'verified_quiet':1,'verified_stimulation':-1},baseline_sources=[{'source':load('baseline-audit.json')['path'],'sha256':load('baseline-audit.json')['raw_sha256']},{'source':a['path'],'sha256':a['raw_sha256']}]))
print('Final site binding and empty measurement fields verified.')
