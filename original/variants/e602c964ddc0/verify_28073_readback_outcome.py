from pathlib import Path
from types import SimpleNamespace
import math,os,sys,json,hashlib,datetime
root=Path.cwd()
pkg=root/'output/pair28073-20261118T070238Z/mea-readback-runtime'
sys.path[:0]=[str(root/'.tools/mea_baseline_deps'),str(pkg/'mea_stimulation_v2')]
os.environ['HDF5_PLUGIN_PATH']=str(root/'.tools/mea_maxwell_hdf5_plugin')
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability
out=Path(sys.argv[1])
def read(p): return json.loads(p.read_text(encoding='utf8'))
b=read(out/'binding.json');q=read(out/'connected-baseline-qc-detail.json');closed=read(out/'closed.json')
a=[json.loads(x) for x in (out/'api_audit.jsonl').read_text(encoding='utf8').splitlines()]
commands=[r['command'] for r in a if r['kind']=='request']
assert not any(c.startswith('sequence_send ') for c in commands)
acks={r['command']:r['reply'] for r in a if r['kind']=='ack'}
assert all(acks[f'mea_set_stimulation_unit {i} 0 0 0 0 0 0'].lower()=='ok' for i in range(32)) and 'mea_set_dac 0 512 512' in acks
assert closed['attempts']==0 and not closed['errors']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()
cfg=SimpleNamespace(threshold_h5_dataset=b['threshold_h5_dataset'])
counts={};paths=[('before',out.parent/'baseline-final-route.raw.h5'),('connected',out/'connected-baseline.raw.h5')]
for label,p in paths:
    if label=='connected':
        assert sha(p)==q['sha256'] and p.stat().st_size==q['raw_bytes']
    with H5Recording(p,cfg) as r:
        assert r.metadata['chip_id']==b['chip_id'] and not r.events and r.metadata['frame_gaps']==0
        if label=='before': mapping=r.mapping
        else: assert mapping==r.mapping
        pre=r.sample(q['roi'],3)
        counts[label]=[sum(stability(pre,r.sample(q['roi'],s))['channel_pass']) for s in (20,40,55)]
assert counts['connected']==[v['passed_channels'] for v in q['windows']]
result={'stage':'closed_at_connected_baseline_qc','sample_id':b['organoid_id'],'chip_id':b['chip_id'],'session_id':b['session_id'],
'started_at':b['started_at'],'closed_at':datetime.datetime.fromtimestamp(closed['time'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),
'output_verification':b['output_verification'],'measured_electrode_voltage_mv':None,'readback':q['device_readback'],'connected_baseline':{'raw_file':str(paths[1][1].resolve()),'sha256_verified':q['sha256'],'seconds':q['metadata']['duration_seconds'],'frame_gaps':q['metadata']['frame_gaps'],'native_spikes':q['spikes'],'mapped_channels':q['mapped_channels'],'roi_count':q['roi_n'],'minimum_passing_channels':q['required_count'],'window_offsets_s':[20,40,55],'reference_s':3,'stable_counts_by_window':counts['connected'],'passed':q['baseline_pass']},'same_channel_comparison_before_routing':{'stable_counts_by_window':counts['before'],'roi_definition':'same 112 connected-run active independent channels on the same channel-electrode mapping; descriptive reference only'},'sequence_send_requests':0,'stimulus_attempts':0,'stimulated_trials':{'H':0,'W':0,'F':0},'response_confusion_matrix':None,'gpu_training_jobs':0,'r2_rewards_issued':0,'cleanup_errors':closed['errors'],'all_32_off_acknowledged':True}
with (out/'test-outcome.json').open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'stable':counts,'sha256':q['sha256'],'closed':result['closed_at'],'attempts':closed['attempts']}))
