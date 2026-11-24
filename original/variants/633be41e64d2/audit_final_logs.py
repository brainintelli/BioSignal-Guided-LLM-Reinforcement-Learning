"""Read only local copies of closed session evidence; no device calls."""
import json,re
from pathlib import Path
r=Path(__file__).resolve().parent
e=r/'final_evidence'
result={'training_admitted':False,'alpha':1,'r2_mode':'record_only','realtime_training':False,'supplemental':{}}
for folder in ['multi_bounded_recording','multi_bounded_resume1','multi_bounded_resume2']:
    p=e/folder
    report=json.loads((p/'progress.json').read_text())
    rows=[json.loads(x) for x in (p/'api_audit.jsonl').read_text().splitlines()]
    intents=[x for x in rows if 'command' in x and 'ack' not in x and 'error' not in x]
    sends=[x['command'] for x in intents if x['command'].startswith('sequence_send ')]
    true=[x for x in sends if not x.startswith('sequence_send bounded_zero_')]
    latest={}
    for x in rows:
        if 'ack' not in x:continue
        c=x.get('command','').split()
        if c and c[0]=='mea_set_stimulation_unit':latest[int(c[1])]=(x['command'],x['ack'])
    off=all(latest.get(u)==(f'mea_set_stimulation_unit {u} 0 0 0 0 0 0','OK') for u in range(32))
    dac=[x for x in rows if x.get('command','').startswith('mea_set_dac ') and 'ack' in x]
    dac_zero=bool(dac) and dac[-1]['command']=='mea_set_dac 0 512 512' and dac[-1]['ack']=='OK'
    if true or report.get('true_attempted'):raise RuntimeError('Unexpected supplemental true output')
    result['supplemental'][folder]={'status':report['status'],'error':report.get('error'),'zero_send_intents':len(sends),
        'true_send_intents':len(true),'true_attempted':report['true_attempted'],'cleanup_errors':report['cleanup_errors'],
        'last_ack_all32_off':off,'last_ack_dac512':dac_zero,'finish_epoch':report['wall']}
last=result['supplemental']['multi_bounded_resume2']
if not last['last_ack_all32_off'] or not last['last_ack_dac512'] or last['cleanup_errors']:raise RuntimeError('Final shutdown not verified')
receipt=json.loads((e/'collection_receipt.json').read_text())
if receipt['maxlab_open_h5']:raise RuntimeError('H5 still open')
for label,p in receipt['processes'].items():
    if label.endswith('_job.json') and p['present']:raise RuntimeError('Supplemental process remains')
result['final_device_status']='all32_off_and_DAC512_acknowledged_no_open_H5_workers_exited'
result['original_stimulus_post_qc']=json.loads((e/'recording_resume2/stimulus_post_qc.json').read_text())['status']
result['original_pulses']=15
result['supplemental_true_pulses']=0
(r/'final_log_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
