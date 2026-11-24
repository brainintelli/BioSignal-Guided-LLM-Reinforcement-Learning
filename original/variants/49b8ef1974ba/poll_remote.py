"""Read status and closed metadata only. Password is provided on stdin."""
import hashlib,json,sys,time
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parent
REMOTE='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();summary={'local_epoch':time.time()}
try:s.stat(REMOTE+'/resume_job.json');folder='recording_resume1';jobname='resume_job.json'
except FileNotFoundError:folder='recording';jobname='job.json'
try:s.stat(REMOTE+'/event_resume_job.json');folder='recording_resume2';jobname='event_resume_job.json'
except FileNotFoundError:pass
summary['active_recording_dir']=folder
names=[jobname,'binding.json']+[folder+'/'+x for x in ['progress.json','identity_qc.json','baseline_qc.json',
       'stimulus_post_qc.json','selected_route.json','sequence_preview.json','raw_manifest.json',
       'recording_times.json','presend_growth.json']]
for name in names:
    target=ROOT/name
    immutable=name.endswith(('_qc.json','selected_route.json','sequence_preview.json','raw_manifest.json','presend_growth.json')) or name=='binding.json'
    if immutable and target.exists():data=target.read_bytes()
    else:
        try:
            with s.open(REMOTE+'/'+name) as f:data=f.read()
        except FileNotFoundError:continue
        target.parent.mkdir(exist_ok=True);target.write_bytes(data)
    obj=json.loads(data)
    if name==jobname:summary['job']=obj
    elif name==folder+'/progress.json':summary['progress']={k:v for k,v in obj.items() if k not in ['stages','baseline_qc','audit','verified_preoutput_recovery','verified_event_setup_recovery']}
    elif name.endswith('_qc.json'):summary[name]={k:v for k,v in obj.items() if k in ['status','errors','chip_id','duration_s','candidate','spike_count','active_independent_channels','independent_channels','max_rail_fraction','events']}
    elif name.endswith('selected_route.json'):summary['route']=obj
for name in ['identity.raw.h5','baseline.raw.h5','stimulus_post.raw.h5']:
    try:
        stat=s.stat(REMOTE+'/'+folder+'/'+name);summary[name]={'bytes':stat.st_size,'mtime':stat.st_mtime}
    except FileNotFoundError:pass
if summary.get('job',{}).get('exit_code') is not None:
    prefix='event_resume_' if folder=='recording_resume2' else 'resume_' if folder=='recording_resume1' else ''
    for name in [prefix+'run.log',prefix+'worker.log',folder+'/api_audit.jsonl']:
        try:s.get(REMOTE+'/'+name,str(ROOT/name))
        except FileNotFoundError:pass
(ROOT/'last_poll.json').write_text(json.dumps(summary,indent=2))
progress=summary.get('progress',{})
compact={'epoch':summary['local_epoch'],'folder':folder,'exit':summary.get('job',{}).get('exit_code'),
         'status':progress.get('status'),'stage':progress.get('stage'),
         'remaining_s':round(progress.get('phase_remaining_seconds',0),1),
         'stimulus_attempted':progress.get('stimulus_attempted'),'acknowledged':progress.get('stimulus_acknowledged'),
         'raw_bytes':{k:v['bytes'] for k,v in summary.items() if k.endswith('.h5')},
         'qc':{k:{'status':v['status'],'errors':v['errors']} for k,v in summary.items() if k.endswith('_qc.json')}}
if progress.get('error'):compact['error']=progress['error']
if progress.get('cleanup_errors'):compact['cleanup_errors']=progress['cleanup_errors']
if progress.get('stage') in ('baseline_qc','cleanup') and summary.get('job',{}).get('exit_code') is None:
    try:
        with s.open('/proc/'+str(int(summary['job']['child_pid']))+'/io') as f:io=f.read().decode()
        compact['audit_io']={k:int(v) for k,v in (line.split(':') for line in io.splitlines()) if k in ('rchar','read_bytes')}
    except FileNotFoundError:pass
print(json.dumps(compact,indent=2));s.close();c.close()
