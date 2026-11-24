import hashlib,json,sys,time
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;base='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();result={'epoch':time.time()}
try:s.stat(base+'/bounded_stable_job.json');folder='multi_bounded_resume2';jobname='bounded_stable_job.json';logname='bounded_stable_run.log'
except FileNotFoundError:folder='multi_bounded_recording';jobname='bounded_job.json';logname='bounded_run.log'
result['folder']=folder
names=[jobname,logname]+[folder+'/'+n for n in ['progress.json','zero_qc.json','zero_batches.json','stimulated_routes.json','true_preview.json','batch_result.json','recording_times.json','connected_routes.json','supplement_source_sha256.json','binding.json']]
for name in names:
    try:
        with s.open(base+'/'+name) as f:data=f.read()
    except FileNotFoundError:continue
    p=root/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
    if name==jobname:result['job']=json.loads(data)
    elif name.endswith('progress.json'):result['progress']=json.loads(data)
    elif name.endswith('zero_qc.json'):
        q=json.loads(data);result['zero_qc']={k:v for k,v in q.items() if k not in ('events','eligible_routes')};result['eligible_count']=len(q['eligible_routes'])
    elif name.endswith('batch_result.json'):
        q=json.loads(data);result['batch_result']={k:v for k,v in q.items() if k!='timings'}
    elif name==logname:result['log_tail']=data.decode(errors='replace')[-1800:]
for name in ['identity.raw.h5','stimulus_post.raw.h5']:
    try:result[name]=s.stat(base+'/'+folder+'/'+name).st_size
    except FileNotFoundError:pass
if 'job' in result:
    try:
        with s.open('/proc/'+str(result['job']['pid'])+'/stat') as f:result['process_state']=f.read().decode().split(') ')[1].split()[0]
    except FileNotFoundError:result['process_state']='absent'
try:
    p=base+'/'+folder+'/api_audit.jsonl';size=s.stat(p).st_size
    with s.open(p) as f:f.seek(max(0,size-3500));tail=f.read().decode(errors='replace')
    result['api_tail']=tail.splitlines()[-8:]
except FileNotFoundError:pass
(root/'bounded_last_poll.json').write_text(json.dumps(result,indent=2))
if 'progress' in result and 'batch_result' in result['progress']:
    result['progress']['batch_result']={k:v for k,v in result['progress']['batch_result'].items() if k!='timings'}
print(json.dumps(result,indent=2));s.close();c.close()
