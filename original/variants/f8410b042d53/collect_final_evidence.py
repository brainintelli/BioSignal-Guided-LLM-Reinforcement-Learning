"""Read-only collection after the worker has exited; no device API calls."""
import hashlib,json,stat,sys,time
from pathlib import Path
import paramiko

ROOT=Path(__file__).resolve().parent
REMOTE='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp()
with s.open(REMOTE+'/event_resume_job.json') as f:job=json.loads(f.read())
if job.get('exit_code') is None:raise RuntimeError('Worker still active; final evidence is not ready')
supplement_jobs={}
for filename in ['bounded_job.json','bounded_resume_job.json','bounded_stable_job.json']:
    try:
        with s.open(REMOTE+'/'+filename) as f:more=json.loads(f.read())
    except FileNotFoundError:continue
    supplement_jobs[filename]=more
    try:
        with s.open('/proc/'+str(more['pid'])+'/stat') as f:state=f.read().decode().split(') ')[1].split()[0]
        if state!='Z':raise RuntimeError('Supplemental worker active: '+filename)
    except FileNotFoundError:pass
out=ROOT/'final_evidence';out.mkdir(exist_ok=True)
inventory=[]
def collect(relative=''):
    for entry in s.listdir_attr(REMOTE+('/'+relative if relative else '')):
        name=(relative+'/' if relative else '')+entry.filename
        if stat.S_ISDIR(entry.st_mode):
            if (not relative and entry.filename in ('recording','recording_resume1','recording_resume2','sequence_upload_only','multi_bounded_recording','multi_bounded_resume1','multi_bounded_resume2')) or (relative.startswith('multi_bounded_') and entry.filename=='slots'):
                collect(name)
            continue
        if not stat.S_ISREG(entry.st_mode):continue
        if Path(name).suffix not in ('.json','.jsonl','.log','.cfg','.txt') or entry.st_size>5*1024*1024:continue
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with s.open(REMOTE+'/'+name) as f:data=f.read()
        dest.write_bytes(data)
        inventory.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
collect()
processes={}
server_pids=[]
for entry in s.listdir('/proc'):
    if not entry.isdigit():continue
    try:
        with s.open('/proc/'+entry+'/comm') as f:comm=f.read().decode().strip()
        if comm=='mxwserver':server_pids.append(int(entry))
    except FileNotFoundError:continue
if len(server_pids)!=1:raise RuntimeError('Require one inspectable current server')
for label,pid in [('maxlab_server',server_pids[0]),('forwarder',3366458),('worker',job.get('worker_pid')),('child',job.get('child_pid'))]+[(k,v['pid']) for k,v in supplement_jobs.items()]:
    if pid is None:continue
    try:
        with s.open('/proc/'+str(pid)+'/cmdline') as f:cmd=f.read().replace(b'\0',b' ').decode(errors='replace')
        processes[label]={'pid':pid,'present':True,'cmdline':cmd}
    except FileNotFoundError:processes[label]={'pid':pid,'present':False}
open_h5=[]
for fd in s.listdir('/proc/'+str(server_pids[0])+'/fd'):
    try:target=s.readlink('/proc/'+str(server_pids[0])+'/fd/'+fd)
    except FileNotFoundError:continue
    if target.endswith('.h5'):open_h5.append(target)
receipt=dict(collected_epoch=time.time(),job=job,metadata=inventory,processes=processes,maxlab_open_h5=open_h5,
             device_commands_sent=0,stimulation_retried=False,supplement_jobs=supplement_jobs)
(out/'collection_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'files':len(inventory),'job':job,'processes':processes,'maxlab_open_h5':open_h5},indent=2))
s.close();c.close()
