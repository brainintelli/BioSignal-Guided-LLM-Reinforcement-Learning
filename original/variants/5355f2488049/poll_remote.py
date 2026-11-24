"""Score-blind compact status; never prints interim evaluation metrics."""
import hashlib,json,sys,time
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent;b='/home/researcher/mea-stim-offline-20261117'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c'
s=c.open_sftp();result={'epoch':time.time()}
for name in ['job.json','experiment01/batch_status.json']:
    try:
        with s.open(b+'/'+name) as f:data=f.read()
    except FileNotFoundError:continue
    (r/Path(name).name).write_bytes(data);value=json.loads(data)
    if name=='job.json':result['job']=value
    else:
        result.update(complete=value['complete'],passed=value['passed'],accepted_cells=sum(len(x['stages'])==3 and all(y['exit_code']==0 for y in x['stages']) for x in value['cells']))
        if value.get('error'):result['error']=value['error']
        if value['cells']:
            last=value['cells'][-1];result['current']={'block':last['block'],'arm':last['arm'],'finished_stages':last['stages']}
            stages=['train','evaluate','verify'];stage=stages[min(2,len(last['stages']))]
            p=b+'/experiment01/'+f"b{last['block']:02d}_{last['arm']}_{stage}.log"
            try:
                size=s.stat(p).st_size
                with s.open(p) as f:f.seek(max(0,size-10000));tail=f.read().decode(errors='replace')
                events=[]
                for line in tail.splitlines():
                    try:q=json.loads(line)
                    except ValueError:continue
                    if isinstance(q,dict) and 'event' in q:events.append({k:q[k] for k in ['event','index','round','phase','context','seconds','applied_steps'] if k in q})
                result['recent_progress']=events[-3:]
                if result.get('error'):result['failure_log_tail']=tail[-2400:]
            except FileNotFoundError:pass
if 'job' in result:
    for k in ['worker_pid','child_pid']:
        try:s.stat('/proc/'+str(result['job'][k]));result[k+'_present']=True
        except FileNotFoundError:result[k+'_present']=False
(r/'latest_status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));s.close();c.close()
