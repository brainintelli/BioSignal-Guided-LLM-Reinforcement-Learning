import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;base='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();out=root/'bounded_failure';out.mkdir(exist_ok=True)
for name in ['multi_bounded_recording/api_audit.jsonl','multi_bounded_recording/progress.json','bounded_run.log']:
    s.get(base+'/'+name,str(out/Path(name).name))
records=[json.loads(x) for x in (out/'api_audit.jsonl').read_text().splitlines()]
first=next((i for i,x in enumerate(records) if 'error' in x),len(records))
result={'first_error_context':records[max(0,first-10):first+3],
        'sequence_send_requests':[x for x in records if 'mono' in x and x.get('command','').startswith('sequence_send')]}
for command in ['ps -p 1551234,3366458,3693252,3602374 -o pid,stat,pcpu,wchan:30,args','ss -lntp']:
    _,stdout,stderr=c.exec_command(command,timeout=10)
    result[command]=stdout.read().decode()+stderr.read().decode()
fds={}
try:
    for fd in s.listdir('/proc/1551234/fd'):
        try:target=s.readlink('/proc/1551234/fd/'+fd)
        except FileNotFoundError:continue
        if fd in ('0','1','2') or '.h5' in target or '.log' in target:fds[fd]=target
except FileNotFoundError:pass
result['server_fds']=fds
(out/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));s.close();c.close()
