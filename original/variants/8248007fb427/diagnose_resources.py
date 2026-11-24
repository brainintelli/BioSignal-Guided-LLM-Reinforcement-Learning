"""Read-only resource samples and stage durations; no model outcome inspection."""
import hashlib,json,sys
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c'
code=r'''
import json,time,subprocess,os,statistics
from pathlib import Path
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    return dict(exit=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
root=Path('/home/researcher/mea-stim-offline-20261117/experiment01')
result=dict(epoch=time.time(),cpu_count=os.cpu_count(),load=os.getloadavg())
result['memory']=Path('/proc/meminfo').read_text()
result['gpu_samples']=[]
for i in range(10):
    result['gpu_samples'].append(dict(epoch=time.time(),data=cmd(['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.max.sm','--format=csv,noheader,nounits'])))
    if i<9:time.sleep(2)
result['gpu_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
result['top_cpu']=cmd(['ps','-eo','pid,ppid,pcpu,pmem,rss,stat,comm','--sort=-pcpu'])['stdout'].splitlines()[:16]
result['vmstat']=cmd(['vmstat','1','5'])
result['pressure']={name:(Path('/proc/pressure')/name).read_text() for name in ['cpu','memory','io']}
status=json.loads((root/'batch_status.json').read_text())
result['accepted']=sum(len(x['stages'])==3 and all(y['exit_code']==0 for y in x['stages']) for x in status['cells'])
result['complete']=status['complete'];result['passed']=status['passed']
result['stages']=[dict(block=x['block'],arm=x['arm'],stages=x['stages']) for x in status['cells']]
print(json.dumps(result,indent=2))
'''
i,o,e=c.exec_command('/home/researcher/meadowmind-gpu/m6-venv/bin/python -',timeout=50)
i.write(code);i.channel.shutdown_write();out=o.read().decode();err=e.read().decode();exit_code=o.channel.recv_exit_status()
if exit_code:raise RuntimeError(err)
value=json.loads(out);(r/'resource_diagnosis.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(out)
c.close()
