import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;src=root.parents[1]/'tools/mea_multi_recording';base='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp();s.mkdir(base+'/bounded_code_v2')
for name in ['bounded.py','current_run.py','multi_prepare.py','multi_protocol.py']:s.put(str(src/name),base+'/bounded_code_v2/'+name)
s.put(str(root/'bounded_routes.json'),base+'/bounded_routes.json')
launcher="""import json,os,subprocess
from pathlib import Path
b=Path('/home/researcher/mea_full_28080_20261117_1406')
log=(b/'bounded_run.log').open('x')
p=subprocess.Popen(['/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python',str(b/'bounded_code_v2/current_run.py'),'--run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,HDF5_PLUGIN_PATH='/pub/soft/MaxLab/so',PYTHONUNBUFFERED='1'))
(b/'bounded_job.json').write_text(json.dumps({'pid':p.pid,'automatic_restart':False}))
print(p.pid)
"""
with s.open(base+'/bounded_launch_once_v2.py','wx') as f:f.write(launcher)
_,out,err=c.exec_command('/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python '+base+'/bounded_launch_once_v2.py')
answer=out.read().decode()+err.read().decode();code=out.channel.recv_exit_status()
(root/'bounded_launch_receipt.log').write_text(answer);print(answer)
s.close();c.close();raise SystemExit(code)
