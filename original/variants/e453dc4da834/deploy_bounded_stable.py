"""Deploy frozen corrected source and run offline tests; separate explicit launch."""
import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;src=root.parents[1]/'tools/mea_multi_recording';base='/home/researcher/mea_full_28080_20261117_1406'
mode=sys.argv[1] if len(sys.argv)==2 else 'deploy'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp()
if mode=='deploy':
    s.mkdir(base+'/bounded_code_v7');s.mkdir(base+'/bounded_code_v7/tests');manifest={}
    for p in list(src.glob('*.py'))+list((src/'tests').glob('*.py')):
        rel=p.relative_to(src).as_posix();s.put(str(p),base+'/bounded_code_v7/'+rel);manifest[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'bounded_v7_source_sha256.json').write_text(json.dumps(manifest,indent=2))
    _,o,e=c.exec_command('/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python -m unittest discover -s '+base+'/bounded_code_v7/tests -v',timeout=60)
    text=o.read().decode()+e.read().decode();code=o.channel.recv_exit_status();(root/'bounded_v7_linux_tests.log').write_text(text)
    if code:print(text);raise SystemExit(code)
    with s.open(base+'/bounded_v7_tests_passed.json','wx') as f:f.write(json.dumps({'status':'passed','source_sha256':manifest}))
    print(text)
elif mode=='launch':
    with s.open(base+'/bounded_v7_tests_passed.json') as f:assert json.loads(f.read())['status']=='passed'
    launcher="""import json,os,subprocess
from pathlib import Path
b=Path('/home/researcher/mea_full_28080_20261117_1406')
log=(b/'bounded_stable_run.log').open('x')
p=subprocess.Popen(['/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python',str(b/'bounded_code_v7/current_run.py'),'--run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,HDF5_PLUGIN_PATH='/pub/soft/MaxLab/so',PYTHONUNBUFFERED='1'))
(b/'bounded_stable_job.json').write_text(json.dumps({'pid':p.pid,'automatic_restart':False}))
print(p.pid)
"""
    with s.open(base+'/bounded_stable_launch_once.py','wx') as f:f.write(launcher)
    _,o,e=c.exec_command('/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python '+base+'/bounded_stable_launch_once.py')
    text=o.read().decode()+e.read().decode();code=o.channel.recv_exit_status();(root/'bounded_stable_launch_receipt.log').write_text(text)
    print(text)
    if code:raise SystemExit(code)
else:raise ValueError('Unknown mode')
s.close();c.close()
