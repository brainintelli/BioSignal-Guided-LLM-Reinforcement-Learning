"""Explicit offline GPU host operations. Password is read from stdin, never saved."""
import hashlib,json,shlex,sys
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent;b='/home/researcher/mea-stim-offline-20261117';py='/home/researcher/meadowmind-gpu/m6-venv/bin/python'
mode=sys.argv[1];c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
if hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()!='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c':raise RuntimeError('Host key changed')
s=c.open_sftp()
def command(cmd,log):
    _,o,e=c.exec_command(cmd,timeout=120);out=o.read().decode()+e.read().decode();code=o.channel.recv_exit_status();(r/log).write_text(out,encoding='utf-8');print(out)
    if code:raise SystemExit(code)
if mode=='deploy':
    s.mkdir(b);s.mkdir(b+'/data')
    for local,remote in [('project_frozen_v2.zip','project_frozen.zip'),('project_source_manifest_v2.json','project_source_manifest.json'),
                         ('model_config.json','model_config.json'),('remote_prepare.py','remote_prepare.py')]:s.put(str(r/local),b+'/'+remote)
    for p in (r/'data').glob('*.json'):s.put(str(p),b+'/data/'+p.name)
    command(py+' '+b+'/remote_prepare.py','remote_preflight.log')
elif mode=='register':
    env='HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site:'+b+'/project'
    command('cd '+b+'/project && '+env+' '+py+' -u -m tools.mea_stim_campaign register --root '+b+'/experiment01 --source '+b+'/data --pin fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61 --model-config '+b+'/model_config.json --project '+b+'/project --godot /home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64','registration_with_original_runtime.log')
    s.get(b+'/experiment01/registration.json',str(r/'registration.json'))
elif mode=='launch':
    for name in ['worker.py','launch_once.py']:
        with s.open(b+'/'+name,'wx') as f:f.write((r/name).read_bytes())
    command(py+' '+b+'/launch_once.py','launch_receipt.log')
    s.get(b+'/launch.json',str(r/'launch.json'))
elif mode=='seal':
    for name in ['seal_delivery.py','start_seal.py']:
        with s.open(b+'/'+name,'wx') as f:f.write((r/name).read_bytes())
    command(py+' '+b+'/start_seal.py','seal_launch.log')
elif mode=='seal_status':
    result={}
    for name in ['archive_receipt.json','seal_job.json']:
        try:
            with s.open(b+'/'+name) as f:data=f.read()
            (r/name).write_bytes(data);result[name]=json.loads(data)
        except FileNotFoundError:pass
    try:
        with s.open(b+'/seal.log') as f:data=f.read()
        (r/'seal.log').write_bytes(data);result['log_tail']=data.decode(errors='replace')[-1600:]
    except FileNotFoundError:pass
    try:result['archive_bytes']=s.stat(b+'/campaign_delivery.zip').st_size
    except FileNotFoundError:pass
    print(json.dumps(result,indent=2))
else:raise ValueError('Unknown explicit operation')
if mode=='deploy':
    s.get(b+'/gpu_preflight_tests.log',str(r/'gpu_preflight_tests.log'));s.get(b+'/preflight_passed.json',str(r/'preflight_passed.json'))
s.close();c.close()
