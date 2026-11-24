"""Explicit isolated actor probe operations; credentials only from stdin."""
import hashlib,json,sys,shlex
from pathlib import Path
import paramiko
r=Path(__file__).resolve().parent;old=r.parent/'mea_stim_offline_20261117'
b='/home/researcher/mea-external-post-20261117-8ba00003';py='/home/researcher/meadowmind-gpu/m6-venv/bin/python'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.0.2.15',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='70b27913e62299123c7d8a979589a47faa5ea04a5668671876357f5b68f6a59c'
c.get_transport().set_keepalive(20);s=c.open_sftp();mode=sys.argv[1]
if mode=='preflight':
    result={}
    for name,cmd in [('gpu','nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'),('processes','nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader')]:
        _,o,e=c.exec_command(cmd,timeout=15);out=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err;result[name]=out.strip()
    (r/'gpu_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
elif mode=='deploy':
    if json.loads((r/'gpu_preflight.json').read_text())['processes']:raise RuntimeError('GPU has another compute process; do not overlap')
    s.mkdir(b)
    for src,name in [(r/'probe_input.json','probe_input.json'),(r/'input_verification.json','input_verification.json'),(r/'actor_probe.py','actor_probe.py'),
                     (old/'model_config.json','model_config.json'),(old/'project_source_manifest_v2.json','project_source_manifest.json')]:
        with s.open(b+'/'+name,'wx') as f:f.write(src.read_bytes())
    print('Isolated actor probe inputs uploaded')
elif mode in ('run','verify'):
    env='HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site:/home/researcher/mea-stim-offline-20261117/project'
    _,o,e=c.exec_command(env+' '+py+' -u '+b+'/actor_probe.py --mode '+mode,timeout=180)
    with (r/(mode+'.log')).open('x',encoding='utf-8') as log:
        for line in iter(o.readline,''):print(line.rstrip(),flush=True);log.write(line)
        err=e.read().decode();log.write(err)
        if err:print(err,flush=True)
    if o.channel.recv_exit_status()!=0:raise SystemExit(1)
elif mode=='download':
    hashes={}
    for name in ['model_probe.json','model_probe_verification.json','actor_probe.py','probe_input.json','input_verification.json']:
        with s.open(b+'/'+name) as f:data=f.read()
        if name.startswith('model_probe'):
            with (r/name).open('xb') as f:f.write(data)
        elif data!=(r/name).read_bytes():raise ValueError('Remote input/script differs')
        hashes[name]=hashlib.sha256(data).hexdigest()
    receipt=json.loads((r/'model_probe_verification.json').read_text())
    assert receipt['passed'] and receipt['model_probe_sha256']==hashes['model_probe.json'] and receipt['actor_script_sha256']==hashes['actor_probe.py']
    with (r/'gpu_delivery.json').open('x') as f:json.dump(dict(passed=True,files=hashes),f,indent=2)
    print(json.dumps(dict(passed=True,files=hashes),indent=2))
else:raise ValueError('Unknown action')
s.close();c.close()
