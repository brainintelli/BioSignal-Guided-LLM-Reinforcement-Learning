"""Explicit offline launch after model, source and idle-GPU gates."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;project=r/'project';root=r/'experiment01'
if any((r/n).exists() for n in ('launch.json','launch.claim','worker.log','job.json')) or (root/'parallel_status.json').exists():raise FileExistsError('Already attempted; no automatic restart')
manifest=json.loads((r/'project_source_manifest.json').read_text())
for name,h in manifest.items():
    if hashlib.sha256((project/name).read_bytes()).hexdigest()!=h:raise ValueError('Source changed: '+name)
pre=json.loads((r/'gpu_preflight.json').read_text())
if pre.get('passed') is not True or pre.get('training_performed') is not False or pre.get('r2_enabled') is not False:raise ValueError('GPU preflight required')
tests=json.loads((r/'gpu_tests_receipt.json').read_text())
if tests.get('passed') is not True:raise ValueError('GPU tests required')
sys.path.insert(0,str(project))
from tools.mea_shuffle_campaign import registration
from agent.mea_shuffle import PIN
from agent.mea_protocol import digest
godot=Path('/home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64')
reg=registration(root,r/'data',PIN,r/'model_config.json',project,godot)
if pre['model_initial']!=reg['common_initial']:raise ValueError('Model identity')
occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if occupied:raise RuntimeError('GPU occupied; do not compete')
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip().splitlines()
if len(gpus)!=8:raise ValueError('Expected8GPUs')
with (r/'launch.claim').open('x') as f:f.write(datetime.now(timezone.utc).isoformat())
with (r/'worker.log').open('x') as log:
    p=subprocess.Popen([sys.executable,str(r/'worker.py')],cwd=project,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,registered_cells=32,gpu_bindings=gpus,registration_sha256=digest(reg),
    automatic_restart=False,r2_enabled=False,alpha=1.,realtime_training=False,device_commands_sent=0,
    started_utc=datetime.now(timezone.utc).isoformat(),
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),worker_sha256=hashlib.sha256((r/'worker.py').read_bytes()).hexdigest())
(r/'launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
