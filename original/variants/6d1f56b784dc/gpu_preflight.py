import hashlib,importlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;project=r/'project';sys.path.insert(0,str(project))
import torch
from agent.mea_shuffle import PIN,cell_config,PROTOCOL
from tools.mea_shuffle_train import new_policy,state,validate_common_initial
from tools.mea_p0 import read_json,sha
from agent.mea_stim_offline import StimRecording
from agent.mea_contract import observation_key
from agent.m9_env import GodotEnv
if torch.cuda.device_count()!=8:raise ValueError('Expected eight GPUs')
policy=new_policy(r/'model_config.json',cell_config(0,'after'));original=state(policy);validate_common_initial(original)
godot=Path('/home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64')
with GodotEnv(str(godot),str(project),map_mode='infinite_freeze',navigation_steps=4) as env:
    obs=env.reset(9900001)['observation']
    data=StimRecording(r/'data',pin=PIN,phase='after',split='train');snapshot=data.snapshot(0,observation_key(obs))
    prepared=policy.prepare([obs],[snapshot]);torch.manual_seed(9900002)
    sampled=policy.act(prepared,with_reference=True)
    if not torch.isfinite(torch.tensor(sampled['values'])).all():raise ValueError('Nonfinite GPU forward')
    if state(policy)!=original:raise ValueError('Forward altered weights')
runtime={}
for name in ('torch','transformers','peft','fla','causal_conv1d','causal_conv1d_cuda'):
    module=importlib.import_module(name);path=Path(module.__file__)
    files=list(path.parent.rglob('*.py')) if path.name=='__init__.py' and name in ('fla','causal_conv1d') else [path]
    runtime[name]={'origin':str(path),'version':getattr(module,'__version__',None),'files':{str(p):sha(p) for p in files}}
receipt=dict(passed=True,model_initial=original,source_pin=PIN,scripted_engineering_forward=True,
    training_performed=False,model_outcomes_selected=False,r2_enabled=False,device_commands_sent=0,
    torch_cuda=torch.version.cuda,devices=[torch.cuda.get_device_name(i) for i in range(8)],runtime=runtime,
    python=sys.version,godot_sha256=sha(godot))
(r/'gpu_preflight.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='runtime'}),flush=True)
