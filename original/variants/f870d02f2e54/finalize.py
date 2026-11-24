"""Verify completed diagnostics, write explicit stopped statuses, seal evidence."""
import hashlib,json,platform,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
base=Path('/home/researcher/meadowmind-gpu');project=base/'m8s3-batch2-project';diag=base/'m8s3-batch-diagnostics'
sys.path.insert(0,str(project))
from tools.benchmark_m8s3_batch import verify
from tools.benchmark_m8s3 import sha,write
import torch,transformers,peft,triton
summaries={}
for name,folder in [('survival_batch_diagnostic2','equal_update2'),('survival_batch_control2','control2')]:
 run=base/'m8s3-runs'/name;reg=json.loads((run/'registration.json').read_text());verify(reg)
 comparison=json.loads((diag/folder/'comparison.json').read_text());audit=json.loads((diag/folder/'independent_audit.json').read_text())
 assert comparison['sources_and_inputs_unchanged'] and comparison['probability_consistency_passed']
 assert comparison['speed_gate']['equal_work'] and comparison['speed_gate']['numeric_checks']
 assert audit['reload_consistency_passed'] and audit['sources_and_inputs_unchanged']
 assert not comparison['formal_training_started'] and not comparison['checkpoint_writes']
 if name=='survival_batch_diagnostic2':assert comparison['speed_gate']['passed']
 for arm in comparison['stability'].values():
  assert arm['actual_actor_steps']==arm['actual_critic_steps']==24
  assert arm['optimizers_initially_empty'] and arm['actor_changed'] and arm['critic_changed'] and arm['reference_unchanged']
 status=dict(status='diagnostic_completed',primary_speed_gate_passed=comparison['speed_gate']['passed'],registration_sha256=reg['registration_sha256'],
  speed_gate=comparison['speed_gate'],independent_audit_passed=True,formal_training_status='not_started',formal_training_started=False,checkpoint_writes=0,
  comparison_sha256=sha(diag/folder/'comparison.json'),audit_sha256=sha(diag/folder/'independent_audit.json'))
 write(run/'status.json',status);summaries[name]=status
extra=json.loads((diag/'extended_probe2.json').read_text())
assert extra['synthetic_reorder_logp_max']==0 and extra['gradient']['finite'] and extra['actor_unchanged']
checkpoints=[]
for folder in [diag,base/'m8s3-runs/survival_batch_diagnostic1',base/'m8s3-runs/survival_batch_diagnostic2',base/'m8s3-runs/survival_batch_control1',base/'m8s3-runs/survival_batch_control2']:
 checkpoints.extend(str(p) for p in folder.rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.safetensors','.bin','.ckpt'))
assert not checkpoints,checkpoints
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not compute,compute
processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
active=[line for line in processes.splitlines() if ('/python ' in line or '/python -' in line or 'godot/Godot' in line) and any(x in line for x in ['agent.m8_train','agent.m8x_train','agent.m8b_train','m8x_remote_job','agent.m8x_server','benchmark_m8s3_batch.py','audit_m8s3_batch.py'])]
# ps may include this command's parent shell text; actual worker commands only.
active=[line for line in active if not any(x in line for x in ['bash -','sh -','finalize.py'])]
engines=[line for line in processes.splitlines() if '/godot/Godot' in line and '--headless' in line and not any(x in line for x in ['bash -','sh -','finalize.py'])]
assert not engines,engines
assert not active,active
environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,triton=triton.__version__,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
final=dict(created_utc=datetime.now(timezone.utc).isoformat(),summaries=summaries,environment=environment,new_checkpoint_files=checkpoints,gpu_compute_processes=compute,
 active_training_or_service_processes=active,game_engine_processes=engines,formal_training_started=False,checkpoint_writes=0,final_sources_and_inputs_verified=True)
write(diag/'final_integrity.json',final)
mapping={p.relative_to(diag).as_posix():p for p in diag.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='delivery_manifest.json' and '__pycache__' not in p.parts}
for label,name in [('final','survival_batch_diagnostic2'),('control','survival_batch_control2'),('superseded','survival_batch_diagnostic1')]:
 for filename in ['registration.json','status.json']:
  mapping[filename.replace('.json','_'+label+'.json')]=base/'m8s3-runs'/name/filename
 if label!='control':mapping['source_snapshot_'+label+'.zip']=base/'m8s3-runs'/name/'source_snapshot.zip'
manifest={name:dict(sha256=sha(path),bytes=path.stat().st_size) for name,path in sorted(mapping.items())}
write(diag/'delivery_manifest.json',manifest);mapping['delivery_manifest.json']=diag/'delivery_manifest.json'
with zipfile.ZipFile(base/'m8s3_batch_delivery_evidence.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for name,path in sorted(mapping.items()):archive.write(path,name)
print(json.dumps(dict(**final,archive_sha256=sha(base/'m8s3_batch_delivery_evidence.zip'),archive_bytes=(base/'m8s3_batch_delivery_evidence.zip').stat().st_size),ensure_ascii=False),flush=True)
