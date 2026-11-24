import hashlib,json,shutil,zipfile
from pathlib import Path
old=Path(__file__).resolve().parent;r=old.parent/'mea_28080_shuffle_20261117_v2';r.mkdir(exist_ok=False)
with zipfile.ZipFile(old/'project_frozen.zip') as z:z.extractall(r/'project')
for name,h in json.loads((old/'project_source_manifest.json').read_text()).items():
    if hashlib.sha256((r/'project'/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozenfirstattempt changed')
shutil.copytree(old/'data',r/'data')
for name in ('remote_prepare.py','gpu_preflight.py','gpu_tests.py','launch_once.py','worker.py','poll_status.py','independent_audit.py','native_check.py','complete_local_fixtures.py','seal_project.py','test_deployment.py','training_flow.mmd'):
    shutil.copy2(old/name,r/name)
amendment=dict(reason='All8firstcells hit CUDAOOM during firstPPOupdate on24GB4090; previouslyvalidated.16 CPUactivationoffload omitted from48GBparenttrainer',
    first_attempt_root=str(old),failed_attempt_kept=True,change='restoreexistingCPU_pinned_preserve_strides_saved_tensor_hooks',
    unchanged=['source_data','initial_tensor_pins','seed_blocks','32cells','384decisions','microbatch4','minibatch128','actor2_critic16_steps_per_round','common150decisionholdout','statistics','R2off'],
    new_model_scores_inspected=False,automatic_retry=False)
(r/'technical_amendment.json').write_text(json.dumps(amendment,indent=2))
print(json.dumps({'revision':str(r)}))
