"""Seal the new experiment; this command never constructs or trains a policy."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8x_kv2_registration import CONFIG,seal,verify,sha


def main():
    import torch
    import triton
    from agent.m8x_identity import execution_identity
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/CONFIG).read_text())
    initial=Path(config['initial_checkpoint']);parent=Path(config['parent_checkpoint']);model=Path(config['model'])
    previous=json.loads((initial.parent/'registration.json').read_text())
    for name,digest in previous['source_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Preserved source changed: '+name)
    prior=json.loads(Path(config['previous_registration']).read_text())
    from agent.m8x_kv_registration import verify as verify_previous
    verify_previous(root,prior)
    if prior['registration_sha256']!=config['previous_registration_sha256']:raise ValueError('Wrong preceding trial')
    torch.set_num_threads(8)
    execution=execution_identity(config,torch)
    execution.update(triton=triton.__version__,cpu_threads=torch.get_num_threads(),
        initial_actor_sha256=json.loads((initial/'policy.json').read_text())['actor_sha256'],
        optimized_arithmetic='FP32 versioned LoRA merge; BF16 projection; analytical FP32 adapter gradients',
        norm_implementation='frozen zero-centered Triton RMSNorm; no torch.compile',
        historical_core_hashes_verified=len(previous['source_hashes']),previous_frozen_sources_verified=len(prior['source_hashes']),grouped_parallel_projections='MLP gate/up, linear attention QKV/z/b/a, attention Q/K/V; FP32 block adapter gradients; fused MLP SiLU')
    inputs={}
    for label,folder in [('initial',initial),('reference',parent)]:
        for pattern in ('adapter/*.safetensors','adapter/*.json','policy.json','critic.pt'):
            for path in folder.glob(pattern):inputs[label+'/'+path.relative_to(folder).as_posix()]=str(path)
    for path in model.iterdir():
        if path.is_file() and path.suffix in ('.safetensors','.json','.jinja','.txt'):
            inputs['base/'+path.name]=str(path)
    reg=seal(root,config['output'],config,execution,inputs)
    verify(root,reg)
    print(json.dumps(dict(registration_sha256=reg['registration_sha256'],sources=len(reg['source_hashes']),
        input_files=len(inputs),output=config['output'],formal_training_status='not_started'),indent=2))

if __name__=='__main__':main()
