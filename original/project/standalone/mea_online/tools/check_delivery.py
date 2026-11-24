"""Read-only file/runtime verification. Does not load Qwen or open MEA/GPU."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for rel,expected in manifest['files'].items():
        target=(root/rel).resolve()
        if not target.is_relative_to(root) or sha(target)!=expected:raise ValueError('Delivery digest mismatch: '+rel)
    from mea_online.config import validate_config
    config=validate_config(json.loads((root/'configs/gpu.unbound.json').read_text(encoding='utf-8')))
    if config['binding'] is not None or config['calibration'] is not None:raise ValueError('Unbound delivery required')
    for name in ('adapter','device','trainer','calibration','m9_runtime','distributed','cli'):
        importlib.import_module('mea_online.'+name)
    if 'torch' in sys.modules:raise ValueError('Static import unexpectedly loaded torch')
    paths={name:dict(path=value,exists=Path(value).is_dir()) for name,value in config['model_paths'].items()}
    if not all(item['exists'] for item in paths.values()):raise ValueError('Model directory missing')
    warm=Path(config['model_paths']['warm_checkpoint'])
    checkpoint=json.loads((warm/'manifest.json').read_text(encoding='utf-8'))
    for rel,expected in checkpoint['files'].items():
        target=(warm/rel).resolve()
        if not target.is_relative_to(warm) or sha(target)!=expected:raise ValueError('Warm checkpoint file mismatch: '+rel)
    from mea_online.m9_runtime import EXPECTED_ACTOR
    if checkpoint['actor_sha256']!=EXPECTED_ACTOR:raise ValueError('Warm actor identity differs')
    policy=json.loads((warm/'policy.json').read_text(encoding='utf-8'))
    identity=dict(contract='m9_research_v1',map_mode='infinite_freeze',navigation_steps=4,
                  state_variant='reserves',reward_variant='bottleneck',gamma_tick=.99985,critic_features=108)
    if any(policy.get(k)!=v for k,v in identity.items()):raise ValueError('Warm policy identity differs')
    if not (warm/'reload_probe.json').is_file() or not Path(config['godot']).is_file():raise ValueError('Probe/Godot unavailable')
    packages={}
    for name in ('torch','numpy','transformers','peft','safetensors','flash-linear-attention','causal-conv1d'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    if any(packages[x] is None for x in ('torch','numpy','transformers','peft','safetensors')):raise ValueError('ML package missing')
    result=dict(schema='mea.online.target-static-check.v1',passed=True,files_verified=len(manifest['files']),
      python=platform.python_version(),packages=packages,model_paths=paths,warm_checkpoint_files_verified=len(checkpoint['files']),
      warm_actor_manifest_sha256=checkpoint['actor_sha256'],warm_identity=identity,
      identity_bound=False,r2_enabled=False,stimulation=False,real_mea=False,real_qwen_loaded=False,
      real_gpu=False,real_nccl=False,torch_imported=False,
      limitations=['Full Qwen base/parent model load and migration probe not executed','Hardware identity/unit/latency and SDK service qualification pending'])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
