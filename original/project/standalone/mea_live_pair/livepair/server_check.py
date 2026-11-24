"""Read-only training-host readiness check without starting a biological session."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import torch
from mea_online.m9_runtime import EXPECTED_ACTOR, validate_runtime_config
from mea_state2_r2.features import digest, fit
from .protocol import PROTOCOL
from .worker import Worker


def reduce_check(rank, init):
    import torch.distributed as dist
    from datetime import timedelta
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method=init,rank=rank,world_size=8,timeout=timedelta(seconds=75))
    try:
        x=torch.tensor(rank+1,device=f'cuda:{rank}')
        dist.all_reduce(x)
        if int(x)!=36:raise ValueError('Collective result differs')
    finally:
        dist.destroy_process_group()


def main():
    base=Path.home()/'mea-state2-r2-20261118/run02'
    assets=Path.home()/'mea-r2-22554/runtime_20261116_01/assets'
    model=json.loads((base/'model_config.json').read_text())
    manifest=json.loads((Path(model['warm_checkpoint'])/'manifest.json').read_text())
    if manifest['actor_sha256']!=EXPECTED_ACTOR or not all(Path(p).exists() for p in model.values()):
        raise ValueError('Warm start/model paths differ')
    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpus)!=8 or any('4090' not in g for g in gpus):raise ValueError('Eight 4090s required')
    norm=fit([[0.,0.,0.,0.]])
    cfg=dict(binding_sha256='0'*64,normalization=norm,normalization_id=digest(norm),r2_enabled=True,
             stimulation=True,source='live',arm='ordered',devices=list(range(8)),batch_size=16,
             model_paths=model,experiment=PROTOCOL)
    validate_runtime_config(cfg)
    godot=assets/'godot/Godot_v4.7.2-stable_linux.x86_64'
    project=base/'project'
    if not godot.is_file() or not (project/'project.godot').is_file():raise ValueError('Godot project absent')
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(reduce_check,args=((Path(directory)/'init').as_uri(),),nprocs=8)
    print(json.dumps(dict(host=socket.gethostname(),gpus=gpus,nccl=torch.distributed.is_nccl_available(),
             nccl_all_reduce_sum=36,
             project=str(project),godot=str(godot),model_paths=model,actor=EXPECTED_ACTOR,
             free_bytes=shutil.disk_usage(base).free,live_training_started=False)))


if __name__=='__main__':main()
