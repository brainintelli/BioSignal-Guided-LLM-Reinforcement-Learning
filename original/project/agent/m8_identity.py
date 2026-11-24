"""Explicit source and execution identities for M8 (protocol paths pending freeze)."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

BASE_WEIGHT_SHA='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
from agent.m8_registration import CORE_FILES


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def config_sha(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def source_identity(project):
    return {name:sha(Path(project)/name) for name in CORE_FILES}


def verify_source_identity(project,expected,core_files=CORE_FILES):
    if set(expected)!=set(core_files):raise ValueError('Frozen source set differs from exact core files')
    for name in core_files:
        path=(Path(project)/name).resolve()
        if not path.is_relative_to(Path(project).resolve()) or sha(path)!=expected[name]:
            raise ValueError('Frozen source changed: '+name)


def claim_ledger(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())


def read_ledger(path):
    raw=Path(path).read_bytes()
    if not raw.endswith(b'\n'):raise ValueError('Incomplete ledger event; never silently discard it')
    events=[json.loads(line) for line in raw.splitlines()]
    previous=None
    for event in events:
        payload={k:v for k,v in event.items() if k!='event_sha256'}
        if event.get('previous_event_sha256')!=previous or event.get('event_sha256')!=config_sha(payload):
            raise ValueError('Ledger hash chain changed')
        previous=event['event_sha256']
    return events


def append_ledger_event(path,event):
    path=Path(path)
    previous=read_ledger(path) if path.exists() else []
    keys=('attempt','phase','registration_sha256','selected_checkpoint','selected_actor_sha256')
    if previous:
        last=previous[-1]
        if last['status']!='started' or event['status'] not in ('completed','failed'):
            raise ValueError('This phase has already been claimed or completed')
        if any(event.get(k)!=last.get(k) for k in keys):raise ValueError('Ledger execution identity changed')
    elif event.get('status')!='started':raise ValueError('First event must claim the phase')
    payload={**event,'previous_event_sha256':previous[-1]['event_sha256'] if previous else None}
    payload['event_sha256']=config_sha(payload)
    with path.open('a' if previous else 'x',encoding='utf-8') as stream:
        stream.write(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    return payload


def runtime_identity(torch):
    return dict(python=sys.version,packages={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','numpy','fla-core')},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def execution_identity(config,torch):
    model=Path(config['model']);parent=Path(config['parent_checkpoint'])
    direct=model/'model.safetensors';index=model/'model.safetensors.index.json'
    if direct.exists()==index.exists():raise ValueError('Missing or ambiguous official weights')
    if direct.exists():weight=direct
    else:
        names=set(json.loads(index.read_text())['weight_map'].values())
        if len(names)!=1:raise ValueError('Expected the verified single official weight shard')
        weight=(model/next(iter(names))).resolve()
        if not weight.is_relative_to(model.resolve()):raise ValueError('Weight outside model path')
    if sha(weight)!=BASE_WEIGHT_SHA:raise ValueError('Official base weights changed')
    restored={name:sha(parent/name) for name in ('policy.json','critic.pt')}
    restored.update({p.relative_to(parent).as_posix():sha(p) for p in sorted((parent/'adapter').iterdir()) if p.is_file()})
    if 'adapter/adapter_model.safetensors' not in restored:raise ValueError('Missing parent adapter')
    return dict(runtime=runtime_identity(torch),config_sha256=config_sha(config),base_weight_file=weight.name,
        base_weight_sha256=BASE_WEIGHT_SHA,parent_restore_sha256=restored,
        tokenizer_files_sha256={p.name:sha(p) for p in sorted(model.iterdir()) if p.is_file() and p.suffix in ('.json','.jinja','.txt')})
