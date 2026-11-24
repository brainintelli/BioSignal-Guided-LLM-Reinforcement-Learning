"""Independent immutable registration for the unpadded KV experiment."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
from agent.m8x_identity import CORE_FILES

from agent.m8x_kv_registration import SOURCES as PREVIOUS_SOURCES
CONFIG='configs/m8x_kv_attempt2.json'
SOURCES=(*PREVIOUS_SOURCES,'agent/m8x_grouped_lora.py','agent/m8x_kv2_registration.py',
         'agent/m8x_fused_mlp.py','tests/test_m8x_fused_mlp.py',
         'tools/benchmark_m8x_kv2.py','tools/register_m8x_kv2.py',
         'tests/test_m8x_grouped_lora.py','tests/test_m8x_kv2_registration.py',
         CONFIG,'docs/M8X_KV2_PROTOCOL.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def validate(project,config):
    expected=json.loads((Path(project)/CONFIG).read_text(encoding='utf-8'))
    if canonical(config)!=canonical(expected):raise ValueError('Config differs from the registered new experiment')
    perf=config.get('performance',{})
    if (config.get('protocol')!='m8x_unpadded_kv_v2' or config.get('resume') is not None
            or config.get('formal_training_status')!='not_started' or config.get('release_authorized') is not False
            or perf.get('minimum_total_speedup')!=2.0 or perf.get('strictly_greater') is not True
            or perf.get('zero_update_logp_tolerance')!=1e-6 or perf.get('checkpoint_writes')!=0
            or config.get('output','').rstrip('/').split('/')[-1]!='kv_attempt2'):
        raise ValueError('Invalid new-experiment status, speed gate or output identity')


def seal(project,output,config,execution,input_files):
    root=Path(project).resolve();output=Path(output)
    validate(root,config)
    sources={name:sha(root/name) for name in SOURCES}
    inputs={name:dict(path=str(Path(path).resolve()),sha256=sha(path)) for name,path in input_files.items()}
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'frozen_source.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for name in SOURCES:stream.write(root/name,name)
    reg=dict(protocol='m8x_unpadded_kv_v2',created_utc=datetime.now(timezone.utc).isoformat(),
        config=json.loads(canonical(config)),source_hashes=sources,input_files=inputs,
        execution_identity=execution,archive_sha256=sha(archive),archive_path=str(archive.resolve()),
        formal_training_status='not_started',release_authorized=False,
        old_rollout_reuse=False,old_optimizer_reuse=False,performance_outcome='pending')
    reg['registration_sha256']=hashlib.sha256(canonical(reg).encode()).hexdigest()
    for name in ('registration.json','preregistration.json'):
        with (output/name).open('x',encoding='utf-8') as stream:json.dump(reg,stream,ensure_ascii=False,indent=2)
    return reg


def verify(project,reg):
    root=Path(project).resolve()
    payload={k:v for k,v in reg.items() if k!='registration_sha256'}
    if hashlib.sha256(canonical(payload).encode()).hexdigest()!=reg.get('registration_sha256'):
        raise ValueError('Registration payload changed')
    validate(root,reg['config'])
    if set(reg['source_hashes'])!=set(SOURCES):raise ValueError('Source inventory changed')
    for name,digest in reg['source_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Source changed: '+name)
    for name,row in reg['input_files'].items():
        if sha(row['path'])!=row['sha256']:raise ValueError('Model input changed: '+name)
    if sha(reg['archive_path'])!=reg['archive_sha256']:raise ValueError('Source archive changed')
    return True
