"""One bounded offline job; no device control, no scheduler, no credentials."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path('/home/researcher/mea-long-20261116')
PROJECT=ROOT/'project'
PYTHON='/home/researcher/meadowmind-gpu/m6-venv/bin/python'
GODOT='/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(pin):
    status={'phase':'starting','stimulation':False,'r2_enabled':False,'alpha':1,'realtime_training':False,
            'source_manifest_sha256':pin}
    def persist():
        status['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        p=ROOT/'training_status.partial.json';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(ROOT/'training_status.json')
    persist()
    source=ROOT/'long_28080_admitted_20261116';out=ROOT/'training28080_01'
    if sha(source/'manifest.json')!=pin:raise ValueError('external source pin differs')
    source_manifest=json.loads((source/'manifest.json').read_text())
    sys.path.insert(0,str(PROJECT))
    from agent.mea_h5_replay import resolve_accepted_run
    accepted_run=resolve_accepted_run(source,source_manifest)
    for name,item in source_manifest['files'].items():
        p=(source/name).resolve()
        if not p.is_relative_to(source.resolve()) or p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:
            raise ValueError('backup/source file differs: '+name)
    receipt=dict(source_host='192.0.2.101',backup_host='192.0.2.15',manifest_sha256=pin,accepted_run=accepted_run.name,
        all_file_hashes_verified=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path=ROOT/'backup_receipt.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        for key in ('source_host','backup_host','manifest_sha256','accepted_run','all_file_hashes_verified'):
            if old[key]!=receipt[key]:raise ValueError('existing backup receipt differs')
    else:
        with receipt_path.open('x') as f:json.dump(receipt,f,indent=2)
    expected=json.loads((ROOT/'project_source_manifest_v2.json').read_text())
    for name,h in expected.items():
        if sha(PROJECT/name)!=h:raise ValueError('frozen software differs: '+name)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',
        PYTHONPATH='/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site:'+str(PROJECT))
    status['phase']='independent_admission_verification';persist()
    audit_env=env|{'PYTHONPATH':str(ROOT/'audit_deps_h5py3151')+':'+env['PYTHONPATH']}
    with (ROOT/'admission_verify_attempt02.log').open('x') as log:
        code=subprocess.call([PYTHON,'-m','tools.mea_h5_admit','verify',str(accepted_run),
            '--chip-id','28080'],cwd=PROJECT,env=audit_env,stdout=log,stderr=subprocess.STDOUT)
    status['admission_verify_exit']=code
    if code:
        status['phase']='admission_verify_failed';persist();return code
    for phase in ('train','verify'):
        status['phase']=phase;persist()
        cmd=[PYTHON,'-m','tools.mea_long_train',phase,'--output',str(out),'--source',str(source),
            '--manifest-sha256',pin,'--model-config',str(ROOT/'model_config.json'),'--project',str(PROJECT),'--godot',GODOT]
        with (ROOT/(phase+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            status[phase+'_pid']=child.pid;persist();code=child.wait()
        status[phase+'_exit']=code
        if code:
            status['phase']=phase+'_failed';persist();return code
    result=json.loads((out/'independent_verification.json').read_text())
    if result['passed'] is not True:raise ValueError('verification did not pass')
    manifest={p.relative_to(out).as_posix():{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    with (ROOT/'accepted_training_manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2)
    report=json.loads((out/'report.json').read_text())
    status.update(phase='accepted',decisions=result['decisions'],rounds=len(result['rounds']),
        elapsed_ticks=report['elapsed_ticks'],terminals=report['terminals'],
        accepted_manifest_sha256=sha(ROOT/'accepted_training_manifest.json'))
    persist();return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);args=p.parse_args()
    raise SystemExit(main(args.manifest_sha256))
