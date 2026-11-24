"""Freeze an isolated deployment; never connect hardware or load model weights."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'standalone/mea_online'))
from mea_online.config import template

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def emit(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')

def main():
    out=ROOT/'output/mea_online_20261117'
    dest=out/'delivery';dest.mkdir(exist_ok=False)
    dependencies=json.loads((out/'runtime_dependencies.json').read_text(encoding='utf-8'))
    files=set(dependencies['files'])
    for rel,expected in dependencies['files'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('Pinned dependency changed: '+rel)
    for directory in ('assets','configs','scenes','scripts','godot_tests','standalone/mea_online'):
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    files.update(['project.godot','agent/__init__.py','tools/__init__.py','tests/__init__.py',
      'tests/test_mea_p0.py','tests/test_m8s3_contract.py','tests/test_m6_semantic_contract.py',
      'docs/superpowers/specs/2026-11-17-mea-online-training.md',
      'docs/superpowers/plans/2026-11-17-mea-online-training.md'])
    for rel in sorted(files):
        source=ROOT/rel
        # Namespace packages may have no __init__.py; preserve that layout.
        if not source.is_file() and source.name=='__init__.py':continue
        target=dest/'project'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copyfile(ROOT/'standalone/mea_online/README.md',dest/'README.md')
    (dest/'native/runtime').mkdir(parents=True)
    (dest/'validation-bin').mkdir()
    shutil.copyfile(out/'adapter_real_build/mea_window_capture',dest/'native/mea_window_capture')
    shutil.copyfile(out/'adapter_stub_build/mea_window_capture',dest/'validation-bin/mea_window_capture_simulation')
    runtime=ROOT/'output/mea_timing/timing_delivery_20261116/timing_software_20261116_v5/runtime'
    for name in ('libstdc++.so.6','libgcc_s.so.1'):shutil.copyfile(runtime/name,dest/'native/runtime'/name)
    provenance=json.loads((out/'adapter_real_ubuntu22_preflight.json').read_text(encoding='utf-8'))
    provenance['binary_sha256']=sha(dest/'native/mea_window_capture')
    emit(dest/'native/provenance.json',provenance)
    target_root='/home/researcher/mea-online_20261117_01'
    c=template();c.update(model_paths=dependencies['external_paths'],project=target_root+'/project',
      godot='/home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64',
      registry=target_root+'/session-registry')
    emit(dest/'configs/gpu.unbound.json',c)
    device=json.loads((ROOT/'standalone/mea_online/cpp/device.unbound.json').read_text(encoding='utf-8'))
    device['producer'].update({k:provenance[k] for k in ('binary_sha256','header_sha256','library_sha256')})
    emit(dest/'configs/device.unbound.json',device)
    launcher='''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/home/researcher/mea-r2-22554/runtime_20261116_01/assets
export PYTHONPATH="$ROOT/project/standalone/mea_online:$ROOT/project:$RUNTIME/overlay:$RUNTIME/site"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
exec "$RUNTIME/python/bin/python3.12" -m mea_online.cli "$@"
'''
    (dest/'gpu-entry.sh').write_text(launcher,encoding='utf-8',newline='\n')
    manifest=dict(schema='mea.online.delivery.v1',target=target_root,identity_bound=False,
      r2_enabled=False,stimulation=False,files={p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    emit(dest/'manifest.json',manifest)
    archive=out/'mea-online_20261117_01.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in sorted(dest.rglob('*')):
            if not p.is_file():continue
            info=tar.gettarinfo(str(p),arcname=p.relative_to(dest).as_posix())
            info.uid=info.gid=0;info.uname=info.gname=''
            info.mode=0o755 if p.name in ('gpu-entry.sh','mea_window_capture','mea_window_capture_simulation') else 0o644
            with p.open('rb') as stream:tar.addfile(info,stream)
    receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest['files']))
    emit(out/'delivery_receipt.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
