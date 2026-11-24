"""Freeze a fresh bounded experiment and snapshot all its exact source bytes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.m8s3_batch_contract import validate_config
from agent.m8s3_diag_contract import registration_digest
from tools.benchmark_m8s3 import sha, write
from tools.benchmark_m8s3_batch import verify

NEW_FILES = ['agent/m8s3_batch_layout.py', 'agent/m8s3_batch_policy.py', 'agent/m8s3_batch_contract.py',
             'tools/benchmark_m8s3_batch.py', 'tools/audit_m8s3_batch.py', 'tools/register_m8s3_batch.py',
             'tools/probe_m8s3_batch.py', 'tests/test_m8s3_batch.py', 'configs/m8s3_batch_attempt1.json']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--parent-registration', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    root = Path(config['project'])
    parent = json.loads(Path(args.parent_registration).read_text())
    from agent.m8s3_diag_contract import verify_registration_digest
    verify_registration_digest(parent)
    sources = dict(parent['source_hashes'])
    for name, digest in sources.items():
        if sha(root / name) != digest:
            raise ValueError('Parent source changed: ' + name)
    sources.update({name: sha(root / name) for name in NEW_FILES})
    assert len(parent['input_files']) == 16
    assert all(not x['path'].endswith('critic.pt') for x in parent['input_files'].values())
    reg = dict(protocol=config['protocol'], created_utc=datetime.now(timezone.utc).isoformat(),
               config=config, source_hashes=sources, input_files=parent['input_files'],
               parent_registration_sha256=parent['registration_sha256'],
               initialization='checkpoint_0032 actor only; new 96-feature critic and empty AdamW in each arm; new Godot rollouts',
               baseline='B4x1536 state, registered native/grouped projections, shared graph-local target KV',
               candidate='B4x896/1280/1537 per-state canonical buckets, at least one masked leading token; max logical 1536; grouped/fused projections; same target KV; no static prefix cache',
               primary_speed_gate='384 decisions, 24 actor and 24 critic steps per arm; aggregate and median speedup >2; numerical gates',
               numeric_gate='reorder/grad/pre-update/reload/cache errors <=1e-6; no KL early stop; finite metrics',
               formal_training_status='not_started', checkpoint_writes=0, release_authorized=False)
    reg['registration_sha256'] = registration_digest(reg)
    verify(reg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'registration.json', reg)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(root / name, name)
    write(out / 'status.json', dict(status='registered', formal_training_status='not_started',
                                   checkpoint_writes=0, registration_sha256=reg['registration_sha256']))
    print(json.dumps(dict(registration_sha256=reg['registration_sha256'], sources=len(sources), inputs=16)))


if __name__ == '__main__':
    main()
