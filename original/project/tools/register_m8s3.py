"""Freeze new-world equal-update diagnostic; excludes incompatible old critics."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8s3_diag_contract import validate_config,registration_digest
from tools.benchmark_m8s3 import sha,verify,write
NEW_FILES=['agent/m8s3_diag_contract.py','agent/m8s3_padded_policy.py','tools/benchmark_m8s3.py',
           'tools/register_m8s3.py','tools/audit_m8s3_diagnostic.py','tools/probe_m8s3_training.py',
           'tests/test_m8s3_diagnostic.py','configs/m8s3_diag_attempt1.json']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    c=json.loads(Path(a.config).read_text());validate_config(c);root=Path(c['project']);base=Path(c['model']).parents[1]
    previous=json.loads((base/'m8s2-runs/semantic_diagnostic2/registration.json').read_text())
    source=json.loads((root/'source_manifest_final.json').read_text())
    for name,h in source.items():
        if sha(root/name)!=h:raise ValueError('New-world source differs: '+name)
    source.update({name:sha(root/name) for name in NEW_FILES})
    inputs={k:v for k,v in previous['input_files'].items() if not v['path'].endswith('critic.pt') and k!='understanding_dataset'}
    assert len(inputs)==16 and all('critic.pt' not in x['path'] for x in inputs.values())
    r=dict(protocol=c['protocol'],created_utc=datetime.now(timezone.utc).isoformat(),config=c,source_hashes=source,input_files=inputs,
           parent_input_registration_sha256=previous['registration_sha256'],world_source_manifest_sha256=sha(root/'source_manifest_final.json'),
           formal_training_status='not_started',checkpoint_writes=0,release_authorized=False,
           initialization='checkpoint_0032 actor only; fresh 96-feature critic, fresh AdamW states and fresh Godot rollouts in each arm',
           baseline=f"same new state/semantic trie, {c['baseline_projection_layout']} projections and graph-local target KV; state B4x1536, no static/persistent KV",
           candidate='same state/semantic trie; compact prefills, grouped projections, system KV and batched target suffixes',
           primary_speed_gate='both actual 384 decisions + 24 actor + 24 critic steps; aggregate AND median >2x; numeric gates',
           numeric_gate='reordered/grad/pre-update/reload/updated-cache logp or probability error <=1e-6; no KL early stop; finite metrics',
           limitations='BF16 layout differences disclosed separately; 24 updates do not establish long-term convergence')
    r['registration_sha256']=registration_digest(r);verify(r)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write(out/'registration.json',r)
    print(json.dumps(dict(registration_sha256=r['registration_sha256'],sources=len(source),inputs=len(inputs),path=str(out/'registration.json'),formal_training_started=False)))
if __name__=='__main__':main()
