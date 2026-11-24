"""Freeze a diagnostic-only semantic protocol and all inherited model inputs."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.benchmark_m8s import sha,verify,write

NEW_FILES=['agent/m8s_state.py','agent/m8s_contract.py','agent/m8s_trie.py','agent/m8s_policy.py',
           'tools/probe_m8s.py','tools/build_m8s_dataset.py','tools/benchmark_m8s.py','tools/register_m8s.py',
           'tests/test_m8s_contract.py','configs/m8s_diag_attempt1.json']

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());root=Path(c['project']);out=Path(a.output)
    old=json.loads(Path(c['previous_registration']).read_text())
    for name,h in old['source_hashes'].items():
        if sha(root/name)!=h:raise ValueError('Frozen source differs: '+name)
    source={**old['source_hashes'],**{name:sha(root/name) for name in NEW_FILES}}
    inputs={**old['input_files'],'understanding_dataset':{'path':c['dataset'],'sha256':sha(c['dataset'])}}
    r=dict(protocol='semantic_state_action_diagnostic_v1',created_utc=datetime.now(timezone.utc).isoformat(),
           config=c,source_hashes=source,input_files=inputs,previous_registration_sha256=old['registration_sha256'],
           formal_training_status='not_started',checkpoint_writes=0,release_authorized=False,
           initial_weights='checkpoint_0032; fresh optimizer and rollouts',
           action_probability='semantic verb softmax * finite legal semantic-target trie distribution',
           comprehension_scoring='96 balanced yes/no facts with common diagnostic instructions; 32 requested legal moves',
           primary_speed_gate='same 3x128 decisions, 24 actor steps each; total and median strictly >2x original B4x1536',
           short_stability_gate='finite gradients and losses, no KL early stopping, frozen reference, logp drift <=1e-6')
    payload=json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    r['registration_sha256']=hashlib.sha256(payload).hexdigest()
    verify(r);out.mkdir(parents=True,exist_ok=False);write(out/'registration.json',r)
    print(json.dumps({'registration_sha256':r['registration_sha256'],'sources':len(source),'inputs':len(inputs),'path':str(out/'registration.json'),'formal_training_started':False}))

if __name__=='__main__':main()
