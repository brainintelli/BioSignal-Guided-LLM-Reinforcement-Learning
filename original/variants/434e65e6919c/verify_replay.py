"""Independently replay every new registered performance trajectory in Godot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agent.m8x_env import GodotEnv
from agent.m8x_contract import SurvivalReward,encode,target_messages
from tools.audit_m6_run import assert_equivalent

parser=argparse.ArgumentParser()
parser.add_argument('--godot',required=True)
args=parser.parse_args()
folder=Path(__file__).resolve().parent
reg=json.loads((folder/'registration.json').read_text())
payload={k:v for k,v in reg.items() if k!='registration_sha256'}
canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
assert hashlib.sha256(canonical.encode()).hexdigest()==reg['registration_sha256']
assert (folder/'registration.json').read_bytes()==(folder/'preregistration.json').read_bytes()
assert hashlib.sha256((folder/'frozen_source.zip').read_bytes()).hexdigest()==reg['archive_sha256']
with zipfile.ZipFile(folder/'frozen_source.zip') as archive:
    assert set(archive.namelist())==set(reg['source_hashes'])
    for name,digest in reg['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
counts={};natural_deaths={}
for mode in reg['config']['performance']['arms']:
    rows=[json.loads(x) for x in (folder/(mode+'_transitions.jsonl')).read_text(encoding='utf-8').splitlines()]
    assert len(rows)==384
    checked=deaths=0
    for slot in range(4):
        selected=[r for r in rows if r['env']==slot]
        state=None;seed=None
        with GodotEnv(args.godot,str(ROOT)) as env:
            for row in selected:
                if state is None:
                    seed=row['seed'];state=env.reset(seed,'open3');tracker=SurvivalReward(state['observation'])
                assert row['seed']==seed
                assert row['source']=='model' and not row['truncated']
                assert_equivalent(state['observation'],row['observation'],'observation')
                encoded=encode(state['observation']);v,t=row['verb_index'],row['target_index']
                assert_equivalent(encoded['messages'],row['model_input'],'messages')
                assert_equivalent(encoded['branches'][v][t],row['action'],'legal_action')
                assert_equivalent(target_messages(encoded,v) if len(encoded['branches'][v])>1 else None,row['target_input'],'semantic_target')
                nxt=env.step(row['action'])
                assert_equivalent(nxt['transition'],row['transition'],'transition')
                assert_equivalent(nxt['observation'],row['next_observation'],'next_observation')
                assert_equivalent(tracker.score(nxt['transition'],nxt['observation']),row['reward'],'reward')
                assert nxt['terminated']==row['terminal']
                checked+=1;deaths+=int(nxt['terminated'])
                state=None if nxt['terminated'] else nxt
    counts[mode]=checked;natural_deaths[mode]=deaths
result=dict(passed=True,real_godot_replayed_steps=sum(counts.values()),arms=counts,natural_deaths=natural_deaths,
            sources_and_archive_verified=len(reg['source_hashes']),registration_sha256=reg['registration_sha256'],
            rewards_recomputed=sum(counts.values()),formal_training_started=False)
(folder/'replay_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
