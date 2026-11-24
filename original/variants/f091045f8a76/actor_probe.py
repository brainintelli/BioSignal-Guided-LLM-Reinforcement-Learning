"""Isolated provisional actor-only probe. No trusted snapshot, optimizer or device."""
import argparse,hashlib,json,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['run','verify'],required=True);args=p.parse_args()
r=Path(__file__).resolve().parent;project=Path('/home/researcher/mea-stim-offline-20261117/project');sys.path.insert(0,str(project))
from agent.mea_protocol import canonical,digest
from agent.m9_contract import encode
from agent.mea_snapshot import FEATURE_NAMES
from agent.m9_env import GodotEnv
from agent.m8s_trie import choice_tree
from tools.mea_offline import evidence
from tools import mea_stim_train as trainlib
from agent.mea_stim_offline import cell_config
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pin=read('input_verification.json')['probe_input_sha256']
assert sha(r/'probe_input.json')==pin
data=read('probe_input.json');q=data['quality'];w=data['window']
assert data['chip_id']=='22021' and data['source_sha256']=='8ba000037b053f47e70c9576808b9d6f2e3609bf6d85dd8e2ffa55496f465771'
assert data['schema']=='mea.external_spikes.actor_input.v1' and w['feature_names']==list(FEATURE_NAMES)
assert q['training_admitted'] is False and q['raw_quality_verified'] is False and q['coverage_basis']=='observed_spike_frame_bounds_only'
assert data['alpha']==1 and not data['r2_reward_enabled'] and not data['realtime_training'] and data['optimizer_updates']==0
assert len(data['mapping'])==734 and w['guard_seconds']==61 and w['end_frame']-w['start_frame']==20000
for name,h in json.loads((r/'project_source_manifest.json').read_text()).items():assert sha(project/name)==h
import torch
policy=trainlib.new_policy(r/'model_config.json',cell_config(0,'none'));initial=trainlib.state(policy);trainlib.validate_common_initial(initial)
policy.model.eval();policy.critic.eval();policy.reference_model.eval()
notice='\n外部MEA输入仅用于离线推理接入检查。数据只含尖峰，原始波形、连续性及刺激伪迹未核验；与游戏动作无因果对应。'
def prepare(obs,visible):
    row=encode(obs,'reserves');row['messages'][1]['content']+=notice
    row['messages'][1]['content']+='\n湿件观测：'+canonical(dict(zip(FEATURE_NAMES,w['features'])) if visible else None)
    row.update(contract='mea.external_spikes.actor_inference.v1',external_source_quality=q,
        provisional_state2=dict(zip(FEATURE_NAMES,w['features'])) if visible else None,
        state2_input_sha256=pin,training_admitted=False)
    ids=policy._tokens(row['messages']);row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,normalized_targets={},semantic_trees={})
    for v,branch in enumerate(row['branches']):
        if len(branch)>1:row['semantic_trees'][v]=choice_tree([policy.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]],policy.tokenizer.eos_token_id)
    assert 'wetware_snapshot' not in row and 'audited_finalized_h5' not in canonical(evidence(row))
    return row
results=[]
with GodotEnv('/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64',str(project),map_mode='infinite_freeze',navigation_steps=4) as env:
    for block in range(8):
        obs=env.reset(7400100+block)['observation'];rows=[prepare(obs,False),prepare(obs,True)]
        with torch.inference_mode():
            vd=policy.verb_distribution(rows);verbs=vd.probs.argmax(-1).tolist();td=policy.target_distribution(rows,verbs);targets=td.probs.argmax(-1).tolist()
        for i,name in enumerate(['none','provisional_state2']):
            results.append(dict(block=block,condition=name,observation=obs,prepared=evidence(rows[i]),verb_probabilities=vd.probs[i].tolist(),
                selected_verb=verbs[i],conditional_target_probabilities=td.probs[i].tolist(),selected_target=targets[i],
                proposed_game_action=rows[i]['branches'][verbs[i]][targets[i]]))
        policy.clear_prefix_cache();print(canonical(dict(event='actor_probe_completed',block=block)),flush=True)
final=trainlib.state(policy);assert initial==final
assert all(p.grad is None for p in policy.model.parameters()) and all(p.grad is None for p in policy.critic.parameters())
report=dict(schema='mea.external_spikes.actor_probe.v1',rows=results,source_sha256=data['source_sha256'],input_sha256=pin,
    initial=initial,final=final,parameters_unchanged=True,mode='actor_only_inference',critic_evaluated=False,optimizer_updates=0,
    environment_actions_executed=0,device_commands_sent=0,realtime_training=False,alpha=1.,r2_reward_enabled=False,
    raw_quality_verified=False,formal_training_admitted=False,biological_effect_verified=False,
    contexts=8,unique_neural_windows=1,prepared_inputs=16,replication_note='8gamecontexts,not8biologicalreplicates; noscoreorlearningclaim.')
if args.mode=='run':
    report['collection_pid']=os.getpid()
    with (r/'model_probe.json').open('x',encoding='utf-8') as f:f.write(canonical(report)+'\n')
else:
    old=read('model_probe.json');collection=old.pop('collection_pid');assert collection!=os.getpid()
    # Exact serialized prompts/tokens/outputs; weights unchanged in both processes.
    assert old==report
    with (r/'model_probe_verification.json').open('x',encoding='utf-8') as f:json.dump(dict(passed=True,fresh_process=True,model_probe_sha256=sha(r/'model_probe.json'),input_sha256=pin,
        actor_script_sha256=sha(Path(__file__)),contexts=8,prepared_inputs=16,exact_outputs_recomputed=True,parameters_unchanged=True,formal_training_admitted=False),f,indent=2)
print(canonical(dict(status='passed',mode=args.mode,parameters_unchanged=True,formal_training_admitted=False)),flush=True)
