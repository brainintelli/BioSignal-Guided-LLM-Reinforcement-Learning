"""Independent engine/token replay and fresh-checkpoint logp audit of diagnostics."""
import gc,json,math,statistics,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from agent.m8x_policy import Policy
from agent.m8x_grouped_lora import GroupedPolicy
from agent.m8s2_policy import BatchedSemanticPolicy as SemanticPolicy
from agent.m8s2_kernels import pin_kernels
from agent import m8x_contract as old,m8s2_contract as new
from agent.m8x_causal_kv import normalize_target
from agent.m8s_trie import choice_tree
from agent.m8x_env import GodotEnv
from tools.audit_m6_run import assert_equivalent
from tools.benchmark_m8s2 import sha,verify,write

warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*',module=r'fla\.ops\.gated_delta_rule\.chunk')
torch.set_num_threads(8)
pin_kernels()
reg=json.loads(Path(sys.argv[1]).read_text());base=Path(sys.argv[2]);c=reg['config'];verify(reg)
tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
def tokens(messages):
    return tokenizer.encode(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
stub=tokenizer.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
records={mode:[json.loads(x) for x in (base/(mode+'_transitions.jsonl')).read_text().splitlines()] for mode in c['arms']}
summary={}
common=[r['observation'] for r in records['baseline'] if r['update']==1][:4]
for mode in c['arms']:
    cls={'baseline':Policy,'old_optimized':GroupedPolicy,'semantic':SemanticPolicy}[mode]
    p=cls(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=True,seed=c['rng_seed'],
          **({} if mode=='baseline' else {'registration':reg}))
    initial=p.digest();first=[r for r in records[mode] if r['update']==1];errors=[]
    with torch.no_grad():
        for start in range(0,len(first),4):
            batch=first[start:start+4];prepared=p.prepare([r['observation'] for r in batch]);verbs=[r['verb_index'] for r in batch]
            v=torch.tensor(verbs,device='cuda');t=torch.tensor([r['target_index'] for r in batch],device='cuda')
            vd,td=p.distributions(prepared,verbs);values=(vd.log_prob(v)+td.log_prob(t)).tolist()
            errors.extend(abs(value-r['log_probability']) for value,r in zip(values,batch))
    profile=[]
    if mode!='baseline':
        prepared=p.prepare(common)
        with torch.no_grad():
            for repeat in range(4):
                for row in prepared:
                    if len(row['branches'][1])<2:continue
                    with p._scope([row],False):
                        torch.cuda.synchronize();start=time.perf_counter();p.verb_distribution([row]);torch.cuda.synchronize();middle=time.perf_counter()
                        p.target_distribution([row],[1]);torch.cuda.synchronize();end=time.perf_counter()
                    if repeat:profile.append({'verb_seconds':middle-start,'navigation_target_seconds':end-middle})
    summary[mode]={'first_round_all_logps_checked':len(errors),'first_round_logp_error_max':max(errors),
                   'reload_logp_within_1e_6':max(errors)<=1e-6,'weights_unchanged':p.digest()==initial,
                   'component_profile_samples':len(profile),
                   'component_profile':None if not profile else {k:statistics.median(x[k] for x in profile) for k in profile[0]}}
    # Preserve a failed numerical gate, but still complete the independent
    # engine/token/reward replay. Do not discard unrelated evidence.
    assert len(errors)==128 and summary[mode]['weights_unchanged'] and all(math.isfinite(x) for x in errors)
    print(json.dumps({'event':'fresh_checkpoint_logps','mode':mode,**summary[mode]}),flush=True)
    del p,prepared,vd,td;gc.collect();torch.cuda.empty_cache()

for mode in c['arms']:
    envs={};states={};trackers={};counts={'records':0,'verb_tokens':0,'target_tokens':0,'single_tile_moves':0};node_counts=[];lengths=[]
    try:
        for r in records[mode]:
            i=r['env']
            if i not in envs:
                envs[i]=GodotEnv(c['godot'],c['project']);states[i]=envs[i].reset(r['seed'],'open3');trackers[i]=old.SurvivalReward(states[i]['observation'])
            elif states[i].get('terminated'):
                states[i]=envs[i].reset(r['seed'],'open3');trackers[i]=old.SurvivalReward(states[i]['observation'])
            obs=states[i]['observation'];assert_equivalent(obs,r['observation'],'observation')
            contract=new if mode=='semantic' else old;row=contract.encode(obs);v,t=r['verb_index'],r['target_index']
            assert_equivalent(row['messages'],r['model_input'],'messages');assert_equivalent(row['branches'][v][t],r['action'],'action')
            ids=tokens(row['messages']);assert ids==r['executed_verb_token_ids'];counts['verb_tokens']+=1;lengths.append(len(ids))
            if len(row['branches'][v])>1:
                if mode=='semantic':target=ids
                else:
                    target=tokens(contract.target_messages(row,v))
                    if mode!='baseline':target=normalize_target(ids,target,len(ids)-len(stub),1536)
                assert target==r['target_prompt_token_ids'];counts['target_tokens']+=1
            if mode=='semantic':
                assert row['target_texts'][v][t]==r['semantic_target']
                if len(row['branches'][v])>1:
                    seq=[tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]]
                    nodes,paths=choice_tree(seq,tokenizer.eos_token_id);node_counts.append(len(nodes))
            nxt=envs[i].step(r['action']);assert_equivalent(nxt['transition'],r['transition'],'transition');assert_equivalent(nxt['observation'],r['next_observation'],'next observation')
            scored=trackers[i].score(nxt['transition'],nxt['observation']);assert_equivalent(scored,r['reward'],'reward')
            if r['action']['verb'] in ('move','navigate'):
                before=nxt['transition']['before']['player'];after=nxt['transition']['after']['player']
                assert abs(before['x']-after['x'])+abs(before['y']-after['y'])==1 and nxt['transition']['elapsed_ticks']==1
                counts['single_tile_moves']+=1
            assert nxt['terminated']==r['terminal'];states[i]=nxt;counts['records']+=1
        summary[mode]['engine_replay']=counts
        summary[mode]['input_tokens']={'mean':statistics.mean(lengths),'max':max(lengths)}
        summary[mode]['semantic_branch_nodes']=None if not node_counts else {'count':len(node_counts),'mean':statistics.mean(node_counts),'max':max(node_counts),'total':sum(node_counts)}
        print(json.dumps({'event':'engine_replayed','mode':mode,**summary[mode]['engine_replay']}),flush=True)
    finally:
        for env in envs.values():env.close()
verify(reg)
result={'registration_sha256':reg['registration_sha256'],'audit_script_sha256':sha(__file__),'arms':summary,
        'reload_consistency_passed':all(x['reload_logp_within_1e_6'] for x in summary.values()),
        'sources_and_inputs_unchanged':True,'checkpoint_writes':0,'formal_training_started':False}
write(base/'independent_audit.json',result)
print(json.dumps({'event':'audit_completed','total_replayed':sum(s['engine_replay']['records'] for s in summary.values())}),flush=True)
if not result['reload_consistency_passed']:raise SystemExit(1)
