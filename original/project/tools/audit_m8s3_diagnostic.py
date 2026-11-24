"""Fresh-process probability reproduction and independent Godot replay of all rows."""
import gc,json,math,statistics,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.benchmark_m8s3 import setup,make,verify,write,sha,selected_logps,critic_digest,emit
def main():
    reg=json.loads(Path(sys.argv[1]).read_text());base=Path(sys.argv[2]);verify(reg);c=SimpleNamespace(**reg['config']);setup()
    import torch
    from transformers import AutoTokenizer
    from agent.m8s3_contract import encode,SurvivalReward
    from agent.m8s3_env import GodotEnv
    from tools.audit_m6_run import assert_equivalent
    tokenizer=AutoTokenizer.from_pretrained(c.model,local_files_only=True)
    records={mode:[json.loads(x) for x in (base/(mode+'_transitions.jsonl')).read_text().splitlines()] for mode in c.arms}
    comparison=json.loads((base/'comparison.json').read_text());summary={}
    for mode in c.arms:
        p=make(c,mode);initial=p.digest();first=[r for r in records[mode] if r['update']==1]
        prepared=p.prepare([r['observation'] for r in first]);verbs=[r['verb_index'] for r in first];targets=[r['target_index'] for r in first]
        values=selected_logps(p,prepared,verbs,targets);errors=[abs(x-r['log_probability']) for x,r in zip(values,first)]
        assert len(errors)==128 and all(math.isfinite(e) for e in errors)
        summary[mode]=dict(first_round_all_logps_checked=len(errors),first_round_logp_error_max=max(errors),reload_logp_within_1e_6=max(errors)<=1e-6,
            weights_unchanged=p.digest()==initial,initial_critic_matches=critic_digest(p)==comparison['stability'][mode]['initial_critic'])
        emit(event='reload',mode=mode,**summary[mode]);del p,prepared;gc.collect();torch.cuda.empty_cache()
    for mode in c.arms:
        envs={};states={};trackers={};counts=dict(records=0,verb_tokens=0,single_tile_moves=0);lengths=[]
        try:
            for r in records[mode]:
                i=r['env']
                if i not in envs:
                    envs[i]=GodotEnv(c.godot,c.project);states[i]=envs[i].reset(r['seed'],c.profile);trackers[i]=SurvivalReward(states[i]['observation'])
                elif states[i].get('terminated'):
                    states[i]=envs[i].reset(r['seed'],c.profile);trackers[i]=SurvivalReward(states[i]['observation'])
                obs=states[i]['observation'];assert_equivalent(obs,r['observation'],'observation')
                row=encode(obs);v,t=r['verb_index'],r['target_index']
                assert_equivalent(row['messages'],r['model_input'],'messages');assert_equivalent(row['branches'][v][t],r['action'],'action')
                ids=tokenizer.encode(tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                assert ids==r['executed_verb_token_ids'] and row['target_texts'][v][t]==r['semantic_target']
                counts['verb_tokens']+=1;lengths.append(len(ids))
                nxt=envs[i].step(r['action']);assert_equivalent(nxt['transition'],r['transition'],'transition');assert_equivalent(nxt['observation'],r['next_observation'],'next')
                scored=trackers[i].score(nxt['transition'],nxt['observation']);assert_equivalent(scored,r['reward'],'reward')
                if r['action']['verb'] in ('move','navigate'):
                    before=nxt['transition']['before']['player'];after=nxt['transition']['after']['player']
                    assert abs(before['x']-after['x'])+abs(before['y']-after['y'])==1 and nxt['transition']['elapsed_ticks']==1
                    counts['single_tile_moves']+=1
                assert nxt['terminated']==r['terminal'];states[i]=nxt;counts['records']+=1
            assert counts['records']==384
            summary[mode]['engine_replay']=counts;summary[mode]['input_tokens']=dict(mean=statistics.mean(lengths),min=min(lengths),max=max(lengths))
            emit(event='replayed',mode=mode,**counts)
        finally:
            for env in envs.values():env.close()
    verify(reg)
    passed=all(x['reload_logp_within_1e_6'] and x['weights_unchanged'] and x['initial_critic_matches'] for x in summary.values())
    result=dict(registration_sha256=reg['registration_sha256'],audit_script_sha256=sha(__file__),arms=summary,reload_consistency_passed=passed,
                sources_and_inputs_unchanged=True,formal_training_started=False,checkpoint_writes=0)
    write(base/'independent_audit.json',result);emit(event='audit_completed',passed=passed,total_replayed=768)
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
