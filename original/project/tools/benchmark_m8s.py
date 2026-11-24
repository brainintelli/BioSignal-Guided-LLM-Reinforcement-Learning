"""Registered old/new protocol diagnostic; bounded updates, no checkpoint save."""
import argparse,gc,hashlib,json,math,random,statistics,subprocess,sys,time,warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace,MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def emit(**value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def verify(reg):
    for path,digest in reg['source_hashes'].items():
        if sha(Path(reg['config']['project'])/path)!=digest:raise ValueError('Source changed: '+path)
    for item in reg['input_files'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Input changed: '+item['path'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();reg=json.loads(Path(args.registration).read_text());verify(reg)
    c=SimpleNamespace(**reg['config']);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy; existing work untouched')
    # This FLA warning is a length-vs-head-count heuristic. Qwen supplies B,T,H,K;
    # short semantic suffixes legitimately have T < H. No other warning is hidden.
    warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*',module=r'fla\.ops\.gated_delta_rule\.chunk')
    import torch
    from agent.m8x_policy import Policy
    from agent.m8x_grouped_lora import GroupedPolicy
    from agent.m8s_policy import SemanticPolicy
    from agent.m8x_contract import SurvivalReward,target_messages as old_target
    from agent.m8s_contract import target_messages as new_target
    from agent.m8x_causal_kv import normalize_target
    from agent.m8x_train import finish_trajectory
    from agent.m8_train import update
    from agent.m8x_env import GodotEnv
    from tools.benchmark_m8x_kv2 import speed_gate
    torch.set_num_threads(8)
    data=json.loads(Path(c.dataset).read_text());lookup={r['id']:r for r in data['observations']}
    fixed=[data['observations'][i*(len(data['observations'])-1)//31]['observation'] for i in range(32)]
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>c.wall_clock_limit_seconds:raise TimeoutError('Diagnostic time limit')
    def timed(fn):
        torch.cuda.synchronize();t=time.perf_counter();result=fn();torch.cuda.synchronize();return result,time.perf_counter()-t
    def make(mode):
        cls={'baseline':Policy,'old_optimized':GroupedPolicy,'semantic':SemanticPolicy}[mode]
        options={} if mode=='baseline' else {'registration':reg}
        return cls(c.model,checkpoint=c.initial_checkpoint,parent_checkpoint=c.parent_checkpoint,
                   seed=c.rng_seed,max_tokens=1536,with_reference=True,**options)
    @torch.no_grad()
    def joint(p,prepared,reference=False):
        tables=[]
        for row in prepared:
            vd=p.verb_distribution([row],reference);parts=[]
            for v,branch in enumerate(row['branches']):
                if branch:
                    td=p.target_distribution([row],[v],reference)
                    parts.extend((vd.probs[0,v]*td.probs[0,:len(branch)]).tolist())
            if abs(sum(parts)-1)>1e-5:raise ValueError('Joint probability is not normalized')
            tables.append(parts)
        return tables
    def difference(a,b):
        diffs=[];kls=[];changed=0
        for x,y in zip(a,b):
            assert len(x)==len(y)
            diffs.append(max(abs(u-v) for u,v in zip(x,y)))
            kls.append(sum(u*math.log(max(u,1e-30)/max(v,1e-30)) for u,v in zip(x,y)))
            changed+=max(range(len(x)),key=x.__getitem__)!=max(range(len(y)),key=y.__getitem__)
        return dict(max_abs=max(diffs),mean_kl=statistics.mean(kls),max_kl=max(kls),argmax_changes=changed)
    @torch.no_grad()
    def comprehension(p,mode):
        answers=[];yesno=p._single_ids(('是','否'))
        for q in data['questions']:
            deadline();row=p.prepare([lookup[q['observation_id']]['observation']])[0]
            messages=[{'role':'system','content':data['common_rules']},
                      {'role':'user','content':row['messages'][1]['content']+'\n事实问题：'+q['question']}]
            ids=p._tokens(messages);qrow={'kv_prefix_ids':ids}
            with p._scope([qrow],False):logits=p._logits([ids],yesno)[0]
            probs=torch.softmax(logits,dim=-1).tolist();prediction=probs[0]>=probs[1]
            answers.append({**q,'prediction':prediction,'correct':prediction==q['answer'],'probability_yes':probs[0]})
        categories={category:sum(x['correct'] for x in answers if x['category']==category)/12 for category in sorted({x['category'] for x in answers})}
        result=dict(correct=sum(x['correct'] for x in answers),total=len(answers),accuracy=sum(x['correct'] for x in answers)/len(answers),by_category=categories,answers=answers)
        write(out/(mode+'_comprehension.json'),result)
        grounding=[]
        for q in data['grounding']:
            row=p.prepare([lookup[q['observation_id']]['observation']])[0]
            messages=(new_target if mode=='semantic' else old_target)(row,0)
            messages[-1]['content']+=' 本轮测试动作含义，请'+q['direction']+'走一格。'
            original=p._tokens(messages)
            row['normalized_targets'][0]=normalize_target(row['input_ids'],original,row['assistant_boundary'],1536)
            td=p.target_distribution([row],[0]);i=int(td.probs[0].argmax())
            action=row['branches'][0][i];expected=row['branches'][0].index(q['expected_action'])
            grounding.append({**q,'predicted_action':action,'correct':action==q['expected_action'],
                              'expected_action_probability':float(td.probs[0,expected])})
        gr=dict(correct=sum(x['correct'] for x in grounding),total=len(grounding),answers=grounding)
        write(out/(mode+'_grounding.json'),gr)
        emit(event='understanding',mode=mode,correct=result['correct'],total=result['total'],categories=categories,grounding_correct=gr['correct'],grounding_total=gr['total'])
        return {'facts_correct':result['correct'],'facts_total':result['total'],'categories':categories,
                'grounding_correct':gr['correct'],'grounding_total':gr['total']}
    comparisons={};measurements={};understanding={};baseline_prob=None
    write(out/'manifest.json',{'registration_sha256':reg['registration_sha256'],'config':reg['config'],
          'formal_training_started':False,'checkpoint_writes':0,'timing_scope':'3 x (128 real decisions + 2 PPO epochs + 8 critic epochs); loading/checks excluded'})
    for mode in c.arms:
        deadline();p=make(mode);initial=p.digest();write(out/(mode+'_policy.json'),p.metadata())
        emit(event='arm_loaded',mode=mode,digest=initial)
        if mode!='baseline':understanding[mode]=comprehension(p,mode)
        prepared=p.prepare(fixed)
        verbs=[];targets=[]
        for i,row in enumerate(prepared):
            legal=[v for v,b in enumerate(row['branches']) if b];v=legal[i%len(legal)]
            verbs.append(v);targets.append(i%len(row['branches'][v]))
        def fixed_forward():
            with torch.no_grad():
                for start in range(0,32,4):
                    batch=prepared[start:start+4];v=verbs[start:start+4]
                    p.distributions(batch,v);p.distributions(batch,v,True);p.values(batch)
        _,warm=timed(fixed_forward);times=[timed(fixed_forward)[1] for _ in range(3)]
        tables={'actor':joint(p,prepared),'reference':joint(p,prepared,True)}
        if baseline_prob is None:baseline_prob=tables
        old=[]
        with torch.no_grad():
            for start in range(0,32,4):
                v=torch.tensor(verbs[start:start+4],device='cuda');t=torch.tensor(targets[start:start+4],device='cuda')
                vd,td=p.distributions(prepared[start:start+4],v.tolist());old.extend((vd.log_prob(v)+td.log_prob(t)).tolist())
        reorder=grad_error=0.
        for size in (1,3,4):
            order=list(reversed(range(32)))
            with torch.no_grad():
                for start in range(0,32,size):
                    ix=order[start:start+size];v=torch.tensor([verbs[i] for i in ix],device='cuda');t=torch.tensor([targets[i] for i in ix],device='cuda')
                    vd,td=p.distributions([prepared[i] for i in ix],v.tolist());values=(vd.log_prob(v)+td.log_prob(t)).tolist()
                    reorder=max(reorder,max(abs(x-old[i]) for x,i in zip(values,ix)))
        for i in range(8):
            vd,td=p.distributions([prepared[i]],[verbs[i]])
            value=float((vd.log_prob(torch.tensor([verbs[i]],device='cuda'))+td.log_prob(torch.tensor([targets[i]],device='cuda'))).detach()[0])
            grad_error=max(grad_error,abs(value-old[i]));del vd,td
        numeric=dict(reorder_max=reorder,grad_mode_max=grad_error)
        if mode=='semantic':
            check=[r for r in prepared if len(r['branches'][1])>1][:4]
            with torch.no_grad():cached=p.target_distribution(check,[1]*len(check)).probs.detach().cpu().tolist()
            original=p._semantic_logits
            def full(self,row,ids,token_ids,reference):
                model=self.reference_model if reference else self.model
                output=self._backbone(model,ids)
                return torch.nn.functional.linear(output.last_hidden_state[:,-1,:].float(),model.get_output_embeddings().weight[token_ids].float())[0]
            try:
                p._semantic_logits=MethodType(full,p)
                with torch.no_grad():uncached=p.target_distribution(check,[1]*len(check)).probs.detach().cpu().tolist()
            finally:p._semantic_logits=original
            numeric['cached_vs_full_prefill']=difference(uncached,cached)
            del original
        assert reorder<=1e-6 and grad_error<=1e-6 and p.digest()==initial
        measurements[mode]=dict(warmup_seconds=warm,fixed_forward_seconds=times,**numeric,
                                actor_vs_old=difference(baseline_prob['actor'],tables['actor']),reference_vs_old=difference(baseline_prob['reference'],tables['reference']))
        write(out/(mode+'_probabilities.json'),tables);write(out/'performance.json',measurements)
        emit(event='probability_checks',mode=mode,**measurements[mode])
        del p,prepared;gc.collect();torch.cuda.empty_cache()
        # Fresh identical checkpoint and fresh optimizers for every timed arm.
        p=make(mode);initial=p.digest();ref_initial=p.reference_digest()
        actor_opt=torch.optim.AdamW([q for q in p.model.parameters() if q.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0.)
        critic_opt=torch.optim.AdamW(p.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0.)
        random.seed(c.rng_seed);torch.manual_seed(c.rng_seed)
        envs=[];states=[];trackers=[];reports=[];seeds=list(c.seeds);resets=[0]*4;action_counts=Counter()
        try:
            for seed in seeds:
                env=GodotEnv(c.godot,c.project);envs.append(env);state=env.reset(seed,'open3');states.append(state);trackers.append(SurvivalReward(state['observation']))
            with (out/(mode+'_transitions.jsonl')).open('x',encoding='utf-8') as trace:
                for number in range(3):
                    deadline();trajectories=[[] for _ in envs];ticks=0
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    for step in range(32):
                        prepared=p.prepare([s['observation'] for s in states]);sampled=p.act(prepared,with_reference=True)
                        for i,env in enumerate(envs):
                            action=sampled['actions'][i];nxt=env.step(action);scored=trackers[i].score(nxt['transition'],nxt['observation'])
                            terminal=nxt['terminated'];v=sampled['verbs'][i];t=sampled['targets'][i]
                            row=dict(prepared=prepared[i],verb=v,target=t,value=sampled['values'][i],logp=sampled['logp'][i],verb_logp=sampled['verb_logp'][i],
                                reference_verb=sampled['reference_verb'][i],reference_target=sampled['reference_target'][i],
                                reward=scored['reward'],duration=scored['elapsed_ticks'],terminal=terminal,boundary=terminal,next_obs=nxt['observation'])
                            trajectories[i].append(row);ticks+=row['duration'];action_counts[action['verb']]+=1
                            selected_text=(prepared[i]['target_texts'][v][t] if mode=='semantic' else None)
                            record=dict(mode=mode,update=number+1,step=step,env=i,seed=seeds[i],source='model',observation=states[i]['observation'],
                                model_input=prepared[i]['messages'],verb_index=v,target_index=t,semantic_target=selected_text,
                                executed_verb_token_ids=prepared[i]['input_ids'],
                                target_prompt_token_ids=p._target_tokens(prepared[i],v) if len(prepared[i]['branches'][v])>1 else None,
                                action=action,transition=nxt['transition'],next_observation=nxt['observation'],reward=scored,terminal=terminal,
                                log_probability=row['logp'],value=row['value'])
                            trace.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                            if terminal:
                                resets[i]+=1;seeds[i]=c.seeds[i]+4*resets[i];states[i]=env.reset(seeds[i],'open3');trackers[i]=SurvivalReward(states[i]['observation'])
                            else:states[i]=nxt
                    rows=[]
                    for trajectory in trajectories:
                        with torch.no_grad():value=0. if trajectory[-1]['terminal'] else float(p.values(p.prepare([trajectory[-1]['next_obs']]))[0])
                        rows.extend(finish_trajectory(trajectory,value,c.gamma_tick,c.gae_lambda))
                    torch.cuda.synchronize();rollout=time.perf_counter()-start
                    metrics,elapsed=timed(lambda:update(p,actor_opt,critic_opt,rows,c))
                    finite=all(math.isfinite(x) for x in metrics.values() if type(x) in (float,int))
                    refresh_error=0.
                    if mode!='baseline':
                        check=[r['prepared'] for r in rows[:4]];vs=[r['verb'] for r in rows[:4]]
                        with torch.no_grad():
                            vd,td=p.distributions(check,vs);before=(vd.probs.clone(),td.probs.clone())
                            p.execution_cache.revisions.clear()
                            for module in p.model.modules():
                                if hasattr(module,'_versioned_merged_weight'):module._versioned_merged_weight.revision=None
                                if hasattr(module,'_grouped_projection'):module._grouped_projection.group.revision=None
                            vd,td=p.distributions(check,vs)
                            refresh_error=max(float((vd.probs-before[0]).abs().max()),float((td.probs-before[1]).abs().max()))
                        assert refresh_error<=1e-6 and p._kv_scope is None
                    report=dict(update=number+1,decisions=128,world_ticks=ticks,rollout_seconds=rollout,update_seconds=elapsed,finite=finite,
                                refresh_error=refresh_error,peak_allocated_bytes=torch.cuda.max_memory_allocated(),**metrics)
                    reports.append(report);write(out/(mode+'_stability.json'),reports);emit(event='update',mode=mode,**report)
            p.assert_reference_frozen();comparisons[mode]=dict(updates=reports,actor_changed=p.digest()!=initial,
                reference_unchanged=p.reference_digest()==ref_initial,action_counts=dict(action_counts),natural_deaths=sum(resets))
        finally:
            for env in envs:env.close()
        del p,actor_opt,critic_opt,rows,trajectories,prepared;gc.collect();torch.cuda.empty_cache()
    gate=speed_gate(comparisons['baseline']['updates'],comparisons['semantic']['updates'])
    old_gate=speed_gate(comparisons['baseline']['updates'],comparisons['old_optimized']['updates'])
    oldtime=sum(x['rollout_seconds']+x['update_seconds'] for x in comparisons['old_optimized']['updates'])
    newtime=sum(x['rollout_seconds']+x['update_seconds'] for x in comparisons['semantic']['updates'])
    verify(reg)
    result=dict(registration_sha256=reg['registration_sha256'],understanding=understanding,performance=measurements,stability=comparisons,
        speed_gate=gate,old_optimized_speed_gate=old_gate,semantic_vs_old_optimized_speedup=oldtime/newtime,
        probability_consistency_passed=all(x['reorder_max']<=1e-6 and x['grad_mode_max']<=1e-6 for x in measurements.values()),
        sources_and_inputs_unchanged=True,formal_training_started=False,checkpoint_writes=0,seconds=time.monotonic()-started)
    write(out/'comparison.json',result);emit(event='completed',speed_gate=gate,old_speed_gate=old_gate,semantic_vs_old_optimized=oldtime/newtime)

if __name__=='__main__':main()
