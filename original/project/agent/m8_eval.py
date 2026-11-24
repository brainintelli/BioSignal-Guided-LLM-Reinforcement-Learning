"""Draft M8 batched stochastic evaluator; final held-out authorization not yet frozen.

No CLI can open held-out worlds in this draft. Each environment owns its CUDA
generator; batching changes only the physical forwarding work, never sampling.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

DECODING = 'categorical verb then target; temperature=1; independent episode CUDA generator'


def target_encoder(contract):
    if contract=='open_world_hierarchical_v2':
        from agent.m6_contract import target_messages
    elif contract=='open_world_semantic_v4':
        from agent.m8_contract import target_messages
    else:raise ValueError('Unknown policy observation contract')
    return target_messages


def episode_seed(prefix,profile,world_seed,replicate):
    if not isinstance(prefix,str) or not prefix or type(world_seed) is not int or type(replicate) is not int or replicate<0:
        raise ValueError('Invalid RNG identity')
    return int.from_bytes(hashlib.sha256(f'{prefix}|{profile}|{world_seed}|{replicate}'.encode()).digest()[:8],'big')%(2**63-1)


def summarize_episodes(episodes,seeds,replicates,horizon):
    expected={(s,r) for s in seeds for r in replicates}
    actual=[(e['seed'],e['replicate']) for e in episodes]
    if len(actual)!=len(expected) or len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Incomplete or repeated world/replicate matrix')
    for e in episodes:
        life=e['survival_ticks']
        if type(life) is not int or not 0<life<=horizon:
            raise ValueError('Invalid exact horizon survival')
        if type(e['terminal']) is not bool or type(e['alive_at_horizon']) is not bool:
            raise ValueError('Invalid endpoint flags')
        if e['terminal']==e['alive_at_horizon'] or (e['alive_at_horizon'] and life!=horizon):
            raise ValueError('Death/time censoring disagree')
    successes=sum(e['alive_at_horizon'] for e in episodes)
    return dict(mean_survival_ticks=sum(e['survival_ticks'] for e in episodes)/len(episodes),
                alive_at_horizon=successes,survival_rate=successes/len(episodes),
                episode_count=len(episodes),independent_world_count=len(seeds))


def evaluate(policy,c,seeds,destination,arm='candidate'):
    import torch
    from agent.rl_env import GodotEnv
    from agent.rl_contract import transition_at_horizon
    from agent.m8_contract import SurvivalReward
    from agent.m8_identity import source_identity
    metadata=policy.metadata()
    contract=metadata['contract']
    actor_digest=policy.digest()
    if c.eval_batch_size not in (1,4):raise ValueError('Evaluation batch must be frozen as 1 or 4')
    target_messages=target_encoder(contract)
    destination=Path(destination)
    trace=destination.with_suffix('.jsonl')
    if destination.exists() or trace.exists():
        raise FileExistsError('Preserve previous assessment evidence')
    reports=[]
    identities=[(seed,replicate) for replicate in c.replicates for seed in seeds]
    with torch.no_grad(),trace.open('x',encoding='utf-8') as stream:
        for start in range(0,len(identities),c.eval_batch_size):
            group=identities[start:start+c.eval_batch_size]
            envs=[GodotEnv(c.godot,c.project) for _ in group]
            try:
                states=[e.reset(seed,c.profile) for e,(seed,_) in zip(envs,group)]
                trackers=[SurvivalReward(s['observation']) for s in states]
                generators=[torch.Generator(device=policy.device_name).manual_seed(episode_seed(c.rng_prefix,c.profile,*pair)) for pair in group]
                batch=[dict(seed=s,replicate=r,rng_seed=episode_seed(c.rng_prefix,c.profile,s,r),decisions=0,
                            source_counts={'model':0,'fallback':0},actions=Counter()) for s,r in group]
                active=list(range(len(group)))
                while active:
                    prepared=policy.prepare([states[i]['observation'] for i in active])
                    vd=policy.verb_distribution(prepared)
                    before=[hashlib.sha256(generators[i].get_state().cpu().numpy().tobytes()).hexdigest() for i in active]
                    verbs=[int(torch.multinomial(vd.probs[j],1,generator=generators[i]).item()) for j,i in enumerate(active)]
                    td=policy.target_distribution(prepared,verbs)
                    targets=[int(torch.multinomial(td.probs[j],1,generator=generators[i]).item()) for j,i in enumerate(active)]
                    remaining=[]
                    for j,i in enumerate(active):
                        action=prepared[j]['branches'][verbs[j]][targets[j]]
                        nxt=envs[i].step(action)
                        transition=transition_at_horizon(nxt['transition'],c.horizon)
                        reward=trackers[i].score(transition);final=transition['after']
                        terminal=bool(final['dead']);truncated=final['tick']>=c.horizon and not terminal
                        after=hashlib.sha256(generators[i].get_state().cpu().numpy().tobytes()).hexdigest()
                        record=dict(arm=arm,seed=group[i][0],replicate=group[i][1],rng_seed=batch[i]['rng_seed'],
                            observation=states[i]['observation'],model_input=prepared[j]['messages'],
                            target_input=target_messages(prepared[j],verbs[j]) if len(prepared[j]['branches'][verbs[j]])>1 else None,
                            verb_index=verbs[j],target_index=targets[j],verb_probabilities=vd.probs[j].tolist(),
                            target_probabilities=td.probs[j].tolist(),rng_before_sha256=before[j],rng_after_sha256=after,
                            action=action,transition=nxt['transition'],reward=reward,terminal=terminal,truncated=truncated,
                            scored_horizon=c.horizon,source='model',salt=prepared[j]['salt'],
                            contract=contract,temperature=1.,decoding=DECODING,actor_sha256=actor_digest)
                        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                        batch[i]['decisions']+=1;batch[i]['source_counts']['model']+=1;batch[i]['actions'][action['verb']]+=1
                        states[i]=nxt
                        if terminal or truncated:
                            batch[i].update(survival_ticks=final['tick'],terminal=terminal,alive_at_horizon=truncated,final=final)
                        else:remaining.append(i)
                    active=remaining
                reports.extend(batch)
                print(json.dumps(dict(event='stochastic_group',arm=arm,
                    identities=[list(pair) for pair in group],lifetimes=[r['survival_ticks'] for r in batch])),flush=True)
            finally:
                for env in envs:env.close()
    if policy.digest()!=actor_digest:raise RuntimeError('Assessment changed actor weights')
    report=dict(arm=arm,profile=c.profile,horizon=c.horizon,seeds=list(seeds),replicates=list(c.replicates),
                rng_prefix=c.rng_prefix,decoding=DECODING,temperature=1.,contract=contract,
                source_sha256=source_identity(c.project),episodes=reports,actor_sha256=policy.digest(),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),**summarize_episodes(reports,seeds,c.replicates,c.horizon))
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


def authorize_test(request):
    """Recompute the full integrity-reviewed selection before opening test worlds."""
    from agent.m8_identity import sha,read_ledger
    from agent.m8_registration import verify_registration
    from tools.compare_m8_survival import build_result
    run=Path(request['run']).resolve()
    registration=json.loads((run/'registration.json').read_text(encoding='utf-8'))
    config=registration['config']
    if request.get('mode')!='test' or run!=Path(config['output']).resolve():
        raise ValueError('Only canonical registered held-out test is supported')
    verify_registration(config['project'],registration)
    if sha(run/'frozen_source.zip')!=registration['archive_sha256']:
        raise ValueError('Training archive changed')
    review_path=Path(request['selection_review'])
    review=json.loads(review_path.read_text(encoding='utf-8'))
    review_request=json.loads(Path(request['selection_review_request']).read_text(encoding='utf-8'))
    rebuilt=build_result(review_request,config['project'])
    if (review!=rebuilt or review.get('phase')!='selection' or review.get('decision')!='eligible_for_test'
            or review.get('evidence_complete') is not True or Path(review_request['training_run']).resolve()!=run
            or review.get('registration_sha256')!=sha(run/'registration.json')):
        raise ValueError('Selection review did not independently unlock this test')
    checkpoint=run/review['selected_checkpoint']
    if checkpoint.resolve().parent!=run or checkpoint.name not in ('checkpoint_0008','checkpoint_0016','checkpoint_0024','checkpoint_0032'):
        raise ValueError('Selected checkpoint path differs')
    ledger=run/'ledger_test.jsonl'
    if ledger.exists():
        read_ledger(ledger)
        raise FileExistsError('Test phase already claimed; no retry or new output bypass')
    output=Path(request['output']).resolve()
    if output.exists() or output==run or output.parent!=run.parent:
        raise ValueError('Test output must be a new sibling of the canonical training run')
    return config,registration,review,checkpoint,ledger,output


def main():
    import argparse
    import gc
    from datetime import datetime,timezone
    from types import SimpleNamespace
    import torch
    from agent.m8_identity import sha,execution_identity,append_ledger_event
    from agent.m8_policy import Policy
    from agent.m6_policy import Policy as LegacyPolicy
    from agent.m8_policy_identity import PARENT_ACTOR_SHA256
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    args=parser.parse_args();request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    config,registration,review,checkpoint,ledger,output=authorize_test(request)
    torch.set_num_threads(8)
    if execution_identity(config,torch)!=registration['execution_identity']:
        raise ValueError('Assessment execution identity differs from training')
    c=SimpleNamespace(**{**config,'replicates':config['test_replicates']})
    event=dict(attempt=c.attempt,phase='test',status='started',registration_sha256=sha(Path(request['run'])/'registration.json'),
               selected_checkpoint=checkpoint.name,selected_actor_sha256=review['selected_actor_sha256'],evidence_sha256={})
    append_ledger_event(ledger,event)
    output.mkdir()
    authorization=dict(request=request,registration_sha256=event['registration_sha256'],config=config,
                       execution_identity=registration['execution_identity'],source_sha256=registration['source_hashes'],
                       selection_review_sha256=sha(request['selection_review']),selected_checkpoint=str(checkpoint),
                       selected_actor_sha256=review['selected_actor_sha256'])
    (output/'authorization.json').write_text(json.dumps(authorization,indent=2),encoding='utf-8')
    try:
        for arm in ('original_v2','semantic_parent','candidate'):
            if arm=='original_v2':
                policy=LegacyPolicy(c.model,checkpoint=c.parent_checkpoint,seed=c.seed,max_tokens=c.max_tokens)
            else:
                policy=Policy(c.model,parent_checkpoint=c.parent_checkpoint,
                              checkpoint=checkpoint if arm=='candidate' else None,seed=c.seed,max_tokens=c.max_tokens,with_reference=False)
            expected=review['selected_actor_sha256'] if arm=='candidate' else PARENT_ACTOR_SHA256
            if policy.digest()!=expected:raise ValueError('Assessment actor identity changed')
            evaluate(policy,c,c.test_seeds,output/(arm+'.json'),arm=arm)
            del policy;gc.collect();torch.cuda.empty_cache()
        evidence={name+'.json':sha(output/(name+'.json')) for name in ('original_v2','semantic_parent','candidate')}
        append_ledger_event(ledger,{**event,'status':'completed','evidence_sha256':evidence})
        (output/'complete.json').write_text(json.dumps(dict(status='completed',mode='test',
            registration_sha256=event['registration_sha256'],evidence_sha256=evidence,
            finished_utc=datetime.now(timezone.utc).isoformat(),release_authorized=False),indent=2),encoding='utf-8')
    except BaseException as error:
        append_ledger_event(ledger,{**event,'status':'failed','error':type(error).__name__+': '+str(error)})
        raise


if __name__=='__main__':main()
