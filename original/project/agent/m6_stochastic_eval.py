"""Separately registered stochastic M6 assessment; no changes to PPO or the world."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile

from agent.m6_test import FROZEN_FILES

VERSION = 'm6_stochastic_v1'
DECODING = 'categorical verb then target; temperature=1; independent episode CUDA generator'
VALIDATION_SEEDS = list(range(181000,181004))
TEST_SEEDS = list(range(202000,202020))
REPLICATES = list(range(4))
BASE_WEIGHT_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
PROTOCOL = 'docs/M6_STOCHASTIC_DIAGNOSTIC.md'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def base_weight_file(model):
    """Resolve the official one-shard layout without changing loader precedence."""
    model=Path(model)
    direct=model/'model.safetensors';index=model/'model.safetensors.index.json'
    if direct.is_file() and not index.exists():return direct
    if index.is_file() and not direct.exists():
        names=set(read(index)['weight_map'].values())
        if len(names)!=1:raise ValueError('Expected the verified single-shard 0.8B model')
        name=next(iter(names))
        if Path(name).name!=name or not name.endswith('.safetensors'):
            raise ValueError('Invalid model shard path')
        path=model/name
        if path.is_file():return path
    raise ValueError('Missing or ambiguous official 0.8B weight layout')


def episode_seed(world_seed,replicate):
    if type(world_seed) is not int or type(replicate) is not int or replicate not in REPLICATES:
        raise ValueError('Invalid world/replicate identity')
    value=hashlib.sha256(f'{VERSION}|{world_seed}|{replicate}'.encode()).digest()
    return int.from_bytes(value[:8],'big') % (2**63-1)


def validate_probabilities(probabilities,legal_mask):
    if len(probabilities)!=len(legal_mask) or not probabilities:
        raise ValueError('Probability/mask dimensions differ')
    if any(not math.isfinite(p) or p<0 or p>1 for p in probabilities):
        raise ValueError('Invalid model probabilities')
    if any(p!=0 for p,legal in zip(probabilities,legal_mask) if not legal):
        raise ValueError('Illegal target has nonzero probability')
    if abs(sum(probabilities)-1)>1.e-5:
        raise ValueError('Model probabilities do not sum to one')


def summarize_episodes(episodes,seeds):
    expected={(seed,replicate) for seed in seeds for replicate in REPLICATES}
    actual=[(r['seed'],r['replicate']) for r in episodes]
    if len(actual)!=len(expected) or len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Incomplete or duplicate world/replicate matrix')
    for r in episodes:
        if r['rng_seed']!=episode_seed(r['seed'],r['replicate']):
            raise ValueError('Episode RNG identity differs from registration')
        life=r['survival_ticks']
        if type(life) is not int or not 0<life<=900:
            raise ValueError('Invalid exact tick survival')
        if type(r['terminal']) is not bool or type(r['alive_at_horizon']) is not bool:
            raise ValueError('Invalid terminal flags')
        if r['terminal']==r['alive_at_horizon'] or (r['alive_at_horizon'] and life!=900):
            raise ValueError('Death and time limit flags disagree')
        sources=r['source_counts']
        if (type(r['decisions']) is not int or r['decisions']<=0 or sources.get('model')!=r['decisions']
            or any(v!=0 for k,v in sources.items() if k!='model')):
            raise ValueError('Unaccounted or non-model actions')
    worlds=[dict(seed=s,survival_ticks=sum(r['survival_ticks'] for r in episodes if r['seed']==s)/4) for s in seeds]
    return dict(world_means=worlds,mean_survival_ticks=sum(r['survival_ticks'] for r in worlds)/len(worlds),
                episode_count=len(episodes),independent_world_count=len(worlds),
                alive_at_horizon=sum(r['alive_at_horizon'] for r in episodes))


def source_identity(c):
    root=Path(c['project'])
    files=(*FROZEN_FILES,'agent/m6_test.py','agent/m6_stochastic_eval.py',PROTOCOL)
    return {name:sha(root/name) for name in files}


def authorize(request):
    """Pure filesystem gate; no world instance or test observation is created."""
    run=Path(request['run']).resolve();registration=read(run/'preregistration.json');c=registration['config']
    with zipfile.ZipFile(run/'frozen_source.zip') as frozen:
        if json.loads(frozen.read('run_config.json'))!=c:
            raise ValueError('Training registration differs from its original frozen archive')
    if run!=Path(c['output']).resolve() or run.name!='small_v2_stable':
        raise ValueError('Use the original registered run, not a copied output to bypass a ledger')
    if c['resume'] is not None or c['profile']!='open1' or c['horizon']!=900 or c['max_tokens']!=1536:
        raise ValueError('This diagnostic is locked to fresh stable-v2 conditions')
    if c['validation_seeds']!=VALIDATION_SEEDS or c['updates']!=16 or c['envs']!=4 or c['seed']!=16213:
        raise ValueError('Original schedule or validation worlds differ')
    train=set(range(c['train_seed_start'],c['train_seed_start']+c['train_seed_count']))
    if train & set(VALIDATION_SEEDS+TEST_SEEDS) or set(c['test_seeds']) & set(TEST_SEEDS):
        raise ValueError('New independent test split overlaps a previous split')
    completed=read(run/'complete.json');summary=read(run/'training_summary.json')
    metrics=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines() if line]
    if (completed['mode']!='train' or [r['update'] for r in metrics]!=list(range(1,17))
        or summary['decisions']!=8192 or metrics[-1]['decisions']!=8192
        or not summary['weight_change_verified']):
        raise ValueError('All sixteen updates and 8192 real decisions must have completed')
    if (summary['source_counts'].get('model')!=8192
        or any(v!=0 for k,v in summary['source_counts'].items() if k!='model')
        or any(r['decisions']!=r['update']*512 for r in metrics)):
        raise ValueError('Completed training decision accounting differs')
    if read(run/'job/finished.json')['exit_code']!=0:
        raise ValueError('Training did not exit successfully')
    hashes=source_identity(c)
    for file in FROZEN_FILES:
        if hashes[file]!=registration['source_sha256'][file]:
            raise ValueError('Frozen production source changed: '+file)
    checkpoint=run/'checkpoint_0016'
    candidate=read(run/'validation_0016.json')['actor_sha256']
    if candidate!=summary['final_actor_sha256']:
        raise ValueError('Final checkpoint identity differs from completed training')
    identity=dict(protocol=VERSION,source_sha256=hashes,base_actor_sha256=read(run/'before.json')['actor_sha256'],
                  candidate_actor_sha256=candidate,adapter_sha256=sha(checkpoint/'adapter/adapter_model.safetensors'),
                  checkpoint=str(checkpoint),decoding=DECODING,replicates=REPLICATES,
                  training_preregistration_sha256=sha(run/'preregistration.json'),
                  training_source_archive_sha256=sha(run/'frozen_source.zip'))
    mode=request['mode']
    if mode not in ('validation','test'):raise ValueError('Unsupported assessment mode')
    seeds=VALIDATION_SEEDS if mode=='validation' else TEST_SEEDS
    # Canonical ledger is independent of caller output names or supplied paths.
    ledger=run/f'{VERSION}_{mode}_ledger.json'
    if ledger.exists():raise FileExistsError('This registered assessment has already been opened')
    if mode=='test':
        validation=Path(request['validation_output']);done=read(validation/'complete.json')
        if done['status']!='completed' or done['mode']!='validation' or done['identity']!=identity:
            raise ValueError('Completed validation identity/source differs')
        previous_ledger=read(run/f'{VERSION}_validation_ledger.json')
        if previous_ledger!=done or Path(done['output']).resolve()!=validation.resolve():
            raise ValueError('Validation output is not bound to the original canonical ledger')
        baseline,after=read(validation/'base.json'),read(validation/'candidate.json')
        for name,report,actor in (('base',baseline,identity['base_actor_sha256']),('candidate',after,candidate)):
            if (sha(validation/(name+'.json'))!=done['result_sha256'][name+'.json']
                or sha(validation/(name+'.jsonl'))!=report['trace_sha256']):
                raise ValueError('Validation evidence changed after completion')
            if (report['actor_sha256']!=actor or report['decoding']!=DECODING or report['profile']!='open1'
                or report['horizon']!=900 or report['seeds']!=VALIDATION_SEEDS or report['replicates']!=REPLICATES):
                raise ValueError('Sampling validation conditions differ')
            recomputed=summarize_episodes(report['episodes'],VALIDATION_SEEDS)
            if any(report[k]!=v for k,v in recomputed.items()):
                raise ValueError('Sampling validation summary differs from all raw episodes')
        if after['mean_survival_ticks']<1.15*baseline['mean_survival_ticks']:
            raise ValueError('Sampling validation is below 15%; test worlds remain sealed')
    if Path(request['output']).exists():raise FileExistsError('Preserve existing assessment output')
    return c,identity,seeds,ledger


def runtime_identity(torch):
    return dict(python=sys.version,packages={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','numpy','fla-core')},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def evaluate(policy,c,seeds,path):
    import torch
    from agent.m6_contract import SurvivalReward,target_messages,VERSION as CONTRACT
    from agent.rl_contract import transition_at_horizon
    from agent.rl_env import GodotEnv
    episodes=[];began=time.monotonic()
    def rng_sha(generator):return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    with path.with_suffix('.jsonl').open('x',encoding='utf-8') as log,torch.no_grad():
        for replicate in REPLICATES:
            for start in range(0,len(seeds),c.envs):
                group=seeds[start:start+c.envs];envs=[GodotEnv(c.godot,c.project) for _ in group]
                try:
                    states=[env.reset(seed,c.profile) for env,seed in zip(envs,group)]
                    trackers=[SurvivalReward(s['observation']) for s in states]
                    generators=[torch.Generator(device=policy.device_name).manual_seed(episode_seed(seed,replicate)) for seed in group]
                    reports=[dict(seed=seed,replicate=replicate,rng_seed=episode_seed(seed,replicate),decisions=0,
                        source_counts=dict(model=0,fallback=0),actions=Counter(),damage=0.,zero_thirst_ticks=0,
                        zero_hunger_ticks=0,discounted_return=0.,cumulative_discount=1.) for seed in group]
                    active=list(range(len(group)))
                    while active:
                        prepared=policy.prepare([states[i]['observation'] for i in active])
                        before_rng={i:rng_sha(generators[i]) for i in active}
                        vd=policy.verb_distribution(prepared)
                        vprobs=vd.probs.tolist()
                        for j,row in enumerate(prepared):validate_probabilities(vprobs[j],row['verb_mask'])
                        verbs=[int(torch.multinomial(vd.probs[j],1,generator=generators[i]).item()) for j,i in enumerate(active)]
                        td=policy.target_distribution(prepared,verbs);tprobs=td.probs.tolist()
                        for j,row in enumerate(prepared):
                            n=len(row['branches'][verbs[j]])
                            validate_probabilities(tprobs[j],[k<n for k in range(len(tprobs[j]))])
                        # Always a second categorical draw, including singleton branches.
                        targets=[int(torch.multinomial(td.probs[j],1,generator=generators[i]).item()) for j,i in enumerate(active)]
                        remaining=[]
                        for j,i in enumerate(active):
                            action=prepared[j]['branches'][verbs[j]][targets[j]]
                            nxt=envs[i].step(action);transition=transition_at_horizon(nxt['transition'],c.horizon)
                            reward=trackers[i].score(transition);final=transition['after']
                            terminal=bool(final['dead']);truncated=final['tick']>=c.horizon and not terminal
                            record=dict(seed=group[i],replicate=replicate,rng_seed=episode_seed(group[i],replicate),
                                observation=states[i]['observation'],model_input=prepared[j]['messages'],
                                target_input=target_messages(prepared[j],verbs[j]) if len(prepared[j]['branches'][verbs[j]])>1 else None,
                                verb_index=verbs[j],target_index=targets[j],verb_probabilities=vprobs[j],target_probabilities=tprobs[j],
                                rng_before_sha256=before_rng[i],rng_after_sha256=rng_sha(generators[i]),
                                action=action,transition=nxt['transition'],reward=reward,terminal=terminal,truncated=truncated,
                                scored_horizon=c.horizon,source='model',greedy=False,decoding=DECODING)
                            log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                            report=reports[i];report['decisions']+=1;report['source_counts']['model']+=1
                            report['actions'][action['verb']]+=1
                            report['discounted_return']+=report['cumulative_discount']*reward['reward']
                            report['cumulative_discount']*=reward['discount']
                            report['damage']+=sum(x['damage'] for x in transition['trace'])
                            report['zero_thirst_ticks']+=sum(x['thirst']<=0 for x in transition['trace'])
                            report['zero_hunger_ticks']+=sum(x['hunger']<=0 for x in transition['trace'])
                            states[i]=nxt
                            if terminal or truncated:
                                report.update(survival_ticks=final['tick'],terminal=terminal,alive_at_horizon=truncated,
                                    unique_positions=len(trackers[i].seen),final=final,overshoot=nxt['observation']['tick']-final['tick'])
                            else:remaining.append(i)
                        active=remaining
                    episodes.extend(reports);log.flush()
                    print(json.dumps(dict(event='stochastic_group',arm=path.stem,replicate=replicate,
                        seeds=group,lifetimes=[r['survival_ticks'] for r in reports])),flush=True)
                finally:
                    for env in envs:env.close()
    result=dict(profile=c.profile,horizon=c.horizon,decoding=DECODING,temperature=1.,contract=CONTRACT,
        seeds=seeds,replicates=REPLICATES,episodes=episodes,actor_sha256=policy.digest(),
        seconds=time.monotonic()-began,trace_sha256=sha(path.with_suffix('.jsonl')),**summarize_episodes(episodes,seeds))
    write(path,result);return result


def main():
    import torch
    from agent.m6_policy import Policy
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    request=read(a.request);c,identity,seeds,ledger=authorize(request)
    model=Path(c['model'])
    weight_file=base_weight_file(model)
    if sha(weight_file)!=BASE_WEIGHT_SHA:raise ValueError('Official frozen base weights changed')
    tokenizer_files={path.name:sha(path) for path in sorted(model.iterdir()) if path.is_file() and path.suffix in ('.json','.jinja','.txt')}
    torch.set_num_threads(8);runtime=runtime_identity(torch)
    if request['mode']=='test':
        prior=read(Path(request['validation_output'])/'authorization.json')
        if prior['runtime']!=runtime or prior['tokenizer_files_sha256']!=tokenizer_files:
            raise ValueError('Runtime or tokenizer differs from sampling validation')
    state=dict(status='started',mode=request['mode'],identity=identity,created_utc=now(),seeds=seeds,replicates=REPLICATES,
               output=str(Path(request['output']).resolve()))
    with ledger.open('x',encoding='utf-8') as stream:json.dump(state,stream,indent=2)
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=False)
    config={**c,'test_seeds':TEST_SEEDS}
    write(output/'authorization.json',dict(request=request,config=config,identity=identity,runtime=runtime,
        tokenizer_files_sha256=tokenizer_files,base_weight_file=weight_file.name,base_weight_sha256=BASE_WEIGHT_SHA))
    with zipfile.ZipFile(output/'assessment_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in identity['source_sha256']:archive.write(Path(c['project'])/file,file)
        archive.writestr('request.json',json.dumps(request,ensure_ascii=False,indent=2))
    cfg=SimpleNamespace(**config)
    for name,checkpoint,expected in [('base',None,identity['base_actor_sha256']),
                                    ('candidate',identity['checkpoint'],identity['candidate_actor_sha256'])]:
        policy=Policy(cfg.model,checkpoint=checkpoint,seed=cfg.seed,max_tokens=cfg.max_tokens)
        if policy.digest()!=expected:raise ValueError('Actor identity differs before '+name)
        if checkpoint and sha(Path(checkpoint)/'adapter/adapter_model.safetensors')!=identity['adapter_sha256']:
            raise ValueError('Candidate weights changed after authorization')
        evaluate(policy,cfg,seeds,output/(name+'.json'))
        if policy.digest()!=expected:raise ValueError('Assessment changed actor weights')
        del policy;gc.collect();torch.cuda.empty_cache()
    state.update(status='completed',finished_utc=now(),result_sha256={name:sha(output/name) for name in ['base.json','candidate.json']})
    write(output/'complete.json',state);write(ledger,state)
    print(json.dumps(dict(event='stochastic_assessment_complete',**state)),flush=True)


if __name__=='__main__':main()
