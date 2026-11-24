"""Bounded actual-CUDA B4/online equivalence on eight fixed training states.

Requires completed offline acceptance. No optimizer, HTTP or fresh world steps.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.m8s3_online_gate import verify_offline_gate
from agent.m8s3_training import sha,verify_files
from agent.m8s3_server import sample_stochastic,decision_identity
from agent.m6_live_session import live_seed
from tools.m8x_live_sampler_probe import require_idle_gpu,write_new


def select_pairs(rows,config):
    groups={i:[] for i in range(4)}
    for index,row in enumerate(rows):
        if row['update']!=1: raise ValueError('Only first training rollout permitted')
        slot=row['env']
        if slot not in groups: raise ValueError('First cohort differs from four slots')
        if len(groups[slot])<2:
            seed=row['seed']
            if (not config['train_seed_start']<=seed<config['train_seed_start']+config['train_seed_count']
                    or seed in config['validation_seeds']+config['test_seeds']):
                raise ValueError('Held-out or unregistered observation forbidden')
            groups[slot].append((index,row))
        if all(len(v)==2 for v in groups.values()): break
    for pair in groups.values():
        if len(pair)!=2: raise ValueError('Incomplete fixed training pair')
        a,b=pair[0][1],pair[1][1]
        if (a['seed']!=b['seed'] or a['transition']['after']['dead']
                or a['next_observation']!=b['observation'] or a['observation']['tick']>=b['observation']['tick']):
            raise ValueError('Original training states must be consecutive')
    return [groups[i] for i in range(4)]


def training_pairs(gate):
    path=Path(gate['config']['output'])/'rollout_0001.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return select_pairs((json.loads(x) for x in stream),gate['config']),path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    gate=verify_offline_gate(args.run)
    verify_files(ROOT,json.loads((Path(args.run)/'registration.json').read_text()))
    pairs,path=training_pairs(gate)
    require_idle_gpu()
    from tools.benchmark_m8s3 import setup
    setup()
    import torch
    from agent.m8s3_run import make_policy,load_checkpoint
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    c=SimpleNamespace(**gate['config'])
    p=make_policy(c,reference=False);load_checkpoint(p,gate['checkpoint']);p.eval()
    actor=p.digest()
    if actor!=gate['actor_sha256']:raise ValueError('Actor changed')
    class Stream:
        def __init__(self,seed):self.generator=torch.Generator(device='cuda').manual_seed(seed)
        def snapshot(self):return self.generator.get_state().cpu().numpy().tobytes()
    contexts=[dict(session_id='m8s3-probe-'+uuid.uuid4().hex,episode=0,world_seed=pair[0][1]['seed']) for pair in pairs]
    offline=[Stream(live_seed(x)) for x in contexts];online=[Stream(live_seed(x)) for x in contexts]
    fixtures=[[] for _ in pairs];comparisons=[]
    with torch.no_grad():
        for step in range(2):
            prepared=p.prepare([pair[step][1]['observation'] for pair in pairs])
            before=[hashlib.sha256(s.snapshot()).hexdigest() for s in offline]
            with p._scope(prepared,False):
                vd=p.verb_distribution(prepared)
                verbs=[int(torch.multinomial(vd.probs[i],1,generator=s.generator)) for i,s in enumerate(offline)]
                td=p.target_distribution(prepared,verbs)
                targets=[int(torch.multinomial(td.probs[i],1,generator=s.generator)) for i,s in enumerate(offline)]
            for i,pair in enumerate(pairs):
                index,row=pair[step]
                actual=sample_stochastic(p,p.prepare([row['observation']]),online[i],torch)
                expected=dict(action=prepared[i]['branches'][verbs[i]][targets[i]],verb_index=verbs[i],target_index=targets[i],
                    verb_probabilities=vd.probs[i].tolist(),target_probabilities=td.probs[i].tolist(),
                    rng_before_sha256=before[i],rng_after_sha256=hashlib.sha256(offline[i].snapshot()).hexdigest(),
                    agent_context=contexts[i],rng_seed=str(live_seed(contexts[i])),tick=row['observation']['tick'])
                equal=(actual['verb']==verbs[i] and actual['target']==targets[i] and actual['action']==expected['action']
                    and actual['verb_probabilities']==expected['verb_probabilities']
                    and actual['target_probabilities']==expected['target_probabilities'] and online[i].snapshot()==offline[i].snapshot())
                comparisons.append(dict(env=i,step=step,exact_equal=equal,singleton_target=len(prepared[i]['branches'][verbs[i]])==1))
                fixtures[i].append(dict(source_index=index,observation=row['observation'],expected=expected))
    passed=all(x['exact_equal'] for x in comparisons) and p.digest()==actor
    write_new(args.output,dict(protocol='m8s3_live_probe_v1',passed=passed,actual_cuda=True,
        identity=decision_identity(actor,gate['registration_sha256'],c.verb_exploration_epsilon),
        checkpoint=gate['checkpoint'],training_trace=str(path),training_trace_sha256=sha(path),
        contexts=contexts,fixtures=fixtures,comparisons=comparisons,weights_unchanged=p.digest()==actor,
        source_sha256={name:sha(ROOT/name) for name in ('tools/m8s3_live_sampler_probe.py','agent/m8s3_server.py','agent/m6_live_session.py')},
        new_world_steps=0,optimizer_steps=0,held_out_observations=0,http_executed=False,
        scope='Eight fixed training observations; canonical B4 vs online sampler; not survival or HTTP evidence'))
    print(json.dumps(dict(passed=passed,observations=8,actual_cuda=True)),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
