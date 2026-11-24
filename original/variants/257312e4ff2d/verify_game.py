"""Replay final holdout actions independently through the same Godot game rules."""
import argparse
import json
import math
from pathlib import Path

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import observation_key
from livepair.predictability import outcome
from mea_state2_r2.protocol import GAMMA


def check(root, executable, project, seed, arm):
    study=json.loads((root/'design.json').read_text())
    folder=root/'models'/str(seed)/arm
    results=json.loads((folder/'result.json').read_text())['evaluation']
    traces=[json.loads(line) for line in (folder/'holdout.jsonl').read_text().splitlines()]
    if [r['seed'] for r in results]!=study['holdout_seeds'] or [r['world'] for r in traces]!=study['holdout_seeds']:
        raise ValueError('Unexpected evaluation worlds')
    decisions=0
    with GodotEnv(str(executable),str(project),map_mode='infinite_freeze',navigation_steps=4) as env:
        for trajectory,metrics in zip(traces,results):
            start=env.reset(trajectory['world'])
            obs=start['observation']
            if start['starting_scene']!=study['starting_scene'] or any(obs['player'][k]!=v for k,v in study['starting_reserves'].items()):
                raise ValueError('Unexpected initial game state')
            tracker=Reward(obs,'bottleneck',GAMMA)
            visited=obs['visited_count']
            r1=0.
            for i,row in enumerate(trajectory['rows']):
                if observation_key(obs)!=row['context']:
                    raise ValueError('Observation mismatch')
                next_state=env.step(row['action'])
                obs=next_state['observation']
                transition=next_state['transition']
                score=tracker.score(transition,obs)['reward']
                eligible=(i+1)%3==0 and not next_state['terminated']
                sign=(1 if outcome(transition)['correct'] else -1) if eligible else 0
                if not math.isclose(score,row['r1'],abs_tol=1e-12) or transition['after']!=row['after'] or sign!=row['r2']:
                    raise ValueError('Replayed action or score differs')
                r1+=score
                decisions+=1
            if (not math.isclose(r1,metrics['r1_sum'],abs_tol=1e-12) or obs['visited_count']-visited!=metrics['new_cells'] or
                len(trajectory['rows'])!=metrics['decisions'] or (len(trajectory['rows'])<study['eval_decisions'] and not next_state['terminated'])):
                raise ValueError('Evaluation endpoint mismatch')
    return dict(seed=seed,arm=arm,worlds=len(traces),decisions=decisions)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('root','executable','project'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--arm',choices=['baseline','sham'],required=True)
    args=parser.parse_args()
    result=check(**vars(args))
    (args.root/'models'/str(args.seed)/args.arm/'replay-audit.json').write_text(json.dumps(result))
