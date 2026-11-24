"""Replay saved, sampled actions on the unmodified Godot environment and extract native ticks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

FIELDS=('visited_count','health','hunger','thirst')


def digest(obj):
    text=json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src,'sha256').hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def replay_seed(task):
    from agent.m9_env import GodotEnv
    arm,seed,games,project,godot,horizon=task
    results={}
    with GodotEnv(str(godot),str(project),map_mode='infinite_freeze',navigation_steps=4) as env:
        for gs,steps in games.items():
            obs=env.reset(int(gs))['observation']
            initial=steps[0]['before']
            series=[[0,initial['health'],initial['hunger'],initial['thirst']]]
            elapsed=0
            crossed=None
            end_of_decision=False
            first_tick=obs['tick']
            for idx,step in enumerate(steps,1):
                where=f'{arm} training {seed} game {gs} decision {idx}'
                require(digest(obs)==step['obs_hash'],where+' initial observation')
                out=env.step(step['action'])
                tr=out['transition']
                require(out['terminated']==step['terminal'] and tr['action']==step['action'] and
                        len(tr['trace'])==tr['elapsed_ticks']==step['duration'],where+' transition')
                for side in ('before','after'):
                    require({k:tr[side][k] for k in FIELDS}==step[side],where+' '+side)
                for tick in tr['trace']:
                    elapsed+=1
                    require(tick['tick']==first_tick+elapsed,where+' missing game tick')
                    values=[tick['visited_count']-initial['visited_count'],tick['health'],tick['hunger'],tick['thirst']]
                    if elapsed<=horizon:
                        series.append(values)
                    if elapsed==horizon:
                        crossed=idx
                        end_of_decision=tick['tick']==tr['after']['tick']
                require(all(tr['trace'][-1][k]==tr['after'][k] for k in FIELDS),where+' native tick endpoint')
                obs=out['observation']
            terminal=steps[-1]['terminal']
            if elapsed<horizon:
                require(terminal,'Game lacks data through the registered time')
                series.extend([list(series[-1]) for _ in range(horizon-elapsed)])
            require(len(series)==1+horizon,'Clock horizon differs')
            results[gs]=dict(series=series,verified_decisions=len(steps),actual_ticks=elapsed,
                             terminal_tick=elapsed if terminal else None,decision_at_horizon=crossed,
                             horizon_on_decision_boundary=end_of_decision)
    return arm,str(seed),results


def run(args):
    pack=json.loads(args.input.read_bytes())
    project,godot=args.project,args.godot
    hashes={f.relative_to(project).as_posix():sha(f) for f in project.rglob('*')
            if f.is_file() and not any(x.startswith('.') for x in f.relative_to(project).parts)
            and f.suffix in ('.gd','.tscn','.tres','.godot','.json')}
    # The prior study used the default JSON spacing for the game asset manifest.
    native=dict(godot=sha(godot),project=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                files=len(hashes))
    require(native==pack['engine'],'Game version differs from the evaluated models')
    tasks=[(arm,seed,games,project,godot,pack['horizon'])
           for arm,records in pack['arms'].items() for seed,games in records.items()]
    output={}
    with ProcessPoolExecutor(max_workers=min(4,len(tasks))) as workers:
        for arm,seed,games in workers.map(replay_seed,tasks):
            output.setdefault(arm,{})[seed]=games
    result=dict(engine=native,source_sha256=sha(args.input),tick_horizon=pack['horizon'],arms=output)
    temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,allow_nan=False,separators=(',',':')),encoding='utf8')
    temp.replace(args.output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('input','output','godot','project'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
