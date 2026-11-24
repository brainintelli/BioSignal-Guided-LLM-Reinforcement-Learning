"""Re-score the original six archived decisions at their common sixth game tick."""
import collections
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from agent.m9_env import GodotEnv
from agent.m9_contract import Reward, potential

STUDY = ROOT/'output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
ARMS = ['masked', 'baseline', 'sham', 'regular', 'irregular']
GAMMA = .99985


def observer(game):
    for dirname in ['scripts', 'configs', 'godot_tests']:
        shutil.copytree(ROOT/dirname, game/dirname)
    shutil.copy2(ROOT/'project.godot', game/'project.godot')
    source = (ROOT/'standalone/mea_live_pair/supplied_session.gd').read_text(encoding='utf8')
    start = 'func handle(request: Dictionary) -> Dictionary:\n'
    assert source.count(start) == 1
    (game/'scripts/mea_supplied_session.gd').write_text(source.replace(start,start+
        '\tif request.get("op", "") == "landmark":\n\t\treturn {"ok": true, "id": request.id, "landmark": world.archive_landmark}\n'),encoding='utf8')
    runner = game/'godot_tests/m9_environment.gd'
    runner.write_text(runner.read_text(encoding='utf8').replace('scripts/m9_environment_session.gd','scripts/mea_supplied_session.gd'),encoding='utf8')
    script = game/'scripts/m9_world.gd'
    source = script.read_text(encoding='utf8')
    for token,addition in [('var navigation_steps: int = 1','\nvar archive_landmark: Dictionary = {}'),
                           ('\t\tsuper._advance(1)\n\t\t_sync_deadlines()','\n\t\tif tick == 6:\n\t\t\tarchive_landmark = observation().duplicate(true)')]:
        assert source.count(token)==1
        source=source.replace(token,token+addition)
    script.write_text(source,encoding='utf8')


def main():
    design=json.loads((STUDY/'design.json').read_bytes())
    seeds, worlds = design['training_seeds'],design['holdout_seeds']
    assert design['eval_decisions']==6
    groups=collections.defaultdict(list)
    for arm in ARMS:
        for i,seed in enumerate(seeds):
            directory=STUDY/('pre-stim-baseline' if arm=='baseline' else '')/'models'/str(seed)/arm
            items=[json.loads(line) for line in (directory/'holdout.jsonl').read_text(encoding='utf8').splitlines()]
            assert [x['world'] for x in items]==worlds
            for j,item in enumerate(items):
                rows=item['rows']
                assert len(rows)==6 and rows[-1]['after']['tick']>=6
                key=(item['world'],json.dumps([r['action'] for r in rows],sort_keys=True))
                groups[key].append((arm,i,j,rows))
    values={a:np.full((8,48,5),np.nan) for a in ARMS}
    with tempfile.TemporaryDirectory(prefix='22021-game-observer-') as tmp:
        game=Path(tmp)
        observer(game)
        with GodotEnv(str(ROOT/'.tools/godot/Godot_v4.7.2-stable_win64_console.exe'),str(game),
                      map_mode='infinite_freeze',navigation_steps=4) as env:
            for (world, actions), copies in groups.items():
                initial=env.reset(world)
                obs=initial['observation']
                assert initial['starting_scene']=='supplied_95' and obs['world_tick']==0
                visited=obs['visited_count']
                tracker=Reward(obs,'bottleneck',GAMMA)
                cumulative=0.
                cut=None
                for k,action in enumerate(json.loads(actions)):
                    prev=tracker.previous
                    step=env.step(action);obs=step['observation'];tr=step['transition']
                    score=tracker.score(tr,obs)['reward']
                    for _,_,_,rows in copies:
                        assert tr['after']==rows[k]['after'] and math.isclose(score,rows[k]['r1'],abs_tol=1e-11)
                    if cut is None and tr['after']['tick']>=6:
                        frame=next(t for t in tr['trace'] if t['tick']==6)
                        b=obs if tr['after']['tick']==6 else env.call('landmark')['landmark']
                        assert b['world_tick']==6 and b['visited_count']==frame['visited_count'] and not frame['dead']
                        dt=6-tr['before']['tick']
                        living=sum(.01*GAMMA**t for t in range(dt))
                        reward=cumulative+living+GAMMA**dt*potential(b,'bottleneck')-prev
                        cut=[reward,b['visited_count']-visited,frame['health'],frame['thirst'],frame['hunger']]
                    cumulative+=score
                assert cut is not None
                for a,i,j,_ in copies:values[a][i,j]=cut
    assert all(np.isfinite(v).all() for v in values.values())
    rng=np.random.default_rng(220211119)
    si=rng.integers(8,size=(30000,8,1));mi=rng.integers(48,size=(30000,1,48))
    results={}
    for arm,ctrl in [('sham','masked'),('baseline','masked'),('regular','sham'),('irregular','sham')]:
        delta=values[arm]-values[ctrl]
        q=.05/(2*(2 if arm=='baseline' else 6))
        results[arm]={}
        for k,metric in enumerate(['r1','cells','health','water','food']):
            d=delta[:,:,k]
            bounds=np.quantile(d[si,mi].mean(axis=(1,2)),[q,1-q])
            results[arm][metric]=dict(mean=float(d.mean()),interval=bounds.tolist(),positive_seeds=int(sum(d.mean(axis=1)>1e-10)))
    summary=dict(tick=6,archived_games=sum(len(g) for g in groups.values()),unique_trajectories=len(groups),
                 matched_action_ends=6*sum(len(g) for g in groups.values()),results=results)
    (OUT/'clock6.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary))


if __name__=='__main__':main()
