"""Replay saved actions through the archive's native tick-60 observer. No policy is run."""
import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STRESS = RUN / 'stress'
BASE = Path.home() / 'mea-state2-r2-20261118/run02/project'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'standalone/mea_state2_r2'))
sys.path.insert(0, str(BASE / 'standalone/mea_binary_r2'))
sys.path.insert(0, str(RUN))
from agent.m9_env import GodotEnv
from agent.m9_contract import potential
from mea_state2_r2.protocol import GAMMA
from mea_state2_r2.features import digest
from resources import GODOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def manifest(path):
    files = [path/'project.godot']
    for folder in ['scripts', 'configs', 'godot_tests']:
        files.extend(p for p in (path/folder).rglob('*') if p.is_file())
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(files)}


def prepare(spec):
    assert sha(BASE/'agent/rl_env.py') == 'c452aa83fe70bd6727f5b177c1ccd1fab19aa2192a100d4caa7ee09cdfa765dd'
    recipe = HERE/'make_observer.py'
    assert sha(recipe) == 'd975282b71083042d22e186f8597538bd56ea25184ab6d10b010962ae2bed909'
    identities = {}
    for scene in spec['scenes']:
        source = STRESS/'scenes'/scene/'plain'
        before = manifest(source)
        assert digest(before) == spec['game_pins'][scene]['plain']
        stage = HERE/'native'/scene
        stage.mkdir(parents=True, exist_ok=True)
        snapshot = stage/'audit-game'
        if not snapshot.exists():
            shutil.copyfile(recipe, stage/'make_observer.py')
            (stage/'resources.py').write_text('from pathlib import Path\nPROJECT = Path(' + repr(str(source)) + ')\n')
            subprocess.run([sys.executable, str(stage/'make_observer.py')], check=True)
        after = manifest(snapshot)
        changed = sorted(k for k in before if before[k] != after[k])
        assert before.keys() == after.keys() and changed == ['scripts/m9_world.gd', 'scripts/mea_supplied_session.gd']
        identities[scene] = {'source_pin':digest(before), 'observed_game_pin':digest(after), 'changed_files':changed}
    write(HERE/'native-sources.json', {'recipe_sha256':sha(recipe), 'engine_sha256':sha(GODOT), 'transport_sha256':sha(BASE/'agent/rl_env.py'), 'reward_source_sha256':sha(BASE/'agent/m9_contract.py'), 'scenes':identities})


def replay_game(env, path, spec, seed, arm, scene, world):
    document = read(path)
    game = document['game']
    assert document['seed'] == seed and document['scene'] == scene and game['mode'] == arm and game['world'] == world
    assert document['frozen'] == spec['states'][f'{seed}/{arm}']
    reply = env.reset(world)
    obs = reply['observation']
    assert reply['starting_scene'] == 'stress_' + scene
    assert obs['world_tick'] == 0 and all(obs['player'][k] == v for k,v in spec['scenes'][scene].items())
    start_cells = obs['visited_count']
    first = {'drink':None, 'eat':None}
    previous = potential(obs, 'bottleneck')
    native_r1 = 0.
    old_r1 = 0.
    cuts = 0
    mapping = spec['shuffle'][str(world)] if arm == 'shuffled' else list(range(60))
    horizon = spec['horizon']
    for row in game['rows']:
        action = row['action']
        t = row['tick']
        assert t == obs['world_tick'] < horizon and action in obs['allowed_actions']
        win = spec['windows'][mapping[t]]
        assert row['window'] == win['index'] and row['events'] == win['events_sha']
        assert row['features'] == ([0.]*4 if arm=='masked' else win['normalized'])
        step = env.step(action)
        tr = step['transition']
        obs = step['observation']
        assert digest(tr) == row['transition_sha'] == row['resource_event']['tr_sha']
        trace = [v for v in tr['trace'] if v['tick'] <= horizon]
        assert [v['tick'] for v in trace] == list(range(t+1, t+1+len(trace))) and trace
        end = trace[-1]
        cut = len(trace) != len(tr['trace'])
        assert row['cut'] == cut and row['end_tick'] == end['tick']
        assert row['new_cells_at_end'] == end['visited_count'] - start_cells
        assert row['r2_contribution'] == 0
        expected_event = {'start': t, 'end':end['tick'],'water0':tr['before']['thirst'], 'food0':tr['before']['hunger'],'water1':end['thirst'],'food1':end['hunger'],'tr_sha':row['transition_sha']}
        for verb, resource in [('drink','thirst'), ('eat','hunger')]:
            moment = next((v['tick'] for v in trace if action['verb'] == verb and v[resource] > tr['before'][resource]), None)
            expected_event[verb] = moment
            if moment is not None and first[verb] is None:
                first[verb] = moment
        assert expected_event == row['resource_event']
        boundary = env.call('landmark')['landmark'] if cut else obs
        assert boundary['world_tick'] == end['tick'] and boundary['visited_count'] == end['visited_count']
        assert all(abs(end[k] - boundary['player'][k]) <= .050001 for k in ['health','hunger','thirst'])
        after = potential(boundary, 'bottleneck', end['dead'])
        living = sum(GAMMA**i * .01 * int(not v['dead']) for i,v in enumerate(trace))
        score = living + GAMMA**len(trace)*after - previous
        if not cut:
            assert math.isclose(score,row['reward'],rel_tol=0,abs_tol=1e-11)
        previous = after
        native_r1 += score
        old_r1 += row['reward']
        cuts += int(cut)
    assert game['decisions'] == len(game['rows']) and cuts in (0,1) and bool(cuts) == game['cut']
    assert first == game['first_success_tick'] and game['restricted_wait'] == {k:60 if v is None else v for k,v in first.items()}
    assert game['end_tick'] == end['tick'] and (end['tick'] == horizon or end['dead'])
    assert game['survived'] == int(not end['dead']) and game['new_cells'] == end['visited_count'] - start_cells
    assert all(game[k] == end[k] for k in ['health','hunger','thirst'])
    assert math.isclose(old_r1, game['r1'], abs_tol=1e-11,rel_tol=0)
    return {'scene':scene,'seed':seed,'arm':arm,'world':world,'decisions':len(game['rows']),'end_tick':end['tick'],'survived':int(not end['dead']),'cells':game['new_cells'],'drink_wait':game['restricted_wait']['drink'],'eat_wait':game['restricted_wait']['eat'],'cut':cuts,'r1_recorded':old_r1,'r1_native':native_r1}


def replay_job(job):
    seed, scene, arm, worlds = job
    spec = read(STRESS/'design.json')
    folder = STRESS/'games'/str(seed)/scene/arm
    pin = digest(spec)
    frozen = read(folder/'frozen.json')
    assert frozen == {'scene':scene,'seed':seed,'arm':arm,'state':spec['states'][f'{seed}/{arm}'],'worlds':len(spec['worlds']),'spec':pin}
    # Limit CPU affinity inherited by the engine without altering recorded game settings.
    available = sorted(os.sched_getaffinity(0)); ix = (seed + spec['arms'].index(arm)) % len(available)
    os.sched_setaffinity(0, {available[ix],available[(ix+1)%len(available)]})
    try:
        with GodotEnv(str(GODOT),str(HERE/'native'/scene/'audit-game'),map_mode='infinite_freeze',navigation_steps=4) as env:
            return [replay_game(env,folder/f'{w}.json',spec,seed,arm,scene,w) for w in worlds]
    finally:
        os.sched_setaffinity(0, available)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=10)
    args = parser.parse_args()
    spec = read(STRESS/'design.json'); prepare(spec)
    games = []
    seeds = spec['training_seeds'][:1] if args.smoke else spec['training_seeds']
    worlds = spec['worlds'][:3] if args.smoke else spec['worlds']
    jobs = [(s,scene,a,worlds) for s in seeds for scene in spec['scenes'] for a in spec['arms']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(replay_job,job):job[:3] for job in jobs}
        for count,future in enumerate(as_completed(futures),1):
            try: games.extend(future.result())
            except Exception as err: raise RuntimeError(f'replay failed at {futures[future]}') from err
            if count%12==0: print(f'{count}/{len(jobs)} groups, {len(games)} games',flush=True)
    games.sort(key=lambda r:tuple(r[k] for k in ['scene','seed','arm','world']))
    assert len(games) == len(jobs)*len(worlds)
    cut = [r for r in games if r['cut']]
    diffs = [abs(r['r1_native']-r['r1_recorded']) for r in games]
    receipt = dict(episodes=len(games),decisions=sum(r['decisions'] for r in games),primary_and_wait_mismatches=0,transition_digest_mismatches=0,episodes_crossing_tick60=len(cut),r1_differences_over_1e_10=sum(v>1e-10 for v in diffs),max_abs_r1_difference=max(diffs))
    if not args.smoke:
        assert len(games)==18432 and receipt['decisions']==669715
        output = HERE/'native-replay.csv'
        with output.with_suffix('.csv.tmp').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(games[0]));writer.writeheader();writer.writerows(games)
        output.with_suffix('.csv.tmp').replace(output)
        receipt.update(csv_sha256=sha(output),source_receipt_sha256=sha(HERE/'native-sources.json'),verifier_sha256=sha(Path(__file__)),scope='Post-evaluation replay of stored actions. No refitting or new action sampling.')
        write(HERE/'native-replay.json',receipt)
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
