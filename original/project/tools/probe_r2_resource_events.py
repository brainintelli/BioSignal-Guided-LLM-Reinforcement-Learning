"""Measure native R1 on scripted resource events in two isolated development scenes."""
import argparse
import hashlib
import json
import random
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv

STUDY = ROOT / 'output/mea_predictability_22021_20261119T042416Z'
HISTORY = ROOT / 'output/mea_predictability_22021_20261118T093720Z'
VERIFICATION = STUDY / 'offline-confirmatory-results/verification'
GAMMA = .99985
HORIZON = 60
SCENES = {'thirst': ('drink', 1.6), 'hunger': ('eat', 4.)}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def as_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def collect_ids(design):
    identifiers = set()
    for key in ('worlds', 'holdout_seeds', 'training_seeds'):
        identifiers.update(i for i in design.get(key, []) if type(i) is int)
    for key in ('episodes', 'restart_worlds'):
        for worlds in design.get(key, {}).values():
            identifiers.update(worlds)
    return identifiers


def prepare_plan(executable):
    readings = ['offline-confirmatory-results/design.json', 'offline-stress-three-arms/design.json',
                'offline-stress-three-arms/stress/design.json']
    histories = ['offline-r2-emergency-drink/design.json', 'offline-r2-emergency-drink/evaluation/plan.json',
                 'offline-r2-bounded-005/design.json', 'offline-r2-bounded-005/thirst-60-paired/design.json']
    registered = set()
    for path in [STUDY / name for name in readings] + [HISTORY / name for name in histories]:
        registered.update(collect_ids(json.loads(path.read_bytes())))
    with tarfile.open(VERIFICATION / 'evaluated-evidence.tar.gz', 'r|gz') as tar:
        for entry in tar:
            if entry.name == 'design.json':
                registered.update(collect_ids(json.load(tar.extractfile(entry))))
                break
        else:
            raise ValueError('Archived training design missing')
    recovery = json.loads((VERIFICATION / 'source-recovery.json').read_bytes())
    registered.update(recovery['development_maps'])
    worlds = random.Random(220210293).sample(range(670000000, 671000000), 32)
    assert not registered.intersection(worlds)
    return dict(worlds=worlds, candidate_maps_excluded=len(registered), wait_ticks=[0, 12, 24], horizon=HORIZON,
                scenarios=SCENES, gamma_tick=GAMMA, reward_variant='bottleneck',
                action_rule='wait until planned tick; take legal eat/drink, else navigate toward nearest available resource by path length then id; otherwise wait',
                event='first legal target action with actual reserve gain from a pre-action reserve <=20, within horizon and alive',
                missing_event='retain map and delay with null event', native_roles=['plain', 'audit'],
                game_source_pins={s: recovery['restored_scene_pins'][s] for s in SCENES},
                godot_sha256=sha(executable), reward_source_sha256=sha(ROOT / 'agent/m9_contract.py'),
                protocol_sha256=sha(Path(__file__)), model_updates=0, auxiliary_reward_applied=0)


def stage_sources(directory, pins):
    with tarfile.open(VERIFICATION / 'recovered-scenes.tar.gz') as tar:
        for entry in tar:
            parts = entry.name.split('/', 3)
            if entry.isdir() or len(parts) != 4 or parts[0] != 'pinned-scenes' or parts[1] not in pins:
                continue
            scene, role, filename = parts[1:]
            assert role in pins[scene] and entry.isfile()
            target = directory / scene / role / filename
            assert target.resolve().is_relative_to(directory / scene / role)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(tar.extractfile(entry).read())
    for scene, roles in pins.items():
        for role, expected in roles.items():
            root = directory / scene / role
            file_pins = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
            assert canonical_digest(file_pins) == expected


def choose(obs, scene, delay):
    wait = {'verb': 'wait'}
    if obs['world_tick'] < delay:
        return wait
    legal = obs['allowed_actions']
    action = {'verb': SCENES[scene][0]}
    if action in legal:
        return action
    kinds = {'water'} if scene == 'thirst' else {'berry', 'mushroom'}
    targets = [(t['path_steps'], t['id']) for t in obs['nearby']
               if t['kind'] in kinds and t['available'] and {'verb': 'navigate', 'target_id': t['id']} in legal]
    return {'verb': 'navigate', 'target_id': min(targets)[1]} if targets else wait


def measure(game, audit, world, scene, delay):
    start = game.reset(world)
    obs = start['observation']
    witness = audit.reset(world)
    assert start['starting_scene'] == witness['starting_scene'] == 'stress_' + scene
    assert obs == witness['observation'] and obs['player'][scene] == SCENES[scene][1]
    tracker = Reward(obs, 'bottleneck', GAMMA)
    result = dict(scene=scene, world=world, delay=delay, event=None, decisions=0)
    while obs['world_tick'] < HORIZON:
        action = choose(obs, scene, delay)
        assert action in obs['allowed_actions']
        step = game.step(action)
        other = audit.step(action)
        tr, obs = step['transition'], step['observation']
        assert tr == other['transition'] and obs == other['observation'] and step['terminated'] == other['terminated']
        score = tracker.score(tr, obs)
        result['decisions'] += 1
        before, after = tr['before'][scene], tr['after'][scene]
        if tr['after']['tick'] <= HORIZON and not step['terminated'] and action['verb'] == SCENES[scene][0] and before <= 20 and after > before:
            result['event'] = dict(tick=tr['after']['tick'], ticks=score['elapsed_ticks'], reserve_before=before,
                                   reserve_after=after, r1=score['reward'], components=score['components'],
                                   transition_sha256=canonical_digest(tr))
            break
        if step['terminated']:
            break
    return result


def run(executable, output):
    plan = prepare_plan(executable)
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / 'dev-probe-plan.json'
    plan_bytes = as_bytes(plan)
    if plan_file.exists():
        if plan_file.read_bytes() != plan_bytes:
            raise ValueError('A different probe plan already exists in this output directory')
    else:
        with plan_file.open('xb') as stream:
            stream.write(plan_bytes)
    records = []
    with tempfile.TemporaryDirectory(prefix='mea-event-scale-') as temporary:
        directory = Path(temporary).resolve()
        stage_sources(directory, plan['game_source_pins'])
        for scene in SCENES:
            settings = dict(map_mode='infinite_freeze', navigation_steps=4)
            with GodotEnv(str(executable), directory / scene / 'plain', **settings) as game, \
                 GodotEnv(str(executable), directory / scene / 'audit', **settings) as audit:
                for world in plan['worlds']:
                    for delay in plan['wait_ticks']:
                        records.append(measure(game, audit, world, scene, delay))
    assert len({(r['scene'], r['world'], r['delay']) for r in records}) == len(records) == 192
    data = dict(plan_sha256=sha(plan_file), episodes=len(records), decisions=sum(r['decisions'] for r in records),
                positive_events=sum(r['event'] is not None for r in records), records=records)
    evidence = output / 'dev-probe.json'
    contents = as_bytes(data)
    if evidence.exists() and evidence.read_bytes() != contents:
        raise ValueError('This replay disagrees with the previous probe; use a separate output directory to investigate')
    if not evidence.exists():
        temp = evidence.with_suffix('.tmp')
        temp.write_bytes(contents)
        temp.replace(evidence)
    print(f'{evidence}  {sha(evidence)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--godot', type=Path, default=ROOT / '.tools/godot/Godot_v4.7.2-stable_win64.exe')
    parser.add_argument('--output', type=Path, default=STUDY / 'offline-r2-scale-exploration')
    args = parser.parse_args()
    run(args.godot, args.output)
