"""Evaluate one frozen 22554 checkpoint on paired maps, using native Godot ticks."""
import hashlib
import json
import sys
from pathlib import Path

from clock_replay import sha, require
from mea_online.m9_runtime import M9Runtime, digest
from train_long_recorded import decisions, read, windows


def save(path, data):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    tmp.replace(path)


def verify_engine(cfg, expected):
    project = Path(cfg['project'])
    files = {p.relative_to(project).as_posix(): sha(p) for p in project.rglob('*')
             if p.is_file() and p.suffix in {'.gd', '.tscn', '.tres', '.godot', '.json'}
             and not any(k.startswith('.') for k in p.relative_to(project).parts)}
    got = dict(files=len(files), godot=sha(Path(cfg['godot'])),
               project=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    require(got == expected, 'Game engine or project changed')


def run(root, job_id):
    import torch
    plan = read(root/'protocol.json')
    job = plan['models'][job_id]
    out = root/'runs'/job_id
    out.mkdir(parents=True, exist_ok=True)
    cp = Path(job['checkpoint'])
    reg = read(cp.parent/'registration.json')
    cfg = reg['config']
    arm = cfg['arm']
    require(sha(cp/'manifest.json') == job['manifest'] and digest(reg) == job['registration']
            and arm == job['arm'] and cfg['seed'] == job['training_seed'] and cfg['stimulation'] is False,
            'Checkpoint or registration differs')
    verify_engine(cfg, plan['engine'])
    order, prime = [], None
    if arm != 'pure':
        require(sha(root/'replay.json') == plan['source_sha'], 'Recording differs')
        data = read(root/'replay.json')
        order = windows(data, arm, False)
        if arm == 'r2only':
            order = [(i, None, phase, None) for i, _, phase, _ in order]
        else:
            prime = read(root/'preceding-quiet.json')['prime']
            require(digest(prime) == plan['prime_digest'], 'Priming window changed')
        require(reg['test_pairs'] == [list(t[:3]) for t in order], 'Holdout windows changed')
    require(not set(plan['maps']) & (set(reg['eval_seeds']) | set(reg['budget']['training_seeds'])),
            'Fresh seeds overlap with earlier games')
    torch.manual_seed(cfg['seed'])
    rt = M9Runtime(cfg, out, frozen_evaluation=True)
    try:
        rt.core.load_checkpoint(cp, digest(reg))
        require(rt.state() == job['state'], 'Reloaded weights differ')
        old_seed = reg['eval_seeds'][0]
        original = read(cp.parent/'eval-trace.json')[str(old_seed)]
        for i, (_, step, _, _) in enumerate(decisions(rt, cfg, order, prime, old_seed, 4, False)):
            ref = original[i]
            require(all(step[k] == ref[k] for k in ('action','obs_hash','duration','context_donor','after'))
                    and max(abs(x-y) for x,y in zip(step['verb_probabilities'],ref['verb_probabilities'])) < 1e-6,
                    'Archived map does not reproduce')
        require(i == 3, 'Incomplete archived probe')
        horizon = max(plan['ticks'])
        for seed in plan['maps']:
            dest = out/f'{seed}.json'
            if dest.is_file():
                previous=read(dest)
                require(previous['checkpoint'] == job['manifest'] and previous['seed'] == seed
                        and previous['protocol'] == sha(root/'protocol.json'), 'Cached game differs')
                continue
            ticks, actions = [], []
            terminal_tick = None
            for row, step, _, _ in decisions(rt, cfg, order, prime, seed, horizon, False):
                tr = row['transition']
                if not ticks:
                    initial = tr['before']
                    first_hash = step['obs_hash']
                    ticks.append([0, initial['health'], initial['hunger'], initial['thirst']])
                start = len(ticks)-1
                legal = {a['verb'] for branch in row['prepared']['branches'] for a in branch}
                actions.append([start, step['action']['verb'], 'eat' in legal, 'drink' in legal])
                require(len(tr['trace']) == step['duration'], 'Incomplete action trace')
                for t in tr['trace']:
                    require(t['tick'] == initial['tick'] + len(ticks), 'Discontinuous game clock')
                    ticks.append([t['visited_count']-initial['visited_count'], t['health'], t['hunger'], t['thirst']])
                if row['terminal']:
                    terminal_tick = len(ticks)-1
                    break
                if len(ticks)>horizon:
                    break
            require(len(ticks)>horizon or terminal_tick is not None, 'Game ended early')
            if terminal_tick is not None and len(ticks)<=horizon:
                ticks += [ticks[-1]]*(horizon+1-len(ticks))
            save(dest, dict(seed=seed, checkpoint=job['manifest'], protocol=sha(root/'protocol.json'),
                            first_obs_hash=first_hash, ticks=ticks[:horizon+1], actions=actions,
                            terminal_tick=terminal_tick))
        require(rt.state() == job['state'], 'Evaluation changed model weights')
        save(out/'done.json', dict(probe_decisions=4, games=len(plan['maps']), final_state=rt.state(),
                                   game_hashes={s:sha(out/f'{s}.json') for s in plan['maps']}))
    finally:
        rt.close()


if __name__ == '__main__':
    run(Path(sys.argv[1]), sys.argv[2])
