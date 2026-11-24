"""One-shot, CPU-only diagnosis of immutable completed training/evaluation traces."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re


def trace_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def evaluation_diagnosis(path):
    counts = Counter()
    per_seed = defaultdict(Counter)
    for row in trace_rows(path):
        obs, action = row['observation'], row['action']
        body = obs['player']
        seed = row['seed']
        verb = action['verb']
        counts['decisions'] += 1
        counts['action_' + verb] += 1
        per_seed[seed]['decisions'] += 1
        water = [x for x in obs['nearby'] if x['kind'] == 'water']
        drink_legal = any(x['verb'] == 'drink' for x in obs['allowed_actions'])
        if body['thirst'] < 20:
            counts['thirst_below20'] += 1
            counts['low_thirst_water_seen'] += bool(water)
            counts['low_thirst_drink_legal'] += drink_legal
            counts['low_thirst_drink_selected'] += verb == 'drink'
            counts['low_thirst_navigate_water'] += verb == 'navigate' and any(x['id'] == action.get('target_id') for x in water)
            per_seed[seed]['low_thirst_no_water'] += not bool(water)
        if body['hunger'] < 20:
            counts['hunger_below20'] += 1
            counts['low_hunger_has_food'] += obs['inventory']['berries'] > 0
            counts['low_hunger_eat'] += verb == 'eat'
        if verb == 'move':
            after = row['transition']['after']['player']
            delta = (after['x']-body['x'], after['y']-body['y'])
            counts['move_delta_' + str(delta)] += 1
        if row['transition']['after']['dead']:
            final = row['transition']['after']
            counts['deaths'] += 1
            counts['death_thirst0'] += final['thirst'] == 0
            counts['death_hunger0'] += final['hunger'] == 0
            counts['death_with_food'] += final['inventory']['berries'] > 0
            counts['death_at_edge'] += min(final['player']['x'],final['player']['y']) <= 2 or max(final['player']['x'],final['player']['y']) >= 94
    return dict(totals=dict(counts), per_seed={str(k):dict(v) for k,v in per_seed.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.run)
    rows = [json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
    recent = rows[-32:]
    screens = {}
    for path in sorted(root.glob('selection_*.json')):
        if not re.fullmatch(r'selection_\d{4}\.json', path.name):
            continue
        report = json.loads(path.read_text())
        screens[path.stem] = dict(summary=report['summary'], seconds=report['seconds'])
    snapshots = {}
    for update in sorted({1,32,64,96,128,160,192,rows[-1]['update']}):
        path = root/f'rollout_{update:04d}.jsonl.gz'
        if not path.exists():
            continue
        slots = {}
        for row in trace_rows(path):
            obs = row['observation']
            slots[row['env']] = dict(seed=row['seed'], tick=obs['tick'], body=obs['player'], inventory=obs['inventory'])
        snapshots[str(update)] = slots
    diagnostics = {}
    for update in (64,96,192):
        path = root/f'selection_{update:04d}.jsonl.gz'
        if path.exists():
            diagnostics[str(update)] = evaluation_diagnosis(path)
    episodes=[json.loads(x) for x in (root/'episodes.jsonl').read_text().splitlines()]
    result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),updates=rows[-1]['update'],
        decisions=rows[-1]['decisions'],recent_mean_seconds=sum(x['rollout_seconds']+x['update_seconds'] for x in recent)/len(recent),
        recent_kl_stops=sum(x['kl_early_stop'] for x in recent),recent_actor_steps=sum(x['optimizer_steps'] for x in recent),
        completed_natural_episodes=len(episodes),screens=screens,training_slot_snapshots=snapshots,evaluation_diagnoses=diagnostics)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
