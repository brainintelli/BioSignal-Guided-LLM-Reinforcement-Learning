"""Pure contracts shared by pilot, training, and held-out assessment."""
import hashlib
import json
import math
from pathlib import Path

from tools.m8b_metrics import matrix, summarize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def validate_seed_split(*groups):
    seen = set()
    for group in groups:
        if not group or any(type(x) is not int for x in group):
            raise ValueError('Nonempty integer seed sets required')
        if len(set(group)) != len(group) or seen.intersection(group):
            raise ValueError('Repeated or overlapping seeds')
        seen.update(group)


def selection_eligible(candidate, parent):
    if set(matrix(candidate, 16)) != set(matrix(parent, 16)):
        raise ValueError('Screening world matrix differs')
    a, b = summarize(candidate), summarize(parent)
    return (a['successes'] >= 15 and a['early_deaths'] == 0
            and a['rmst'] - b['rmst'] >= -90
            and (a['successes'] - b['successes']) / 16 >= -.05
            and a['cvar20'] >= .9 * b['cvar20'])


def training_slots(update, batch_size, pool_size):
    """Rotate complete persistent cohorts; pausing never resets an episode."""
    if (type(update) is not int or update < 1 or type(batch_size) is not int or batch_size not in (4,8,16,32)
            or type(pool_size) is not int or not batch_size <= pool_size <= 128 or pool_size % batch_size):
        raise ValueError('Invalid persistent environment pool')
    start = ((update - 1) * batch_size) % pool_size
    return list(range(start, start + batch_size))


def should_stop_training(candidates, parent, config):
    count = config.get('stop_after_consecutive_perfect', 0)
    if count < 2 or len(candidates) < count or candidates[-1][0] < config['minimum_updates']:
        return False
    return all(summarize(r['episodes'])['successes'] == 16
               and selection_eligible(r['episodes'], parent) for _, r in candidates[-count:])


def estimate_seconds(*, decisions, training_seconds_per_decision,
                     evaluation_decisions, evaluation_seconds_per_decision, fixed_seconds):
    values = (decisions, training_seconds_per_decision, evaluation_decisions,
              evaluation_seconds_per_decision, fixed_seconds)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Invalid timing observations')
    central = (decisions * training_seconds_per_decision
               + evaluation_decisions * evaluation_seconds_per_decision + fixed_seconds)
    return dict(central_seconds=central, planning_seconds=central * 1.5,
                convergence_guaranteed=False,
                scope='Compute budget forecast; no inference that this many decisions attains the survival target')


def claim_test(directory, actor_digest):
    with (Path(directory) / 'test_claim.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(actor_sha256=actor_digest, single_use=True), stream)


def authorize_test(directory, config):
    """Recompute selection and require independent evidence before test exposure."""
    root = Path(directory)
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    completed = read('training_complete.json')
    actual_updates = completed['updates']
    expected_decisions = actual_updates * config['envs'] * config['rollout_steps']
    if (not 1 <= actual_updates <= config['updates'] or completed['decisions'] != expected_decisions
            or actual_updates % config['validation_every']):
        raise ValueError('Registered training budget is incomplete')
    replay = read('independent_replay.json')
    if replay.get('passed') is not True:
        raise ValueError('Independent replay did not pass')
    for name, item in replay['files'].items():
        if sha(root / name) != item['sha256']:
            raise ValueError('Audited trace changed')
    for number in range(1, actual_updates + 1):
        row = replay['files'].get(f'rollout_{number:04d}.jsonl.gz', {})
        if row.get('transitions') != config['envs'] * config['rollout_steps']:
            raise ValueError('Incomplete replay coverage of registered training')
    parent = read('parent_selection.json')['episodes']
    candidates = {}
    for number in range(config['validation_every'], actual_updates + 1, config['validation_every']):
        name = f'selection_{number:04d}'
        report = read(name + '.json')
        if name + '.jsonl.gz' not in replay['files']:
            raise ValueError('Selection trace was not independently replayed')
        candidates[number] = report
    if actual_updates < config['updates'] and not should_stop_training(list(candidates.items()), parent, config):
        raise ValueError('Early completion does not meet the registered stopping rule')
    winner = max(candidates, key=lambda n: (summarize(candidates[n]['episodes'])['successes'],
        summarize(candidates[n]['episodes'])['cvar20'], summarize(candidates[n]['episodes'])['rmst'], -n))
    chosen = candidates[winner]
    if not selection_eligible(chosen['episodes'], parent):
        raise ValueError('Selection does not qualify for held-out testing')
    checkpoint = root / f'checkpoint_{winner:04d}'
    saved = read('selection.json')
    reload = json.loads((checkpoint / 'independent_reload.json').read_text(encoding='utf-8'))
    if (saved['update'] != winner or Path(saved['checkpoint']).resolve() != checkpoint.resolve()
            or saved['actor_sha256'] != chosen['actor_sha256']
            or reload.get('passed') is not True or reload['logp_error'] > 1e-6
            or reload['actor_sha256'] != chosen['actor_sha256']):
        raise ValueError('Selected checkpoint or independent probability audit differs')
    return checkpoint


def verify_files(root, registration):
    for name, digest in registration['source_hashes'].items():
        if sha(Path(root) / name) != digest:
            raise ValueError('Source changed: ' + name)
    for item in registration['input_files'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Input changed: ' + item['path'])
