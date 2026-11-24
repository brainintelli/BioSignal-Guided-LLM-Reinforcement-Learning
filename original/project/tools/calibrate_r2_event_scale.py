"""Rescore archived R1-only training batches. No policy, simulator or device is started."""
import argparse
import hashlib
import json
import math
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)] + [str(ROOT / 'standalone' / name) for name in
                               ('mea_live_pair', 'mea_binary_r2', 'mea_state2_r2')]
from agent.rl_contract import gae
from livepair.predictability import outcome
from mea_state2_r2.features import digest
from replay_reward import ReplayRewardMix, emergency_drink_label

CURRENT = ROOT / 'output/mea_predictability_22021_20261119T042416Z'
HISTORY = ROOT / 'output/mea_predictability_22021_20261118T093720Z'
GAMMA = 0.99985
LAMBDA = 0.999
AMPLITUDE = 0.10
WEIGHTS = {'event_1pct': .10, 'event_2p5pct': .25, 'event_5pct': .50}
PROTOCOLS = {'low_water_drink': 'offline-r2-emergency-drink',
             'signed_game_label': 'offline-r2-bounded-005'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


@dataclass
class Batch:
    seed: int
    arm: str
    index: int
    r1: np.ndarray
    sign: np.ndarray
    ticks: list
    values: list
    next_values: list
    terminal: list
    boundary: list
    advantages: list
    water: list
    can_drink_low: list

    def recompute(self, rewards):
        return gae(rewards, self.values, self.next_values, self.ticks,
                   self.terminal, self.boundary, GAMMA, LAMBDA)

    def per_tick(self):
        return np.abs(self.r1) / [sum(GAMMA ** t for t in range(dt)) for dt in self.ticks]


def parse_batch(raw, seed, arm, index, protocol, world):
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 12
    labels, opportunity, water = [], [], []
    for i, row in enumerate(rows):
        transition, score = row['transition'], row['r1_score']
        before, after = transition['before'], transition['after']
        assert row['index'] == i
        if i == 0:
            assert row['seed'] == world
        assert row['duration'] == transition['elapsed_ticks'] == score['elapsed_ticks']
        assert row['duration'] >= 1 and row['terminal'] == after['dead']
        assert row['boundary'] == (row['terminal'] or i == len(rows) - 1)
        if not row['boundary']:
            assert rows[i + 1]['seed'] == row['seed'] and row['next_value'] == rows[i + 1]['value']
        menu = row['observation']['allowed_actions']
        assert row['action'] in menu and row['action'] == transition['action']
        low_opportunity = before['thirst'] <= 20 and any(a['verb'] == 'drink' for a in menu)
        label = emergency_drink_label(row['action']['verb'], before['thirst'], after['thirst'], 20.)
        assert not label or low_opportunity
        if protocol == 'signed_game_label':
            eligible = (i + 1) % 3 == 0 and not row['terminal']
            label = (1 if outcome(transition)['correct'] else -1) if eligible else 0
        assert row['r2'] == label and (not row['terminal'] or label == 0)
        assert score['variant'] == 'bottleneck' and score['gamma_tick'] == GAMMA
        assert row['reward'] == score['reward'] and row['contribution'] == 0
        assert math.isclose(row['reward'], sum(score['components'].values()), rel_tol=0, abs_tol=1e-12)
        labels.append(label)
        opportunity.append(low_opportunity)
        water.append(before['thirst'])
    batch = Batch(seed, arm, index, np.array([r['reward'] for r in rows]), np.array(labels),
                  *[[r[key] for r in rows] for key in
                    ('duration', 'value', 'next_value', 'terminal', 'boundary', 'advantage')],
                  water, opportunity)
    advantages, returns = batch.recompute(batch.r1)
    np.testing.assert_allclose(advantages, batch.advantages, rtol=0, atol=1e-10)
    np.testing.assert_allclose(returns, [r['return'] for r in rows], rtol=0, atol=1e-10)
    return batch


def current_training(pins):
    verification = CURRENT / 'offline-confirmatory-results/verification'
    archive = verification / 'evaluated-evidence.tar.gz'
    index_path = verification / 'SHA256SUMS.evaluated'
    for row in (verification / 'transfer.sha256').read_text().splitlines():
        expected, name = row.split('  ', 1)
        if name in (archive.name, index_path.name):
            path = verification / name
            assert sha(path) == expected
            pins[path.relative_to(ROOT).as_posix()] = expected
    assert all(p.relative_to(ROOT).as_posix() in pins for p in (archive, index_path))
    index = {name: pin for pin, name in (r.split('  ', 1) for r in index_path.read_text().splitlines())}
    batches, registrations, design = {}, set(), None
    with tarfile.open(archive, 'r|gz') as tar:
        for entry in tar:
            if entry.name == 'design.json':
                raw = tar.extractfile(entry).read()
                assert sha_bytes(raw) == index[entry.name]
                design = json.loads(raw)
                assert design['beta'] == 0 and design['alpha'] == 1
                assert design['starting_reserves'] == {'health': 100, 'hunger': 95, 'thirst': 95}
            parts = entry.name.split('/')
            if len(parts) != 4 or parts[0] != 'models':
                continue
            _, seed, arm, name = parts
            if name == 'registration.json':
                raw = tar.extractfile(entry).read()
                assert sha_bytes(raw) == index[entry.name]
                reg = json.loads(raw)
                assert reg['training_seed'] == int(seed) and reg['arm'] == arm
                assert reg['config']['gamma_tick'] == GAMMA and reg['config']['gae_lambda'] == LAMBDA
                registrations.add((int(seed), arm))
            if not name.startswith('train-'):
                continue
            batch_id = int(name.removeprefix('train-').removesuffix('.jsonl'))
            key = int(seed), arm, batch_id
            assert key not in batches and design is not None
            raw = tar.extractfile(entry).read()
            assert sha_bytes(raw) == index[entry.name]
            batches[key] = parse_batch(raw, *key, 'low_water_drink', design['episodes'][seed][batch_id - 1])
    expected_keys = {(s, a, b) for s in design['training_seeds'] for a in design['arms'] for b in range(1, 5)}
    assert set(batches) == expected_keys and len(batches) == 384
    assert registrations == {(s, a) for s, a, _ in batches}
    return list(batches.values())


def historical_training(protocol, pins):
    root = HISTORY / PROTOCOLS[protocol]
    design = read(root / 'design.json')
    pins[(root / 'design.json').relative_to(ROOT).as_posix()] = sha(root / 'design.json')
    batches = []
    for seed in design['training_seeds']:
        folder = root / f'models/{seed}/r1'
        manifest = folder / 'verified-transfer.json'
        files = read(manifest)
        pins[manifest.relative_to(ROOT).as_posix()] = sha(manifest)

        def verified(name):
            path = folder / name
            raw = path.read_bytes()
            assert sha_bytes(raw) == files[path.relative_to(root).as_posix()]
            return raw

        reg = json.loads(verified('registration.json'))
        assert reg['study_digest'] == digest(design)
        assert reg['beta'] == 0 and reg['alpha'] == 1 and reg['training_seed'] == seed and reg['arm'] == 'r1'
        assert reg['config']['gamma_tick'] == GAMMA and reg['config']['gae_lambda'] == LAMBDA
        assert reg['reward_mix'] == design['reward_mix']
        for b, world in enumerate(design['episodes'][str(seed)], 1):
            batches.append(parse_batch(verified(f'train-{b}.jsonl'), seed, 'r1', b, protocol, world))
    assert len(design['training_seeds']) == 8 and len(batches) == 32
    return design, batches


def distribution(values):
    if not len(values):
        return None
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                    map(float, np.quantile(values, [0, .1, .5, .9, 1]))))


def describe(batches):
    labels = np.concatenate([b.sign for b in batches])
    active = labels != 0
    native = np.abs(np.concatenate([b.r1 for b in batches]))
    per_tick = np.concatenate([b.per_tick() for b in batches])
    seed_counts = defaultdict(int)
    for b in batches:
        seed_counts[b.seed] += int(np.count_nonzero(b.sign))
    return dict(steps=len(native), batches=len(batches), model_seeds=len(seed_counts),
                models=len({(b.seed, b.arm) for b in batches}), arms=sorted({b.arm for b in batches}),
                positive=int(sum(labels > 0)), negative=int(sum(labels < 0)), silent=int(sum(labels == 0)),
                active_batches=sum(bool(np.any(b.sign)) for b in batches), events_per_seed=dict(seed_counts),
                low_water_legal_drink_steps=sum(sum(b.can_drink_low) for b in batches),
                min_water_before=min(min(b.water) for b in batches), abs_r1=distribution(native),
                abs_r1_per_discounted_tick=distribution(per_tick), active_abs_r1=distribution(native[active]),
                active_abs_r1_per_discounted_tick=distribution(per_tick[active]))


def fit_reference(batches):
    events = [r for b in batches for r, z in zip(b.per_tick(), b.sign) if z and r > 0]
    if not events:
        raise ValueError('This training subset has no nonzero R1 on labeled events')
    return float(median(events))


def normalized(advantages):
    values = np.asarray(advantages, dtype=np.float32)
    return (values - values.mean()) / max(values.std(), 1e-8)


def compare(batches, mix):
    """A null mix means a hypothetical unit label. A zero beta is the actual R1-only control."""
    additive, changes, drifts, batch_shares = [], [], [], []
    baseline_stds = []
    sample_ratios = []
    flips, direction_changes = 0, 0
    for b in batches:
        if mix is None:
            contribution = b.sign.astype(float)
        else:
            contribution = np.asarray([mix.score(float(v), int(z), dt)[1]
                                       for v, z, dt in zip(b.r1, b.sign, b.ticks)])
            bound = mix.beta * mix.scale_fraction * np.abs(b.r1)
            assert np.all(np.abs(contribution) <= bound + 1e-15)
        total = b.r1 + contribution
        assert np.all(contribution[b.sign == 0] == 0)
        advantages, _ = b.recompute(total)
        standard, proposed = normalized(b.advantages), normalized(advantages)
        baseline_stds.append(float(np.std(b.advantages)))
        changes.append(proposed.astype(float) - standard)
        drifts.append(float(np.sqrt(np.mean(changes[-1] ** 2))))
        flips += int(np.count_nonzero(np.sign(proposed) != np.sign(standard)))
        direction_changes += int(np.count_nonzero(np.sign(total) != np.sign(b.r1)))
        denominator = np.abs(b.r1).sum()
        batch_shares.append(float(np.abs(contribution).sum() / denominator) if denominator else None)
        active = (b.sign != 0) & (b.r1 != 0)
        sample_ratios.extend(np.abs(contribution[active] / b.r1[active]).tolist())
        additive.append(contribution)
    native = np.concatenate([b.r1 for b in batches])
    sig = np.concatenate([b.sign for b in batches])
    delta = np.concatenate(additive)
    result = dict(
        active_step_abs_contribution=distribution(np.abs(delta[sig != 0])),
        active_step_ratio=distribution(sample_ratios), sum_abs_ratio=float(np.abs(delta).sum() / np.abs(native).sum()),
        batch_sum_abs_ratio=distribution([r for r in batch_shares if r is not None]),
        larger_than_native_steps=int(np.count_nonzero(np.abs(delta) > np.abs(native))),
        total_reward_sign_changes=direction_changes, standardized_advantage_sign_changes=flips,
        standardized_advantage_change_rms=float(np.sqrt(np.mean(np.concatenate(changes) ** 2))),
        max_batch_standardized_advantage_change_rms=max(drifts),
        active_batches_standardized_advantage_change_rms=distribution([d for b, d in zip(batches, drifts) if np.any(b.sign)]),
        active_batches_r1_only_advantage_std=distribution([s for b, s in zip(batches, baseline_stds) if np.any(b.sign)]),
        undefined_active_ratios=int(np.count_nonzero((sig != 0) & (native == 0))))
    if mix is not None:
        assert result['larger_than_native_steps'] == direction_changes == 0
        if mix.beta == 0:
            assert result['standardized_advantage_change_rms'] == 0
    return result


def analyze(output):
    pins = {}
    current = current_training(pins)
    result = dict(mode='offline_fixed_trajectory_rescoring', new_policy_updates=0,
                  device_commands=0, reward_units='native reward per decision',
                  gamma_tick=GAMMA, gae_lambda_tick=LAMBDA, r1_weight=1, r2_scale_fraction=AMPLITUDE,
                  current_training=describe(current), protocols={})
    for protocol in PROTOCOLS:
        spec, batches = historical_training(protocol, pins)
        fit_ids = spec['training_seeds'][:4]
        check_ids = spec['training_seeds'][4:]
        fit = [b for b in batches if b.seed in fit_ids]
        check = [b for b in batches if b.seed in check_ids]
        assert len(fit) == len(check) == 16 and not set(fit_ids) & set(check_ids)
        ref = fit_reference(fit)
        old_ref = spec['reward_mix']['reference_per_tick']
        mixes = {'r1_only': (ref, 0), 'legacy_5pct_limit': (old_ref, .50)}
        mixes.update({key: (ref, weight) for key, weight in WEIGHTS.items()})
        params = {key: ReplayRewardMix(reference, AMPLITUDE, 1., beta, GAMMA)
                  for key, (reference, beta) in mixes.items()}
        params['raw_unit_diagnostic'] = None
        sections = {}
        for key, subset in [('fit', fit), ('check', check), ('all_descriptive', batches)]:
            sections[key] = dict(counts=describe(subset),
                                 rescoring={k: compare(subset, mix) for k, mix in params.items()})
        result['protocols'][protocol] = dict(
            study=spec['study'], context_sha256=spec['context_sha'], calibration_seeds=fit_ids,
            check_seeds=check_ids, calibration='median(abs(R1)/discounted_duration) on active nonzero-R1 fit steps',
            reference_per_tick=ref, fit_leave_one_seed_out={s: fit_reference([b for b in fit if b.seed != s]) for s in fit_ids},
            candidates={k: dict(reference_per_tick=r, beta=beta, effective_cap=beta * AMPLITUDE)
                        for k, (r, beta) in mixes.items()}, sections=sections)
    drink_ref = result['protocols']['low_water_drink']['reference_per_tick']
    result['current_training']['rescoring'] = {
        name: compare(current, ReplayRewardMix(drink_ref, AMPLITUDE, 1., beta, GAMMA))
        for name, beta in WEIGHTS.items()}
    for rel in ('tools/calibrate_r2_event_scale.py', 'standalone/mea_live_pair/replay_reward.py',
                'standalone/mea_live_pair/livepair/predictability.py', 'agent/rl_contract.py', 'agent/m8_train.py'):
        pins[rel] = sha(ROOT / rel)
    result['input_sha256'] = dict(sorted(pins.items()))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'scale-audit.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(target)
    print(f'{target}  {sha(target)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CURRENT / 'offline-r2-scale-exploration')
    analyze(parser.parse_args().output)
