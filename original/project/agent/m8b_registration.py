"""A separate preregistration for the user-requested open2 balance amendment."""
from datetime import datetime, timezone
import json
from pathlib import Path

from agent.m8_registration import CORE_FILES as LEGACY_CORE_FILES

PROTOCOL = 'm8_balanced_reliable_v1'
CONFIG_FILES = ('configs/m8b_protocol.json', 'configs/m8b_balanced_attempt1.json',
                'configs/m8_balance_v1.json')
CORE_FILES = (*LEGACY_CORE_FILES,
    'scripts/balanced_world.gd', 'scripts/m8b_environment_session.gd',
    'godot_tests/m8b_environment.gd', 'agent/m8b_env.py',
    'agent/m8b_contract.py', 'agent/m8b_policy.py', 'agent/m8b_policy_identity.py',
    'agent/m8b_train.py', 'agent/m8b_eval.py', 'agent/m8b_identity.py',
    'agent/m8b_registration.py', 'tools/m8b_metrics.py',
    'tools/audit_m8b_run.py', 'tools/audit_m8b_stochastic_rng.py',
    'tools/compare_m8b_survival.py', 'docs/M8B_PROTOCOL.md', *CONFIG_FILES)


def validate_config(project, config):
    if type(config.get('attempt')) is not int or config['attempt'] != 1:
        raise ValueError('Only one separately registered balanced attempt exists')
    expected = json.loads((Path(project) / CONFIG_FILES[1]).read_text(encoding='utf-8'))
    if config != expected:
        raise ValueError('Configuration differs from the registered balanced attempt')
    fixed = dict(profile='open2', world_version=4, environment_version='m8_balanced_v1',
        contract='open_world_balanced_v1', mode='train', resume=None,
        updates=24, envs=4, rollout_steps=512, horizon=1800, training_horizon=None,
        temperature=1, eval_batch_size=4, max_tokens=1536, microbatch=4,
        gamma_tick=.9995, gae_lambda=.999, validation_every=8,
        replicates=[0], test_replicates=[0], bootstrap_samples=10000,
        bootstrap_seed=912800, rng_prefix=PROTOCOL)
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError('Frozen balanced-world or training invariants changed')
    for key, count in (('validation_seeds', 16), ('test_seeds', 100)):
        seeds = config[key]
        if (len(seeds) != count or len(set(seeds)) != count
                or any(type(seed) is not int for seed in seeds)):
            raise ValueError('Invalid independent world matrix: ' + key)
    if type(config['train_seed_start']) is not int or type(config['train_seed_count']) is not int or config['train_seed_count'] != 2048:
        raise ValueError('Invalid training seed partition')
    train = set(range(config['train_seed_start'], config['train_seed_start'] + config['train_seed_count']))
    val, test = set(config['validation_seeds']), set(config['test_seeds'])
    if train & val or train & test or val & test:
        raise ValueError('Seed partitions overlap')


def create_registration(project, config, execution, archive):
    from agent.m8b_identity import sha, source_identity
    from agent.m8_policy_identity import PARENT_ACTOR_SHA256
    validate_config(project, config)
    sources = source_identity(project)
    return dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
        config=config, source_hashes=sources, source_sha256=sources,
        config_hashes={p: sha(Path(project) / p) for p in CONFIG_FILES},
        balance_sha256=sha(Path(project) / CONFIG_FILES[2]),
        parent_actor_sha256=PARENT_ACTOR_SHA256, execution_identity=execution,
        archive_sha256=sha(archive), no_demonstrations=True, no_task_goals=True,
        no_policy_fallback=True, training_reset='natural death only',
        reference='independent frozen released M6 parent under the same new input',
        selection='three completed checkpoints: successes, fractional CVaR20, RMST, earliest',
        old_world_results_are_new_world_results=False)


def verify_registration(project, registration):
    from agent.m8b_identity import sha, verify_source_identity
    from agent.m8_policy_identity import PARENT_ACTOR_SHA256
    if registration.get('source_hashes') != registration.get('source_sha256'):
        raise ValueError('Source aliases disagree')
    if registration.get('protocol') != PROTOCOL or registration.get('parent_actor_sha256') != PARENT_ACTOR_SHA256:
        raise ValueError('Balanced protocol or parent identity differs')
    validate_config(project, registration['config'])
    verify_source_identity(project, registration['source_hashes'])
    expected = {p: sha(Path(project) / p) for p in CONFIG_FILES}
    if registration.get('config_hashes') != expected or registration.get('balance_sha256') != expected[CONFIG_FILES[2]]:
        raise ValueError('Balanced rules or registered configs changed')
    for key in ('no_demonstrations', 'no_task_goals', 'no_policy_fallback'):
        if registration.get(key) is not True:
            raise ValueError('Reward-only registration changed: ' + key)
