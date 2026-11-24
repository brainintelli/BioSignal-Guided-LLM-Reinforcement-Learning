"""Exact M8 preregistration. No model sampling and no implicit source discovery."""
from datetime import datetime,timezone
import json
from pathlib import Path

PROTOCOL='m8_reliable_v1'
CONFIG_FILES=('configs/m8_protocol.json','configs/m8_reliable_attempt1.json','configs/m8_reliable_attempt2.json')
CORE_FILES=(
    'project.godot','scripts/world_state.gd','scripts/rl_environment_session.gd','godot_tests/rl_environment.gd',
    'agent/rl_contract.py','agent/rl_env.py','agent/m6_contract.py','agent/m6_policy.py','agent/m6_semantic_contract.py',
    'agent/m6_test.py','agent/m6_stochastic_eval.py','agent/m6_train.py',
    'agent/m8_contract.py','agent/m8_policy.py','agent/m8_policy_identity.py','agent/m8_train.py','agent/m8_eval.py',
    'agent/m8_identity.py','agent/m8_registration.py','tools/m8_metrics.py','tools/audit_m8_run.py','tools/audit_m8_stochastic_rng.py',
    'tools/compare_m8_survival.py','tools/audit_m6_run.py','tools/rl_audit_io.py','docs/M8_RL_DESIGN.md',*CONFIG_FILES)


def validate_config(project,config):
    attempt=config.get('attempt')
    if type(attempt) is not int or attempt not in (1,2):raise ValueError('Only two registered attempts exist')
    expected=json.loads((Path(project)/f'configs/m8_reliable_attempt{attempt}.json').read_text(encoding='utf-8'))
    if config!=expected:raise ValueError('Configuration differs from the frozen registered attempt')
    train=set(range(config['train_seed_start'],config['train_seed_start']+config['train_seed_count']))
    val=set(config['validation_seeds']);test=set(config['test_seeds'])
    if train&val or train&test or val&test:raise ValueError('Seed partitions overlap')


def create_registration(project,config,execution,archive):
    from agent.m8_identity import sha,source_identity
    from agent.m8_policy_identity import PARENT_ACTOR_SHA256
    validate_config(project,config)
    sources=source_identity(project)
    return dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),config=config,
                source_hashes=sources,source_sha256=sources,
                config_hashes={p:sha(Path(project)/p) for p in CONFIG_FILES},
                parent_actor_sha256=PARENT_ACTOR_SHA256,execution_identity=execution,
                archive_sha256=sha(archive),no_demonstrations=True,no_task_goals=True,no_policy_fallback=True,
                training_reset='natural death only',reference='independent frozen released parent',
                selection='all four validations: successes, fractional CVaR20, RMST, earliest; see frozen m8_metrics')


def verify_registration(project,registration):
    from agent.m8_identity import sha,verify_source_identity
    from agent.m8_policy_identity import PARENT_ACTOR_SHA256
    if registration.get('source_hashes')!=registration.get('source_sha256'):
        raise ValueError('Source aliases disagree')
    if registration.get('protocol')!=PROTOCOL or registration.get('parent_actor_sha256')!=PARENT_ACTOR_SHA256:
        raise ValueError('Protocol/parent registration identity differs')
    validate_config(project,registration['config'])
    verify_source_identity(project,registration['source_hashes'])
    expected={p:sha(Path(project)/p) for p in CONFIG_FILES}
    if registration.get('config_hashes')!=expected:raise ValueError('Registered configs changed')
