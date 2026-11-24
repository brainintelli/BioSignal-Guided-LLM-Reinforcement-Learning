"""Frozen source/runtime bindings for the independent exploration trial."""
from pathlib import Path
from agent.m8b_registration import CORE_FILES as PRESERVED_CORE_FILES
from agent.m8_identity import (BASE_WEIGHT_SHA, sha, config_sha, claim_ledger,
    read_ledger, append_ledger_event, execution_identity as legacy_execution_identity,
    runtime_identity as legacy_runtime_identity)
from agent.m8x_runtime import kernel_identity

CORE_FILES = (*PRESERVED_CORE_FILES,
    'configs/m8_exploration_v1.json', 'scripts/exploration_world.gd',
    'scripts/m8x_environment_session.gd', 'godot_tests/m8x_environment.gd',
    'agent/m8b_water_potential.py', 'agent/m8x_contract.py', 'agent/m8x_env.py',
    'agent/m8x_policy.py', 'agent/m8x_policy_identity.py', 'agent/m8x_runtime.py',
    'agent/m8x_train.py', 'agent/m8x_eval.py', 'agent/m8x_identity.py',
    'tools/audit_m8x_step.py', 'agent/m8x_registration.py',
    'configs/m8x_protocol.json', 'configs/m8x_exploration_attempt1.json', 'configs/m8x_runtime.json',
    'tools/m8x_metrics.py', 'tools/audit_m8x_run.py', 'tools/audit_m8x_stochastic_rng.py',
    'tools/compare_m8x_survival.py', 'tools/m8x_remote_job.py', 'docs/M8X_PROTOCOL.md')


def source_identity(project):
    return {name: sha(Path(project) / name) for name in CORE_FILES}


def verify_source_identity(project, expected):
    if set(expected) != set(CORE_FILES):
        raise ValueError('Exploration source inventory differs')
    root=Path(project).resolve()
    for name in CORE_FILES:
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=expected[name]:
            raise ValueError('Exploration source changed: '+name)


def runtime_identity(torch):
    return {**legacy_runtime_identity(torch), 'kernel_identity': kernel_identity()}


def execution_identity(config, torch):
    result=legacy_execution_identity(config, torch)
    result['runtime']['kernel_identity']=kernel_identity()
    return result
