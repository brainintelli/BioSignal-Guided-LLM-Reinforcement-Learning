"""Balanced-world identities; legacy cryptographic/runtime helpers are unchanged."""
from pathlib import Path
from agent.m8_identity import (BASE_WEIGHT_SHA, sha, config_sha, claim_ledger,
    read_ledger, append_ledger_event, runtime_identity, execution_identity)
from agent.m8b_registration import CORE_FILES


def source_identity(project):
    return {name: sha(Path(project) / name) for name in CORE_FILES}


def verify_source_identity(project, expected, core_files=CORE_FILES):
    if set(expected) != set(core_files):
        raise ValueError('Balanced frozen source set differs from exact core files')
    root = Path(project).resolve()
    for name in core_files:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected[name]:
            raise ValueError('Balanced frozen source changed: ' + name)
