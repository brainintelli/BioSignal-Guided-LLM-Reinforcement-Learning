"""Opt-in pure observation adapter; existing M9 policy/trainer is unchanged."""
from agent import m9_contract
from agent.m8s3_supply import readiness_features
from agent.mea_protocol import canonical, digest
from agent.mea_snapshot import FEATURE_NAMES, WetwareSnapshot
from agent.mea_replay import RecordingSnapshot, RECORDING_CONTRACT

VERSION = 'mea_rl_p0_v1'


def observation_key(obs, state_variant='reserves'):
    # Hash the positive factual allowlist, never raw observation extras/rewards.
    return digest(m9_contract.encode(obs, state_variant))


def encode(obs, snapshot, state_variant='reserves'):
    if not isinstance(snapshot, (WetwareSnapshot, RecordingSnapshot)):
        raise ValueError('Validated immutable snapshot required')
    row = m9_contract.encode(obs, state_variant)
    if digest(row) != snapshot.context_id:
        raise ValueError('Snapshot belongs to a different observation/representation')
    row['messages'][1]['content'] += '\n湿件观测：' + canonical(dict(zip(FEATURE_NAMES, snapshot.feature_values)))
    if isinstance(snapshot, RecordingSnapshot):
        row['messages'][1]['content'] += '；离线无刺激记录，与本次游戏动作无因果对应。'
    row.update(contract=VERSION, base_contract=m9_contract.VERSION, wetware_snapshot=snapshot.to_dict(),
               wetware_features=list(snapshot.normalized_values),
               critic_input=row['features'] + readiness_features(row) + list(snapshot.normalized_values))
    if isinstance(snapshot, RecordingSnapshot): row['contract'] = RECORDING_CONTRACT
    return row
