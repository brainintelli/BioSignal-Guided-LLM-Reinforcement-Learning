"""Pure prompt/critic adapter for the new schema."""
from agent import m9_contract
from agent.m8s3_supply import readiness_features
from agent.mea_contract import observation_key
from .features import NAMES, validate_snapshot
from .protocol import ARMS

CONTRACT = 'mea.state2_r2.digital.v1'


def encode(obs, snapshot, arm, normalization_id):
    if arm not in ARMS:
        raise ValueError('Unregistered model arm')
    measured = validate_snapshot(snapshot, observation_key(obs), normalization_id)
    values = [0., 0., 0., 0.] if arm in ('A', 'B') else list(measured)
    row = m9_contract.encode(obs, 'reserves')
    text = ','.join(f'{name}={value:.3f}' for name, value in zip(NAMES, values))
    row['messages'][1]['content'] += '\nMEA标准化观测：' + text + '。历史记录，与本次游戏动作无因果对应。'
    row.update(contract=CONTRACT, wetware_snapshot=snapshot, wetware_features=values,
               critic_input=row['features'] + readiness_features(row) + values)
    return row
