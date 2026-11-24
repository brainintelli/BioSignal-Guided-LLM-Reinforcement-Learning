"""New registration boundary for canonical state batch execution."""
from agent.m8s3_diag_contract import fixed_config as previous_config
from agent.m8s3_diag_contract import validate_config as previous_validate
from pathlib import PurePosixPath

PROTOCOL = 'survival_v3_canonical_batch_equal_update_v2'


def fixed_config(project, remote_root, baseline_projection_layout='native'):
    config = previous_config(project, remote_root, baseline_projection_layout)
    config.update(protocol=PROTOCOL, candidate_layout='canonical_b4_896_1280_1537_masked',
                  seeds=[884010, 884011, 884012, 884013], rng_seed=20261112)
    return config


def validate_config(config):
    expected = fixed_config(config.get('project', ''),
                            str(PurePosixPath(config.get('model', '/missing')).parents[1]),
                            config.get('baseline_projection_layout'))
    if set(config) != set(expected):
        raise ValueError('Unregistered parameter set')
    new_fields = ('protocol', 'candidate_layout', 'seeds', 'rng_seed')
    for key in new_fields:
        if type(config[key]) is not type(expected[key]) or config[key] != expected[key]:
            raise ValueError('Unregistered batch diagnostic value: ' + key)
    previous = previous_config(config['project'], str(PurePosixPath(config['model']).parents[1]),
                               config['baseline_projection_layout'])
    translated = {k: v for k, v in config.items() if k != 'candidate_layout'}
    for key in ('protocol', 'seeds', 'rng_seed'):
        translated[key] = previous[key]
    previous_validate(translated)
