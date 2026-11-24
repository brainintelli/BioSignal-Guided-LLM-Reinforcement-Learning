"""Separate open3 preregistration; construction requires a complete source set."""
from datetime import datetime, timezone
import json
from pathlib import Path
from agent.m8x_identity import CORE_FILES, source_identity, verify_source_identity, sha
from agent.m8_policy_identity import PARENT_ACTOR_SHA256

PROTOCOL='m8_exploration_reliable_v1'
CONFIG_FILES=('configs/m8x_protocol.json','configs/m8x_exploration_attempt1.json',
              'configs/m8_exploration_v1.json','configs/m8x_runtime.json')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def validate_config(project, config):
    expected=json.loads((Path(project)/CONFIG_FILES[1]).read_text(encoding='utf-8'))
    if canonical(config)!=canonical(expected):
        raise ValueError('Configuration differs from the exploration attempt')
    fixed=dict(protocol=PROTOCOL,rng_prefix=PROTOCOL,profile='open3',world_version=5,
        environment_version='m8_exploration_v1',contract='open_world_exploration_v1',
        mode='train',resume=None,attempt=1,updates=48,envs=4,rollout_steps=512,
        horizon=1800,training_horizon=None,validation_every=16,temperature=1,
        eval_batch_size=4,max_tokens=1536,microbatch=4,gamma_tick=.9995,gae_lambda=.999,
        actor_lr=1e-5,zero_update_logp_tolerance=1e-6,entropy_schedule_decisions=98304,
        train_seed_start=100000,train_seed_count=4096,
        validation_seeds=list(range(941000,941016)),test_seeds=list(range(951000,951100)),
        replicates=[0],test_replicates=[0],bootstrap_seed=912800,bootstrap_samples=10000,
        kernel_wheel_sha256='6a115fad2d4f096260b79f15170dd7bf6be651e711616939f332f64a324cc41d')
    if any(canonical(config.get(key))!=canonical(value) for key,value in fixed.items()):
        raise ValueError('Exploration budget, world, sampling or runtime invariant changed')


def verify_kernel_runtime(project, runtime):
    expected=json.loads((Path(project)/CONFIG_FILES[3]).read_text(encoding='utf-8'))
    if canonical(runtime.get('kernel_identity'))!=canonical(expected):
        raise ValueError('Actual GPU implementation differs from the tested runtime')


def create_registration(project, config, execution, archive):
    validate_config(project,config)
    verify_kernel_runtime(project,execution['runtime'])
    sources=source_identity(project)
    return dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),
        config=config,source_hashes=sources,source_sha256=sources,
        config_hashes={name:sha(Path(project)/name) for name in CONFIG_FILES},
        balance_sha256=sha(Path(project)/CONFIG_FILES[2]),parent_actor_sha256=PARENT_ACTOR_SHA256,
        execution_identity=execution,archive_sha256=sha(archive),no_demonstrations=True,
        no_task_goals=True,no_policy_fallback=True,fresh_environment_decisions=98304,
        replay_sample_uses=0,training_reset='natural death only',
        reference='independent frozen released M6 parent under the same new input',
        selection='three completed checkpoints: successes, fractional CVaR20, RMST, earliest',
        old_world_results_are_new_world_results=False)


def verify_registration(project, registration):
    if (registration.get('protocol')!=PROTOCOL
            or registration.get('parent_actor_sha256')!=PARENT_ACTOR_SHA256
            or registration.get('source_hashes')!=registration.get('source_sha256')):
        raise ValueError('Exploration registration identity differs')
    validate_config(project,registration['config'])
    verify_kernel_runtime(project,registration['execution_identity']['runtime'])
    verify_source_identity(project,registration['source_hashes'])
    expected={name:sha(Path(project)/name) for name in CONFIG_FILES}
    if (registration.get('config_hashes')!=expected
            or registration.get('balance_sha256')!=expected[CONFIG_FILES[2]]
            or type(registration.get('fresh_environment_decisions')) is not int
            or registration['fresh_environment_decisions']!=98304
            or type(registration.get('replay_sample_uses')) is not int
            or registration['replay_sample_uses']!=0
            or registration.get('training_reset')!='natural death only'):
        raise ValueError('Exploration rules, exposure or runtime binding differs')
    for flag in ('no_demonstrations','no_task_goals','no_policy_fallback'):
        if registration.get(flag) is not True:raise ValueError('Training provenance changed: '+flag)
