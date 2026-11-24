"""Immutable budget and provenance rules for the open4 bounded comparison."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath as Path
import statistics

PROTOCOL='survival_v3_equal_update_diagnostic_v1'

def fixed_config(project, remote_root, baseline_projection_layout='grouped'):
    r=Path(remote_root)
    return dict(protocol=PROTOCOL,project=str(project),model=str(r/'models/Qwen3.5-0.8B'),
        godot=str(r/'godot/Godot_v4.7.2-stable_linux.x86_64'),
        initial_checkpoint=str(r/'m8b-runs/exploration_attempt1/checkpoint_0032'),
        parent_checkpoint=str(r/'m6-runs/small_v2_stable/checkpoint_0016'),
        kernel_overlay=str(r/'m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site'),
        profile='open4',world_version=6,environment_version='m8_survival_v2',
        contract='open_world_semantic_survival_v3',reset_critic=True,fresh_optimizers=True,
        fresh_rollouts=True,resume=None,arms=['baseline','optimized'],prefix_layout='system_only',
        baseline_projection_layout=baseline_projection_layout,baseline_target_layout='shared_batched_suffix',envs=4,rollout_steps=32,updates=3,epochs=2,minibatch=32,microbatch=4,
        critic_epochs=8,critic_minibatch=128,actor_lr=5e-7,critic_lr=3e-4,
        reference_kl=.005,clip=.15,target_kl=.03,entropy_coef=.002,
        actor_grad_clip=1.,critic_grad_clip=1.,gamma_tick=.9995,gae_lambda=.999,
        max_tokens=1536,gradient_checkpointing=False,temperature=1.,
        zero_update_logp_tolerance=1e-6,seeds=[883910,883911,883912,883913],rng_seed=20261111,
        wall_clock_limit_seconds=3600,formal_training_status='not_started',
        checkpoint_writes=0,release_authorized=False)

def validate_config(config):
    layout=config.get('baseline_projection_layout')
    if layout not in ('grouped','native'):raise ValueError('Unknown baseline projection layout')
    expected=fixed_config(config.get('project',''),str(Path(config.get('model','/missing')).parents[1]),layout)
    paths={'project','model','godot','initial_checkpoint','parent_checkpoint','kernel_overlay'}
    if set(config)!=set(expected):raise ValueError('Unregistered parameter set')
    for key,value in expected.items():
        if key in paths:
            if not isinstance(config[key],str) or not config[key]:raise ValueError('Invalid path: '+key)
        elif type(config[key]) is not type(value) or config[key]!=value:
            raise ValueError('Unregistered diagnostic value: '+key)

def physical_batches(inputs,max_tokens=1536,pad_id=0):
    if not inputs or any(not x or len(x)>max_tokens for x in inputs):
        raise ValueError('Empty or overflowing sequence; never truncate')
    for start in range(0,len(inputs),4):
        real=inputs[start:start+4];physical=real+[real[-1]]*(4-len(real))
        yield dict(real_rows=len(real),input_ids=[[pad_id]*(max_tokens-len(x))+list(x) for x in physical],
                   attention_mask=[[0]*(max_tokens-len(x))+[1]*len(x) for x in physical])

def registration_digest(reg):
    payload={k:v for k,v in reg.items() if k!='registration_sha256'}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def verify_registration_digest(reg):
    if reg.get('registration_sha256')!=registration_digest(reg):raise ValueError('Registration content hash differs')

def speed_gate(baseline,candidate):
    equal=(len(baseline)==len(candidate)==3 and all(
        a['decisions']==b['decisions']==128 and a['optimizer_steps']==b['optimizer_steps']==8
        and a['critic_optimizer_steps']==b['critic_optimizer_steps']==8 for a,b in zip(baseline,candidate)))
    valid=all(x['finite'] and not x['kl_early_stop'] and x['approx_kl_max']<=.03 and
        max(x['zero_update_logp_max'],x['pre_update_all_logp_max'],x['refresh_error'])<=1e-6 for x in baseline+candidate)
    a=[x['rollout_seconds']+x['update_seconds'] for x in baseline]
    b=[x['rollout_seconds']+x['update_seconds'] for x in candidate]
    if not a or not b or not all(math.isfinite(t) and t>0 for t in a+b):
        return dict(passed=False,equal_work=equal,numeric_checks=False)
    total=sum(a)/sum(b);median=statistics.median(a)/statistics.median(b)
    return dict(passed=equal and valid and total>2 and median>2,equal_work=equal,numeric_checks=valid,
                all_three_rounds_speedup=total,median_speedup=median,
                baseline_round_seconds=a,candidate_round_seconds=b)
