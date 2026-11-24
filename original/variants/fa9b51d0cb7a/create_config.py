import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
c=json.loads((root/'configs/m8x_exploration_attempt1.json').read_text())
c.update(protocol='m8x_unpadded_kv_v1',rng_prefix='m8x_unpadded_kv_v1',
    initial_checkpoint='/home/researcher/meadowmind-gpu/m8b-runs/exploration_attempt1/checkpoint_0032',
    initialization='warm_start_weights_fresh_optimizer_and_rollouts',
    policy_layout='unpadded_full_verb_kv_merged_v1',
    target_template='explicit_empty_thinking_previous_assistant_v1',
    seed=20261107,train_seed_start=120000,
    validation_seeds=list(range(943000,943016)),test_seeds=list(range(953000,953100)),
    output='/home/researcher/meadowmind-gpu/m8x-kv-runs/kv_attempt1',
    formal_training_status='not_started',release_authorized=False,
    performance=dict(minimum_total_speedup=2.0,strictly_greater=True,
        primary_metric='median_full_rollout_plus_ppo_update_seconds_same_decision_budget',
        arms=['baseline','compact','optimized'],updates=3,envs=4,decisions_per_update=128,
        fused_rms_norm=True,natural_death_reset=True,reset_seed_stride=4,
        seeds=[883810,883811,883812,883813],rng_seed=20261109,
        preserve_epochs_minibatch_microbatch=True,checkpoint_writes=0,
        zero_update_logp_tolerance=1e-6,probability_states=64,
        fixed_workload_states=64,wall_clock_limit_seconds=1800))
(root/'configs/m8x_kv_attempt1.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
