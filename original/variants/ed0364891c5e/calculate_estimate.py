"""Reproduce compute-time scenarios from downloaded real experiment evidence."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.m8s3_training import estimate_seconds, write

root = Path(__file__).resolve().parent
evidence = root / 'evidence'
read = lambda name: json.loads((evidence / name).read_text(encoding='utf-8'))
metrics = [json.loads(line) for line in (evidence / 'pilot3_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
full = [r for r in metrics if not r['kl_early_stop'] and r['optimizer_steps'] == 32]
if not full:
    raise ValueError('No complete-update timing sample; do not substitute early-stopped work')
training_rate = sum(r['rollout_seconds'] + r['update_seconds'] for r in full) / (512 * len(full))
evaluations = [read('pilot3_' + name + '.json') for name in ('initial_evaluation', 'selection_0004', 'selection_0008')]
eval_rate = sum(r['seconds'] for r in evaluations) / sum(r['decisions'] for r in evaluations)
legacy = read('legacy_timing.json')
audit = read('supplemental_audit_timing.json')
audit_rate = audit['seconds'] / audit['transitions']
legacy_projection = legacy['evaluation_seconds'] * 25
scenarios = []
for updates in (64, 128, 256, 512, 1024):
    decisions = updates * 512
    # 16 initial candidate + 16 same-rule parent; every screen has 16 worlds;
    # final candidate/parent test each has 100 worlds. Assume every new-rule
    # episode takes the maximum 1800 decisions, so early deaths do not make the
    # planning budget look artificially cheap.
    worlds = 32 + (updates // 32) * 16 + 200
    evaluation_decisions = worlds * 1800
    audits = (decisions + evaluation_decisions) * audit_rate
    checkpoint_and_launch = (updates // 16) * 1.3 + 120
    value = estimate_seconds(decisions=decisions, training_seconds_per_decision=training_rate,
        evaluation_decisions=evaluation_decisions, evaluation_seconds_per_decision=eval_rate,
        fixed_seconds=legacy_projection + audits + checkpoint_and_launch)
    scenarios.append(dict(updates=updates, fresh_decisions=decisions,
        training_hours=decisions*training_rate/3600,
        new_rule_evaluation_hours=evaluation_decisions*eval_rate/3600,
        legacy_evaluation_hours=legacy_projection/3600, audit_hours=audits/3600,
        total_hours=value['central_seconds']/3600, planning_hours=value['planning_seconds']/3600))
write(root / 'estimate.json', dict(training_seconds_per_decision=training_rate,
    full_update_samples=len(full), total_pilot_updates=len(metrics), evaluation_seconds_per_decision=eval_rate,
    audit_seconds_per_transition=audit_rate, legacy_observed=legacy, scenarios=scenarios,
    safety_factor=1.5, maximum_training_seconds=259200, hard_job_seconds=345600,
    convergence_guaranteed=False,
    convergence_time='Not identifiable from four diagnostic worlds and 4096 new decisions; use independent screening/test gates',
    assumptions='Same node and canonical layout; complete PPO work; new-rule assessment uses maximum 1800 decisions/world; legacy scales four measured worlds; workload mix may change'))
print(json.dumps(scenarios, ensure_ascii=False, indent=2))
