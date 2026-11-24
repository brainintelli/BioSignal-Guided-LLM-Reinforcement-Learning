"""Reproduce budget estimates from this run's downloaded GPU evidence."""
from datetime import datetime
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]))
from agent.m8s3_training import estimate_seconds,write

read=lambda name:json.loads((root/'evidence'/name).read_text(encoding='utf-8'))
metrics=[json.loads(x) for x in (root/'evidence/pilot_treatment_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
full=[x for x in metrics if not x['kl_early_stop'] and x['optimizer_steps']==32]
if not full:raise ValueError('No full PPO update sample; early stopping is not a throughput gain')
training_rate=sum(x['rollout_seconds']+x['update_seconds'] for x in full)/(2048*len(full))
evaluations=[read('pilot_treatment_'+name+'.json') for name in ('initial_evaluation','selection_0002')]
evaluation_rate=sum(x['seconds'] for x in evaluations)/sum(x['decisions'] for x in evaluations)
finished=read('pilot_treatment_finished.json');active=read('pilot_treatment_active_child.json')
if active['stage']!='replay' or finished['error'] is not None:raise ValueError('Require completed clean replay stage')
audit=read('pilot_treatment_independent_replay.json')
audit_seconds=(datetime.fromisoformat(finished['finished_utc'])-datetime.fromisoformat(active['launched_utc'])).total_seconds()
audit_rate=audit_seconds/audit['transitions']
historical=json.loads((root.parent/'rl_training_20261108/estimate.json').read_text(encoding='utf-8'))
legacy_projection=historical['legacy_observed']['evaluation_seconds']*25
scenarios=[]
for updates in (16,32,64,128,256):
    decisions=updates*2048
    worlds=32+(updates//8)*16+200
    evaluation_decisions=worlds*1800
    replay=(decisions+evaluation_decisions)*audit_rate
    value=estimate_seconds(decisions=decisions,training_seconds_per_decision=training_rate,
        evaluation_decisions=evaluation_decisions,evaluation_seconds_per_decision=evaluation_rate,
        fixed_seconds=replay+legacy_projection+(updates//4)*2+120)
    scenarios.append(dict(updates=updates,decisions=decisions,training_hours=decisions*training_rate/3600,
        evaluation_hours=evaluation_decisions*evaluation_rate/3600,audit_hours=replay/3600,
        legacy_hours=legacy_projection/3600,total_hours=value['central_seconds']/3600,
        planning_hours=value['planning_seconds']/3600))
result=dict(training_seconds_per_decision=training_rate,full_update_samples=len(full),
    evaluation_seconds_per_decision=evaluation_rate,audit_seconds_per_transition=audit_rate,
    observed_audit_seconds=audit_seconds,scenarios=scenarios,safety_factor=1.5,
    convergence_guaranteed=False,convergence_time=None,
    scope='Finite registered compute budget including final test if eligible; excludes human/HTTP/rendered-game acceptance',
    assumptions='Same node and physical B4 layout; full PPO updates only; evaluation assumes maximum1800 decisions per world; audit timing includes stage polling/loading; old-rule100-world cost reused from measured unchanged legacy path; changing action mix may change throughput')
write(root/'estimate.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
