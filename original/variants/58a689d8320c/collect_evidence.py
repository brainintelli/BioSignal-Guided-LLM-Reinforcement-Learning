"""Read-only campaign summary and an explicit compute-budget projection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import subprocess

BASE = Path('/home/researcher/meadowmind-gpu')
OUT = BASE / 'm9-preflight-v2'

def read(path):
    return json.loads(Path(path).read_text())

def stage_seconds(job, name):
    stages = read(job / 'stages.json')
    index = next(i for i, s in enumerate(stages) if s['stage'] == name)
    start = read(job / (name + '.json'))['started_utc']
    end = (read(job / (stages[index + 1]['stage'] + '.json'))['started_utc']
           if index + 1 < len(stages) else read(job / 'finished.json')['finished_utc'])
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()

reg = read(BASE / 'm9-runs/research_attempt2_registration/registration.json')
formal_reg = read(BASE / 'm9-runs/research_attempt3_registration/registration.json')
assert read(OUT / 'v3_equivalence.json')['training_and_evaluation_code_identical']
root = Path(reg['config']['output'])
smoke_job = BASE / 'm9-training-jobs/research2_smoke'
calibration_job = BASE / 'm9-training-jobs/research2_calibration'
smoke_finish = read(smoke_job / 'finished.json')
calibration_finish = read(calibration_job / 'finished.json')
assert smoke_finish['status'] == 'smoke_completed'
assert calibration_finish['status'] == 'calibration_completed'
rows = []
for arm in reg['config']['arms']:
    p = root / 'smoke' / (arm + '_r0')
    metric = read(p / 'metrics.jsonl')
    initial, final = read(p / 'initial.json'), read(p / 'evaluation.json')
    audit, reload = read(p / 'audit.json'), read(p / 'checkpoint_0001/independent_reload.json')
    complete = read(p / 'training_complete.json')
    assert audit['passed'] and reload['passed'] and complete['initial_actor'] != complete['final_actor']
    rows.append(dict(arm=arm,metric=metric,audit=audit,reload=reload,
        evaluation_seconds=initial['seconds']+final['seconds'],
        evaluation_decisions=initial['decisions']+final['decisions'],
        audit_seconds=stage_seconds(smoke_job,'smoke_'+arm+'_audit')))
cal = root / 'calibration/control_r0'
metrics = [json.loads(line) for line in (cal / 'metrics.jsonl').read_text().splitlines()]
assert len(metrics) == 2 and all(m['optimizer_steps'] == 32 and not m['kl_early_stop'] for m in metrics)
cal_audit = read(cal / 'audit.json')
cal_reload = read(cal / 'checkpoint_0002/independent_reload.json')
control = next(r for r in rows if r['arm']=='control')
base_rollout = statistics.mean(m['rollout_seconds'] for m in metrics)
base_update = statistics.mean(m['update_seconds'] for m in metrics)
train_seconds = {r['arm']:base_rollout*r['metric']['rollout_seconds']/control['metric']['rollout_seconds']
    +base_update*r['metric']['update_seconds']/control['metric']['update_seconds'] for r in rows}
eval_rates = {r['arm']:r['evaluation_seconds']/r['evaluation_decisions'] for r in rows}
audit_base = stage_seconds(calibration_job,'calibration_control_audit')/cal_audit['transitions']
audit_rates = {r['arm']:audit_base*(r['audit_seconds']/r['audit']['transitions'])
    /(control['audit_seconds']/control['audit']['transitions']) for r in rows}
scenarios = []
arms = list(train_seconds)
for i,a in enumerate(arms):
    for b in arms[i+1:]:
        train = 16*sum(train_seconds.values())+64*(train_seconds[a]+train_seconds[b])
        evaluation = 32*5400*sum(eval_rates.values())+48*5400*(eval_rates[a]+eval_rates[b])+200*5400*max(eval_rates[a],eval_rates[b])
        audit = (64*2048+48*5400)*(audit_rates[a]+audit_rates[b])+200*5400*max(audit_rates[a],audit_rates[b])
        scenarios.append(dict(confirmation_arms=[a,b],training_hours=train/3600,
            evaluation_hours=evaluation/3600,audit_hours=audit/3600,
            total_hours=(train+evaluation+audit)/3600+1))
budget = dict(assumptions='All evaluation worlds reach5400; at most one decision per tick; candidate/parent final200 included; one hour fixed overhead; short-smoke arm ratios applied to full-batch calibration. No convergence estimate.',
    formal_updates=288,formal_decisions=589824,max_evaluation_worlds=616,
    min_estimated_hours=min(s['total_hours'] for s in scenarios),
    max_estimated_hours=max(s['total_hours'] for s in scenarios),
    planning_allowance=1.25,hard_wall_hours=formal_reg['config']['hard_wall_seconds']/3600,
    scenarios=scenarios,training_seconds_per_update=train_seconds,evaluation_seconds_per_decision=eval_rates,
    audit_seconds_per_transition=audit_rates)
gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
result = dict(created_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=reg['registration_sha256'],
    formal_registration_sha256=formal_reg['registration_sha256'],
    smoke=rows,calibration=dict(metrics=metrics,audit=cal_audit,reload=cal_reload),
    smoke_finished=smoke_finish,calibration_finished=calibration_finish,gpu_compute_processes=gpu,budget=budget)
(OUT/'experiment_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(dict(smoke_arms=len(rows),smoke_transitions=sum(r['audit']['transitions'] for r in rows),
    calibration_transitions=cal_audit['transitions'],base_rollout_seconds=base_rollout,base_update_seconds=base_update,
    estimate_hours=[budget['min_estimated_hours'],budget['max_estimated_hours']],gpu_compute_processes=gpu),ensure_ascii=False))
