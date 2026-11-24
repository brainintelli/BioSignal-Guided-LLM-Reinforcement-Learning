"""Recompute bounded remaining-time scenarios and validate this inspection."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from agent.m9_metrics import summarize, select_arm

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

snapshot_path = OUT/'evidence/snapshot.json'
digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
assert digest == 'c112fae9b2832ec8eb703a976ac57c02baf4d966c69ad4a21d524c74aecd9d2f'
s = read(snapshot_path)
all_metrics = [m for run in s['runs'] for m in run['metrics']]
assert len(all_metrics) == 224
assert sum(run['decisions'] for run in s['runs']) == 458752
rank_rows = []
for run in s['runs']:
    for evaluation in run['evaluations'].values():
        assert summarize(evaluation['report']['episodes']) == evaluation['report']['summary']
    if run['phase'] == 'explore':
        rank_rows.append(dict(arm=run['name'].rsplit('_r',1)[0],replicate=int(run['name'][-1]),
            summary=run['evaluations']['evaluation']['report']['summary']))
assert select_arm(rank_rows,list(s['registration_config']['arms'])) == s['ranking']['ranking']

run = next(r for r in s['runs'] if r['phase'] == 'confirm')
p = run['partial_evaluation']
assert summarize(p['episodes']) == p['completed_subset_summary']
seeds = {e['seed'] for e in p['episodes']}
paired_initial = [e for e in run['evaluations']['initial']['report']['episodes'] if e['seed'] in seeds]
assert len(paired_initial) == len(p['episodes']) == 13
assert p['max_possible_30day_successes'] == 22
assert not s['finished'] and not s['final_test_files'] and not s['selection']

budget = read(ROOT/'output/m9_research_20261111/evidence/experiment_summary.json')['budget']
audit = budget['audit_seconds_per_transition']
freeze = [r for r in s['runs'] if r['name'].startswith('infinite_freeze_')]
freeze_train = [statistics.mean(m['rollout_seconds']+m['update_seconds'] for m in r['metrics']) for r in freeze]
freeze_eval = [e['report']['seconds']/e['report']['decisions'] for r in freeze for e in r['evaluations'].values()]
combined_eval = [e['report']['seconds']/e['report']['decisions'] for r in s['runs'] if r['name'].startswith('combined_') for e in r['evaluations'].values()]
partial_seconds = (datetime.fromisoformat(p['captured_utc'])-datetime.fromisoformat(p['evaluation_started_utc'])).total_seconds()
combined_eval.append(partial_seconds/p['complete_records'])
remaining_ticks = (24-p['started_worlds'])*5400 + sum(5400-v['tick'] for v in p['active'].values())
combined_audit_transitions = run['decisions'] + run['evaluations']['initial']['report']['decisions'] + p['complete_records'] + remaining_ticks
freeze_audit_transitions = 64*2048 + 48*5400
confirmation_audit_seconds = combined_audit_transitions*audit['combined'] + freeze_audit_transitions*audit['infinite_freeze']
now = datetime.fromisoformat(s['checked_utc'])
zone = timezone(timedelta(hours=8))
scenarios = []
for label, reducer in [('observed_fast',min),('observed_mean',statistics.mean),('observed_slow',max)]:
    training = 64*reducer(freeze_train)
    confirmation_eval = remaining_ticks*reducer(combined_eval) + 48*5400*reducer(freeze_eval)
    final_eval = 200*5400*reducer(freeze_eval)
    final_audit = 200*5400*audit['infinite_freeze']
    stop_seconds = training+confirmation_eval+confirmation_audit_seconds+3600
    test_seconds = stop_seconds+final_eval+final_audit
    scenarios.append(dict(speed=label,training_hours=training/3600,confirmation_eval_hours=confirmation_eval/3600,
        confirmation_audit_hours=confirmation_audit_seconds/3600,final_eval_hours=final_eval/3600,
        final_audit_hours=final_audit/3600,fixed_overhead_hours=1,
        if_confirmation_fails_hours=stop_seconds/3600,if_freeze_unlocks_test_hours=test_seconds/3600,
        fail_finish_cst=(now+timedelta(seconds=stop_seconds)).astimezone(zone).isoformat(),
        test_finish_cst=(now+timedelta(seconds=test_seconds)).astimezone(zone).isoformat()))
est = dict(checked_utc=s['checked_utc'],remaining_optimization_updates=64,
    remaining_combined_evaluation_tick_upper_bound=remaining_ticks,
    freeze_training_seconds_per_update=freeze_train,freeze_evaluation_seconds_per_decision=freeze_eval,
    combined_evaluation_seconds_per_decision=combined_eval,
    audit_seconds_per_transition={a:audit[a] for a in ['combined','infinite_freeze']},
    confirmation_audit_transition_upper_bounds=dict(combined=combined_audit_transitions,infinite_freeze=freeze_audit_transitions),
    scenarios=scenarios,
    hard_deadline_cst=(now+timedelta(seconds=s['hard_wall_remaining_seconds'])).astimezone(zone).isoformat(),
    hard_remaining_hours=s['hard_wall_remaining_seconds']/3600,
    assumptions=[
        'Combined already has two failures; cannot qualify for >=23/24, so only infinite_freeze can unlock final testing.',
        'Upper-bound remaining evaluation decisions by remaining ticks; all remaining episodes reach 5400.',
        'Freeze rates use two completed exploration trainings and their four completed evaluations.',
        'Combined evaluation rates use completed evaluations plus current elapsed time per captured trace row.',
        'Audits extrapolate original full-batch calibration rates adjusted by arm smoke ratios; not measured current full confirmation audits.',
        'Includes both 64-update confirmation audits, reload/setup overhead of one hour, and conditional final candidate100+parent100.',
        'Ranges are throughput scenarios, not confidence intervals or convergence promises; online acceptance excluded.'
    ])
(OUT/'evidence/remaining_time_estimate.json').write_text(json.dumps(est,ensure_ascii=False,indent=2),encoding='utf-8')
verification = dict(snapshot_sha256=digest,complete_evaluation_reports=sum(len(r['evaluations']) for r in s['runs']),
    all_summaries_recomputed=True,ranking_recomputed=True,partial_endpoints=13,
    paired_initial_summary=summarize(paired_initial),partial_after_summary=p['completed_subset_summary'],
    total_updates=224,total_decisions=458752,total_ticks=sum(r['ticks'] for r in s['runs']),
    max_pre_update_error=max(m['pre_update_all_logp_error'] for m in all_metrics),
    kl_early_stops=sum(m['kl_early_stop'] for m in all_metrics),
    final_test_available=False,source_files_verified=s['source_files_verified'],
    input_files_rehashed_this_check=False,independent_reload_pending=run['checkpoints'][0]['reload'] is None,
    full_native_audit_rerun_this_check=False)
(OUT/'evidence/verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(verification=verification,estimate=est),ensure_ascii=True,indent=2))
