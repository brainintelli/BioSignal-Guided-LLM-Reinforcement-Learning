"""Independent checks over the frozen trial's downloaded evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics

folder=Path(__file__).resolve().parent
result=json.loads((folder/'comparison.json').read_text())
registration=json.loads((folder/'registration.json').read_text())
assert result['manifest']['registration_sha256']==registration['registration_sha256']
assert result['formal_training_started'] is False and result['checkpoints_saved']==0
totals={};medians={};sampling={};updating={};memory={};lengths={}
for mode in ('baseline','compact','optimized'):
    arm=result['stability'][mode]
    assert arm['actor_changed'] and arm['reference_unchanged']
    rows=arm['updates']
    assert len(rows)==3
    for row in rows:
        assert row['decisions']==128 and row['optimizer_steps']==8
        assert row['finite'] and not row['kl_early_stop']
        assert row['zero_update_logp_max']<=1e-6
        assert row['weight_cache_refresh_max']<=1e-6 and row['kv_scope_released']
        assert all(math.isfinite(v) for v in row.values() if type(v) in (float,int))
    sampling[mode]=sum(r['rollout_seconds'] for r in rows)
    updating[mode]=sum(r['update_seconds'] for r in rows)
    totals[mode]=sampling[mode]+updating[mode]
    medians[mode]=statistics.median(r['rollout_seconds']+r['update_seconds'] for r in rows)
    memory[mode]=max(r['peak_allocated_bytes'] for r in rows)/2**30
    traces=[json.loads(x) for x in (folder/(mode+'_transitions.jsonl')).read_text(encoding='utf-8').splitlines()]
    assert len(traces)==384
    assert len({(r['update'],r['step'],r['env']) for r in traces})==384
    assert all(r['source']=='model' and not r['truncated'] for r in traces)
    verb=[len(r['executed_verb_token_ids']) for r in traces]
    target=[len(r['executed_target_token_ids']) for r in traces if r['executed_target_token_ids'] is not None]
    lengths[mode]=dict(verb_min=min(verb),verb_mean=statistics.mean(verb),verb_max=max(verb),
                      target_min=min(target),target_mean=statistics.mean(target),target_max=max(target),
                      input_cap_hits=sum(n==1536 for n in verb+target))
    assert max(verb+target)<=1536
    p=result['performance'][mode]
    assert p['zero_update_reordered_max']<=1e-6 and p['zero_update_grad_mode_max']<=1e-6
total_speedup=totals['baseline']/totals['optimized']
median_speedup=medians['baseline']/medians['optimized']
assert abs(total_speedup-result['speed_gate']['all_three_rounds_speedup'])<1e-12
assert abs(median_speedup-result['speed_gate']['median_speedup'])<1e-12
assert (total_speedup>2 and median_speedup>2)==result['speed_gate']['passed']
audit=dict(evidence_consistent=True,passed=result['passed'],registration_sha256=registration['registration_sha256'],
           decisions_per_arm=384,actor_steps_per_arm=24,totals=totals,medians=medians,
           sampling=sampling,updating=updating,peak_GiB=memory,lengths=lengths,
           total_speedup=total_speedup,median_speedup=median_speedup,
           speedup_vs_compact=totals['compact']/totals['optimized'],
           formal_training_started=False,checkpoints_saved=0)
(folder/'independent_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
