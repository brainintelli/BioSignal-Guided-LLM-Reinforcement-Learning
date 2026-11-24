"""Independent row-level arithmetic checks; no analysis implementation imports."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'mea_22554_baseline_20261116'
OUT=ROOT/'stress_v1'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
analysis=json.loads((PREVIOUS/'analysis_routed_v1/analysis.json').read_text())
report=json.loads((OUT/'analysis.json').read_text())
identity=json.loads((OUT/'identity.json').read_text())
for key,path in [('script',ROOT/'stress_candidates.py'),('protocol',ROOT/'analysis_protocol.md'),('analysis',OUT/'analysis.json'),('trials',OUT/'all_trials.csv')]:
    assert sha(path)==identity[key+'_sha256']
assert sha(PREVIOUS/'analysis_routed_v1/candidate_profile.json')==report['profile_sha256']
p=analysis['candidates']['profile']
assert not p['r2_training_enabled'] and not report['r2_training_enabled']
assert report['alpha']==1 and not report['source_full_structural_passed']
with (OUT/'all_trials.csv').open(newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==7028
keys=[(r['window_index'],r['condition'],r['repetition']) for r in rows]
assert len(set(keys))==len(rows)
groups=defaultdict(list)
past=[]
for w in analysis['windows']:
    x=[math.log1p(w['rate_hz_per_channel']),w['active_channel_fraction'],w['coincident_event_fraction']]
    past.append([(v-c)/s for v,c,s in zip(x,p['center'],p['scale'])])
max_error=0.0
def close(a,b):
    global max_error
    error=abs(a-b);max_error=max(max_error,error)
    assert error<1e-12,(a,b)
for r in rows:
    i=int(r['window_index']);cond=r['condition'];rep=int(r['repetition'])
    assert 31<=i<59 and int(r['history_last_window'])==i-1
    assert rep in range(1 if cond=='observed' else 50)
    assert r['source']==('observed' if cond=='observed' else 'synthetic_diagnostic')
    w=analysis['windows'][i];n=round(w['rate_hz_per_channel']*4250)
    expected_count=n
    if cond.startswith('inject'):
        expected_count+=max(1,round(n*int(cond.split('_')[1])/100))
    elif cond=='drop_20_percent': expected_count-=round(.2*n)
    assert int(r['event_count'])==expected_count
    close(float(r['rate_hz_per_channel']),expected_count/4250)
    if cond in ('observed','channel_circular_shift'):
        close(float(r['active_channel_fraction']),w['active_channel_fraction'])
    if cond=='observed': close(float(r['cooccurrence']),w['coincident_event_fraction'])
    x=[math.log1p(float(r['rate_hz_per_channel'])),float(r['active_channel_fraction']),float(r['cooccurrence'])]
    z=[(v-c)/s for v,c,s in zip(x,p['center'],p['scale'])]
    baseline=math.exp(-sum(v*v for v in z)/6)
    distance=min(math.sqrt(sum((a-b)**2 for a,b in zip(z,h))/3) for h in past[:i])
    novelty=min(distance/p['novelty_distance_p95'],1.0)
    close(float(r['baseline_proximity']),baseline)
    close(float(r['novelty_raw_distance']),distance)
    close(float(r['novelty']),novelty)
    original=analysis['candidates']['scores'][i-31]
    close(float(r['baseline_delta']),baseline-original['baseline_proximity'])
    close(float(r['novelty_delta']),novelty-original['novelty'])
    groups[cond].append(r)
def quantile(values,q):
    v=sorted(values);position=(len(v)-1)*q;lo=math.floor(position);hi=math.ceil(position)
    return v[lo]+(v[hi]-v[lo])*(position-lo)
for cond,rr in groups.items():
    counts=Counter(r['window_index'] for r in rr)
    assert set(counts)==set(map(str,range(31,59)))
    assert set(counts.values())=={1 if cond=='observed' else 50}
    g=report['conditions'][cond]
    for key in ('baseline_proximity','novelty','baseline_delta','novelty_delta','cooccurrence'):
        values=[float(r[key]) for r in rr]
        for metric,value in [('mean',statistics.mean(values)),('median',statistics.median(values)),('p05',quantile(values,.05)),('p95',quantile(values,.95))]:
            close(g[key][metric],value)
    close(g['novelty_clipped_fraction'],sum(float(r['novelty'])==1 for r in rr)/len(rr))
for name,lo,hi in [('validation_first_half',31,45),('validation_second_half',45,59)]:
    for key in ('rate_hz_per_channel','active_channel_fraction','coincident_event_fraction'):
        values=[w[key] for w in analysis['windows'][lo:hi]]
        for metric,value in [('mean',statistics.mean(values)),('median',statistics.median(values)),('p05',quantile(values,.05)),('p95',quantile(values,.95))]:
            close(report['temporal'][name][key][metric],value)

conditions=list(groups)
labels=['Observed','Channel-wise time shifts','Add 1% random events','Add 5% random events','Add 20% random events','Drop 20% events']
fig,axes=plt.subplots(1,2,figsize=(12.5,5.6),sharey=True)
for ax,metric,title in zip(axes,['baseline_proximity','novelty'],['Baseline proximity (not health)','Natural-fluctuation novelty']):
    for y,cond in enumerate(conditions):
        rr=groups[cond]
        means=[statistics.mean(float(r[metric]) for r in rr if int(r['window_index'])==i) for i in range(31,59)]
        ax.barh(y,statistics.mean(means),height=.5,color='#abc7df' if y==0 else '#ead2ba',alpha=.7)
        ax.scatter(means,[y+(i-13.5)*.012 for i in range(28)],s=16,color='#215d88' if y==0 else '#9b5620',alpha=.65,zorder=3)
        ax.text(1.025,y,f'{statistics.mean(means):.3f}',va='center',fontsize=10)
    ax.set_title(title);ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.grid(axis='x',alpha=.2)
    ax.set_yticks(range(len(labels)),labels);ax.set_xlabel('Frozen-profile score [0, 1]')
axes[0].invert_yaxis()
fig.suptitle('Chip 22554: synthetic input sensitivity',fontsize=16)
fig.text(.01,.025,'28 validation windows from one 5-minute recording; synthetic points average 50 trials/window.\nOriginal full-file QC failed; mapped subset only. Frozen history; R2 disabled; no task-benefit inference.',fontsize=10)
fig.tight_layout(rect=(0,.10,.96,.94))
for suffix in ['png','svg']: fig.savefig(OUT/f'candidate_stress.{suffix}',dpi=170)
result=dict(verified_rows=len(rows),max_arithmetic_error=max_error,all_checks_passed=True,
    profile_unchanged=True,verification_scope='independent score/count/aggregate recomputation from retained features; not an independent regeneration of synthetic event arrays',
    verifier_sha256=sha(Path(__file__)),analysis_sha256=sha(OUT/'analysis.json'),trials_sha256=sha(OUT/'all_trials.csv'))
with (OUT/'independent_verification.json').open('x') as f: json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
