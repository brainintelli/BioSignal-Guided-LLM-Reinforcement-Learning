"""Frozen-profile offline perturbation diagnostics. Synthetic data are not training rollouts."""
from pathlib import Path
import csv
import hashlib
import json
import math
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'mea_22554_baseline_20261116'
SOURCE = PREVIOUS/'source/22554_baseline.raw.h5'
ANALYSIS = PREVIOUS/'analysis_routed_v1/analysis.json'
EXPECTED = '45a341c80d7ac60eb2b9ba846dce926f9e7fd4b6fb27f0b7b8c963798640f84c'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def features(events, lo, width, channels):
    if not len(events):
        return np.array([0.,0.,0.])
    assert np.all((events[:,0] >= lo) & (events[:,0] < lo+width))
    bins = (events[:,0]-lo)//200
    pairs = np.unique(np.column_stack([bins,events[:,1]]), axis=0)
    counts = np.bincount(pairs[:,0], minlength=width//200)
    fraction = np.mean(counts[bins] >= 2)
    return np.array([math.log1p(len(events)/(channels*5)), len(np.unique(events[:,1]))/channels, fraction])


def score(values, history, center, scale):
    z = (values-center)/scale
    return math.exp(-.5*np.mean(z*z)), float(np.sqrt(np.mean((history-z)**2, axis=1)).min())


def summary(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p05=float(np.quantile(values,.05)),p95=float(np.quantile(values,.95)))


def main():
    output = ROOT/'stress_v1'
    if output.exists():
        raise FileExistsError(output)
    assert sha(SOURCE) == EXPECTED
    original_profile_hash = sha(PREVIOUS/'analysis_routed_v1/candidate_profile.json')
    analysis = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    profile = analysis['candidates']['profile']; windows = analysis['windows']
    assert not profile['r2_training_enabled'] and not profile['source_full_structural_passed']
    assert all(profile['active_features']) and profile['split']['validation']==[31,59]
    center=np.array(profile['center']);scale=np.array(profile['scale']);novelty_scale=profile['novelty_distance_p95']
    real_features=np.array([[math.log1p(w['rate_hz_per_channel']),w['active_channel_fraction'],w['coincident_event_fraction']] for w in windows])
    normalized=(real_features-center)/scale
    with h5py.File(SOURCE,'r') as f:
        store=f['data_store/data0000'];data=store['spikes'][:]
        channel_ids=store['settings/mapping']['channel'].astype(np.int64)
        event_array=np.column_stack([data['frameno'],data['channel']]).astype(np.int64)
    channel_to_index={int(c):i for i,c in enumerate(channel_ids)}
    rng=np.random.default_rng(225542026); rows=[];checks=0
    conditions=['observed','channel_circular_shift','inject_1_percent','inject_5_percent','inject_20_percent','drop_20_percent']
    for i in range(31,59):
        w=windows[i];lo=w['start_frame'];hi=w['end_frame'];width=hi-lo
        selected=event_array[(event_array[:,0]>=lo)&(event_array[:,0]<hi)&np.isin(event_array[:,1],channel_ids)]
        measured=features(selected,lo,width,len(channel_ids))
        assert np.max(np.abs(measured-real_features[i]))<1e-13
        expected=analysis['candidates']['scores'][i-31]
        bp,raw=score(measured,normalized[:i],center,scale)
        assert abs(bp-expected['baseline_proximity'])<1e-12
        assert abs(min(raw/novelty_scale,1.)-expected['novelty'])<1e-12
        observed_counts={int(c):int(np.sum(selected[:,1]==c)) for c in np.unique(selected[:,1])}
        occupied={(int(f)-lo)*len(channel_ids)+channel_to_index[int(c)] for f,c in selected}
        for condition in conditions:
            repetitions=1 if condition=='observed' else 50
            for repetition in range(repetitions):
                if condition=='observed':
                    changed=selected
                elif condition=='channel_circular_shift':
                    offsets={c:int(rng.integers(0,width)) for c in observed_counts}
                    changed=np.array([[lo+(int(f)-lo+offsets[int(c)])%width,int(c)] for f,c in selected],dtype=np.int64)
                    assert len(changed)==len(selected)
                    assert {int(c):int(np.sum(changed[:,1]==c)) for c in np.unique(changed[:,1])}==observed_counts
                elif condition.startswith('inject'):
                    fraction=int(condition.split('_')[1])/100
                    count=max(1,round(len(selected)*fraction));new_codes=set()
                    while len(new_codes)<count:
                        candidate=int(rng.integers(0,width*len(channel_ids)))
                        if candidate not in occupied:new_codes.add(candidate)
                    added=np.array([[lo+k//len(channel_ids),int(channel_ids[k%len(channel_ids)])] for k in sorted(new_codes)],dtype=np.int64)
                    changed=np.concatenate([selected,added])
                    assert len(changed)==len(selected)+count
                else:
                    remove=round(.2*len(selected));indices=rng.choice(len(selected),len(selected)-remove,replace=False)
                    changed=selected[indices]
                    assert len(changed)==len(selected)-remove
                feature=features(changed,lo,width,len(channel_ids));bp,nd=score(feature,normalized[:i],center,scale)
                assert len(np.unique(changed,axis=0))==len(changed)
                if condition=='channel_circular_shift':assert np.array_equal(feature[:2],measured[:2])
                rows.append(dict(window_index=i,repetition=repetition,condition=condition,source='observed' if condition=='observed' else 'synthetic_diagnostic',
                    event_count=len(changed),rate_hz_per_channel=float(math.expm1(feature[0])),active_channel_fraction=float(feature[1]),
                    cooccurrence=float(feature[2]),baseline_proximity=float(bp),novelty=min(nd/novelty_scale,1.),
                    novelty_raw_distance=float(nd),baseline_delta=float(bp-expected['baseline_proximity']),
                    novelty_delta=float(min(nd/novelty_scale,1.)-expected['novelty']),history_last_window=i-1))
                checks+=1
    aggregated={}
    for condition in conditions:
        subset=[row for row in rows if row['condition']==condition]
        aggregated[condition]=dict(observations=len(subset),biological_samples=1,
            baseline_proximity=summary([r['baseline_proximity'] for r in subset]),novelty=summary([r['novelty'] for r in subset]),
            baseline_delta=summary([r['baseline_delta'] for r in subset]),novelty_delta=summary([r['novelty_delta'] for r in subset]),
            cooccurrence=summary([r['cooccurrence'] for r in subset]),novelty_clipped_fraction=float(np.mean([r['novelty']==1 for r in subset])))
    temporal={}
    for name,lo,hi in [('validation_first_half',31,45),('validation_second_half',45,59)]:
        temporal[name]={key:summary([w[key] for w in windows[lo:hi]]) for key in ('rate_hz_per_channel','active_channel_fraction','coincident_event_fraction')}
    assert sha(PREVIOUS/'analysis_routed_v1/candidate_profile.json')==original_profile_hash
    report=dict(schema='mea.r2.unstimulated.stress.v1',source_sha256=EXPECTED,profile_sha256=original_profile_hash,
        r2_training_enabled=False,alpha=1,source_full_structural_passed=False,synthetic_conditions_are_not_real_recordings=True,
        seed=225542026,repetitions=50,independent_biological_samples=1,checked_rows=checks,temporal=temporal,conditions=aggregated,
        history='original past observed windows; each perturbation affects current window only',
        limitations=['within-session only','not on-policy rollouts','no task benefit evidence','synthetic noise is not a measured noise distribution'])
    output.mkdir(exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (output/'all_trials.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'identity.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'analysis_protocol.md'),'analysis_sha256':sha(output/'analysis.json'),'trials_sha256':sha(output/'all_trials.csv')},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
