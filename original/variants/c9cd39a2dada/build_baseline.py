"""Sensitivity comparison with the same sample and routing, preceding the first nonzero output."""
from pathlib import Path
import json
from build_inputs import RECORD, ROOT, FS, read, write, read_record, window
from mea_state2_r2.features import digest, transform

def main():
    spec=read(ROOT/'design.json')
    dest=ROOT/'pre-stim-baseline'; dest.mkdir(exist_ok=True)
    plan=read(RECORD/'design.json'); roi=plan['roi']
    sham=read(ROOT/'sham.json')
    first,last,sp,markers=read_record('warmup',read(RECORD/'raw-hashes.json'),roi)
    receipt=read(RECORD/'warmup.json')
    first_trial=read(RECORD/'trial_01-evidence.json')['metadata']['first_frame']
    assert not markers and last < first_trial and not receipt['zero_reference_site_rails'] and receipt['route_confirmed']
    assert receipt['metadata']['chip_id']=='22021' and receipt['metadata']['frame_gaps']==0
    assert not read(ROOT.parent/'setup-attempt/end.json')['attempts_total']
    series=dict(train=[],bootstrap=[],holdout=[])
    for block in range(4):
        for j in range(13):
            t=2+13*block+j
            series['bootstrap' if j==12 else 'train'].append(window(sp,first,last,first+t*FS,1000+t,0,t,roi))
    for t in range(58,64): series['holdout'].append(window(sp,first,last,first+t*FS,1000+t,0,t,roi))
    train_end=max(w['frame_end'] for w in series['train']+series['bootstrap'])
    eval_start=min(w['frame_start'] for w in series['holdout'])
    assert train_end+4*FS<=eval_start and max(w['frame_end'] for w in spec['baseline_windows'])<=train_end
    for rows in series.values():
        for w in rows:w['normalized']=list(transform(w['features'],sham['normalization']))
    ctx=sham|dict(state2_source='device_spontaneous_before_first_stimulation',series=series)
    base=spec|dict(arms=['baseline'],contexts={'baseline':digest(ctx)},parent_sha=digest(spec),contrasts=[['baseline','masked']],
        physical_training_blocks=[],physical_holdout_blocks=[],holdout_seconds=[58,64],training_seconds=[2,54],
        sensitivity='Pre-first-stimulation spontaneous recording only. Compare with the exact same masked models and worlds. Secondary, two endpoints with 97.5% marginal intervals.',
        interval=dict(draws=30000,seed=632202,quantiles=[.0125,.9875]))
    write(dest/'design.json',base);write(dest/'baseline.json',ctx)
    for n in ('train.py','run_study.py','verify_game.py','resources.py','model_config.json'):
        s=(ROOT/n).read_text(encoding='utf8')
        if n in ('train.py','verify_game.py'):
            a="choices=['masked','sham','regular','irregular']";assert a in s;s=s.replace(a,"choices=['baseline']")
        (dest/n).write_text(s,encoding='utf8')
    print('frames, final prestim first frame',first,last,first_trial,'holdout starts',eval_start)
if __name__=='__main__':main()
