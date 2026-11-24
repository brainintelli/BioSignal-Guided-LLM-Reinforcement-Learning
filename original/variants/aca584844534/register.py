"""Bind the archived five-arm training protocol to the second 22021 sample."""
import json
import shutil
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OLD=ROOT/'output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
RECORD=OUT.parent/'cpp-crossover-v1'
sys.path[:0]=[str(OLD),str(ROOT/'.tools/mea_baseline_deps'),str(ROOT/'standalone/mea_state2_r2')]
import build_inputs as raw
from mea_state2_r2.features import digest,fit,transform
from build_inputs import read,write,window

def main():
    raw.RECORD=RECORD
    prev=read(OLD/'design.json')
    binding=read(RECORD/'binding.json')
    plan=read(RECORD/'design.json')
    hashes=read(RECORD/'raw-hashes.json')
    checks=read(RECORD/'raw-readback-check.json')
    roi=plan['roi']
    assert binding['chip_id']==prev['chip']=='22021' and binding['organoid_id']!=read(OLD/'sham.json')['sample']
    assert binding['prior_nonzero_attempts']==0 and plan['prior_attempts']==0 and len(set(roi))==len(roi)==324
    assert len(hashes)==19 and len(checks)==len(plan['trials'])==18
    first,last,sp,ev=raw.read_record('warmup',hashes,roi)
    warm=read(RECORD/'warmup.json')
    assert not ev and warm['route_confirmed'] and not warm['zero_reference_site_rails'] and warm['metadata']['frame_gaps']==0
    assert last<checks_first_frame() and warm['metadata']['fs']==raw.FS
    calibration=[window(sp,first,last,first+i*raw.FS,11-i,0,i,roi) for i in range(12,36)]
    norm=fit([w['features'] for w in calibration])
    series={a:{k:[] for k in ('train','bootstrap','holdout')} for a in ('sham','regular','irregular','baseline')}
    for i in [*range(2,54),*range(58,64)]:
        key='holdout' if i>=58 else 'bootstrap' if (i-2)%13==12 else 'train'
        series['baseline'][key].append(window(sp,first,last,first+i*raw.FS,1000+i,0,i,roi))
    for i,(trial,check) in enumerate(zip(plan['trials'],checks,strict=True),1):
        name='trial_%02d'%i
        evidence=read(RECORD/(name+'-evidence.json'))
        start,stop,sp,events=raw.read_record(name,hashes,roi)
        assert trial==evidence['plan'] and events==evidence['frames'] and check['index']==i
        assert check['chip']=='22021' and check['condition']==trial['condition'] and check['pulses']==len(trial['events'])
        assert evidence['pattern_verified'] and evidence['intended_edges_present'] and not evidence['late_site_rail']
        assert evidence['metadata']['fs']==raw.FS and evidence['metadata']['frame_gaps']==0
        for off in (range(2,15) if trial['block']<=4 else [2,7,12]):
            key='holdout' if trial['block']>4 else 'bootstrap' if off==14 else 'train'
            series[trial['condition']][key].append(window(sp,start,stop,events[0]+off*raw.FS,i*100+off,i,off,roi))
    for records in series.values():
        assert {k:len(v) for k,v in records.items()}==dict(train=48,bootstrap=4,holdout=6)
        assert max(w['frame_end'] for k in ['train','bootstrap'] for w in records[k])<=min(w['frame_start'] for w in records['holdout'])
        for rows in records.values():
            for w in rows:w['normalized']=list(transform(w['features'],norm))
    source=digest(hashes)
    aux=OUT/'pre-stim-baseline'
    aux.mkdir(exist_ok=True)
    contexts={}
    for a in [*prev['arms'],'baseline']:
        base=(OLD/'pre-stim-baseline' if a=='baseline' else OLD)
        old_context=read(base/(a+'.json'))
        ctx=old_context|dict(sample=binding['organoid_id'],normalization=norm,source_sha256=source,roi_sha=digest(roi),
                             series=series['sham' if a=='masked' else a])
        write((aux if a=='baseline' else OUT)/(a+'.json'),ctx)
        contexts[a]=digest(ctx)
    spec=prev|dict(session_id=binding['session_id'],contexts={a:contexts[a] for a in prev['arms']},
                   source_sha=source,baseline_windows=calibration,organoid_id=binding['organoid_id'],
                   replication_of=digest(prev),paired_with_original_worlds=True,
                   supplemental_clock=6)
    assert not set(spec['holdout_seeds']) & {w for ep in spec['episodes'].values() for w in ep}
    write(OUT/'design.json',spec)
    b=read(OLD/'pre-stim-baseline/design.json')
    write(aux/'design.json',b|{k:spec[k] for k in ('source_sha','session_id','baseline_windows','organoid_id','paired_with_original_worlds','supplemental_clock')}|
          dict(contexts={'baseline':contexts['baseline']},replication_of=digest(b),parent_sha=digest(spec)))
    for folder,origin in [(OUT,OLD),(aux,OLD/'pre-stim-baseline')]:
        for name in ('train.py','verify_game.py','model_config.json'):
            shutil.copyfile(origin/name,folder/name)
    script=(OLD/'compare.py').read_text(encoding='utf8')
    (OUT/'compare.py').write_text(script.replace('48 个新留存游戏世界','48 张原方案留存地图 · 第二枚样本'),encoding='utf8')
    print(binding['organoid_id'],len(roi),len(spec['training_seeds']),len(spec['holdout_seeds']),source)

def checks_first_frame():
    return read(RECORD/'trial_01-evidence.json')['metadata']['first_frame']

if __name__=='__main__':main()
