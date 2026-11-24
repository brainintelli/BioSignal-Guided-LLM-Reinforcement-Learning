"""Derive registered inputs again from the archived physical recordings only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tarfile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent.parent
raw_dir = STUDY/'cpp-crossover-v1'
recipe = STUDY/'offline-stress-three-arms/register.py'
spec = importlib.util.spec_from_file_location('archive_feature_extraction', recipe)
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)


def main():
    raw_manifest = reg.read(raw_dir/'raw-hashes.json')
    physical = reg.read(raw_dir/'design.json')
    chip = reg.read(raw_dir/'binding.json')['chip_id']
    roi = physical['roi']
    with tarfile.open(HERE/'evaluated-evidence.tar.gz', 'r:gz') as archive:
        ctx = json.load(archive.extractfile('context.json'))
        execution = json.load(archive.extractfile('stress/design.json'))
        settings = json.load(archive.extractfile('design.json'))
    assert len(raw_manifest) == 19 and chip == ctx['chip'] == '22021' and len(roi) == len(set(roi)) == 324
    assert reg.digest(roi) == ctx['roi_sha'] == execution['roi_sha']
    assert reg.digest(raw_manifest) == ctx['source_sha256'] == execution['source_sha']
    a,b,spikes,events = reg.read_record('warmup',raw_manifest,roi)
    assert not events
    calibration = reg.fit([reg.window(spikes,a,b,a+(12+i)*reg.FS,-i-1,0,12+i,roi)['features'] for i in range(24)])
    assert calibration == ctx['normalization'] == execution['normalization']
    train, bootstrap, test = [], [], []
    for i,trial in enumerate(physical['trials'],1):
        name = f'trial_{i:02d}'
        ev = reg.read(raw_dir/(name+'-evidence.json'))
        a,b,spikes,events = reg.read_record(name,raw_manifest,roi)
        assert events == ev['frames'] and trial == ev['plan'] and trial['block'] == 1+(i-1)//3
        for offset in (range(8,13 if i%3==0 else 12) if i<=12 else range(2,12)):
            w=reg.window(spikes,a,b,events[0]+offset*reg.FS,i*100+offset,i,offset,roi)
            w['normalized']=list(reg.transform(w['features'],calibration))
            dest = test if i>12 else bootstrap if offset==12 else train
            dest.append(w)
    assert train==ctx['series']['train'] and bootstrap==ctx['series']['bootstrap'] and test==execution['windows']
    assert max(w['frame_end'] for w in train+bootstrap) <= min(w['frame_start'] for w in test)
    assert [len(train),len(bootstrap),len(test)] == [48,4,60]
    for windows,orders in [(train,settings['training_order']),(test,execution['shuffle'])]:
        for order in orders.values():
            assert sorted(order)==list(range(len(windows)))
            for j in range(0,len(order),2):
                assert order[j]%2==0 and order[j+1]==order[j]+1 and order[j]!=j
                assert windows[j]['trial']==windows[order[j]]['trial']==windows[order[j+1]]['trial']
                assert windows[order[j]]['frame_end']==windows[order[j+1]]['frame_start']
    result=dict(raw_h5_files=len(raw_manifest),raw_bytes=sum(x['bytes'] for x in raw_manifest.values()),chip=chip,roi_channels=len(roi),calibration_windows=24,training_windows=48,training_bootstrap_windows=4,heldout_windows=60,physical_training_blocks=settings['physical_training_blocks'],physical_evaluation_blocks=settings['physical_holdout_blocks'],nonoverlapping_frames=True,training_permutations_checked=len(settings['training_order']),evaluation_permutations_checked=len(execution['shuffle']),features_match=True,recording_manifest_digest=reg.digest(raw_manifest),registration_extractor_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())
    (HERE/'source-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    main()
