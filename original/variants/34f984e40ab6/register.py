"""Register the signal screen and comparison before inspecting gameplay outcomes."""
import hashlib
import json
import os
import random
import runpy
from pathlib import Path

out=Path(__file__).resolve().parent
src=out.parent/'22554_full_20261119_1224'
rd=lambda p:json.loads(p.read_bytes())
def store(p,value):
    data=json.dumps(value,indent=2,allow_nan=False).encode()
    if p.exists() and p.read_bytes()!=data:raise ValueError('Changed input '+str(p))
    p.write_bytes(data)

def main():
    old=rd(out.parent/'22554_fresh_maps_20261119/protocol.json')
    files={r['path']:r for r in rd(src/'remote-manifest.json')}
    source={}
    for r in rd(src/'closed-h5-audit.json'):
        k=r.get('site')
        if not k and 'site-selection' in r['path']:k='quiet'
        if not k:continue
        item=files[r['path']]
        sha=hashlib.sha256()
        with (src/'records'/item['rel']).open('rb') as f:
            for block in iter(lambda:f.read(4194304),b''):sha.update(block)
        assert sha.hexdigest()==item['sha256'] and r['gaps']==0 and r['chip']=='22554'
        source[k]=dict(path=r['path'],sha256=item['sha256'])
        if k!='quiet':
            receipt=rd(src/'records/session'/f'{k}-result.json')
            assert receipt['onset_visible'] and receipt['stim_channel_recovered_at_12s'] and r['route_match'] and len(r['events'])==1
            source[k]['event']=receipt['event_frame']
    assert set(source)==set(['quiet',*'HWF'])
    maps=random.Random(225541119).sample(range(51000000,59000000),12)
    seeds=list(range(2255401,2255407))
    plan=dict(source=source,sites=rd(src/'records/session/site-plan.json')['suggested'], offsets=[-10,-7,-4,12,32,52],
              binding=hashlib.sha256((src/'binding.json').read_bytes()).hexdigest(), training_seeds=seeds,
              heldout=dict(zip(seeds,'HWFHWF')), maps=maps,ticks=[48,192],steps=96,batch=12,
              engine=old['engine'],pure={str(s):old['models'][f'pure-{s}'] for s in seeds},mix=old['reward_mix'],
              screen='Per file, passive-active electrodes at least 200 um from all outputs; six entire 1s windows relative to -12s. Fixed intersection of rail-free channels with positive MAD, ratio 0.67 to 1.50 and median shift <=5 ADC in every window. At least 50 channels per file. Entire file split, panel computed independently per file. No noise correction/interpolation.',
              shuffle='Within each pre/post phase derange all six training feature windows with fixed seed. Same prime, event ordering, reward sign, and feature multiset. Both measured policies get identical ordered features from the unseen file at evaluation.',
              metrics='At tick 48 and 192: new cells, drink and eat actions, water, satiety, health, deaths; completed reserve increases separately. Tick 192 primary. Equal-weight paired maps and training seeds. Exploratory direction for each endpoint must beat BOTH controls in mean, in >=4/6 seeds, and in each of three heldout-site means; persistence at 48 reported. Paired crossed-bootstrap intervals are conditional on this one sample. No biological replication.')
    store(out/'protocol.json',plan)
    v=runpy.run_path(str(Path(os.environ['TEMP'])/'codex_live22554_1119.py'))
    with v['connect']() as c:
        data=json.loads(v['execute'](c,(out/'extract_hwf.py').read_text(encoding='utf8'),[json.dumps(plan)],timeout=240))
    store(out/'windows.json',data)
    print({k:(x['candidates'],len(x['channels']),len(x['windows'])) for k,x in data.items()})

if __name__=='__main__':main()
