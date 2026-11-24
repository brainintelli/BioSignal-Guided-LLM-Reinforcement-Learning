"""Lock the new held-out worlds, contrasts and frozen model inputs before inference."""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    pilot = ARCHIVE/'severe/design.json'
    spec = read(pilot)
    original = read(ARCHIVE.parent/'design.json')
    worlds = random.Random(20261118224048).sample(range(182000000,183000000),48)
    prior = {179999999}
    for path in ARCHIVE.parents[2].rglob('design.json'):
        if HERE in path.parents: continue
        obj=read(path)
        if not isinstance(obj,dict): continue
        for key in ('worlds','holdout_seeds','training_seeds','seeds'):
            prior.update(x for x in obj.get(key,[]) if isinstance(x,int))
        for ids in obj.get('episodes',{}).values() if isinstance(obj.get('episodes'),dict) else ():
            if isinstance(ids,list): prior.update(x for x in ids if isinstance(x,int))
    assert not set(worlds).intersection(prior)
    for world in worlds:
        rng=random.Random(world+22021)
        while True:
            blocks=rng.sample(range(12),12)
            if all(a!=b for a,b in enumerate(blocks)): break
        seq=[5*b+i for b in blocks for i in range(5)]
        assert sorted(seq)==list(range(60))
        if world==worlds[0]: spec['shuffle']={}
        spec['shuffle'][str(world)]=seq
    assert original['alpha']==1 and original['beta']==0 and len(set(original['training_seeds']))==8
    assert sha(ARCHIVE.parent.parent/spec['source_file'])==spec['source_sha']
    spec.update(worlds=worlds,seeds=original['training_seeds'],bootstrap={'draws':30000,'seed':22021183},
        parent_pilot_design_sha256=sha(pilot),frozen_runner_sha256=sha(ARCHIVE/'freeze.py'),
        historical_seed_ids_checked=len(prior),alpha=1,beta=0,
        study_note='Prospective map holdout conditional on one archived organoid trace, chosen after an eight-map pilot; pilot games excluded.',
        decision_rule='Primary: ordered minus independently trained masked control in probability of being alive at tick 60. Crossed paired seed/world bootstrap, two-sided 95% percentile interval. A lower bound above zero supports gain; an upper bound below zero supports loss. Temporal diagnostic: same weights with five-second blocks permuted. Time survived (capped at 60), native R1, and added cells are descriptive secondary outcomes. All worlds and seeds retained; no interim decision, tuning or early stopping.')
    with (HERE/'design.json').open('x',encoding='utf8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print('Worlds:',len(worlds),'seeds:',len(spec['seeds']),'distinct prior IDs:',len(prior),'design SHA256:',sha(HERE/'design.json'))
if __name__=='__main__': main()
