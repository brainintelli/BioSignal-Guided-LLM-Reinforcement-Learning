"""Verify immutable per-game outputs, then estimate crossed, paired contrasts."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ['survived', 'r1', 'new_cells', 'end_tick', 'health', 'thirst', 'hunger', 'decisions']

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def main(stress=False, severe=False):
    stress = stress or severe
    folder = ROOT / "severe" if severe else ROOT / "stress" if stress else ROOT
    spec = read(folder/'design.json')
    seeds, worlds, modes = (spec[k] for k in ('seeds', 'worlds', 'modes'))
    study_sha = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    assert len(set(spec['aligned_electrodes'])) == 322
    wins = spec['windows']
    assert len(set(tuple(w['normalized']) for w in wins)) == len(wins) == 60
    assert all(a['frame_end'] == b['frame_start'] for a,b in zip(wins, wins[1:]))
    games, csvrows, filehashes = {}, [], {}
    for i, seed in enumerate(seeds):
        for mode in modes:
            if mode == 'initial_ordered' and i: continue
            p = folder/'runs'/str(seed)/(mode+'.json')
            rr = read(p)
            assert rr['study_sha'] == study_sha and rr['mode']==mode and rr['seed']==seed
            loc = ROOT.parent/('models' if mode=='control_masked' else 'pre-stim-baseline/models')/str(seed)/('masked' if mode=='control_masked' else 'baseline')
            expected = read(loc/'registration.json')['initial'] if mode=='initial_ordered' else read(loc/'update.json')['after']
            assert rr['frozen']==expected and [g['world'] for g in rr['games']]==worlds
            filehashes[p.relative_to(folder).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
            for game in rr['games']:
                world=game['world']
                assert game['mode']==mode and game['end_tick'] <= 60 and (game['end_tick']==60 or not game['survived'])
                rows=game['rows']
                assert len(rows)==game['decisions'] and rows[0]['tick']==0 and rows[-1]['end_tick']==game['end_tick']
                assert all(a['end_tick']==b['tick'] and not a['cut'] for a,b in zip(rows,rows[1:]))
                assert math.isclose(sum(r['reward'] for r in rows),game['r1'],abs_tol=1e-12)
                for r in rows:
                    t=r['tick']; assert t < r['end_tick'] and len(r['transition_sha'])==64
                    at=spec['shuffle'][str(world)][t] if mode=='trained_shuffled' else t
                    w=wins[at]; vec=[0.]*4 if mode in ('trained_masked','control_masked') else w['normalized']
                    assert (r['features'],r['window'],r['events'])==(vec,w['index'],w['events_sha'])
                games[mode,seed,world]=game
                csvrows.append({'mode':mode,'seed':seed,'world':world,**{k:game[k] for k in METRICS}})
    assert len(games)==(96 if severe else 64 if stress else 264) and len(filehashes)==(12 if severe else 8 if stress else 33)
    with (folder/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=csvrows[0]);wr.writeheader();wr.writerows(csvrows)
    means={}
    for mode in modes:
        data=[g for (m,_,_),g in games.items() if m==mode]
        means[mode]={'n':len(data),'deaths':sum(1-g['survived'] for g in data),**{k:float(np.mean([g[k] for g in data])) for k in METRICS},
                    'min_health':min(g['health'] for g in data)}
    rng=np.random.default_rng(spec['bootstrap']['seed'])
    ns,nw=len(seeds),len(worlds);bs=spec['bootstrap']['draws']
    si=rng.integers(0,ns,(bs,ns));wi=rng.integers(0,nw,(bs,nw))
    effects={}
    for a,b in spec['primary_comparisons']+([spec['diagnostic_comparison']] if spec['diagnostic_comparison'] else []):
        eff={}
        for k in (('survived','end_tick') if stress else ('survived','r1','new_cells')):
            mat=np.array([[games[a,s,w][k]-games[b,seeds[0] if b=='initial_ordered' else s,w][k] for w in worlds] for s in seeds])
            samples=mat[si[:,:,None],wi[:,None,:]].mean(axis=(1,2))
            tail=.025 if stress else .05/12
            lower,upper=np.quantile(samples,[tail,1-tail])
            eff[k]=dict(mean=float(mat.mean()),interval=[float(lower),float(upper)] if k not in ('survived','end_tick') or mat.any() else None,pairs_higher=int((mat>0).sum()),pairs_lower=int((mat<0).sum()),ties=int((mat==0).sum()),positive_seeds=int((mat.mean(1)>0).sum()),positive_worlds=int((mat.mean(0)>0).sum()))
        effects[a+'__'+b]=eff
    result={'design_sha':study_sha,'episodes':len(games),'action_decisions':sum(g['decisions'] for g in games.values()),
        'biological_samples':1,'source_h5':spec['source_file'],'mode_means':means,'effects':effects,
        'interval_note':'95% paired crossed bootstrap over 4 training seeds and 8 worlds; survival primary, truncated survival time descriptive.' if stress else 'Crossed paired resampling of 8 training seeds and 8 held-out worlds; 30000 draws; 99.17% individual intervals for the six primary-comparison secondary outcomes. Diagnostic contrast displayed at the same level. Survival interval omitted if no discordant pairs.',
        'run_files_sha256':filehashes}
    (folder/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'means':means,'comparisons':effects,'episodes':len(games),'actions':result['action_decisions']}))
if __name__=='__main__':
    import sys
    main('--stress' in sys.argv,'--severe' in sys.argv)
