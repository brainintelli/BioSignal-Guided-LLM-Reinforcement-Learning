"""Register three starting conditions, one common set of games and endpoint families."""
from pathlib import Path
import hashlib,json,random,sys
H=Path(__file__).resolve().parent
T=H.parent;archive=T/'baseline-replication-64';session=T.parent;work=session.parents[1]
sys.path.insert(0,str(work/'standalone/mea_state2_r2'))
from mea_state2_r2.features import digest

def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    prev=read(archive/'design.json');tr=read(T/'design.json');ctx=read(T/'context.json')
    assert sha(session/prev['source_file'])==prev['source_sha'] and prev['normalization']==ctx['normalization'] and prev['roi_sha']==ctx['roi_sha']
    assert prev['previous_study_sha']==digest(tr) and prev['training_audit_sha']==sha(T/'training-audit.json')
    assert len(prev['physical_electrodes'])==322 and len(prev['windows'])==60 and tr['alpha']==1 and tr['beta']==0
    scenes=read(H/'scenes.json');pilot=read(H/'scene-check.json');prior=set(pilot['worlds'])
    def collect(v):
        if isinstance(v,int):prior.add(v)
        elif isinstance(v,list):
            for i in v:collect(i)
        elif isinstance(v,dict):
            for i in v.values():collect(i)
    for nm in ('design.json','protocol.json','plan.json'):
        for file in (work/'output').rglob(nm):
            if H in file.parents:continue
            d=read(file)
            if not isinstance(d,dict):continue
            for key in ('worlds','holdout_seeds','training_seeds','seeds','episodes','restart_worlds','training_worlds'):
                if key in d:collect(d[key])
    maps=random.Random(22021111971).sample(range(248000000,249000000),64)
    assert len(set(maps))==64 and not set(maps)&prior
    sh={}
    for w in maps:
        rng=random.Random(22021111972+w);order=[]
        for off in range(0,60,20):
            while True:
                a=rng.sample(range(10),10)
                if all(i!=x for i,x in enumerate(a)):break
            order.extend(off+2*j+k for j in a for k in (0,1))
        assert sorted(order)==list(range(60));sh[str(w)]=order
    fields=('chip','sample','source_sha','source_file','source_frames','physical_electrodes','roi_sha','normalization','states','windows','training_seeds','horizon')
    s={k:prev[k] for k in fields}
    s.update(scenes=scenes,game_pins=pilot['pins'],pilot_sha=sha(H/'scene-check.json'),arms=['ordered','masked','shuffled'],
        alpha=1,beta=0,reward_note='R1 unchanged; drink R2 diagnostics only; no model update',worlds=maps,shuffle=sh,
        previous_eval_design_sha=digest(prev),training_design_sha=digest(tr),training_audit_sha=sha(T/'training-audit.json'),
        play_sha=sha(T/'play60.py'),runner_sha=sha(H/'evaluate.py'),prior_ids_checked=len(prior),
        draws=40000,bootstrap_seed=22021111973,
        primary='Six survival contrasts: ordered minus masked and shuffled within each scene. Per-family nominal 95%, 99.1667% two-sided paired crossed intervals. Stable gain needs lower bound above zero and >=6 of 8 paired training seeds positive for the contrast. Both controls required for a time-order-specific gain in a scene.',
        targeted_decisions='four secondary contrasts: time to successful hydration in thirst and time to successful eating in hunger, each versus two comparators; two-sided 98.75% intervals, faster is better; at least 6/8 seeds favorable for stable change',
        other='Both resource times, observed success counts, mean restricted survival and new cells in every scene. Other timing and exploration intervals descriptive 95%.',
        first_event='First native game tick in the corresponding executed legal drink/eat action with a verified increase in thirst/hunger reserve from the action start, at or before tick 60. Unachieved including death before event assigned 60 for restricted waiting; actual first-success ticks and counts retained separately.',
        evaluation='all 24 previously fitted checkpoints frozen; use one shared new map list, per-map action RNG and MEA shuffle across all scenes and arms; distinct raw H5 relative to training; no live stimulation')
    with (H/'design.json').open('x',encoding='utf8') as f:json.dump(s,f,ensure_ascii=False,indent=2)
    print('locked',digest(s),len(maps),len(prior))
if __name__=='__main__':main()
