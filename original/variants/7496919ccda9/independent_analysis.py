"""Independent saved-R1 and paired-statistic audit; no experiment module imports."""
import argparse,hashlib,itertools,json,math,random,statistics
from pathlib import Path
ARMS=('baseline','state','reward','both','shuffled')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mean(x):return sum(x)/len(x)
def statistics_for(d):
    m=mean(d);eps=1e-12*max(1,abs(m));extreme=0
    for mask in range(256):
        value=mean([x*(1 if mask&(1<<i) else -1) for i,x in enumerate(d)])
        extreme+=abs(value)>=abs(m)-eps
    rng=random.Random(5320000);draws=[]
    # Match the registered Python random.choices index sampling, independently.
    for _ in range(20000):draws.append(sum(d[math.floor(rng.random()*8)] for i in range(8))/8)
    draws.sort()
    def q(a):
        k=a*19999;i=int(k);return draws[i]*(1-(k-i))+draws[min(i+1,19999)]*(k-i)
    return dict(mean=m,p=extreme/256,ci=[q(.025),q(.975)])
def audit(root):
    analysis=read(root/'analysis.json');reg=read(root/'registration.json');launch=read(root/'execution_plan.json');pin=sha(root/'registration.json')
    assert analysis['status']=='complete' and launch['code']['independent_analysis.py']==sha(Path(__file__)) and launch['registration_sha256']==pin
    means={};total_decisions=0
    for block,arm in itertools.product(range(8),ARMS):
        out=root/f'b{block:02d}_{arm}';report=read(out/'report.json');receipt=read(out/'independent_verification.json')
        assert report['passed'] and receipt['passed'] and receipt['registration_sha256']==pin and receipt['collection_manifest_sha256']==sha(out/'collection_manifest.json')
        assert receipt['verification_pid']!=report['collection_pid'];assert report['update']['before']==reg['common_initial']
        phase_values={'pre':[],'post':[]}
        for name in ('pre0','pre1','train','post0','post1'):
            events=[json.loads(s) for s in (out/name/'rollout.jsonl').read_text(encoding='utf-8').splitlines()];tick=0;value=0;raw=0
            for i,e in enumerate(events):
                row=e['training'];r1=e['r1_score']['reward'];u=e['snapshot']['score']['u'];weight=.05 if name=='train' and arm in ('reward','both','shuffled') else 0.
                assert abs(row['reward']-(r1+weight*u))<1e-14 and e['reward_record']['aux_weight']==weight
                assert e['index']==i and row['duration']==e['r1_score']['elapsed_ticks'];assert -1<=u<=1
                value+=.99985**tick*r1;tick+=row['duration'];raw+=r1
            total_decisions+=len(events);m=read(out/name/'metrics.json');assert abs(value-m['discounted_r1'])<1e-12 and abs(raw-m['r1_sum'])<1e-12 and tick==m['ticks']
            if name!='train':phase_values[name[:-1]].append(value)
            else:assert len(events)==128
        means[block,arm]={p:mean(v) for p,v in phase_values.items()}
    comparisons={}
    for a,b,primary in [('both','baseline',True),('both','shuffled',True),('state','baseline',False),('reward','baseline',False),('both','state',False)]:
        d=[means[i,a]['post']-means[i,b]['post'] for i in range(8)];g=[d[i]-(means[i,a]['pre']-means[i,b]['pre']) for i in range(8)]
        for phase,v in [('post',d),('gain',g)]:
            derived=statistics_for(v);claimed=analysis['primary' if primary else 'secondary'][a+'-'+b][phase]
            assert max(abs(x-y) for x,y in zip(v,claimed['differences']))<1e-12
            assert abs(derived['mean']-claimed['mean_difference'])<1e-12 and derived['p']==claimed['p_two_sided']
            assert max(abs(x-y) for x,y in zip(derived['ci'],claimed['bootstrap_95']))<1e-12
            if primary and phase=='post':assert claimed['p_adjusted_primary']==min(1.,2*derived['p'])
        comparisons[a+'-'+b]=statistics_for(d)
    result=dict(passed=True,cells=40,collection_decisions=total_decisions,comparison_statistics=comparisons,production_modules_imported=False,
        registration_sha256=pin,analysis_sha256=sha(root/'analysis.json'),biological_recordings=1)
    with (root/'independent_analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(passed=True,cells=40,collection_decisions=total_decisions)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
