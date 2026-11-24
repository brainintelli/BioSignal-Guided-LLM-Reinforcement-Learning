"""Standalone byte/receipt/return/statistics audit of the delivered experiment.

Native transitions and GPU arithmetic were checked by the bound server receipts.
Large checkpoint tensors stay on the GPU host, listed explicitly as remote-only.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from statistics import mean
import zipfile


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;gpu=root/'gpu';exp=gpu/'experiment01'
    reg=read(exp/'registration.json');rsha=digest(reg);remote_only=read(gpu/'remote_only_files.json')
    assert reg==read(root/'frozen_registration.json'),'registered experiment changed'
    assert reg['protocol']['source_manifest_sha256']=='2a85126aac33f824b2523ea9459e16d1264317a7453d9a653e739812c37228c1'
    assert reg['protocol']['alpha']==1 and reg['protocol']['r2_enabled'] is False
    with zipfile.ZipFile(gpu/'project_frozen.zip') as z:
        for name,h in reg['sources'].items():assert hashlib.sha256(z.read(name.split('/',1)[1])).hexdigest()==h,name
    scores={};rows_checked=0;remote_count=0
    for block in range(8):
        for arm in ('none','real','shuffled'):
            cell=exp/f'b{block:02d}_{arm}';manifest=read(cell/'manifest.json');report=read(cell/'report.json')
            assert manifest['registration_sha256']==rsha
            assert report['baseline']==reg['common_initial_tensors']
            for name,h in manifest['files'].items():
                f=cell/name
                if f.is_file():assert sha(f)==h,str(f)
                else:
                    key=f.relative_to(gpu).as_posix()
                    assert key in remote_only and remote_only[key]['sha256']==h,key
                    assert f.suffix in {'.pt','.safetensors'},key
                    remote_count+=1
            receipt=read(exp/f'verified_b{block:02d}_{arm}.json')
            assert receipt['block']==block and receipt['arm']==arm
            assert receipt['passed'] is True and receipt['model_verified'] is True
            assert receipt['registration_sha256']==rsha and receipt['manifest_sha256']==sha(cell/'manifest.json')
            assert all(math.isfinite(e) and 0<=e<=1e-6 for e in receipt['statistic_errors'].values())
            scores[block,arm]={}
            for phase in ('pre0','pre1','train','post0','post1'):
                records=[json.loads(s) for s in (cell/phase/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
                metrics=read(cell/phase/'metrics.json');elapsed=0;reward=0.
                for r in records:
                    assert r['reward']==r['r1_score']['reward']==r['reward_record']['total_reward']
                    assert r['reward_record']['alpha']==1 and r['reward_record']['r2_valid'] is False
                    reward+=reg['config']['gamma_tick']**elapsed*r['r1_score']['reward']
                    elapsed+=r['duration'];rows_checked+=1
                assert len(records)==metrics['decisions']
                assert abs(reward-metrics['discounted_r1'])<1e-12 and elapsed==metrics['elapsed_ticks']
                if phase!='train':
                    assert report[phase[:-1]][int(phase[-1])]==metrics
                    scores[block,arm][phase]=reward
                else:assert report['train']==metrics
    analysis=read(exp/'analysis.json')
    assert analysis['status']=='complete' and analysis['replication_units']==8
    for control in ('none','shuffled'):
        for endpoint in ('post','gain'):
            deltas=[]
            for b in range(8):
                d=mean(scores[b,'real']['post'+str(j)]-scores[b,control]['post'+str(j)] for j in range(2))
                if endpoint=='gain':d-=mean(scores[b,'real']['pre'+str(j)]-scores[b,control]['pre'+str(j)] for j in range(2))
                deltas.append(d)
            saved=analysis['contrasts']['real_minus_'+control][endpoint]
            assert all(abs(a-b)<1e-12 for a,b in zip(deltas,saved['differences']))
            observed=mean(deltas);tol=1e-12*max(1.,abs(observed))
            p=sum(abs(mean(x*s for x,s in zip(deltas,signs)))>=abs(observed)-tol for signs in itertools.product([-1,1],repeat=8))/256
            assert p==saved['p_two_sided'] and min(1.,2*p)==saved['p_bonferroni_two_contrasts']
            rng=random.Random(5000100);draws=sorted(mean(rng.choices(deltas,k=8)) for _ in range(20000))
            ci=[]
            for q in [.025,.975]:
                at=q*19999;i=int(at);ci.append(draws[i]+(draws[i+1]-draws[i])*(at-i))
            assert all(abs(a-b)<1e-12 for a,b in zip(ci,saved['bootstrap_95']))
    result=dict(passed=True,cells=24,episodes=120,rows=rows_checked,source_files=len(reg['sources']),
        remote_only_checkpoint_files=remote_count,registration_sha256=rsha,
        scope='bytes, bound receipts, independently summed R1 and paired statistics; no local GPU execution')
    print(json.dumps(result))


if __name__=='__main__':main()
