"""Independent file/endpoint audit, without importing campaign analysis code."""
import argparse,hashlib,itertools,json,math,random
from pathlib import Path
from statistics import mean
ARMS=('none','before','after','shuffled_after');CONTROLS=('none','before','shuffled_after')
METRICS=('visited_count','discounted_r1','final_health','final_hunger','final_thirst','terminated')
PIN='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'
INITIAL={'actor':'94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0','critic':'52a7c71c972fe83ea149a14d91472f0c4e8a397d3b4a656ffab8a9a58ab2b63e','base':'0ddb7bb4f250d5ec4ed7b0a59180234fe8df0aed660868ea7da6856d3e29a7f0','reference':'c4e0cac8056d8537efad0ff82725665da41c5f0a723aae5c4dbe6d78f1852d1f'}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def stats(d,family):
    assert len(d)==8 and all(math.isfinite(v) for v in d)
    center=mean(d);tol=1e-12*max(1,abs(center));extreme=0
    for mask in range(256):
        signed=sum(v*(1 if mask&(1<<i) else -1) for i,v in enumerate(d))/8
        extreme+=abs(signed)>=abs(center)-tol
    rng=random.Random(8800100)
    draws=sorted(mean(rng.choices(d,k=8)) for _ in range(20000))
    def quantile(f):
        pos=(len(draws)-1)*f;lo=int(pos);return draws[lo]*(1-(pos-lo))+draws[min(lo+1,len(draws)-1)]*(pos-lo)
    return dict(differences=d,mean_difference=center,bootstrap_95=[quantile(.025),quantile(.975)],
        p_two_sided=extreme/256,p_adjusted=min(1,family*extreme/256),positive_pairs=sum(v>0 for v in d),negative_pairs=sum(v<0 for v in d))
def audit(root):
    exp=root/'experiment01';status=read(exp/'parallel_status.json');reg=read(exp/'registration.json')
    assert status['passed'] is True and status['accepted_cells']==32
    assert reg['source_pin']==PIN and reg['common_initial']==INITIAL
    assert reg['protocol']['r2_enabled'] is False and reg['protocol']['alpha']==1.
    regpin=hashlib.sha256(canonical(reg).encode()).hexdigest()
    original_analysis=read(exp/'analysis.json');assert original_analysis['status']=='complete' and len(original_analysis['cells'])==32
    cells={};verified_files=0;train_rows=0;eval_rows=0
    for b,a in itertools.product(range(8),ARMS):
        folder=exp/f'b{b:02d}_{a}';receipt=read(exp/f'verified_b{b:02d}_{a}.json')
        assert receipt['passed'] is True and receipt['fresh_process'] is True
        assert (receipt['block'],receipt['arm'],receipt['source_pin'],receipt['registration_sha256'])==(b,a,PIN,regpin)
        assert receipt['cell_manifest_sha256']==sha(folder/'cell_manifest.json')
        manifest=read(folder/'cell_manifest.json')
        actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}-{'cell_manifest.json'}
        assert actual==set(manifest)
        for name,h in manifest.items():
            path=(folder/name).resolve();assert path.is_relative_to(folder.resolve()) and sha(path)==h
            verified_files+=1
        e=read(folder/'evaluation.json');t=read(folder/'train/report.json');cfg=read(folder/'train/registration.json')['identity']['ablation']
        assert cfg['world_seed']==8100100+1000*b and cfg['eval_world_seed']==8400100+b
        assert t['initial']==e['checkpoint_tensors']['pre']==INITIAL and t['final']==e['checkpoint_tensors']['post']
        assert t['decisions']==384 and t['r2_enabled'] is False and e['r2_enabled'] is False and e['device_commands_sent']==0
        assert len(t['rounds'])==3 and all(r['applied_steps']=={'actor':2,'critic':16} for r in t['rounds'])
        assert receipt['training_receipt_sha256']==sha(folder/'train/independent_verification.json')
        source=t['source'];order=source['assignment']
        assert sorted(order[:384])==list(range(384)) and order[384]==384
        if a=='shuffled_after':assert source['phase']=='after' and all(i!=v for i,v in enumerate(order[:384]))
        else:assert order==list(range(385))
        train_rows+=t['decisions']
        for p,c in itertools.product(('pre','post'),('before','after')):
            m=e['metrics'][p][c];assert 1<=m['decisions']<=150 and (m['terminated'] or m['decisions']==150)
            assert all(math.isfinite(m[k]) for k in METRICS)
            assignment=read(folder/f'{p}_{c}'/'assignment.json')
            assert assignment['order']==list(range(445,596)) and assignment['world_seed']==8400100+b
            eval_rows+=m['decisions']
        cells[b,a]=e['metrics']
    for b in range(8):
        assert cells[b,'before']['pre']==cells[b,'after']['pre']==cells[b,'shuffled_after']['pre']
    avg=lambda b,a,p,k:mean(cells[b,a][p][c][k] for c in ('before','after'))
    groups={a:{p:{k:mean(avg(b,a,p,k) for b in range(8)) for k in METRICS+('elapsed_ticks',)} for p in ('pre','post')} for a in ARMS}
    contrasts={}
    for control in CONTROLS:
        comparisons={}
        for metric in METRICS:
            d=[avg(b,'after','post',metric)-avg(b,control,'post',metric) for b in range(8)]
            s=stats(d,3 if metric=='visited_count' else 15)
            s['context_mean_differences']={c:mean(cells[b,'after']['post'][c][metric]-cells[b,control]['post'][c][metric] for b in range(8)) for c in ('before','after')}
            oracle=original_analysis['contrasts']['after_minus_'+control][metric]
            for key in ('mean_difference','p_two_sided','p_adjusted'):
                assert abs(s[key]-oracle[key])<1e-12,(control,metric,key)
            assert all(abs(x-y)<1e-12 for x,y in zip(s['bootstrap_95'],oracle['bootstrap_95']))
            comparisons[metric]=s
        contrasts['after_minus_'+control]=comparisons
    def stable(control):
        s=contrasts['after_minus_'+control]['visited_count']
        return s['mean_difference']>0 and s['bootstrap_95'][0]>0 and s['p_adjusted']<.05 and s['positive_pairs']>=6 and min(s['context_mean_differences'].values())>0
    replicated=stable('none') and stable('before');temporal=replicated and stable('shuffled_after')
    assert replicated==original_analysis['exploration_replicated'] and temporal==original_analysis['temporal_order_evidence']
    return dict(passed=True,cells=32,training_decisions=train_rows,evaluation_decisions=eval_rows,verified_files=verified_files,
        source_pin=PIN,registration_sha256=regpin,arms=groups,contrasts=contrasts,
        exploration_replicated=replicated,temporal_order_evidence=temporal,r2_enabled=False,new_stimulation=False,
        survival_significance_reachable=False,survival_minimum_adjusted_p=15*2/256,
        paired_values={a:{k:[avg(b,a,'post',k) for b in range(8)] for k in METRICS+('elapsed_ticks',)} for a in ARMS},
        final_terminations={a:sum(cells[b,a]['post'][c]['terminated'] for b in range(8) for c in ('before','after')) for a in ARMS})
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert stats([1.]*8,3)['p_adjusted']==6/256 and stats([0.]*8,15)['p_adjusted']==1
    result=audit(a.root)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','contrasts','paired_values')}))
if __name__=='__main__':main()
