"""Paired estimates across held-out maps and independently trained policy seeds."""
import csv
import hashlib
import json
from pathlib import Path

from report_fixed_clock import ARMS, COLORS, METRICS, STYLES, Figure, _plot_style, _save, np, read, require

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/22554_fresh_maps_20261119'
FOLDERS = {'explored':'探索新格数','health':'生命值','hunger':'饱食','thirst':'水分','deaths':'死亡率'}
PAIRS = [('ordered','pure'),('ordered','shuffled'),('ordered','r2only'),('stateonly','pure'),
         ('shuffled','pure'),('r2only','pure')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(plan):
    seeds=sorted({r['training_seed'] for r in plan['models'].values()})
    maps=plan['maps']; horizons=plan['ticks']; clock=max(horizons)
    assert len(set(maps))==24 and len(seeds)==6 and not set(maps)&set(plan['old_maps'])
    series={a:np.empty((len(seeds),len(maps),clock+1,5)) for a in ARMS}
    observations={}; behavior={}; rows=[]
    reg_sha=digest(OUT/'protocol.json')
    for arm in ARMS:
        behavior[arm]={str(h):{k:0 for k in ('eat','drink','low_food_eligible','low_food_eat',
                                               'low_water_eligible','low_water_drink')} for h in horizons}
        for i,s in enumerate(seeds):
            job=f'{arm}-{s}'; info=plan['models'][job]; dest=OUT/'runs'/job; done=read(dest/'done.json')
            require(done['probe_decisions']==4 and done['final_state']==info['state'] and
                    set(map(int,done['game_hashes']))==set(maps),'Incomplete checkpoint validation')
            for j,g in enumerate(maps):
                path=dest/f'{g}.json';r=read(path)
                require(digest(path)==done['game_hashes'][str(g)] and r['protocol']==reg_sha and
                        r['seed']==g and r['checkpoint']==info['manifest'],'Unmatched game or checkpoint')
                obs=observations.setdefault(g,r['first_obs_hash'])
                require(r['first_obs_hash']==obs,'Game initializations differ across policies')
                x=np.array(r['ticks']); end=r['terminal_tick']; actions=r['actions']
                require(x.shape==(clock+1,4) and x[0,0]==0 and np.isfinite(x).all() and
                        np.all(np.diff(x[:,0])>=0) and 0<=x[:,1:].min()<=x[:,1:].max()<=100.000001,
                        'Invalid native tick trace')
                require(end is None or end>0 and (end>clock or (x[end:]==x[end]).all()),
                        'Terminal state not absorbing')
                require(actions[0][0]==0 and all(a[0]<b[0] for a,b in zip(actions,actions[1:])),
                        'Invalid action clock')
                series[arm][i,j,:,:4]=x
                series[arm][i,j,:,4]=0 if end is None else (np.arange(clock+1)>=end)
                for h in horizons:
                    a=behavior[arm][str(h)];local=dict.fromkeys(a,0)
                    for t,verb,eat_ok,drink_ok in actions:
                        if t>=h:break
                        local['eat']+=int(verb=='eat');local['drink']+=int(verb=='drink')
                        if eat_ok and x[t,2]<60:
                            local['low_food_eligible']+=1;local['low_food_eat']+=int(verb=='eat')
                        if drink_ok and x[t,3]<60:
                            local['low_water_eligible']+=1;local['low_water_drink']+=int(verb=='drink')
                    for key,n in local.items():a[key]+=n
                    rows.append(dict(arm=arm,training_seed=s,map_seed=g,tick=h,
                                     **dict(zip(FOLDERS,series[arm][i,j,h].tolist())),**local))
    require(len(set(observations.values()))==len(maps),'Maps have duplicate initial observations')
    with (OUT/'paired-games.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return series,behavior,seeds


def effects(x,tweights,mweights):
    # Same paired policies share a map; all 24 games share the policy weights for their training seed.
    bootstrap=np.einsum('bi,ijk,bj->bk',tweights,x,mweights,optimize=True)
    low,high=np.quantile(bootstrap,[.025,.975],axis=0)
    out={}
    for k,metric in enumerate(FOLDERS):
        seed=x[:,:,k].mean(axis=1); by_map=x[:,:,k].mean(axis=0)
        out[metric]=dict(mean=float(seed.mean()),interval=[float(low[k]),float(high[k])],by_seed=seed.tolist(),
                         first_12_maps=float(x[:,:12,k].mean()),last_12_maps=float(x[:,12:,k].mean()),
                         positive_seeds=int((seed>1e-9).sum()),positive_maps=int((by_map>1e-9).sum()))
    return out


def report():
    plan=read(OUT/'protocol.json');series,behavior,seeds=collect(plan)
    rng=np.random.default_rng(2026111902)
    tw=rng.multinomial(6,[1/6]*6,size=20000)/6
    mw=rng.multinomial(24,[1/24]*24,size=20000)/24
    means={};contrasts={}
    for h in plan['ticks']:
        means[str(h)]={a:dict(zip(FOLDERS,x[:,:,h].mean(axis=(0,1)).tolist())) for a,x in series.items()}
        contrasts[str(h)]={a+'-'+b:effects(series[a][:,:,h]-series[b][:,:,h],tw,mw) for a,b in PAIRS}
    prim=[contrasts['48'][pair]['hunger'] for pair in ('ordered-pure','ordered-shuffled')]
    result=dict(training_seeds=seeds,map_seeds=plan['maps'],games=720,means=means,contrasts=contrasts,behavior=behavior,
                primary_weak=all(r['mean']>0 and r['positive_seeds']>=4 for r in prim),
                primary_interval_support=all(r['interval'][0]>0 for r in prim),independent_chips=1)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    with _plot_style():
        fig=Figure(figsize=(11.3,7.1),dpi=155,layout='constrained')
        fig.suptitle('芯片 22554｜24 张新地图上的五组表现\n每条曲线汇总 6 个冻结模型，每组 144 局',fontsize=14)
        for j,(ax,(k,label)) in enumerate(zip(fig.subplots(2,2).flat,METRICS.items())):
            for a,x in series.items():
                ax.plot(range(x.shape[2]),x[:,:,:,j].mean(axis=(0,1)),color=COLORS[a],ls=STYLES[a],label=ARMS[a])
            ax.axvline(48,color='#888888',alpha=.5,ls=':')
            ax.set(xlabel='游戏时钟刻',ylabel=label,xlim=(0,192),xticks=[0,48,96,144,192])
            if j==1 and all(np.ptp(v[:,:,:,j])==0 for v in series.values()):ax.set_ylim(99,101)
            if j==0:ax.legend(frameon=False,fontsize=8)
            ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
        _save(fig,OUT/'five-arms.png')
        fig=Figure(figsize=(10.7,6.6),dpi=150,layout='constrained')
        fig.suptitle('有序 MEA 信号的配对差值｜正值表示高于对照\n横线为按训练种子和地图重采样的 95% 区间，小点为各训练种子的均值',fontsize=13)
        labels=['48 刻 · 对纯模型','48 刻 · 对打乱','192 刻 · 对纯模型','192 刻 · 对打乱']
        for ax,(k,name) in zip(fig.subplots(2,2).flat,METRICS.items()):
            ax.axvline(0,color='#777',lw=1)
            for i,(h,pair) in enumerate((h,pair) for h in ('48','192') for pair in ('ordered-pure','ordered-shuffled')):
                v=contrasts[h][pair][k];y=3-i;lo,hi=v['interval'];color='#167b76' if pair=='ordered-shuffled' else '#435989'
                ax.scatter(v['by_seed'],y+np.linspace(-.13,.13,6),s=12,alpha=.52,c=color)
                ax.hlines(y,lo,hi,color=color,lw=2);ax.scatter(v['mean'],y,marker='D',s=27,c=color)
            ax.set(yticks=[3,2,1,0],yticklabels=labels,ylim=(-.4,3.4),xlabel=name+'差值')
            ax.grid(axis='x',alpha=.14);ax.spines[['top','right','left']].set_visible(False)
        _save(fig,OUT/'paired-effects.png')
    write_summary(result)
    return result


def write_summary(result):
    primary=result['primary_weak']
    first=result['contrasts']['48']
    joint=first['ordered-pure']['hunger']; specific=first['ordered-shuffled']['hunger']
    evidence='达到了探索性弱阳性判据' if primary else '未达到探索性弱阳性判据'
    lines=['# 芯片 22554 的新地图验证', '',
           f'预先选定第 48 刻饱食作为验证指标。有序组比纯模型平均 {joint["mean"]:+.2f} 点，比打乱组平均 {specific["mean"]:+.2f} 点。两个比较中分别有 {joint["positive_seeds"]}/6 和 {specific["positive_seeds"]}/6 个训练种子改善。本轮{evidence}。',
           '', '## 实验和判据', '',
           '五组各使用上一轮的 6 个冻结检查点，每个检查点在相同的 24 个新地图种子上运行至第 192 个原生游戏时钟刻或死亡，合计 720 局。评估开始前每个检查点先重现旧地图的前 4 次决策，并核对动作、游戏状态和动作概率。各组使用同一游戏引擎、地图开局和动作采样种子。',
           '', 'R2 沿用 22554 的校准设置，无刺激为正、有刺激为负，每步加权贡献最高为 |R1| 的 5%。打乱组在训练时按试次打乱同一刺激阶段的 MEA 窗口，评估时与有序组读取相同的留出生物窗口。每张新地图均复用同一枚类器官留下的记录。',
           '', '本次将第 48 刻饱食相对纯模型、相对打乱模型的均值均为正，且两个比较均至少有 4/6 个训练种子改善，定义为弱阳性线索。两项 95% 区间下界都高于零时，判为更强支持。探索、生命值、水分、死亡及第 192 刻均为次要指标。区间按训练种子和地图两个维度配对重采样 20,000 次，次要比较使用描述性区间。','',
           '## 同一游戏时钟刻的五组结果','',
           '| 游戏刻 | 模型 | 探索新格数 | 生命值 | 饱食 | 水分 | 死亡局数 / 144 |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for h,arms in result['means'].items():
        for arm,v in arms.items():
            lines.append(f'| {h} | {ARMS[arm]} | {v["explored"]:.2f} | {v["health"]:.2f} | {v["hunger"]:.2f} | {v["thirst"]:.2f} | {round(144*v["deaths"])} |')
    lines+=['','![五组轨迹](five-arms.png)','','## 有序信号相对两种对照的差值','',
            '| 游戏刻 | 比较 | 指标 | 平均差 | 95% 配对区间 | 改善训练种子 |','|---:|---|---|---:|---|---:|']
    for h,contrasts in result['contrasts'].items():
        for pair,label in [('ordered-pure','有序减纯模型'),('ordered-shuffled','有序减打乱')]:
            for metric,name in METRICS.items():
                v=contrasts[pair][metric];lo,hi=v['interval']
                lines.append(f'| {h} | {label} | {name} | {v["mean"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {v["positive_seeds"]}/6 |')
    lines+=['','![两种对照的配对差](paired-effects.png)','','## 第 192 刻前的进食和饮水行为','',
            '| 模型 | 进食次数 | 饮水次数 | 饱食低于 60 时进食 / 可进食决策 | 水分低于 60 时饮水 / 可饮水决策 |',
            '|---|---:|---:|---:|---:|']
    for a in ARMS:
        v=result['behavior'][a]['192']
        lines.append(f'| {ARMS[a]} | {v["eat"]} | {v["drink"]} | {v["low_food_eat"]} / {v["low_food_eligible"]} | {v["low_water_drink"]} / {v["low_water_eligible"]} |')
    lines+=['','分母是策略实际遇到的合法动作机会，会随策略的行走路线和资源储备变化。行为结果描述游戏里的补给决策。独立生物样本数为 1，本轮验证的是已有模型在新地图上的泛化。','',
            '[评估登记](protocol.json) · [逐局数据](paired-games.csv) · [逐种子差值与区间](results.json)']
    checks=OUT/'replay-checks.json'
    if checks.is_file():
        samples=read(checks)
        for sample in samples:
            require(sample['matched_ticks']==192 and sample['map_seed']==result['map_seeds'][0] and
                    digest(OUT/'runs'/sample['model']/f'{sample["map_seed"]}.json')==sample['saved_game_sha'],
                    'Independent replay does not match reported game')
        lines+=['',f'另对纯模型、有序组、打乱组各抽取一个检查点，在第一张新地图上重新生成动作，并用独立游戏进程重放。三局共 {sum(r["decisions"] for r in samples)} 次决策，每局 192 刻的四项指标逐刻吻合。[独立回放记录](replay-checks.json)']
    (OUT/'新地图验证.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    print(json.dumps({k:v for k,v in report().items() if k.startswith('primary')},ensure_ascii=False))
