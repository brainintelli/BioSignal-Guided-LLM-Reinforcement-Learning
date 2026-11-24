import csv
import hashlib
import json
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'standalone/brain_device_session'))
from report_recorded_pulses import Figure, _plot_style, _save, np
from clock_replay import require

def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
ARM={'ordered':'有序实测','shuffled':'打乱信号','pure':'纯 R1'}
COL={'ordered':'#087e79','shuffled':'#d7773a','pure':'#516381'}
MET={'cells':'探索新格数','drink':'饮水次数','eat':'进食次数','water':'水分','food':'饱食','health':'生命值',
     'drink_gain':'饮水后水分净增次数','eat_gain':'进食后饱食净增次数','death':'累计死亡比例'}
MAIN=['cells','drink','eat','water','food','health']

def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#dae0e4',alpha=.6)

def signal_plot():
    long,paired,a=(read(OUT/f) for f in ('windows.json','clean-windows.json','common-mode-sensitivity.json'))
    with _plot_style():
        fig=Figure(figsize=(10.5,4),dpi=160,layout='constrained')
        fig.suptitle('22554  ·  每个位点一次刺激的离线筛查',fontsize=15)
        left,right=fig.subplots(1,2)
        y=np.arange(3);labs=list('HWF')
        for dy,series,color,label in [(-.18,[len(long[k]['channels']) for k in labs],'#9da9b6','六窗同时稳定'),
                                       (.18,[len(paired[k]['channels']) for k in labs],'#087e79','原采集前后配对窗')]:
            left.barh(y+dy,series,height=.29,color=color,label=label)
            for i,n in enumerate(series):left.text(n+6,i+dy,str(n),va='center',fontsize=9)
        left.set(yticks=y,yticklabels=labs,xlim=(0,685),xlabel='通过的通道数（每个位点候选 685 个）')
        left.invert_yaxis();left.legend(frameon=False,loc='center right',bbox_to_anchor=(1,.29),fontsize=8)
        left.set_title('同一位点的前后窗使用同一组电极',fontsize=11)
        for i,k in enumerate(labs):
            nums=[a['results'][k][off]['retained_spikes'] for off in ('-3','12')]
            yv=np.array(nums)/len(paired[k]['channels'])
            right.plot([0,1],yv,'o-',color=['#5276a8','#dd8340','#087e79'][i],lw=2,label=k)
            for j,n in enumerate(nums):
                right.annotate(f'{k} · {n} 个',(j,yv[j]),xytext=((5 if j==1 else -5),4),
                               textcoords='offset points',ha=('left' if j==1 else 'right'),fontsize=8)
        right.set(xticks=[0,1],xticklabels=['刺激前 −3 至 −2 秒','刺激后 12 至 13 秒'],xlim=(-.4,1.4),
                  ylabel='每通道每秒保留尖峰数',title='点旁数字为剔除共模瞬变后保留的尖峰数')
        tidy(left);tidy(right)
        _save(fig,OUT/'signal-qc.png')

def collect(plan):
    seeds,maps=plan['training_seeds'],plan['maps']
    shape=(len(seeds),len(maps),max(plan['ticks'])+1)
    series={a:{k:np.zeros(shape) for k in MET} for a in ARM}
    key=sha(OUT/'clean-protocol.json')
    first={}
    inputs=read(OUT/'clean-windows.json')
    history=OUT.parent/'22554-20261118T090720Z/exploratory/extended-96x48/three-arm-six-seeds'
    baseline={r['training_seed']:r for c in ('original','added') for r in read(history/c/'pure/result.json')['seeds']}
    require(not set(maps)&(set(seeds)|set(read(OUT.parent/'22554_fresh_maps_20261119/protocol.json')['maps'])
                          |set(read(history/'original/pure/result.json')['budget']['game_eval_seeds'])),'Map overlap')
    registrations={}
    rows=[];max_r2=0.;train_steps=0
    opportunity={h:{a:{v:dict(eligible=0,taken=0,net_restored=0.) for v in ('drink','eat')} for a in ARM} for h in plan['ticks']}
    for a in ARM:
        for i,s in enumerate(seeds):
            d=OUT/'runs'/f'{a}-{s}'
            done=read(d/'done.json')
            require(done['arm']==a and done['seed']==s and set(map(int,done['games']))==set(maps),'Incomplete model')
            if a!='pure':
                tr=read(d/'training-trace.json')
                reg=read(d/'registration.json')
                trained=read(d/'trained.json')
                updates=read(d/'updates.json')
                require(reg['windows']==sha(OUT/'clean-windows.json') and reg['plan']==key and
                        trained['final']==done['state'] and trained['checkpoint']['reload_verified'] and
                        all(u['world_size']==8 and u['unique_decisions']==12 for u in updates),'Model or replay lineage differs')
                require(all(trained['initial'][k]==baseline[s]['initial'][k] for k in ('actor','base','reference')),'Warm policy differs from pure R1')
                pairs=reg['pairs']
                for source,donor,phase,wid in pairs:
                    require(wid==inputs[donor]['windows'][int(phase=='post')]['id'],'Donor window changed')
                signature=(reg['config']['normalization'],trained['initial'],sorted(p[3] for p in pairs),[p[2] for p in pairs])
                require(registrations.setdefault(s,signature)==signature,'Unequal arms')
                require(len(tr)==96 and len(read(d/'updates.json'))==8 and reg['hold']==plan['heldout'][str(s)] and
                        reg['config']['reward_mix']==plan['mix'] and reg['hold'] not in {v for p in reg['pairs'] for v in p[:2]},'Training split or budget differs')
                require(all(set((p[0],p[1]))=={p[0]} if a=='ordered' else p[0]!=p[1] for p in reg['pairs']),'Shuffle differs')
                for index,t in enumerate(tr):
                    source,donor,phase,wid=pairs[index%len(pairs)]
                    require((t['r2_trial'],t['next_state2_trial'],t['recorded_phase'],t['next_donor'])==
                            (source,donor,phase,wid) and
                            t['r2_event_sign']==(None if t['terminal'] else (1 if phase=='pre' else -1)), 'R2 and state2 alignment differs')
                    if t['r1']:max_r2=max(max_r2,abs(t['r2_contribution']/t['r1']))
                    require(abs(t['r2_contribution'])<=.05*abs(t['r1'])+1e-14,'R2 too large')
                train_steps+=len(tr)
            else:
                hist=baseline[s]
                require(done['manifest']==hist['checkpoint']['manifest_sha256'] and done['state']==hist['after'] and
                        hist['training_steps']==96 and hist['update_count']==8,'Native R1 checkpoint or budget mismatch')
            for j,g in enumerate(maps):
                p=d/f'map-{g}.json';game=read(p)
                require(sha(p)==done['games'][str(g)] and game['protocol']==key and game['model']==done['manifest']
                        and game['first']==first.setdefault(g,game['first']),'Game identity or file differs')
                z=np.array(game['ticks']);require(z.shape==(shape[2],4) and np.isfinite(z).all() and np.all(np.diff(z[:,0])>=0),'Invalid ticks')
                h=shape[2]-1
                for idx,m in enumerate(('cells','health','food','water')):series[a][m][i,j]=z[:,idx]
                end=game['terminal']
                if end is not None and end<=h:series[a]['death'][i,j,end:]=1
                for start,stop,v,eat_ok,drink_ok in game['actions']:
                    for limit in plan['ticks']:
                        if start>=limit:continue
                        for verb,legal,col_idx in [('eat',eat_ok,2),('drink',drink_ok,3)]:
                            summary=opportunity[limit][a][verb]
                            if legal and z[start,col_idx]<60:
                                summary['eligible']+=1
                                summary['taken']+=int(v==verb)
                            if v==verb and stop<=limit:summary['net_restored']+=max(0.,float(z[stop,col_idx]-z[start,col_idx]))
                    if v not in ('eat','drink') or stop>h:continue
                    require((eat_ok if v=='eat' else drink_ok) and stop>start,'Illegal food or water action')
                    series[a][v][i,j,stop:]+=1
                    col=2 if v=='eat' else 3
                    if z[stop,col]>z[start,col]+1e-6:series[a][v+'_gain'][i,j,stop:]+=1
                for tick in plan['ticks']:
                    rows.append(dict(arm=a,training_seed=s,heldout=plan['heldout'][str(s)],map=g,tick=tick,
                                     **{m:series[a][m][i,j,tick] for m in MET}))
    with (OUT/'paired-games.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save('opportunities.json',opportunity)
    return series,train_steps,max_r2

def estimate(plan,arr):
    rng=np.random.default_rng(225541120)
    n=len(plan['training_seeds']);m=len(plan['maps'])
    # Each replicate uses the same seed and map sample in both arms.
    ii=rng.integers(0,n,(20000,n,1));jj=rng.integers(0,m,(20000,1,m))
    site=np.array([plan['heldout'][str(s)] for s in plan['training_seeds']])
    res={}
    for h in plan['ticks']:
        res[h]={}
        for key in MET:
            data={a:arr[a][key][:,:,h] for a in ARM}
            entry=dict(means={a:float(z.mean()) for a,z in data.items()})
            for ctrl in ('shuffled','pure'):
                diff=data['ordered']-data[ctrl]
                seed_means=diff.mean(axis=1)
                folds={k:float(diff[site==k].mean()) for k in 'HWF'}
                pooled=diff[ii,jj].mean(axis=(1,2))
                entry[ctrl]=dict(delta=float(diff.mean()),ci=np.quantile(pooled,[.025,.975]).tolist(),
                                 per_seed=seed_means.tolist(),positive=int(sum(seed_means>1e-9)),folds=folds,
                                 map_halves=[float(diff[:,:m//2].mean()),float(diff[:,m//2:].mean())],
                                 leave_one_seed_out=((seed_means.sum()-seed_means)/(n-1)).tolist())
            entry['candidate']=all(entry[c]['delta']>1e-9 and entry[c]['positive']>=4 and min(entry[c]['folds'].values())>1e-9 for c in ('pure','shuffled')) if key!='death' else None
            res[h][key]=entry
    return res

def game_plots(plan,arr,res):
    with _plot_style():
        fig=Figure(figsize=(11.1,6.7),dpi=160,layout='constrained')
        fig.suptitle('相同游戏时长的三组表现  ·  6 个训练种子 × 12 张留存地图',fontsize=15)
        for ax,k in zip(fig.subplots(2,3).flat,MAIN):
            for a in ARM:
                ax.plot(arr[a][k].mean(axis=(0,1)),color=COL[a],lw=1.9,
                        ls={'ordered':'-','pure':':','shuffled':'--'}[a],label=ARM[a])
            ax.axvline(48,color='#889398',ls=':',lw=.8)
            ax.set(title=MET[k],xlabel='游戏时钟刻',xticks=[0,48,96,144,192],xlim=(0,192))
            if k=='cells':ax.legend(frameon=False,fontsize=8)
            tidy(ax)
        _save(fig,OUT/'game-curves.png')
        fig=Figure(figsize=(11.1,6.0),dpi=160,layout='constrained')
        fig.suptitle('第 192 刻实测组的配对差值  ·  小点为各训练种子，横线为 95% 区间',fontsize=14)
        for ax,k in zip(fig.subplots(2,3).flat,MAIN):
            ax.axvline(0,lw=.9,color='#8f9a9f',ls='--')
            for y,c in enumerate(('pure','shuffled')):
                r=res[192][k][c]
                ax.hlines(y,*r['ci'],color=COL[c],lw=2.3)
                ax.scatter(r['per_seed'],y+np.linspace(-.14,.14,6),s=16,color=COL[c],alpha=.57)
                ax.scatter(r['delta'],y,s=36,marker='D',color=COL[c])
            ax.set(title=MET[k],yticks=[0,1],yticklabels=['对纯 R1','对打乱'],ylim=(-.55,1.55),xlabel='有序实测 − 对照')
            ax.spines[['left','top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
        _save(fig,OUT/'paired-effects.png')

def report(plan,res,steps,r2):
    out=['# 22554 三个位点的离线回放对照','',
         '探索、补水和进食均未达到预定的可复现弱阳性判据。第 192 刻，有序信号比打乱组平均多探索 4.57 格，但少于纯 R1 34.25 格。饮水次数在六个种子上都低于纯 R1。进食差异接近零。','',
         '2026 年 11 月 19 日采集。每个位点一次刺激，一枚类器官。表中以每局每模型等权计算，动作按完成时刻计数。','',
         '![原始记录筛查](signal-qc.png)','',
         '每份文件先从无刺激记录中选活跃电极，排除距任何刺激点 200 µm 以内的电极。以刺激前 12 秒为参照，能同时通过全部六个窗口的通道为 H 3、W 19、F 10，低于 50 个下限。随后单独登记原采集程序使用的前后配对时窗并进行敏感性分析。刺激前 −3 至 −2 秒与刺激后 12 至 13 秒都通过检查的通道分别为 491、486、474 个。各窗均未触及 ADC 上下限。四个输入为通道平均尖峰率的对数、活跃通道比例、时间集中度与空间熵。距刺激最近的 12 秒全部屏蔽。','',
         'F 后窗原有 258 个尖峰，阵列通道中位电压峰值约 50 µV，H、W 均约 6 µV。对六个窗口统一排除通道中位电压偏移至少 4 个 ADC 码（约 25.2 µV）的时刻及其前后各 2 ms。仅 F 后窗触发，屏蔽 10.5 ms，剔除 40 个尖峰，余下 218 个进入特征计算。H、W 后窗分别保留 30、133 个。未填补被屏蔽的时间。三组游戏使用这版输入，早期配对窗处理在任何留存地图评估前停止。缺少同位点重复和假刺激记录，无法估计多少变化由刺激诱发。每份文件独立筛通道，位点和通道组成也可能影响读数。','',
         '另对三份文件均通过筛查的 268 个相同电极做了复算。按相同规则屏蔽共模瞬变，H 前后为 14 与 16 个尖峰，W 为 100 与 80 个，F 为 25 与 125 个。三个位点的变化方向与各自训练面板一致。[共同电极复算](shared-panel-check.json)','',
         '进入 state2 的刺激后特征如下。按训练折归一化前列数值，尖峰率对数为每通道每秒尖峰数加 1 后取自然对数。','',
         '| 位点 | 尖峰率对数 | 活跃通道比例 | 时间集中度 | 空间熵 |','|---|---:|---:|---:|---:|',
         *['| '+k+' | '+' | '.join(f'{x:.3f}' for x in v['windows'][1]['features'])+' |' for k,v in read(OUT/'clean-windows.json').items()],'',
         '## 训练与评估','',
         f'两台八卡机交叉分配实测与打乱两组。两组各新训练 6 次，共 {steps} 个 PPO 决策。每个模型从相同来源的暖启动策略训练 96 步，完成 8 次八卡更新。每个种子用两份文件训练，第三份整份留出，H、W、F 各留出两次。归一化只使用两份训练文件的刺激前特征。纯 R1 复用此前同种子、同预算的 6 个已训练检查点并重新评估。三个组共 18 个检查点，12 张新地图，216 局。终止后保留终止状态到固定时钟刻。','',
         f'刺激前记录对应正 R2，已确认刺激的后窗对应负 R2。两组中符号和顺序完全相同，打乱组仅交换同一前／后阶段的两份训练来源。留出评估向两组提供相同的有序实测输入。R1 权重 1，R2 幅值系数 0.5，混合权重 0.1，参考值每刻 0.009775。实际辅助项绝对值占非零 R1 的最高比例为 {r2:.2%}。这些均为记录回放，游戏动作无法改变已采集的信号。','',
         '## 游戏结果','', '![三组随时钟的变化](game-curves.png)','', '![第 192 刻的配对差值](paired-effects.png)','',
         '| 刻 | 指标 | 有序实测 | 打乱 | 纯 R1 | 实测减打乱，95% 区间 | 实测减纯 R1，95% 区间 |',
         '|---:|---|---:|---:|---:|---:|---:|']
    for h in plan['ticks']:
        for k in MET:
            v=res[h][k];a=v['means']
            cs=[f"{v[c]['delta']:+.2f} [{v[c]['ci'][0]:+.2f}, {v[c]['ci'][1]:+.2f}]" for c in ('shuffled','pure')]
            out.append(f"| {h} | {MET[k]} | {a['ordered']:.2f} | {a['shuffled']:.2f} | {a['pure']:.2f} | "+' | '.join(cs)+' |')
    out+=['','## 补给机会','',
          '以下为第 192 刻前的描述性汇总。仅在对应储备低于 60 且动作可选时统计选择率。净恢复量按已完成饮水或进食动作前后的对应储备正差求和，再除以 72 局。各组遇到的补给机会数可以不同。','',
          '| 组别 | 低水分时饮水／机会 | 低饱食时进食／机会 | 饮水净恢复量／局 | 进食净恢复量／局 |',
          '|---|---:|---:|---:|---:|']
    ops=read(OUT/'opportunities.json')['192']
    for a,name in ARM.items():
        dr,ea=ops[a]['drink'],ops[a]['eat']
        def rate(g):
            return f"{g['taken']}/{g['eligible']}（{g['taken']/g['eligible']:.1%}）" if g['eligible'] else '0/0'
        out.append(f"| {name} | {rate(dr)} | {rate(ea)} | {dr['net_restored']/72:.2f} | {ea['net_restored']/72:.2f} |")
    out+=['','三组都没有遇到水分低于 60 且可以饮水的决策点，本轮无法检验严重缺水时是否更主动找水。与纯 R1 相比，实测组和打乱组都更少饮水；这三组对照无法将这一差异单独归因于 R2 或 MEA 输入。','',
          '## 弱阳性判读','',
          '以第 192 刻为主。探索、饮水、进食逐项判断。有序组须同时高于两个对照，两个比较各至少 4/6 个训练种子为正，且三种留出位点的平均差分别为正。区间为训练种子与地图两个维度配对重采样 20,000 次，未作多指标校正，也无法计入类器官间差异。留出位点与种子部分绑定，每个位点仅两种子。']
    for k in ['cells','drink','eat','drink_gain','eat_gain','water','food']:
        v=res[192][k]
        out.append(f"\n- {MET[k]}{'达到' if v['candidate'] else '未达到'}探索性方向判据。相对打乱、纯 R1 分别有 {v['shuffled']['positive']}/6、{v['pure']['positive']}/6 个正向种子。")
    x=res[192]['cells']['shuffled']
    idx=int(np.argmax(x['per_seed']))
    out+=['',f"有序相对打乱的探索均值差为 {x['delta']:+.2f} 格。差值最大的种子为 {plan['training_seeds'][idx]}（{x['per_seed'][idx]:+.2f} 格）。六个种子都保留在主分析中；单独检验这个种子的影响时，另外五个种子的平均差为 {x['leave_one_seed_out'][idx]:+.2f} 格。"]
    out+=['','按原登记的地图顺序分为前六张和后六张，另列有序减对照的均值，以检查结论对地图的敏感度。','',
          '| 第 192 刻 | 前六图对打乱 | 后六图对打乱 | 前六图对纯 R1 | 后六图对纯 R1 |','|---|---:|---:|---:|---:|']
    for k in ['cells','drink','eat']:
        vals=[n for c in ('shuffled','pure') for n in res[192][k][c]['map_halves']]
        out.append('| '+MET[k]+' | '+' | '.join(f'{n:+.2f}' for n in vals)+' |')
    out+=['','[预登记主方案](protocol.json) · [短时窗探索方案](paired-protocol.json) · [训练前冻结方案](clean-protocol.json) · [六窗质控](windows.json) · [最终特征及通道](clean-windows.json) · [共模瞬变屏蔽明细](common-mode-sensitivity.json) · [逐局数据](paired-games.csv) · [按种子及留出位点的统计](results.json)']
    (OUT/'实验结果.md').write_text('\n'.join(out)+'\n',encoding='utf8')

if __name__=='__main__':
    signal_plot()
    if '--qc' not in sys.argv:
        plan=read(OUT/'clean-protocol.json')
        arr,steps,ratio=collect(plan)
        res=estimate(plan,arr)
        save('results.json',dict(steps=steps,max_weighted_r2_fraction=ratio,metrics=res))
        game_plots(plan,arr,res)
        report(plan,res,steps,ratio)
        print({k:{a:res[192][k][a]['delta'] for a in ('shuffled','pure')} for k in MAIN})
