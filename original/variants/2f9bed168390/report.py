"""Publish the confirmatory outcomes alongside the separate pilot."""
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
NAMES={'trained_ordered':'真实顺序 MEA','control_masked':'训练与评估均屏蔽 MEA','trained_shuffled':'同模型分块打乱 MEA'}
ORDER=tuple(NAMES)
METRICS={'survived':'存活率差（百分点）','end_tick':'平均观察存活刻数差','r1':'每局原生游戏回报差','new_cells':'新增探索格数差'}

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def endpoints(result,control):
    lower,upper=result['effects_ordered_minus'][control]['survived']['ci95']
    if lower>0:return 'higher'
    if upper<0:return 'lower'
    return 'uncertain'

def figure(result,pilot):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager,pyplot as plt
    f=Path('C:/Windows/Fonts/msyh.ttc')
    if f.exists():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#abbdbe','axes.unicode_minus':False,'font.size':11,'axes.labelcolor':'#213c44','ytick.color':'#213c44','xtick.color':'#41565b'})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.4),gridspec_kw={'width_ratios':[1.0,1.04]})
    fig.subplots_adjust(left=.16,bottom=.24,top=.77,right=.98,wspace=.49)
    for i,m in enumerate(ORDER):
        data=result['modes'][m];x=100*data['survived'];color=['#147477','#869bab','#bc873f'][i]
        ax.barh(i,x,.5,color=color)
        ax.text(x+1.0,i,f"{data['alive']}/{data['n']}   {x:.1f}%",va='center',fontsize=10)
    ax.set(yticks=range(3),yticklabels=['真实顺序','屏蔽 MEA','分块打乱'],ylim=(2.6,-.6),xlabel='第 60 刻存活率（%）',xlim=(0,min(133,max(70,max(100*r['survived'] for r in result['modes'].values())+29))))
    ax.set_title('48 张新地图的生存结果',loc='left',fontweight='bold',pad=13)
    ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    bx.axvline(0,color='#6c8084',ls='--',lw=1)
    for idx,(m,grp) in enumerate([('control_masked','pilot'),('control_masked','repeat'),('trained_shuffled','pilot'),('trained_shuffled','repeat')]):
        eff=pilot['effects']['trained_ordered__'+m]['survived'] if grp=='pilot' else result['effects_ordered_minus'][m]['survived']
        lo,hi=[z*100 for z in (eff['interval'] if grp=='pilot' else eff['ci95'])]
        y=[0,.65,1.85,2.5][idx];color='#9caeb0' if grp=='pilot' else '#147477'
        bx.plot([lo,hi],[y,y],color=color,lw=2.7,solid_capstyle='round')
        bx.scatter(eff['mean']*100,y,color=color,s=38,zorder=3)
    bx.set(yticks=[0,.65,1.85,2.5],yticklabels=['屏蔽 · 上轮','屏蔽 · 本轮','打乱 · 上轮','打乱 · 本轮'],ylim=(3,-.45),xlabel='真实顺序组减去对照组（百分点）')
    bx.grid(axis='x',alpha=.13)
    bx.set_title('同地图、同训练种子的配对差',loc='left',fontweight='bold',pad=13)
    fig.suptitle('芯片 22021  |  濒渴开局的存活复测',x=.035,y=.975,ha='left',fontweight='bold',fontsize=18,color='#173e47')
    fig.text(.035,.055,'本轮每组 384 局，共 8 个训练种子。右图横线为地图与训练种子交叉配对重采样的 95% 区间。两轮使用同一段存档 MEA。',color='#53696d',fontsize=10)
    fig.savefig(HERE/'濒渴存活对照.png',dpi=180)
    plt.close(fig)

def main():
    obj=read(HERE/'comparison.json');pilot=read(HERE.parent/'severe/comparison.json')
    primary=endpoints(obj,'control_masked')
    summary={'higher':'本轮真实 MEA 组的第 60 刻存活率高于屏蔽组，配对差区间下界高于零。','lower':'本轮真实 MEA 组的第 60 刻存活率低于屏蔽组，配对差区间上界低于零。','uncertain':'本轮未能确认接入 MEA 能提高游戏存活率。主要比较的配对区间包含零。'}[primary]
    lines=['# 芯片 22021 的濒渴开局复测','',summary,'','使用一段刺激前的类器官静默记录，回放同一批冻结模型，在 48 张新地图上比较 8 个训练种子。角色从生命 25、水分 1.6、饱食 95 开始，观察到死亡或第 60 个游戏刻。单局每个存档时间窗最多使用一次。','',
           '![实验结果](濒渴存活对照.png)','','| 条件 | 活到第 60 刻 | 存活率 | 平均观察存活刻数 | 原生游戏回报 R1 | 新增探索格数 |',
           '|---|---:|---:|---:|---:|---:|']
    for m in ORDER:
        r=obj['modes'][m]
        lines.append(f"| {NAMES[m]} | {r['alive']}/{r['n']} | {100*r['survived']:.2f}% | {r['end_tick']:.2f} | {r['r1']:.4f} | {r['new_cells']:.2f} |")
    lines+=['','## 真实顺序相对两种对照的差值','','| 指标 | 相对屏蔽组，差值 [95% 区间] | 相对打乱组，差值 [95% 区间] |','|---|---:|---:|']
    for k,title in METRICS.items():
        labels=[]
        for m in ('control_masked','trained_shuffled'):
            r=obj['effects_ordered_minus'][m][k]
            lo,hi=r['ci95'];scale=100 if k=='survived' else 1
            prec=4 if k=='r1' else 2
            labels.append(f"{scale*r['mean']:+.{prec}f} [{scale*lo:+.{prec}f}, {scale*hi:+.{prec}f}]")
        lines.append('| '+title+' | '+' | '.join(labels)+' |')
    for m in ('control_masked','trained_shuffled'):
        r=obj['effects_ordered_minus'][m]['survived']
        lines+=['',f"真实顺序对{NAMES[m]}的 384 对游戏中，仅真实顺序存活 {r['higher']} 对，仅对照存活 {r['lower']} 对，{r['ties']} 对结局相同。按训练种子分别汇总，真实顺序组在 {r['positive_seeds']}/8 个种子中存活率更高。"]
    lines+=['','## 对上轮线索的复核','','上轮只有 8 张地图、4 个训练种子。真实顺序组 13/32，屏蔽组 14/32，打乱组 10/32。复测使用全部 8 个既有训练种子（含上轮的 4 个），48 张地图都未进入上轮、先导开局或已登记的训练世界。结论依据本轮单独估计，上轮数据单列。','','## 证据范围','','- 主要终点与判据在推理开始前锁定。真实顺序与独立屏蔽训练组在第 60 刻的存活率差是主要比较。同一权重下打乱 5 秒块的时间顺序是时序诊断。平均存活刻数截于 60，R1 和探索作为辅助指标。','- 以同一训练种子、同一游戏地图配对，对这两个因素分别重采样 30,000 次，给出双侧 95% 百分位区间。辅助比较使用各自未经多重校正的区间。','- 使用同一颗类器官及同一段 60 秒的历史信号。它曾用于挑选电极，未用于 PPO 更新或标准化拟合。所有模型、物理电极映射、输入窗口、归一化、动作采样和游戏时限沿用上一轮；地图和训练种子决定游戏估计精度。此轮只能检验该存档下的离线游戏表现。','- 本轮使用刺激前的记录，模型奖励只有原生 R1（α=1，R2 权重为 0）。刺激条件和其他类器官的泛化效果需要独立对照。','',f"共 {obj['episodes']:,} 局，{obj['decisions']:,} 次动作。原生游戏与可读取截止刻快照的副本逐动作核对。复核了模型权重摘要和每个动作实际接收的信号。",'','[试验登记](design.json) · [逐局数据](episodes.csv) · [逐种子逐地图配对数据](paired.csv) · [差值、区间与文件摘要](comparison.json) · [上一轮结果](../存档离线验证.md)']
    diagnostics=obj['posthoc_drinking_check']
    assert len(diagnostics)==3 and all(d['drank_and_survived']+d['no_drink_and_died_at_33']==384 and d['first_drink_max']<=32 for d in diagnostics.values())
    n_drink=sum(d['drank_and_survived'] for d in diagnostics.values())
    n_died=sum(d['no_drink_and_died_at_33'] for d in diagnostics.values())
    where=lines.index('## 证据范围')
    lines[where:where]=['## 存活差异对应的游戏行为','',f"事后复核逐动作日志，{n_drink} 个存活角色都在第 33 刻前启动了首次喝水动作；{n_died} 个死亡角色均未执行喝水动作，并在第 33 刻死亡。真实顺序组比屏蔽组少 34 局及时喝水。这个结果定位了游戏行为的差异，尚不能单独判断其来自训练权重变化还是推理时加入 MEA。",'']
    (HERE/'濒渴开局复测.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    figure(obj,pilot)
if __name__=='__main__': main()
