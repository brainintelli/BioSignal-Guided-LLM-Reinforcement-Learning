"""Render the observed paired effects and game survival from the frozen-checkpoint runs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf8'))
labels={'trained_ordered':'自发 MEA 原顺序','trained_shuffled':'同模型，分块打乱','trained_masked':'同模型，屏蔽 MEA','control_masked':'屏蔽 MEA 训练及评估','initial_ordered':'训练起点，原顺序 MEA'}
names={'trained_shuffled':'同权重打乱时序','control_masked':'独立训练的屏蔽组','initial_ordered':'PPO 更新前','trained_masked':'同权重屏蔽输入'}
modes=list(labels)
comparators=['trained_shuffled','control_masked','initial_ordered','trained_masked']

def interval(e,mult,dec):
    ci=e['interval'];base=f"{e['mean']*mult:+.{dec}f}"
    return base if ci is None else base+f" [{ci[0]*mult:+.{dec}f}, {ci[1]*mult:+.{dec}f}]"

def figure(a,b,c):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11.5,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#c6d2d4','axes.labelcolor':'#244147','xtick.color':'#41585c','ytick.color':'#244147'})
    fig,axs=plt.subplots(1,2,figsize=(12.0,4.8),gridspec_kw={'width_ratios':[1.12,1]})
    fig.subplots_adjust(left=.18,right=.97,bottom=.19,top=.78,wspace=.61)
    ax=axs[0];ax.axvline(0,color='#637777',lw=1,ls='--')
    for i,m in enumerate(comparators):
        e=a['effects']['trained_ordered__'+m]['r1'];x=e['mean'];lo,hi=e['interval']
        ax.plot([lo,hi],[i,i],color='#287d87',lw=3,solid_capstyle='round');ax.scatter([x],[i],s=55,color='#144a52',zorder=3)
    ax.set(yticks=list(range(4)),yticklabels=[names[m] for m in comparators],ylim=(3.65,-.65),xlabel='原顺序组减去对应组的平均 R1')
    ax.set_title('标准开局  ·  等时长游戏回报差',loc='left',fontweight='bold',pad=18)
    ax.grid(axis='x',alpha=.15)
    ax=axs[1]; groups=[('trained_ordered',a),('control_masked',a),('trained_ordered',b),('control_masked',b),('trained_ordered',c),('control_masked',c),('trained_shuffled',c)]
    for i,(m,d) in enumerate(groups):
        v=d['mode_means'][m];pct=100*v['survived'];n=v['n'];y=[5.9,5.3,4.1,3.5,2.3,1.7,1.1][i]
        ax.barh(y,pct,height=.42,color='#337f88' if m=='trained_ordered' else '#b59261' if m=='trained_shuffled' else '#96a5b2')
        ax.text(max(pct,0)+2,y,f"{n-v['deaths']}/{n}",va='center',fontsize=10)
    ax.set(yticks=[5.9,5.3,4.1,3.5,2.3,1.7,1.1],yticklabels=['标准 · 原顺序','标准 · 屏蔽','缺水 · 原顺序','缺水 · 屏蔽','濒渴 · 原顺序','濒渴 · 屏蔽','濒渴 · 打乱'],xlim=(0,123),xticks=[0,25,50,75,100],xlabel='第 60 个游戏刻存活率（%）',ylim=(.55,6.45))
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('游戏存活  ·  三种开局',loc='left',fontweight='bold',pad=18)
    fig.suptitle('芯片 22021  |  冻结模型的存档回放',x=.032,y=.97,ha='left',fontsize=18,fontweight='bold',color='#183c42')
    fig.text(.03,.035,'左图横线为种子与地图交叉配对重采样区间（99.17%）。右图标注存活局数 / 总局数。三种场景使用同一颗类器官的存档。',fontsize=9,color='#52666a')
    fig.savefig(ROOT/'离线存活与回报.png',dpi=180,facecolor='white');plt.close(fig)

def main():
    a=read(ROOT/'comparison.json'); b=read(ROOT/'stress/comparison.json'); c=read(ROOT/'severe/comparison.json')
    lines=['# 用芯片 22021 的存档复测游戏存活和训练效果','','原测试固定六次动作，各组实际覆盖的游戏时间不同。本次冻结现有权重，在相同游戏时间上重新比较。三个游戏场景合计 '+str(a['episodes']+b['episodes']+c['episodes'])+' 局，使用同一颗类器官的同一段静默信号。','','![冻结模型回报与存活](离线存活与回报.png)','','## 标准开局的留存结果','','8 个独立训练种子，8 张新地图。训练起点是共用的一份检查点，在每张地图实际推理一次。其余条件各有 64 局。止于首次死亡或第 60 个游戏刻，开局生命 100、水分和饱食各 95。','','| 训练与评估输入 | 第 60 刻存活 | 每局 R1 | 新增探索格数 | 期末生命 | 期末水分 |','|---|---:|---:|---:|---:|---:|']
    for m in modes:
        v=a['mode_means'][m];lines.append(f"| {labels[m]} | {v['n']-v['deaths']}/{v['n']} | {v['r1']:.5f} | {v['new_cells']:.2f} | {v['health']:.2f} | {v['thirst']:.2f} |")
    lines+=['','| 原顺序训练组减去比较组 | R1 差 [区间] | 探索格数差 [区间] |','|---|---:|---:|']
    for m in comparators:
        e=a['effects']['trained_ordered__'+m]
        lines.append(f"| {names[m]} | {interval(e['r1'],1,5)} | {interval(e['new_cells'],1,2)} |")
    lines+=['','每个差值都在相同的游戏地图与动作采样随机数下计算。按训练种子和地图交叉配对重采样 30,000 次。前三种比较的 R1 和探索共六项，各用 99.17% 区间。最后一行是附加诊断，采用相同区间水平。区间仅反映该记录下模型种子和游戏地图的变异。','','## 游戏内缺水测试','','固定开局生命 25、水分 8、饱食 95。另取 8 张新地图和事先指定的前 4 个训练种子。主要结果为第 60 刻是否存活。达到上限仍存活的对局，其观测时长计为 60 刻。各组共享游戏规则、开局资源与动作采样随机数。','','| 训练与评估输入 | 第 60 刻存活 | 平均观测存活刻数（上限 60） | 期末生命 | 期末水分 |','|---|---:|---:|---:|---:|']
    for m in ('trained_ordered','control_masked'):
        v=b['mode_means'][m];lines.append(f"| {labels[m]} | {v['n']-v['deaths']}/{v['n']} | {v['end_tick']:.2f} | {v['health']:.2f} | {v['thirst']:.2f} |")
    e=b['effects']['trained_ordered__control_masked'];s=e['survived']
    lines+=['',f"原顺序组存活率减去屏蔽组为 {interval(s,100,2)} 个百分点。32 对里，仅原顺序组存活 {s['pairs_higher']} 对，仅屏蔽组存活 {s['pairs_lower']} 对，结局相同 {s['ties']} 对。平均观测存活刻数差为 {interval(e['end_tick'],1,2)} 刻。压力测试的区间使用 95% 配对重采样，训练种子只有四个。若无存活结局不同的配对，表中仅给出观察差。",'','## 回放与解释范围','','- 信号来自首次非零刺激前约 65 秒的静默 H5。从第 2 秒开始取连续 60 个一秒窗。与用于训练和标准化的后续记录在硬件帧上分离。这段静默记录曾用于选电极，来自同一生物样本。','- 按电极物理编号和坐标对齐 322 个非刺激电极，沿用训练时的增益、采样率、尖峰阈值与标准化。四项输入是对数放电率、活跃通道比例、时间集中度和空间熵。分块打乱以五秒为一块，同时置换四项 MEA 特征；同一局中每个位置最多读取一次。','- 每个游戏刻对应一个预录窗口，在动作开始刻读取。多刻动作跨过的窗口跳过。两个游戏实例逐动作核对其原生转移和观察，只在最后一个动作跨越第 60 刻时取边界快照并截算 R1。离线重放没有动作到类器官的即时反馈。','- 本次只计算 R1。所有模型冻结，策略和价值模型的参数摘要在评估前后核对。训练起点对照检验已有 PPO 更新，原顺序与打乱对照检验这批权重是否利用已录信号的时间顺序，独立屏蔽训练组评估完整接入方式的差值。','- 此次用的是静默段。原有刺激效果请查看上轮的规律、随机及零输出配对比较。用已有刺激记录复核时须按完整实验轮次留出，固定同一回报定义，区分刺激导致的神经响应和游戏模型从该响应获得的收益。','',f"逐动作审计合计 {a['action_decisions']+b['action_decisions']+c['action_decisions']:,} 次。[标准开局逐局结果](episodes.csv) · [缺水逐局结果](stress/episodes.csv) · [标准开局区间和文件摘要](comparison.json) · [缺水区间和文件摘要](stress/comparison.json) · [标准设计](design.json) · [缺水设计](stress/design.json) · [既有刺激组对照](../离线对照结果.md)"]
    insert=lines.index('## 回放与解释范围')
    high=['## 可出现死亡的濒渴开局','','缺水先导场景中，即使整局不喝水，水分从 8 只降到 3.2。保留先导结果后，另在全新地图使用生命 25、水分 1.6、饱食 95。锁定参数后，用独立开发地图上的持续等待策略验证了角色在第 33 刻死亡。再在 8 张留存地图、4 个训练种子上评估三个条件。主要比较为真实顺序相对独立屏蔽训练组，分块打乱为附加诊断。','','| 条件 | 第 60 刻存活 | 平均观测存活刻数（上限 60） |','|---|---:|---:|']
    for m in ('trained_ordered','trained_shuffled','control_masked'):
        v=c['mode_means'][m];high.append(f"| {labels[m]} | {v['n']-v['deaths']}/{v['n']} | {v['end_tick']:.2f} |")
    for other in ('control_masked','trained_shuffled'):
        e=c['effects']['trained_ordered__'+other];u=e['survived']
        high+=['',f"真实顺序相对{labels[other]}的存活率差为 {interval(u,100,2)} 个百分点，平均观测存活时长差为 {interval(e['end_tick'],1,2)} 刻。32 对中仅真实顺序存活 {u['pairs_higher']} 对，仅对照存活 {u['pairs_lower']} 对，其余 {u['ties']} 对结局相同。区间为游戏地图与训练种子交叉配对重采样的 95% 区间。"]
    high+=['','[濒渴逐局结果](severe/episodes.csv) · [区间与核验摘要](severe/comparison.json) · [测试设计](severe/design.json) · [等待策略的开发地图核验](severe/idle-check.json)','']
    lines[insert:insert]=high
    lines[2:2]=['**濒渴开局中，真实顺序 MEA 组 13/32 局存活，屏蔽组 14/32 局，打乱组 10/32 局。真实顺序相对屏蔽组少存活 1 局；相对打乱组多存活 3 局。这批数据尚不能确认接入 MEA 提高生存率。**','','标准开局中，真实顺序组相对屏蔽训练组平均 R1 高 0.01500（约 2.7%），新增探索格数多 8.00，两项校正区间均跨零。与 PPO 更新前的同输入策略相比，R1 均差为 +0.00045，探索格数均差为 −13.66，两项区间也跨零。','']
    lines+=['','## 仅用存档继续验证','','保持模型和濒渴开局固定，在另一批从未筛选的地图上复测，使用全部 8 个训练种子。事先锁定主要比较为原顺序组对屏蔽训练组、终点为第 60 刻存活。先导中 32 对有 5 对结局不同，这可用于粗估增加游戏世界数后的精度。扩大地图数能提高同一条记录下的统计精度。','','若问题是刺激有没有额外收益，可在已有规律刺激、随机刺激、零输出的完整留存轮次上单独比较，统一 PPO 预算和 R2 权重，避免同一物理轮次的时间窗进入两端。各组先回放相同长度的真实片段，再统一屏蔽后续 MEA，跑到同一游戏时限。这种设置检验短时输入以及既有训练对后续游戏存活的影响。要区分 R2 奖励塑形本身的贡献，可在同一信号回放条件下再比较仅 R1 与 R1 加 R2，最终仍按原生 R1 和存活评价。']
    (ROOT/'存档离线验证.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    figure(a,b,c)
if __name__=='__main__':main()
