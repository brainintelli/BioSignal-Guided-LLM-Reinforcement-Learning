"""Show survival and the cumulative fraction that has actually drunk by each game tick."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def plot(g):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    f=Path('C:/Windows/Fonts/msyh.ttc')
    if f.is_file():
        font_manager.fontManager.addfont(str(f));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#adbec0','axes.unicode_minus':False,'axes.labelcolor':'#243e42'})
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.4))
    fig.subplots_adjust(top=.73,bottom=.22,left=.07,right=.97,wspace=.22)
    for ax,key,title in [(left,'alive_by_tick','仍然存活的比例'),(right,'first_drink_by_tick','已首次喝水的比例')]:
        for arm,name,color,style in [('r1','仅 R1','#6a789c','-'),('ordered','R1 + 缩放 R2','#178075','--')]:
            ax.step(range(61),[100*v for v in g[arm][key]],where='post',label=name,color=color,ls=style,lw=2.2)
        ax.set(xlim=(0,60),ylim=(-2,103),xticks=[0,10,20,30,40,50,60],yticks=[0,25,50,75,100],xlabel='游戏刻',ylabel='%')
        ax.grid(axis='y',alpha=.16);ax.set_title(title,loc='left',fontweight='bold')
    left.legend(loc='lower left',frameon=False,fontsize=10)
    fig.suptitle('芯片 22021  |  濒渴开局的离线游戏表现',x=.045,y=.98,ha='left',color='#203e45',fontsize=17,fontweight='bold')
    fig.text(.045,.86,'8 个训练种子 × 48 张新地图，两个冻结模型条件各 384 局',color='#50696c')
    fig.text(.045,.04,'两条曲线都以该组全部对局为分母。喝水以水分确实增加的动作开始刻计时，死亡前未喝水的角色保留在分母内。',fontsize=9,color='#50696c')
    fig.savefig(HERE/'存活与首次喝水.png',dpi=180,facecolor='white');plt.close(fig)

def main():
    r=read(HERE/'comparison.json');g=r['groups'];effects=r['effects'];d=r['paired_drinks']
    conclusion='三项均达到预设的稳定改善标准。' if r['stable_all'] else '三项结果未同时达到预设的稳定改善标准。'
    lines=['# 缩放 R2 重训模型的濒渴开局对照','',conclusion,'',
        '芯片 22021 的两组冻结检查点均来自上轮等预算 PPO 重训。两组都读取真实 MEA。开局生命 25，水分 1.6，饱食 95。在 48 张从未用于训练或既有留存比较的地图上，逐一配对 8 个训练种子。每局至死亡或第 60 刻结束。','',
        '![存活与首次喝水的时间曲线](存活与首次喝水.png)','','| 指标 | 仅 R1 | R1 + 缩放 R2 | 两组配对差 [98.33% 区间] | 改善的训练种子 |','|---|---:|---:|---:|---:|']
    for k,label in [('survived','第 60 刻存活率'),('end_tick','平均观察存活时长（刻）'),('wait_to_drink_60','全体平均首次喝水等待（刻，未喝计 60）')]:
        a=g['r1'][k];b=g['ordered'][k];e=effects[k];lo,hi=e['interval']
        if k=='survived':
            v1=f"{g['r1']['alive']}/384，{100*a:.2f}%";v2=f"{g['ordered']['alive']}/384，{100*b:.2f}%"
            diff=f"{e['delta']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] 个百分点"
        else:
            v1=f"{a:.2f}";v2=f"{b:.2f}";diff=f"{e['delta']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        lines.append(f"| {label} | {v1} | {v2} | {diff} | {e['favorable_seeds']}/8 |")
    v=effects['survived'];e=effects['wait_to_drink_60'];aud=r['reward_audit']
    neg=sum(row['labels']['-1'] for key,row in aud.items() if key.endswith('/ordered'))
    pos=sum(row['labels']['1'] for key,row in aud.items() if key.endswith('/ordered'))
    n_eval=max(row['max_r2_fraction'] for row in aud.values());n_train=max(read(HERE/'design.json')['training_max_ratio'].values())
    lines+=['','差值均为缩放 R2 组减去仅 R1 组。存活率和观察存活时长越高越好，首次喝水等待越短越好。按训练种子和游戏地图交叉配对重采样 40,000 次，三个终点分别使用双侧 98.33% 区间，控制三项同时判断的名义错误率。单项判为稳定改善，需区间完全朝有利方向且至少 6/8 个训练种子朝有利方向。','',
        f"384 对游戏中，仅缩放 R2 组存活 {v['favorable_pairs']} 对，仅 R1 组存活 {v['unfavorable_pairs']} 对，存活结局相同 {v['ties']} 对。",'','## 首次喝水时间','',
        f"仅 R1 组有 {g['r1']['drank']}/384 局实际喝到水，缩放 R2 组有 {g['ordered']['drank']}/384 局。两组都喝到水的有 {d['both_drank']} 对，只有缩放 R2 组喝到水 {d['only_ordered']} 对，只有仅 R1 组喝到水 {d['only_control']} 对，两组都没喝到 {d['neither']} 对。",
        '',f"在各组实际喝到水的角色中，仅 R1 与缩放 R2 的平均首次喝水刻分别是 {g['r1']['actual_drink_mean']:.2f} 和 {g['ordered']['actual_drink_mean']:.2f}，中位数分别是 {g['r1']['actual_drink_median']:.1f} 和 {g['ordered']['actual_drink_median']:.1f}。两组都喝到水的配对中，首次喝水时间的平均差为 {d['both_drink_time_diff']:+.2f} 刻。这些描述只适用于相应子集。",
        '','主要时间指标保留未喝到水的对局。成功喝水要求游戏执行 drink 动作且水分实际增加，以该动作开始刻计时；第 60 刻前未喝到水（包括先死亡）记 60，表示观察期内未达成。逐局数据另存确认水分增加的结束刻。','',
        '## 奖励与回放核验','',f"复核了两组全部 768 个训练动作及 16 份检查点。缩放 R2 组在训练和本次离线记分中的最大单步贡献分别为 |R1| 的 {n_train*100:.2f}% 与 {n_eval*100:.2f}%，仅 R1 组贡献为零。评估中冻结策略，只观察游戏结果；离线记分未进入模型输入或更新。按原训练标准，缩放 R2 组本次有 {pos} 个正反馈、{neg} 个负反馈。判定正确需要新增探索格且生命、水分、饱食均不低于 85。开局本身的生命和水分很低。",'',
        f"两组在同地图、同游戏刻使用相同动作采样随机数。输入来自同一颗类器官，取当初未用于 PPO 的第 13–18 次物理记录，每次在执行回执后取 [6,16) 秒的 10 个一秒窗口，按原顺序拼接为 60 窗；其中有 6 个窗口也曾用于上轮短程评估。沿用原训练 ROI 和归一化，仅按动作开始刻取窗。本次独立性来自新地图，生物样本数为 1。通过原生游戏与观测副本逐动作核对了 {r['action_decisions']:,} 次转移。",'',
        '[预先登记的设计](design.json) · [逐局结果与喝水时间](episodes.csv) · [逐地图配对数据](paired.csv) · [区间、轨迹与奖励核验汇总](comparison.json)']
    (HERE/'濒渴对照结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    plot(g)
if __name__=='__main__':main()
