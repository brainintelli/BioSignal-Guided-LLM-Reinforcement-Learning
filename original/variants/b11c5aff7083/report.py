"""Render survival, resource use and exploration in the three registered start conditions."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
import numpy as np
H=Path(__file__).resolve().parent
SC={'low_health':'低生命','thirst':'濒渴','hunger':'濒饿'}; AR={'ordered':'真实顺序','masked':'屏蔽','shuffled':'分块打乱'}
def fmt(e,scale=1):
    return f"{scale*e['delta']:+.2f} [{scale*e['lo']:+.2f}, {scale*e['hi']:+.2f}]"
def plot(r):
    fp=Path('C:/Windows/Fonts/msyh.ttc')
    if fp.exists():
        font_manager.fontManager.addfont(str(fp));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.4));fig.subplots_adjust(left=.079,right=.98,wspace=.22,hspace=.48,top=.78,bottom=.12)
    cols=['#16766f','#8091a6','#ae8762'];x=np.arange(3)
    for ax,(key,title,unit) in zip(axs.flat,[('survived','第 60 刻存活率','%'),('cells','第 60 刻或死亡前新增探索格数','格'),('drink_wait','首次成功补水的平均限制等待时间','游戏刻'),('eat_wait','首次成功进食的平均限制等待时间','游戏刻')]):
        fac=100 if key=='survived' else 1
        for j,arm in enumerate(AR):
            vals=[r['groups'][sc][arm][key]*fac for sc in SC]
            ax.bar(x+(j-1)*.235,vals,width=.21,color=cols[j],label=AR[arm])
        ax.set_xticks(x,list(SC.values()));ax.set_ylabel(unit);ax.set_title(title,loc='left',fontsize=12,weight='bold')
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.17);ax.spines['left'].set_visible(False)
        if key=='survived':ax.set_ylim(0,104)
        if key.endswith('_wait'):ax.set_ylim(0,62)
    fig.suptitle('芯片 22021  |  食水压力下的游戏生存决策',x=.04,y=.982,ha='left',fontsize=18,weight='bold',color='#1c3745')
    fig.text(.04,.915,'同一批冻结模型、64 张新地图、8 个训练种子，每个开局每组 512 局。',color='#486474',fontsize=11)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.034,.895),frameon=False,ncol=3)
    fig.text(.04,.035,'首次事件需游戏确认资源实际增加。60 刻内未成功（包括提前死亡）按 60 计；等待时间越短越好。',fontsize=10,color='#486474')
    fig.savefig(H/'三种开局对照.png',dpi=180,facecolor='white');plt.close(fig)
def main():
    r=json.loads((H/'comparison.json').read_bytes());s=json.loads((H/'design.json').read_bytes());gs=r['groups'];effects=r['effects']
    stable=[SC[sc] for sc in SC if effects[sc]['masked']['survived']['stable_gain']]
    specific=[SC[sc] for sc in SC if all(effects[sc][a]['survived']['stable_gain'] for a in ('masked','shuffled'))]
    hurt=[SC[sc] for sc in SC if effects[sc]['masked']['survived']['stable_harm']]
    lines=['# 芯片 22021 的生存决策离线对照','']
    if stable:lines.append('真实顺序相对屏蔽组的第 60 刻存活率达到预设提升标准的场景为'+'、'.join(stable)+'。')
    else:lines.append('三个开局中，真实顺序相对屏蔽组的第 60 刻存活率均未达到预设稳定提升标准。')
    if specific:lines.append('同时优于两项对照的场景为'+'、'.join(specific)+'。')
    if hurt:lines.append('真实顺序相对屏蔽组的存活率在'+'、'.join(hurt)+'出现了跨种子的一致下降。')
    for sc,verb,k in [('thirst','补水','drink_wait'),('hunger','进食','eat_wait')]:
        v=effects[sc]['masked']['survived'];t=effects[sc]['masked'][k]
        lines+=['',f"{SC[sc]}时，真实相对屏蔽组的存活率差为 {fmt(v,100)} 个百分点，首次成功{verb}的全体平均等待差为 {fmt(t)} 刻。"]
    for sc in SC:
        u=effects[sc]['masked']['cells'];v=effects[sc]['shuffled']['cells']
        if u['lo']>0:
            lines+=['',f"{SC[sc]}开局的探索次要指标中，真实相对屏蔽多 {u['delta']:.2f} 格（描述性 95% 区间 {u['lo']:.2f} 至 {u['hi']:.2f}），{u['favorable_seeds']}/8 个种子方向一致；相对打乱的差值为 {fmt(v)} 格。"]
    lines+=['','![三种开局的存活、首次补水和进食、探索](三种开局对照.png)','','## 各组的实际表现','',
        '| 开局 | MEA 输入 | 第 60 刻存活 | 首次补水平均等待（刻） | 成功补水局 | 首次进食平均等待（刻） | 成功进食局 | 平均新增格数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for sc in SC:
        for a in AR:
            g=gs[sc][a];lines.append(f"| {SC[sc]} | {AR[a]} | {g['alive']}/512（{g['survived']:.2%}） | {g['drink_wait']:.2f} | {g['drink_count']}/512 | {g['eat_wait']:.2f} | {g['eat_count']}/512 | {g['cells']:.2f} |")
    lines+=['','首次成功以合法 drink / eat 动作中原生游戏首次确认水分或饱食度高于动作开始值的游戏刻为准。平均等待包括全部 512 局，当局 60 刻前未成功或先死亡均计 60。成功者的实际时间见下表及逐局数据。探索为到第 60 刻或死亡前新揭示的地图格数。','',
            '| 开局 | 成功补水的首次时间中位数，真实 / 屏蔽 / 打乱 | 成功进食的首次时间中位数，真实 / 屏蔽 / 打乱 |','|---|---:|---:|']
    def med(sc,key):
        vals=[gs[sc][a][key+'_confirmed_median'] for a in AR]
        return ' / '.join('无成功局' if v is None else f'{v:g}' for v in vals)
    for sc in SC:lines.append(f"| {SC[sc]} | {med(sc,'drink')} | {med(sc,'eat')} |")
    lines+=['','## 同种子、同地图配对差','',
        '| 开局 | 真实减去的对照 | 存活率差及校正区间（百分点） | 存活改善的种子 | 补水等待差（刻） | 进食等待差（刻） | 探索格数差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for sc in SC:
        for a in ('masked','shuffled'):
            e=effects[sc][a];sv=e['survived'];sv_text=fmt(sv,100)
            if gs[sc]['ordered']['alive']==gs[sc][a]['alive']==512:sv_text='0.00（均无死亡）'
            if gs[sc]['ordered']['alive']==gs[sc][a]['alive']==0:sv_text='0.00（均无存活）'
            lines.append(f"| {SC[sc]} | {AR[a]} | {sv_text} | {sv['favorable_seeds']}/8 | {fmt(e['drink_wait'])} | {fmt(e['eat_wait'])} | {fmt(e['cells'])} |")
    for sc,verb in [('thirst','补水'),('hunger','进食')]:
        allrows=[gs[sc][a]['resource_and_survival'] for a in AR]
        alive_no=sum(v['0_1'] for v in allrows);dead_yes=sum(v['1_0'] for v in allrows)
        lines+=['',f"{SC[sc]}开局，跨三组共有 {alive_no} 局未成功{verb}但活到第 60 刻，{dead_yes} 局成功{verb}后仍死亡。"]
    lines+=['','存活率六项主要比较各用双侧 99.17% 配对区间。预设稳定改善要求区间完全位于有利一侧且至少 6/8 个训练种子有利。判定真实顺序同时优于屏蔽和打乱需两项对照各自达标。预设的针对性次要指标为濒渴时补水等待、濒饿时进食等待，四项对照各用 98.75% 区间；其他等待与探索区间为描述性 95%。等待差取真实减去对照，负值表示提早。按地图与训练种子交叉配对重采样 40,000 次；各比较族分开解释。若一组场景中无死亡，观察到的零差不代表总体效果精度为零。','',
            '## 开局、信号及核验','',
            '| 开局 | 生命 | 水分 | 饱食 |','|---|---:|---:|---:|']
    for sc,v in s['scenes'].items():lines.append(f"| {SC[sc]} | {v['health']} | {v['thirst']} | {v['hunger']} |")
    lines+=['','原生游戏仅修改三项初始储备，保留原地图、资源、随身物品、动作和伤害规则。4 张独立开发图上，濒渴开局静候到第 33 刻死亡，先补水可活到第 60 刻；濒饿开局静候到第 40 刻死亡，先吃东西可存活。低生命、食水充足时静候也能存活。','',
            '使用三个条件各 8 个已训练的 Qwen + LoRA 检查点，训练及评估均使用各自条件。真实组依原顺序回放，屏蔽组在 Actor 与 Critic 中均置零，打乱组在每个 20 秒段内联合置换两秒块。输入为芯片 22021 同一颗类器官首次刺激前的无刺激 H5，60 个连续一秒窗，322 个物理电极与训练路由对齐。三个开局共用全新的地图列表、每张图的置换方式和动作采样随机数。只做离线评估，模型冻结，α=1，R2 只记录。','',
            f"核验 {r['games']:,} 局、{r['pairs']:,} 组种子和地图及开局组合、{r['decisions']:,} 次动作。两份原生游戏逐动作核对状态与转移，复核了资源实际增加、死亡与 60 刻边界、输入窗和检查点权重。单段记录来自一个生物样本，估计区间反映训练种子和游戏地图的变异。",'',
            '[预设方案](design.json) · [开局检查](scene-check.json) · [逐局生存、时间与探索](episodes.csv) · [配对结果](paired.csv) · [区间及各训练种子](comparison.json)']
    (H/'生存决策对照.md').write_text('\n'.join(lines)+'\n',encoding='utf8');plot(r)
if __name__=='__main__':main()
