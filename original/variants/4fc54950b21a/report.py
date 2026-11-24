"""Render the three-arm result and interval plot from the audited frozen receipts."""
from pathlib import Path
import json
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

HERE=Path(__file__).resolve().parent
LABEL={'masked':'屏蔽 MEA','ordered':'真实顺序 MEA','shuffled':'分块打乱 MEA'}

def plot(r):
    path=Path('C:/Windows/Fonts/msyh.ttc')
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.6,4.8),gridspec_kw={'width_ratios':[1,1.17]})
    fig.subplots_adjust(left=.075,right=.97,bottom=.21,top=.72,wspace=.45)
    groups=r['groups'];arms=['masked','ordered','shuffled'];effects=r['effects']
    colors=['#7990a5','#16796f','#a88763']
    vals=[groups[a]['new_cells'] for a in arms]
    ax.bar(range(3),vals,color=colors,width=.53)
    for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.2f}',ha='center',color='#203c45',fontsize=11)
    ax.set(xticks=range(3),xticklabels=['屏蔽','真实顺序','分块打乱'],ylim=(0,max(vals)*1.20),ylabel='平均新增探索格数')
    ax.set_title('三种输入下的探索表现',loc='left',fontsize=13)
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    bx.axvline(0,color='#939fa6',lw=1,ls='--')
    for y,arm in [(1,'masked'),(0,'shuffled')]:
        e=effects[arm]['new_cells']
        bx.scatter(e['per_seed'],y+np.linspace(.10,.24,8),s=16,color='#9aaab2',alpha=.85)
        bx.plot([e['lo'],e['hi']],[y,y],lw=2.8,color='#16796f')
        bx.plot(e['delta'],y,'D',ms=7,color='#154c51')
    bx.set(yticks=[1,0],yticklabels=['真实 − 屏蔽','真实 − 打乱'],ylim=(-.39,1.54),xlabel='配对差值（格）')
    bx.set_title('真实顺序相对两种对照的差值',loc='left',fontsize=13)
    bx.grid(axis='x',alpha=.13);bx.set_axisbelow(True)
    fig.suptitle('芯片 22021  |  接入 MEA 后，60 刻内多发现了多少格？',x=.046,y=.96,ha='left',fontsize=17,color='#203c45',weight='bold')
    fig.text(.046,.845,'标准开局，8 个训练种子 × 48 张留存地图，每组 384 局',color='#4f636d')
    fig.text(.046,.046,'灰点为各训练种子的平均差。菱形为总体配对差，横线为经两项比较校正的 97.5% 区间。',fontsize=10,color='#4f636d')
    fig.savefig(HERE/'探索对照.png',dpi=180,facecolor='white');plt.close(fig)

def main():
    read=lambda name:json.loads((HERE/name).read_text(encoding='utf8'))
    r=read('comparison.json');s=read('design.json');a=r['effects']['masked']['new_cells'];b=r['effects']['shuffled']['new_cells']
    if a['stable'] and b['stable']:verdict='真实输入相对屏蔽和打乱两组均达到预设的探索提升标准。该样本支持真实时序在离线游戏中的增益。'
    elif a['stable']:verdict='真实输入相对屏蔽组达到预设的探索提升标准。真实组相对打乱组的优势仍待验证。'
    elif b['stable']:verdict='真实顺序相对打乱组达到预设标准，但相对屏蔽组未达到标准；这批数据不足以确认接入 MEA 能提升探索。'
    else:verdict='真实输入相对两种对照均未达到预设的稳定探索提升标准。'
    lines=['# 芯片 22021 的 MEA 输入与游戏探索', '',verdict,'','![三组均值和配对差](探索对照.png)','',
        '| 训练和评估的输入 | 60 刻内新增探索格数 | 每局原生回报 R1 | 第 60 刻存活 |','|---|---:|---:|---:|']
    for arm,g in r['groups'].items():
        lines.append(f"| {LABEL[arm]} | {g['new_cells']:.2f} | {g['r1']:.5f} | {round(384*g['survived'])}/384 ({g['survived']:.1%}) |")
    lines+=['','| 真实顺序减去对照 | 新增探索格数差及 97.5% 区间 | 探索改善的训练种子 | R1 差及描述性 95% 区间 |','|---|---:|---:|---:|']
    for ref in ['masked','shuffled']:
        x=r['effects'][ref]['new_cells'];y=r['effects'][ref]['r1']
        lines.append(f"| {LABEL[ref]} | {x['delta']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] | {x['positive_seeds']}/8 | {y['delta']:+.5f} [{y['lo']:+.5f}, {y['hi']:+.5f}] |")
    if all(g['survived']==1 for g in r['groups'].values()):
        lines+=['','三组均无死亡，本次标准开局、60 刻观察期内的存活指标达到上限。']
    lines+=['','## 样本与比较方法','',
        '三组从同一份 Qwen3.5-0.8B + LoRA / Critic 检查点重训。每组 8 个训练种子，每个种子 4 批 × 12 个动作，Actor 更新 8 次、Critic 更新 64 次。角色初始生命 100、水分 95、饱食 95。所有模型仅优化游戏原生回报 R1，α=1，R2 权重为 0。各组使用同一套游戏地图和动作采样随机数；训练地图与本次留存地图均在训练前锁定。','',
        '使用单颗类器官、同一路由 322 个非刺激通道。在已保存的 6 个零输出轮次中，第 2、4、8 轮提供训练窗，第 10、15、18 轮完整留出，各取 20 个连续一秒窗，按试验时间拼接。训练或留存均无跨轮信号块。硬件在这些轮次内零输出，此前交叉实验有过刺激，可能有残留影响。归一化参数来自首次刺激前独立的预热段。','',
        '四项 state2 特征分别为对数放电率、活跃通道比例、时间集中度和空间熵。屏蔽组在 Actor 和 Critic 都置零。打乱组在同一物理轮次内联合置换两秒块，训练按种子、评估按地图预先登记；块内时序和各特征的联合分布保留。模型在动作开始时读取对应游戏刻的一秒窗，多刻动作会跳过中间窗。每张地图使用同一批留存生物记录，生物样本数为 1。','',
        '唯一主指标为开局至第 60 个游戏刻新发现的地图格数（游戏 discovered 计数的净增加，可包含角色视野内揭示的格子）。终点对跨越第 60 刻的动作截算；提前死亡则保留死亡时的计数。每对比较在同一训练种子、同一张新图下取差，对这两个因素分别重采样 40,000 次。两项主比较各用双侧 97.5% 区间以控制整体名义错误率。预设稳定提升需区间下界高于 0，且至少 6/8 个种子的差值为正。次要终点仅作描述。','',
        f"独立重放了全部 1,152 个训练动作，并复核每组 R2 实际贡献均为零。留存游戏共 {r['games']:,} 局，{r['actions']:,} 次决策由两份原生游戏实例逐动作核对。已复核输入窗、数字指纹、评估前后权重和训练预算。新地图和训练种子提高了本批数据的游戏评估精度。所用历史记录也曾用于此前分析，本轮按完整物理轮次分开训练与评估。生物样本只有一个。整个训练与评估离线进行。",'',
        '[预先登记的方案与顺序](design.json) · [逐局数据](episodes.csv) · [逐地图配对](paired.csv) · [种子差值与区间](comparison.json) · [训练核验](training-audit.json)']
    (HERE/'探索对照结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    plot(r)
if __name__=='__main__':main()
