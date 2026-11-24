"""Reproducible paired estimates for the original and the separate-H5 confirmation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

H=Path(__file__).resolve().parent
LABEL={'ordered':'真实训练、真实输入','masked':'屏蔽训练、屏蔽输入','shuffled':'打乱训练、打乱输入',
       'same_masked':'真实训练、仅评估时屏蔽','same_shuffled':'真实训练、仅评估时打乱'}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def fmt(e,d=2):
    return f"{e['delta']:+.{d}f} [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}]"
def plot(o,r):
    fp=Path('C:/Windows/Fonts/msyh.ttc')
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.right':False,'axes.spines.top':False,'font.size':10})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.2,4.65),gridspec_kw={'width_ratios':[1,1.14]})
    fig.subplots_adjust(left=.084,right=.97,top=.72,bottom=.20,wspace=.48)
    fig.suptitle('芯片 22021  |  换一段 MEA 记录后，探索增益能否复现？',x=.037,y=.97,ha='left',weight='bold',fontsize=16,color='#203d47')
    fig.text(.037,.85,'冻结同一批模型。上轮为 48 张地图，本轮为 64 张全新地图，均使用 8 个训练种子。',color='#586c79')
    ax.set_title('60 刻新增探索格数的配对差',loc='left',fontsize=12)
    ax.axvline(0,color='#84949c',linestyle='--',lw=1)
    for y,c in [(1,'masked'),(0,'shuffled')]:
        for off,z,color,lab in [(.12,o,'#8796a9','上一轮'),(-.12,r,'#117c75','本轮')]:
            e=z['effects'][c]['new_cells'];ax.plot([e['lo'],e['hi']],[y+off]*2,color=color,lw=2)
            ax.scatter(e['delta'],y+off,color=color,s=36,marker='o' if off>0 else 'D',zorder=3,label=lab if y==1 else None)
    ax.set(yticks=[0,1],yticklabels=['真实 − 打乱','真实 − 屏蔽'],xlabel='新增探索格数差（格）',ylim=(-.66,1.55));ax.grid(axis='x',alpha=.14)
    ax.legend(frameon=False,fontsize=9,loc='lower right',ncol=2)
    bx.set_title('低探索对局占比（越低越好）',loc='left',fontsize=12)
    low=r['lower_tail'];arr=[[100*low['original_masked']/384,100*r['groups']['masked']['low']],
                            [100*low['original_ordered']/384,100*r['groups']['ordered']['low']]]
    for j,(ys,lab,col) in enumerate(zip(arr,['屏蔽','真实顺序'],['#8594a6','#117c75'])):
        xx=np.arange(2)+(j-.5)*.33
        bx.bar(xx,ys,.29,color=col,label=lab)
        for x,y in zip(xx,ys):bx.text(x,y+.8,f'{y:.1f}%',ha='center',fontsize=9)
    bx.set(xticks=[0,1],xticklabels=['上一轮','本轮'],ylabel='低探索占比（%）',ylim=(0,max(max(a) for a in arr)*1.45))
    bx.legend(frameon=False,loc='upper right',ncol=2,fontsize=9);bx.set_axisbelow(True);bx.grid(axis='y',alpha=.13)
    fig.text(.037,.04,f"左图为配对差值及 97.5% 区间。右图阈值取自上一轮屏蔽组下四分位数，即前 60 刻新增不超过 {low['cutoff']} 格。",fontsize=9,color='#586c79')
    fig.savefig(H/'验证结果.png',dpi=180,facecolor='white');plt.close(fig)

def main():
    o=read(H.parent/'comparison.json');r=read(H/'comparison.json');g=r['groups'];eff=r['effects'];a=eff['masked']['new_cells'];b=eff['shuffled']['new_cells'];l=r['lower_tail'];n=r['pairs']
    if a['stable'] and b['stable']:msg='独立记录和新地图上，真实顺序相对屏蔽和打乱两组均达到预设探索提升标准。'
    elif a['stable']:msg='独立记录和新地图上，真实顺序相对屏蔽组达到预设标准；相对打乱组的时间顺序优势尚待验证。'
    elif a['delta']>0:msg='真实输入相对屏蔽组的探索均差再次为正，但未达到预设稳定提升标准。'
    else:msg='换用独立记录和新地图后，未复现上一轮真实输入相对屏蔽组的正向平均探索差。'
    out=['# 芯片 22021 的探索效果复验','',msg,'','![配对差与低探索占比](验证结果.png)','',
         '| 条件 | 60 刻新增格数 | 每局原生 R1 | 第 60 刻存活 | 低探索对局 |','|---|---:|---:|---:|---:|']
    for c,v in g.items():
        out.append(f"| {LABEL[c]} | {v['new_cells']:.2f} | {v['r1']:.5f} | {round(n*v['survived'])}/{n} | {round(n*v['low'])}/{n} |")
    out+=['','## 接入效果与固定权重诊断','','所有差值均为真实训练、真实输入组减去同一训练种子和地图下的比较组。按种子和地图交叉重采样 40,000 次；两项主要对照和两项固定权重诊断分别构成比较族，新增格数各用 97.5% 双侧区间。预设稳定标准为区间下界大于零，且至少 6/8 个训练种子差值为正。R1 的 95% 区间仅作描述。','',
         '| 比较组 | 探索差及 97.5% 区间（格） | 正向种子 | R1 差及描述性 95% 区间 |','|---|---:|---:|---:|']
    for c in ['masked','shuffled','same_masked','same_shuffled']:
        x=eff[c]['new_cells'];y=eff[c]['r1'];out.append(f"| {LABEL[c]} | {fmt(x)} | {x['positive_seeds']}/8 | {fmt(y,5)} |")
    out+=['','上一轮，在相同训练种子但另 48 张地图、另一次记录上，真实减屏蔽为 '+fmt(o['effects']['masked']['new_cells'])+' 格，真实减打乱为 '+fmt(o['effects']['shuffled']['new_cells'])+' 格。两轮结果单独估计。','',
          '## 新的补充指标','',f"低探索定义为前 60 刻新增不超过 {l['cutoff']} 格，阈值由上一轮屏蔽组的下四分位数单独确定。辅助指标在本轮推理启动后、读取任何本轮逐局结果之前锁定。正差代表接入 MEA 后低探索局减少。",'',
          '| 数据 | 屏蔽组低探索 | 真实组低探索 | 减少的百分点 [描述性 95% 区间] |','|---|---:|---:|---:|']
    for title,nm,den,k1,k2 in [('上一轮','original',384,l['original_masked'],l['original_ordered']),('本轮','new',n,round(n*g['masked']['low']),round(n*g['ordered']['low']))]:
        e=l[nm];out.append(f"| {title} | {k1}/{den} ({k1/den:.2%}) | {k2}/{den} ({k2/den:.2%}) | {e['delta']*100:+.2f} [{e['lo']*100:+.2f}, {e['hi']*100:+.2f}] |")
    low_new=l['new']
    if low_new['lo']>0:
        low_message='在新记录和新地图上，真实组出现低探索对局的比例较低，补充指标的描述性区间完全高于零。这是需要在独立样本继续复核的行为线索。'
    elif low_new['delta']>0:
        low_message='新数据中低探索对局的点估计也较少，差值区间跨零。两次方向一致，尚不足以确认低探索的发生率下降。'
    else:
        low_message='按锁定的 99 格阈值，新数据未重现低探索对局减少的方向。'
    out+=['',low_message]
    if all(row['survived']==1 for row in g.values()):
        out+=['','五种条件下角色全部活到第 60 刻。这次标准开局的结果无法区分生存能力。']
    out+=['','## 数据与解释范围','',
          '本轮使用首次非零刺激前单独保存的无刺激 H5，自第二秒起取连续 60 个一秒窗。核对原始 H5 校验值、全部帧连续性、采样和尖峰阈值；按物理电极编号和坐标将原始路由映射到训练用的 322 个电极。训练时确定的归一化参数原样使用。该记录未用于这 24 个模型的拟合，也未进入上一轮 48 张图的评估；早期项目曾用它筛选电极和评估其他模型。生物样本仍为同一颗类器官。','',
          '所有条件共享相同的地图、动作采样随机数、初始生命 100、水分 95、饱食 95，均在死亡或第 60 个游戏刻结算。指标为游戏新揭开的格数，可包含视野揭示的格子。打乱使用同一组向量，四项特征联合在每个 20 秒区间内置换两秒块。固定权重诊断只切换输入；独立训练的三组用于估计完整接入方式的变化。R2 只记录，训练只用 R1，α=1。本轮未更新权重。','',
          f"总计 {r['games']:,} 局，每种条件 {n} 局；原生游戏和读取截止刻的副本逐动作核对了 {r['decisions']:,} 次转移。已逐局核验实际输入顺序、特征、累计探索格数、权重与终点。区间反映游戏地图和训练种子的变异，无法估计跨类器官的效果。",'',
          '[主要方案](design.json) · [补充指标登记](lower-tail-plan.json) · [逐地图配对](paired.csv) · [逐局结果](episodes.csv) · [各训练种子与区间](comparison.json)']
    (H/'探索效果复验.md').write_text('\n'.join(out)+'\n',encoding='utf8');plot(o,r)
if __name__=='__main__':main()
