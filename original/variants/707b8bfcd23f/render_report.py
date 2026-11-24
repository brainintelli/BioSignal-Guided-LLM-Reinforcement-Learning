"""Render all preregistered outcomes after evidence acceptance."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'.tools/mea_baseline_deps'),str(root/'standalone/mea_state2_r2'),str(root/'standalone/mea_binary_r2'),str(root)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mea_state2_r2.analysis import analyze, METRICS

out=Path(__file__).resolve().parent/'final_report'
out.mkdir(exist_ok=False)
accepted=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
result=analyze(accepted['cells'],accepted['registration_sha256'])
if result['status']!='complete':raise RuntimeError('Incomplete campaign cannot produce effect report')
(out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['axes.unicode_minus']=False
arms=['A0','A','B','C','D']
labels=['A0 原基线','A 继续R1','B 增加R2','C 历史信号','D 打乱信号']
colors=['#6c7481','#91a4be','#3976c7','#329c82','#be873b']
titles=dict(r1_sum='原始 R1 累计值（主指标）',visited_count='探索格数',final_health='结束生命值',
            final_hunger='结束饱食',final_thirst='结束水分',elapsed_ticks='游戏 ticks')
fig,axes=plt.subplots(2,3,figsize=(15,8.5))
for ax,key in zip(axes.flat,titles):
    means=[result['arms']['after'][arm][key] for arm in arms]
    ax.bar(range(5),means,color=colors,width=.65,alpha=.85)
    for i,arm in enumerate(arms):
        values=[row['metrics']['after']['full'][key] for row in accepted['cells'] if row['arm']==arm]
        xs=[i+(j-3.5)*.035 for j in range(8)]
        ax.scatter(xs,values,s=16,color='#26394b',alpha=.55,zorder=3)
        ax.text(i,means[i],f'{means[i]:.3f}' if key=='r1_sum' else f'{means[i]:.2f}',ha='center',va='bottom',fontsize=10)
    ax.set_title(titles[key],fontproperties=font,fontsize=13)
    ax.set_xticks(range(5),labels,fontproperties=font,rotation=15,fontsize=9)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    if key in ('final_health','final_hunger','final_thirst'):ax.set_ylim(0,112)
fig.suptitle('新版 state2 + 二值 R2：同一已训练基线的五组比较',fontproperties=font,fontsize=18)
fig.text(.5,.015,'32个继续训练模型 + 8个冻结基线评价；点为数字种子。28080历史MEA输入，R2为游戏状态驱动的模拟奖惩。',ha='center',fontproperties=font,fontsize=11)
fig.tight_layout(rect=[0,.045,1,.95])
fig.savefig(out/'metrics_overview.png',dpi=165);fig.savefig(out/'metrics_overview.svg')

stat=result['contrasts']['C_minus_B']['r1_sum']
fig,ax=plt.subplots(figsize=(10,6.7))
ds=stat['differences']
ax.barh(range(8),ds,color=['#329c82' if d>0 else '#ca746a' for d in ds])
ax.axvline(0,color='#556270',lw=1)
ax.set_yticks(range(8),[f'种子组 {i+1}' for i in range(8)],fontproperties=font)
mu=stat['mean_difference'];lo,hi=stat['bootstrap_95']
ax.errorbar([mu],[9],xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color='#21394f',capsize=6,lw=2)
fig.text(.15,.02,f'均值 {mu:+.5f}；95%区间 [{lo:+.5f}, {hi:+.5f}]\np={stat["p_two_sided"]:.5f}；正向 {stat["positive_pairs"]}/8',
        fontproperties=font,fontsize=11,va='bottom')
ax.set_yticks(list(range(8))+[9],[f'种子组 {i+1}' for i in range(8)]+['配对均值'],fontproperties=font)
ax.set_ylim(10.8,-.8)
ax.set_xlabel('原始 R1 差值：C 历史信号 − B 仅辅助奖惩',fontproperties=font,fontsize=12)
ax.set_title('主要比较：真实历史观测是否提供额外收益',fontproperties=font,fontsize=15)
ax.grid(axis='x',alpha=.15)
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(out/'primary_paired_effect.png',dpi=165);fig.savefig(out/'primary_paired_effect.svg')

base=result['contrasts']['C_minus_A0']['r1_sum']
if stat['mean_difference']<=0:
    conclusion='本轮主比较未观察到正向提升，不能认定新版历史信号带来稳定增益。'
elif not result['digital_state2_benefit_detected']:
    conclusion='主比较出现正向点估计，但未达到预先登记的稳定提升判据：区间仍跨过0，配对检验未通过。'
elif base['mean_difference']<=0:
    conclusion='历史信号组优于仅R2组，但没有超过原纯软件基线；尚未达到整体改善目标。'
else:
    conclusion='本轮观察到离线数字对照增益；它尚不证明真实闭环湿件有效。'
means=result['arms']['after']
cpre=sum(row['training']['pre_metrics']['full']['r1_sum'] for row in accepted['cells'] if row['arm']=='C')/8
effect=result['contrasts']['C_minus_B']
interpretation=(f'主比较C−B的R1均值差为{stat["mean_difference"]:+.6f}（相对B {100*stat["mean_difference"]/means["B"]["r1_sum"]:+.2f}%），'
    f'正向{stat["positive_pairs"]}/8，p={stat["p_two_sided"]:.6f}。探索格数差{effect["visited_count"]["mean_difference"]:+.3f}，'
    f'生命值差{effect["final_health"]["mean_difference"]:+.2f}，饱食差{effect["final_hunger"]["mean_difference"]:+.2f}，'
    f'水分差{effect["final_thirst"]["mean_difference"]:+.2f}。同一输入接口下，C组训练前后的R1均值为{cpre:.6f}→{means["C"]["r1_sum"]:.6f}；'
    '这一描述性比较也应与组间差值一起阅读，不能把输入差异直接归为训练收益。')
lines=['# 新版 state2 / R2 历史数据再训练结果','',conclusion,'',
       interpretation,'',
       '从同一纯软件强化学习检查点独立启动32个训练模型（8配对数字种子×A/B/C/D），另有8个A0冻结基线评价。每个训练模型384个游戏决策，3轮PPO；实际更新Actor的LoRA与Critic，基础模型和参考权重冻结。R2为模拟刺激事件奖惩，β=0.001；未向MEA发送刺激。','',
       '| 组别 | 原始R1 | 探索格数 | 生命值 | 饱食 | 水分 | 游戏ticks | 死亡/8 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm,label in zip(arms,labels):
    m=result['arms']['after'][arm]
    lines.append(f'| {label} | {m["r1_sum"]:.6f} | {m["visited_count"]:.2f} | {m["final_health"]:.2f} | {m["final_hunger"]:.2f} | {m["final_thirst"]:.2f} | {m["elapsed_ticks"]:.2f} | {int(round(m["terminated"]*8))}/8 |')
lines+=['','主评价固定150决策机会，使用刺激后留存窗口；R1直接逐转移相加，不含R2。自然死亡后按吸收态处理。游戏ticks随动作不同，另报告每tick结果；该比较不等同于相同游戏时长。','',
        '| 原始R1比较 | 配对均值差 | 95%区间 | 双侧p | 正向种子 |','|---|---:|---|---:|---:|']
for name,stats in result['contrasts'].items():
    s=stats['r1_sum'];lo,hi=s['bootstrap_95']
    lines.append(f'| {name} | {s["mean_difference"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {s["p_two_sided"]:.6f} | {s["positive_pairs"]}/8 |')
lines+=['','同一新接口训练前后比较（描述性结果，用于区分输入格式影响与继续训练影响）：','',
        '| 组别 | 训练前R1 | 训练后R1 | 均值变化 | 正R2次数 | 负R2次数 | Actor/Critic实际更新总数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
for arm in arms[1:]:
    rows=[r for r in accepted['cells'] if r['arm']==arm]
    pre=sum(r['training']['pre_metrics']['full']['r1_sum'] for r in rows)/8
    post=result['arms']['after'][arm]['r1_sum']
    positive=sum(r['training']['positive'] for r in rows);negative=sum(r['training']['negative'] for r in rows)
    actor=sum(s['actor'] for r in rows for s in r['training']['applied_steps'])
    critic=sum(s['critic'] for r in rows for s in r['training']['applied_steps'])
    lines.append(f'| {arm} | {pre:.6f} | {post:.6f} | {post-pre:+.6f} | {positive} | {negative} | {actor}/{critic} |')
lines+=['','每tick原始R1为各种子比值的均值（描述性，不替代主指标）：'+
        '；'.join(f'{arm} {sum(r["metrics"]["after"]["full"]["r1_per_tick"] for r in accepted["cells"] if r["arm"]==arm)/8:.7f}' for arm in arms)+'。']
lines+=['','仅 C−B 的 after/R1 是登记的单一主检验。其它比较和指标为未校正探索性分析；完整before场景、前16步、全部配对差值均保存在analysis.json。置信区间反映数字种子变化，不能当作生物重复的不确定性。','',
        '如果 C 超过 A0 但未超过 A/B，不能把继续训练或奖励塑形收益归因于湿件。若探索增加伴随生命/饱食/水分下降，作为权衡报告。150决策属于短时评价；死亡未增加不等于已证明没有负面影响或长期生存优势。','',
        'C−D用于检查历史时序是否超出同分布打乱输入。如果该比较不支持优势，即使C−B正向，也没有证据把收益归因于真实信号的时间顺序；非零输入或一般扰动等解释仍未排除。','',
        '数据来自一枚28080的一次历史记录。state2与当前动作没有实时因果对应；最新22021只用于候选刺激参数，不提供本轮真实训练标签。R2正负不能解释为类器官感到舒适、痛苦或恐惧。','',
        '验证限制：独立验证重放了记录动作、模型概率、价值、原生游戏和奖励，没有独立重演随机动作抽样。标准化前原特征和冻结参数已保存，舍入前标准化数值可重建但未直接逐行保存。','',
        '另行真实终止测试发现冻结v1训练入口会尝试编码死亡状态的空合法动作列表。该分支已在独立v2包修复；本报告仅在完整训练日志确认没有死亡转移时接受v1效果结果。正式运行不混用版本，边界验证记录与效果试验分开保存。短时无训练死亡也限制了对终止后的泛化判断。','',
        f'冻结登记摘要：`{accepted["registration_sha256"]}`。完整证据清单见本轮交付目录。','',
        f'![指标对照]({(out/"metrics_overview.png").as_posix()})','',
        f'![主要配对效果]({(out/"primary_paired_effect.png").as_posix()})']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'report':str(out/'RESULTS.md'),'conclusion':conclusion,'primary':stat},ensure_ascii=False))
