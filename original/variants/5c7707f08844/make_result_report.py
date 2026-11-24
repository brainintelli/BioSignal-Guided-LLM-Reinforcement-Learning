"""Render only fully verified final results; no model/device access."""
import json
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
audit=read(root/'final_audit.json')
if audit.get('passed') is not True or audit.get('cells')!=24:raise ValueError('Complete independent audit required')
original=root.parent/'mea_stim_offline_20261117'
saved=read(original/'delivery/experiment01/analysis.json')
if saved['registration_sha256']!=audit['registration_sha256'] or saved['source_pin']!=audit['source_pin']:
    raise ValueError('Result binding changed')
arms=('none','before','after');names={'none':'不加湿件信号','before':'刺激前信号','after':'刺激后信号'}
sentences={
 'registered_exploratory_advantage':'本轮模型端达到预先登记的探索性优势标准，可视为弱阳性线索；尚不能认定刺激产生了生物学阳性效果。',
 'direction_only_not_positive_evidence':'两项均值对比呈正向趋势，但没有达到预先登记的探索性优势标准，不能据此确认弱阳性。',
 'no_consistent_advantage_across_both_controls':'本轮未见刺激后信号训练相对两个对照产生一致优势，不能认定为弱阳性。'}
conclusion=sentences[audit['classification']]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(13,5.3),gridspec_kw={'width_ratios':[1.12,1]})
colors=['#798692','#2878B5','#D97721']
for block in range(8):
    values=[mean(next(c for c in saved['cells'] if c['block']==block and c['arm']==arm)['post'].values()) for arm in arms]
    axes[0].plot(range(3),values,color='#B2BAC2',alpha=.62,linewidth=1,marker='o',markersize=4)
means=[audit['arms'][a]['post'] for a in arms]
axes[0].scatter(range(3),means,color=colors,s=120,zorder=5,label='8 个种子的均值')
for i,m in enumerate(means):axes[0].annotate(f'{m:.4f}',(i,m),xytext=(-6 if i==2 else 6,8),textcoords='offset points',ha='right' if i==2 else 'left',color=colors[i],weight='bold')
axes[0].set_xticks(range(3),[names[a] for a in arms]);axes[0].set_ylabel('共同留存评估 R1（越高越好）')
axes[0].set_title('灰线连接同一个配对数字种子');axes[0].grid(axis='y',alpha=.2);axes[0].legend(loc='best',frameon=False)
keys=('after_minus_before','after_minus_none');contrast_names=['刺激后 − 刺激前','刺激后 − 不加信号']
for i,key in enumerate(keys):
    q=audit['contrasts'][key]['post'];lo,hi=q['bootstrap_95'];m=q['mean_difference']
    axes[1].plot([lo,hi],[i,i],color=colors[i+1],linewidth=4)
    axes[1].scatter([m],[i],color=colors[i+1],s=70,zorder=4)
    axes[1].annotate(f'差值 {m:+.4f}；校正 p={q["p_bonferroni_two_contrasts"]:.4f}',(m,i),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10)
axes[1].axvline(0,color='#777',linestyle='--',linewidth=1)
axes[1].set_yticks(range(2),contrast_names);axes[1].set_ylim(-.55,1.55)
axes[1].set_xlabel('配对 R1 差值；线段为 bootstrap 95% 区间');axes[1].grid(axis='x',alpha=.2)
axes[1].set_title('两个预定对照；区间跨零表示方向不确定')
fig.suptitle('28080 刺激前后信号的离线训练效果',fontsize=17)
fig.text(.5,.025,'24 个模型 / 8 个配对数字种子 / 同初始参数与训练预算 / R2 关闭\n单枚类器官、单次刺激、无假刺激生物对照；数字种子不是生物重复。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.11,1,.94),w_pad=3)
for ext in ('png','svg'):fig.savefig(root/('model_comparison.'+ext),dpi=180)
plt.close(fig)

lines=['# 28080 刺激后信号训练：独立复核结论','',conclusion,'',
 '已接续并完成既有三组对照，共24个模型、9,216次训练决策。全部训练、共同留存评估及新进程重放核验通过；本核验另行检查封存文件和登记身份，独立复算最终统计。','',
 '| 训练输入 | 初始留存 R1 | 训练后留存 R1 | 训练增量（描述性） |','|---|---:|---:|---:|']
for arm in arms:
    a=audit['arms'][arm];lines.append(f'| {names[arm]} | {a["pre"]:.6f} | {a["post"]:.6f} | {a["post"]-a["pre"]:+.6f} |')
lines+=['','R1 越高越好。各模型都在同样的刺激前与刺激后留存信号上评估，表中对两种上下文取平均；不是各组各选有利测试输入。','',
 '| 预定比较 | 配对 R1 差值 | 95% bootstrap 区间 | 两项比较校正 p |','|---|---:|---|---:|']
for key,label in zip(keys,contrast_names):
    q=audit['contrasts'][key]['post'];lo,hi=q['bootstrap_95']
    lines.append(f'| {label} | {q["mean_difference"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q["p_bonferroni_two_contrasts"]:.6f} |')
lines+=['','判定沿用原登记规则：两项比较均需均值为正、95%区间下界大于零、Bonferroni校正 p<0.05。没有按中途分数挑种子、挑检查点或替换主指标。','',
 f'![模型结果对比]({(root/"model_comparison.png").as_posix()})','',
 '| 训练输入 | 最终生命值 | 饱食储备 | 水分储备 | 探索格数 | 游戏 tick |','|---|---:|---:|---:|---:|---:|']
for arm in arms:
    v=audit['secondary'][arm]
    lines.append('| '+names[arm]+' | '+' | '.join(f'{v[k]:.2f}' for k in ('final_health','final_hunger','final_thirst','visited_count','elapsed_ticks'))+' |')
lines+=['','以上为描述性次指标，不替代 R1 主指标。水分和饱食字段表示剩余储备；游戏 tick 受动作耗时影响。每次评估最多150次决策，不能用该短程记录宣称完整寿命延长或产生怕死、怕渴等感受。',
 '最终评估自然终止次数（每组16次共同上下文评估）：'+ '；'.join(names[a]+f' {audit["natural_termination_counts"][a]}/16' for a in arms)+'。','',
 '数据范围：仅28080单点15脉冲的合格前后记录，统一排除刺激通道640；刺激后从末脉冲结束60秒后取600个1秒窗口。训练384决策，中间隔离60秒，留存评估150决策。没有使用28073质控失败数据。R2关闭，训练奖励严格等于R1，没有新增采集或刺激。','',
 '本结果的推断单位是同一份生物记录条件下的8个数字种子。只有一枚类器官、一次刺激、无随机假刺激生物对照，也未加入打乱信号组，因此不能把模型端差异归因为刺激本身，不能证明生物学习或在线闭环收益。未见优势也不等于证明所有剂量、样本和训练时长均无效。','',
 '独立复核未重跑模型或重演每一步Adam动量；它核验已有新进程重放凭据、真实预算、参数身份、封存哈希并独立重算统计。','',
 f'[独立统计与验收 JSON]({(root/"final_audit.json").as_posix()}) · [完整原始对照报告]({(original/"REPORT.md").as_posix()}) · [逐组与上下文数据]({(original/"comparison.csv").as_posix()}) · [训练输入分布]({(root/"input_distribution.png").as_posix()})','']
(root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(conclusion)
