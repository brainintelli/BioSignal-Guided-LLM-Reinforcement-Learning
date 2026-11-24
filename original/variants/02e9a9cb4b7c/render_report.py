"""Render all prespecified outcomes from the independently audited 32-cell run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ARMS=('none','before','after','shuffled_after')
CONTROLS=('none','before','shuffled_after')
LABELS={'none':'不加信号','before':'刺激前信号','after':'刺激后信号','shuffled_after':'打乱刺激后信号'}
METRICS=('visited_count','discounted_r1','final_health','final_hunger','final_thirst','elapsed_ticks')
TITLES={'visited_count':'探索格数','discounted_r1':'原始 R1（不含 R2）','final_health':'最终生命值',
        'final_hunger':'最终饱食储备','final_thirst':'最终水分储备','elapsed_ticks':'游戏经过 tick',
        'terminated':'自然死亡率差'}
COLORS=('#94a3b8','#5791cf','#d18437','#8b75b7')
PIN='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'


def validate(data):
    if (data.get('passed') is not True or data.get('cells')!=32 or data.get('training_decisions')!=12288
            or data.get('r2_enabled') is not False or data.get('new_stimulation') is not False
            or data.get('source_pin')!=PIN or set(data.get('arms',{}))!=set(ARMS)):
        raise ValueError('Complete independently accepted fixed experiment required')
    for arm in ARMS:
        for metric in METRICS:
            values=data['paired_values'][arm][metric]
            if len(values)!=8 or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                raise ValueError('Eight finite seed-level observations required')
            if abs(mean(values)-data['arms'][arm]['post'][metric])>1e-9:
                raise ValueError('Group mean differs from paired values')
    for control in CONTROLS:
        for metric,stat in data['contrasts']['after_minus_'+control].items():
            if not all(math.isfinite(stat[k]) for k in ('mean_difference','p_adjusted')):
                raise ValueError('Invalid audited contrast')
            if (len(stat['bootstrap_95'])!=2 or not all(math.isfinite(x) for x in stat['bootstrap_95'])
                    or stat['bootstrap_95'][0]>stat['bootstrap_95'][1]):
                raise ValueError('Invalid uncertainty interval')
    if data.get('survival_significance_reachable') is not False:
        raise ValueError('Do not silently change the frozen survival analysis')


def render(audit_path,output):
    data=json.loads(audit_path.read_text(encoding='utf-8'));validate(data)
    output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),
        'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'#f8fafc','axes.facecolor':'white','axes.titleweight':'bold'})

    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,metric in zip(axes.flat,METRICS):
        values=[data['arms'][arm]['post'][metric] for arm in ARMS]
        ax.bar(range(4),values,color=COLORS,width=.65,zorder=2)
        for x,arm in enumerate(ARMS):
            vals=data['paired_values'][arm][metric]
            ax.scatter([x+(i-3.5)*.035 for i in range(8)],vals,s=14,c='#243447',alpha=.6,zorder=3)
            ax.annotate(f'{values[x]:.3f}' if metric=='discounted_r1' else f'{values[x]:.2f}',
                        (x,values[x]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(range(4),[LABELS[a].replace('刺激后信号','刺激后\n信号') for a in ARMS],fontsize=9)
        ax.set_title(TITLES[metric]);ax.grid(axis='y',alpha=.17,zorder=0)
        if min(v for arm in ARMS for v in data['paired_values'][arm][metric])>=0:
            ax.set_ylim(bottom=0)
        ax.margins(y=.20)
    fig.suptitle('28080 新种子复测：四组模型的完整结果',fontsize=19,y=.985)
    fig.text(.5,.015,'柱：8 个数字种子的平均值；点：单个种子（两种共同评估上下文取平均）\n历史记录回放 · R2 关闭 · 1 枚类器官 · 每次评估最多 150 个决策',ha='center',fontsize=11,color='#475569')
    fig.tight_layout(rect=(0,.07,1,.95))
    for ext in ('png','svg'):fig.savefig(output/f'metrics_overview.{ext}',dpi=170)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,metric in zip(axes.flat,METRICS[:-1]+('terminated',)):
        ax.axvline(0,color='#64748b',linewidth=1,linestyle='--')
        for y,control in enumerate(CONTROLS):
            stat=data['contrasts']['after_minus_'+control][metric]
            center=stat['mean_difference'];lo,hi=stat['bootstrap_95']
            ax.plot([lo,hi],[y,y],color='#d18437',linewidth=3)
            ax.scatter([center],[y],color='#9a4c10',s=45,zorder=4)
            ax.annotate(f'Δ={center:+.3f}；校正 p={stat["p_adjusted"]:.4f}',
                        (center,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
        ax.set_yticks(range(3),['后 − '+LABELS[c] for c in CONTROLS],fontsize=9)
        ax.set_ylim(2.55,-.65);ax.set_title(TITLES[metric]);ax.grid(axis='x',alpha=.17)
        ax.margins(x=.28)
    fig.suptitle('刺激后信号相对各对照：配对差值与不确定性',fontsize=18,y=.985)
    fig.text(.5,.015,'点：平均配对差；线：登记方法的 95% bootstrap 区间；零线右侧为数值增加\n探索校正 3 项比较；生存相关校正 15 项。本轮 n=8 的生存精确检验最低校正 p=0.1172，仅作描述。',ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.075,1,.95))
    for ext in ('png','svg'):fig.savefig(output/f'paired_effects.{ext}',dpi=170)
    plt.close(fig)

    replicated=data['exploration_replicated'];temporal=data['temporal_order_evidence']
    if temporal:
        conclusion='本轮探索指标通过了预定复现判定，且刺激后有序信号优于打乱信号；这是同一份记录下的数字种子证据，尚不能确认为真实闭环或生物学收益。'
    elif replicated:
        conclusion='本轮探索指标通过了相对不加信号与刺激前信号的预定复现判定，但未通过相对打乱信号的判定；不能把增益归因于神经信号时序。'
    else:
        conclusion='本轮探索指标未通过预定复现判定，不能认定为稳定弱阳性。所有局部上涨与下降按原登记完整展示。'
    lines=['# 28080 打乱信号对照：独立审计结果','',conclusion,'',
        '共32个模型、8个新配对数字种子×4组；每模型384训练决策、3轮更新；评估使用同样的刺激前和刺激后留存上下文。R2关闭，无新增采集或刺激。','',
        '| 条件 | 探索格数 | 原始R1 | 生命值 | 饱食 | 水分 | 游戏tick | 自然终止/16 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        g=data['arms'][arm]['post']
        lines.append('| '+LABELS[arm]+' | '+' | '.join(f'{g[m]:.4f}' if m=='discounted_r1' else f'{g[m]:.2f}' for m in METRICS)+f' | {data["final_terminations"][arm]}/16 |')
    lines+=['','| 条件 | 训练前R1 | 训练后R1 | 训练前探索 | 训练后探索 |',
            '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        pre=data['arms'][arm]['pre'];post=data['arms'][arm]['post']
        lines.append(f'| {LABELS[arm]} | {pre["discounted_r1"]:.4f} | {post["discounted_r1"]:.4f} | {pre["visited_count"]:.2f} | {post["visited_count"]:.2f} |')
    lines+=['','训练前后均使用共同留存上下文。以上纵向变化仅作描述，不替代预先登记的训练后组间比较，也不证明生物学习。']
    lines+=['','| 探索比较 | 配对差值 | 95%区间 | 校正p | 上涨种子 | 前信号评估差值 | 后信号评估差值 |',
            '|---|---:|---|---:|---:|---:|---:|']
    for control in CONTROLS:
        s=data['contrasts']['after_minus_'+control]['visited_count'];lo,hi=s['bootstrap_95']
        contexts=s['context_mean_differences']
        lines.append(f'| 刺激后 − {LABELS[control]} | {s["mean_difference"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {s["p_adjusted"]:.6f} | {s["positive_pairs"]}/8 | {contexts["before"]:+.3f} | {contexts["after"]:+.3f} |')
    lines+=['','复现规则保持冻结：相对不加信号和刺激前信号均需平均差>0、区间下界>0、校正p<0.05、至少6/8种子上涨且两种评估上下文方向一致。时序证据还需同时优于打乱信号。','',
        '![四组全部指标]('+str((output/'metrics_overview.png').resolve()).replace('\\','/')+')','',
        '![全部预定配对比较]('+str((output/'paired_effects.png').resolve()).replace('\\','/')+')','',
        '生存指标限制：8个配对种子、双侧精确符号翻转检验、15项Bonferroni校正时，最低可能p=15×2/256=0.1171875。因此本轮生存相关差异只作描述，不能以未显著证明没有伤害或没有效果；若探索上涨伴随水分下降，应作为权衡报告。不同动作耗用的游戏tick不同，探索格数需结合时间和储备解读，不能直接等同于探索欲。','',
        '只有一枚类器官的一次历史记录；8个数字种子不是8个生物重复。短程决策上限与零死亡不能证明完整寿命延长。不得把本结果解释为类器官感到怕死、怕渴或已实现在线生物学习。','',
        '先前24模型结果不并入本次统计；首次OOM失败尝试保留，当前版本仅恢复已验证的CPU激活卸载，数据、种子、初始权重、预算和统计方法保持原登记。','',
        f'源数据PIN：`{data["source_pin"]}`；登记摘要：`{data["registration_sha256"]}`。',
        f'独立核验了{data["verified_files"]}个文件、{data["training_decisions"]}条训练转移及{data["evaluation_decisions"]}条评估转移。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    provenance=dict(audit_path=str(audit_path.resolve()),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exploration_replicated=replicated,temporal_order_evidence=temporal,r2_enabled=False,new_stimulation=False)
    (output/'report_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(render(args.audit,args.output)))
