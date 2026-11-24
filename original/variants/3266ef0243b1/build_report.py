"""Generate final descriptive/paired report only from verified full delivery."""
import csv,hashlib,json,sys
from pathlib import Path
from statistics import mean
from datetime import datetime
r=Path(__file__).resolve().parent;frozen=r/'project_v2';sys.path.insert(0,str(frozen))
from agent.mea_stim_offline import analyze,PROTOCOL,validate_budget
from agent.mea_protocol import digest
def read(p):return json.loads(p.read_text(encoding='utf-8'))
if read(r/'delivery_verified.json').get('passed') is not True:raise RuntimeError('Delivery not verified')
d=r/'delivery';exp=d/'experiment01';reg=read(exp/'registration.json');saved=read(exp/'analysis.json');accepted=read(d/'accepted_campaign_manifest.json')
job=read(d/'job.json');batch=read(exp/'batch_status.json')
runtime_s=(datetime.fromisoformat(job['finished_utc'])-datetime.fromisoformat(job['started_utc'])).total_seconds()
stage_means={stage:mean(q['seconds'] for c in batch['cells'] for q in c['stages'] if q['stage']==stage) for stage in ('train','evaluate','verify')}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,h in read(r/'project_source_manifest_v2.json').items():
    if sha(frozen/name)!=h:raise RuntimeError('Local frozen report code changed')
rows=[];cells=[];parameter_starts=[];train_ticks=[];train_tokens=[];eval_rows=0;terminal_count=0;counts={}
for b in range(8):
    for arm in PROTOCOL['arms']:
        out=exp/f'b{b:02d}_{arm}';training=read(out/'train/report.json');evaluation=read(out/'evaluation.json')
        validate_budget(training['decisions'],[x['applied_steps'] for x in training['rounds']]);parameter_starts.append(training['initial'])
        train_ticks.append(training['elapsed_ticks'])
        cells.append(dict(block=b,arm=arm,**{p:{c:evaluation['metrics'][p][c]['discounted_r1'] for c in ('before','after')} for p in ('pre','post')}))
        for p in ('pre','post'):
            for context in ('before','after'):
                m=evaluation['metrics'][p][context];rows.append(dict(block=b,arm=arm,checkpoint=p,context=context,**m))
                eval_rows+=m['decisions']
                if p=='post':terminal_count+=int(m['terminated'])
        for path in out.glob('train/round_*/rollout.jsonl'):
            records=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
            for q in records:
                if q['reward']!=q['r1_score']['reward'] or q['reward_record']['total_reward']!=q['reward']:raise RuntimeError('R1-only violation')
                train_tokens.append(len(q['prepared']['input_ids']))
        counts[arm]=counts.get(arm,0)+training['decisions']
if any(x!=reg['common_initial'] for x in parameter_starts):raise RuntimeError('Initial tensors differ')
computed=analyze(cells)|dict(cells=cells,registration_sha256=digest(reg),source_pin=reg['source_pin'])
if computed!=saved:raise RuntimeError('Local aggregate differs')
for name in accepted['files']:
    if name.endswith(('evaluation.json','report.json','registration.json','analysis.json','rollout.jsonl','independent_verification.json')) or 'verified_b' in name:
        if sha(d/name)!=accepted['files'][name]['sha256']:raise RuntimeError('Report input changed')
with (r/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axes=plt.subplots(1,2,figsize=(12,4.5));colors=['#7E8792','#2878B5','#E38C2B'];labels=['No state2','Before-stim state2','After-stim state2']
for b in range(8):
    values=[mean(next(c for c in cells if c['block']==b and c['arm']==a)['post'].values()) for a in PROTOCOL['arms']]
    axes[0].plot(range(3),values,color='#9AA5B1',alpha=.5,marker='o',markersize=3)
means=[saved['arms'][a]['post'] for a in PROTOCOL['arms']]
axes[0].scatter(range(3),means,c=colors,s=100,zorder=4);axes[0].set_xticks(range(3),labels,rotation=12)
axes[0].set_ylabel('Heldout discounted R1 (mean of two common contexts)');axes[0].set_title('Eight paired digital seeds');axes[0].grid(axis='y',alpha=.2)
for i,key in enumerate(['after_minus_before','after_minus_none']):
    q=saved['contrasts'][key]['post'];lo,hi=q['bootstrap_95'];m=q['mean_difference']
    axes[1].plot([lo,hi],[i,i],color=colors[i+1],linewidth=3);axes[1].scatter([m],[i],color=colors[i+1],s=60)
axes[1].axvline(0,color='#777',linestyle='--');axes[1].set_yticks([0,1],['After minus before','After minus no state2']);axes[1].set_xlabel('Paired R1 difference; bootstrap95% interval');axes[1].grid(axis='x',alpha=.2)
fig.suptitle('Chip28080 offline training comparison: one biological recording, no sham control');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(r/('comparison.'+ext),dpi=180)
plt.close(fig)

conclusion='达到预先约定的本批离线探索性优势条件。' if saved['positive_evidence'] else '未达到预先约定的本批离线探索性优势条件。'
text=['# 刺激前后 state2 的离线训练对照结果','',conclusion,'',
 '本报告只分析芯片28080已完成的单点15脉冲记录。后来多点零幅值失败数据未使用；本次没有连接设备、发送刺激或开启实时训练。R2仅记录，alpha=1。','',
 '| 训练组 | 初始模型留存R1 | 训练后留存R1 | 平均训练增量 |','|---|---:|---:|---:|']
cn={'none':'无state2','before':'刺激前state2','after':'刺激后state2'}
for a in PROTOCOL['arms']:
    q=saved['arms'][a];text.append(f"| {cn[a]} | {q['pre']:.6f} | {q['post']:.6f} | {q['post']-q['pre']:+.6f} |")
text+=['','各组均在同样的刺激前、刺激后留存输入上交叉评估，上表对两种上下文取平均。没有用各组专属测试输入制造分数差异。','',
 '| 预定比较 | 配对差值 | bootstrap95%区间 | 校正后p值 |','|---|---:|---|---:|']
for key,label in [('after_minus_before','刺激后训练 − 刺激前训练'),('after_minus_none','刺激后训练 − 无state2训练')]:
    q=saved['contrasts'][key]['post'];text.append(f"| {label} | {q['mean_difference']:+.6f} | [{q['bootstrap_95'][0]:+.6f}, {q['bootstrap_95'][1]:+.6f}] | {q['p_bonferroni_two_contrasts']:.6f} |")
text+=['','以下是描述性次指标，均取最终模型、8个数字种子及两种共同留存上下文的平均值；不据此替换主指标或挑选阳性结果。','',
 '| 训练组 | 生命值 | 饱食储备 | 水分储备 | 探索格数 | 游戏tick |','|---|---:|---:|---:|---:|---:|']
for a in PROTOCOL['arms']:
    selected=[q for q in rows if q['arm']==a and q['checkpoint']=='post']
    values=[mean(q[k] for q in selected) for k in ('final_health','final_hunger','final_thirst','visited_count','elapsed_ticks')]
    text.append('| '+cn[a]+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
text+=['','游戏字段hunger/thirst表示剩余饱食和水分储备，越高代表储备越多。游戏tick记录实际经过时间；在固定决策上限内，它也受动作耗时影响，不能单独解释为存活更久。','',
 '| 描述性训练增量比较 | 配对增量差值 | bootstrap95%区间 |','|---|---:|---|']
for key,label in [('after_minus_before','刺激后组增量 − 刺激前组增量'),('after_minus_none','刺激后组增量 − 无state2组增量')]:
    q=saved['contrasts'][key]['gain'];text.append(f"| {label} | {q['mean_difference']:+.6f} | [{q['bootstrap_95'][0]:+.6f}, {q['bootstrap_95'][1]:+.6f}] |")
text+=['','差值为正表示刺激后数据训练组较高。整体探索性优势只依据上方最终模型R1的两项预定比较：两者均要求均值为正、区间下界大于0、Bonferroni校正p<0.05；辅助指标和增量表不替换这一规则。不挑选某个种子、上下文或中间检查点。','',
 f'![对比图]({(r/"comparison.png").as_posix()})','',
 '训练前后数据各取600个完整1秒窗口，统一排除刺激通道640；刺激后从最后脉冲结束60秒后取样。两段原始波形全样本检查均无帧缺口、无ADC轨限样本，state2已从H5独立重算。','',
 '每个阶段用384个窗口训练，保留最后bootstrap窗口，中间隔离60秒，再用150个窗口评估。三组各8配对数字种子，每个模型3轮×128=384次决策，actor每轮2次、critic每轮16次实际更新。所有模型从同一原始Qwen＋LoRA＋critic开始，基础模型与参考模型冻结。','',
 f'共24个模型、{sum(counts.values())}次训练决策、{eval_rows}条评估决策完成新进程重放核验。每模型游戏tick实际范围{min(train_ticks)}–{max(train_ticks)}，训练输入token实际范围{min(train_tokens)}–{max(train_tokens)}；同决策预算不等于相同游戏时长或算力。最终评估自然终止{terminal_count}/48次，其余在150决策上限截尾，不能据此宣称完整寿命延长。','',
 '独立核验覆盖输入快照、token、合法动作、原生游戏状态、R1/GAE、保存概率/价值、初始/最终参数和优化器更新计数。未重新数值复现Adam每个动量更新或随机采样流，不能把轨迹核验写成逐比特重训练复现。','',
 '只有一颗类器官、一次刺激前后记录，没有随机假刺激生物对照。8个种子是数字重复，不能当作8颗类器官；结果受时间漂移、记录阶段与短训练/评估时长限制。即使模型分数出现优势，也不能证明刺激的生物因果作用、探索欲或在线闭环效果。未加入打乱state2组，未检验时序特异性。','',
 '工程记录：首次GPU注册因缺少旧卷积运行时路径被拒绝，训练尚未开始；恢复已验证的原环境后四组原始参数哈希通过，没有修改检查点或放宽门槛。完整421项本地测试374通过47环境跳过，GPU33项32通过1跳过。汇总状态提前成功问题经失败回归修复；CLI失败退出码不作为完成依据，采用24份核验回执及完整汇总状态。','',
 f"计算总耗时{runtime_s/60:.2f}分钟。每模型平均训练{stage_means['train']/60:.2f}分钟、留存评估{stage_means['evaluate']/60:.2f}分钟、独立核验{stage_means['verify']/60:.2f}分钟，24个模型顺序运行。17:06前后的18秒资源抽样显示：单个GPU进程约9GiB显存、系统可用内存约112GiB，未见明显换页或I/O等待；GPU利用率在0–96%间波动。这是短时采样，不能作为整个运行期的完整性能剖析。",'',
 f'[逐组与上下文数据]({(r/"comparison.csv").as_posix()}) · [完整机器结果]({(exp/"analysis.json").as_posix()}) · [数据交付校验]({(r/"delivery_verified.json").as_posix()})','',
 '所有检查点、优化器状态、轨迹、日志与封存源码保存在本目录delivery及campaign_delivery.zip。源H5保留在本轮原始采集备份目录。']
(r/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
verification=dict(passed=True,cells=24,training_decisions=sum(counts.values()),evaluation_decisions=eval_rows,analysis_recomputed=True,
                  source_pin=reg['source_pin'],positive_evidence=saved['positive_evidence'],report_sha256=sha(r/'REPORT.md'))
(r/'report_verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2))
