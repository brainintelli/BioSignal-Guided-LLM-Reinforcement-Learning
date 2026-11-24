"""Summarize only a fully verified, sealed campaign copied from the GPU host."""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone,timedelta
from statistics import mean
from agent.mea_long_ablation import PIN,ARMS,analyze
from tools.mea_long_ablation import validate_receipt
from agent.mea_protocol import digest

root=Path(__file__).resolve().parent
delivery=root/'delivery';campaign=delivery/'experiment01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

reg=read(campaign/'registration.json');status=read(campaign/'batch_status.json');result=read(campaign/'analysis.json')
assert status['complete'] is True and status['passed'] is True and result['status']=='complete'
assert reg['protocol']['source_manifest_sha256']==PIN
assert reg['common_initial']==reg['protocol']['original_tensors']
cells=[];decompositions={a:[] for a in ARMS};report_rows={a:[] for a in ARMS};token_counts={a:[] for a in ARMS};verification_rows=0;verified_files=0
for b in range(8):
    for arm in ARMS:
        out=campaign/f'b{b:02d}_{arm}'
        receipt=read(campaign/f'verified_b{b:02d}_{arm}.json')
        validate_receipt(receipt,b,arm,digest(reg),sha(out/'cell_manifest.json'))
        manifest=read(out/'cell_manifest.json')
        assert set(manifest)=={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}-{'cell_manifest.json'}
        for name,h in manifest.items():
            p=(out/name).resolve();assert p.is_relative_to(out.resolve()) and sha(p)==h,name
            verified_files+=1
        report=read(out/'evaluation.json');train=read(out/'train/report.json');independent=read(out/'train/independent_verification.json')
        assert independent['passed'] is True and train['initial']==reg['common_initial']
        assert train['decisions']==1024 and train['heldout_used'] is False
        assert train['alpha']==1 and train['r2_enabled'] is False and train['stimulation'] is False and train['realtime_training'] is False
        assert all(r['applied_steps']=={'actor':2,'critic':16} for r in train['rounds'])
        for rd in range(1,9):
            with (out/f'train/round_{rd:02d}/rollout.jsonl').open(encoding='utf-8') as f:
                token_counts[arm].extend(len(json.loads(line)['prepared']['input_ids']) for line in f)
        assert report['checkpoint_tensors']['pre']==reg['common_initial'] and report['checkpoint_tensors']['post']==train['final']
        cells.append(dict(block=b,arm=arm,**{p:report['metrics'][p]['discounted_r1'] for p in ('pre','post')}))
        report_rows[arm].append(report)
        discounted={'survival':0.,'potential':0.};elapsed=0
        with (out/'post/rollout.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                assert r['reward']==r['r1_score']['reward'] and r['reward_record']['alpha']==1
                for k in discounted:discounted[k]+=.99985**elapsed*r['r1_score']['components'][k]
                elapsed+=r['duration']
        assert abs(sum(discounted.values())-report['metrics']['post']['discounted_r1'])<1e-8
        decompositions[arm].append(discounted)
        verification_rows+=1024+sum(report['metrics'][p]['decisions'] for p in ('pre','post'))
recomputed=analyze(cells)
for k,v in recomputed.items():assert result[k]==v,k
receipt=dict(passed=True,verified_utc=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=PIN,registration_sha256=digest(reg),
    verified_cells=24,verified_cell_files=verified_files,verified_training_decisions=24576,
    verified_training_and_evaluation_rows=verification_rows,analysis_independently_recomputed=True,
    common_original_tensors=reg['common_initial'],positive_evidence=result['positive_evidence'])
(root/'local_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

labels={'none':'无 state2','real':'真实顺序 state2','shuffled':'打乱顺序 state2'}
conclusion=('在这批记录及预登记数字种子下，真实 state2 在固定决策数评估的 R1 指标上同时优于两组对照，达到了本轮预设的阳性判据。'
    if result['positive_evidence'] else '本轮未获得真实 state2 在主要 R1 指标上同时稳定优于两组对照的证据。')
lines=['# 芯片 28080：30 分钟 MEA 三组离线对照','',f'**{conclusion}**','',
    '全程无刺激、R2 只记录、α=1，实时训练关闭。没有连接或控制采集设备。','',
    '## 结果','',
    '每组 8 个配对数字种子，共 24 个单元。每个单元均从原始 combined_0064 权重开始，训练 1,024 次决策（8 轮，每轮 actor 2 次、critic 16 次更新）。前测和后测分别加载初始与最终权重，均在训练结束后只读评估留存数据；最多 600 次决策，遇自然死亡提前结束。主要指标是按游戏 tick 折扣的累计 R1，越高越好。','',
    '| 条件 | 前测平均 R1 | 后测平均 R1 | 前后变化 | 后测平均 tick | 后测存活至截止 |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
tz=timezone(timedelta(hours=8))
times=[datetime.fromisoformat(status[k]).astimezone(tz).strftime('%Y-%m-%d %H:%M:%S') for k in ('started_utc','finished_utc')]
lines.insert(2,f'运行时间：{times[0]} 至 {times[1]}（北京时间）。')
lines.insert(3,'')
for arm in ARMS:
    a=result['arms'][arm];post=[r['metrics']['post'] for r in report_rows[arm]]
    lines.append(f"| {labels[arm]} | {a['pre']:.6f} | {a['post']:.6f} | {a['post']-a['pre']:+.6f} | {mean(p['elapsed_ticks'] for p in post):.1f} | {sum(not p['terminated'] for p in post)}/8 |")
lines+=['','| 主要比较 | 后测平均差 | 配对 bootstrap 95% 区间 | 双侧 p | 两比较修正 p |',
    '| --- | ---: | --- | ---: | ---: |']
for control in ('none','shuffled'):
    c=result['contrasts']['real_minus_'+control]['post'];lo,hi=c['bootstrap_95']
    lines.append(f"| 真实 − {labels[control]} | {c['mean_difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {c['p_two_sided']:.6f} | {c['p_bonferroni_two_contrasts']:.6f} |")
lines+=['','阳性判据在运行前固定：两项主要比较的均值差和区间下限均为正，且两比较修正 p 均小于 0.05。检验为 8 个配对块上的双侧符号翻转；区间来自 20,000 次配对 bootstrap。样本数小，推断依赖配对块差异的交换/对称性假设。','',
    '| 辅助比较：前后增益差 | 平均差 | 95% 区间 |','| --- | ---: | --- |']
for control in ('none','shuffled'):
    c=result['contrasts']['real_minus_'+control]['gain'];lo,hi=c['bootstrap_95']
    lines.append(f"| 真实 − {labels[control]} | {c['mean_difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
real=result['arms']['real']
lines+=['',f"真实组平均 R1 从前测 {real['pre']:.6f} 变为后测 {real['post']:.6f}（变化 {real['post']-real['pre']:+.6f}）。后测组间差异与训练带来的额外增益是两个问题，应分别判断。"]
if (result['contrasts']['real_minus_none']['post']['p_bonferroni_two_contrasts']<.05
    and not result['positive_evidence']):
    lines+=['','本轮真实组相对无 state2 的后测差异通过了两比较修正，但未显示相对打乱组的优势。因此不能将这项差异归因于真实 MEA 的时间顺序，也未达到同时优于两组对照的预设判据。增加输入、特征分布或动作持续时间变化等解释仍需进一步区分。']
if all(c['gain']['bootstrap_95'][0]<=0<=c['gain']['bootstrap_95'][1] for c in result['contrasts'].values()):
    lines+=['','两项前后增益差的区间均覆盖 0，当前结果也不支持真实 state2 带来了额外的强化学习训练收益。未检出优势不等于证明两组等效。']
lines+=['','## 指标解释','',
    '相同决策次数可能对应不同游戏 tick，R1 包含存活时长项和资源势函数项。以下为后测累计 R1 的分解，不能把更长的动作持续时间直接解释成更好的资源管理。固定 600 决策评估不等同于等游戏时长或完整生存期实验。','',
    '| 条件 | 折扣存活时长项 | 折扣势函数项 | 最终 health | 最终 hunger | 最终 thirst |','| --- | ---: | ---: | ---: | ---: | ---: |']
for arm in ARMS:
    d=decompositions[arm];p=[r['metrics']['post'] for r in report_rows[arm]]
    lines.append(f"| {labels[arm]} | {mean(x['survival'] for x in d):.6f} | {mean(x['potential'] for x in d):.6f} | {mean(x['final_health'] for x in p):.2f} | {mean(x['final_hunger'] for x in p):.2f} | {mean(x['final_thirst'] for x in p):.2f} |")
if all(not r['metrics']['post']['terminated'] for a in ARMS for r in report_rows[a]):
    lines+=['','三组后测均为 8/8 存活至评估截止，生存终点在本次观察窗口内没有区分度；不能把 R1 的组间差异直接称为生存能力提升。']
lines+=['','## 数据与可追溯性','',
    '- 数据为芯片 28080 的约 30 分钟无刺激记录；按封存 H5 元数据与原始保存通道交集分析 831 路。硬件映射尚未独立确认，不能用于推断刺激靶点。',
    '- 训练当前窗 0–1023，边界窗 1024；间隔 1025–1084；评估当前窗 1085–1684，边界窗 1685。训练和评估不交叉，但间隔本身不能证明统计独立。',
    '- 四项 MEA 特征作为整组在分区内打乱，边界固定。无 state2 组保持相同 112 维 critic，四个神经输入为零，并去除 actor 的神经观测文本。',
    '- 等预算指决策数、训练轮数与优化步骤相同；神经文本增加输入 token，因此不声称三组 FLOPs 或运行时长相同。',
    '- 所有单元保留，不按成绩替换种子或挑选模型。初始 actor、critic、base、reference 张量均与先前已审计原始权重指纹一致。',
    f'- 独立新进程验证全部 24 个单元，共 {verification_rows:,} 条训练与评估记录，包括原生游戏回放、输入、策略/价值统计、GAE 和优化器步数。本地逐文件核验并重算统计结果。',
    '- 验证没有逐位重跑整个随机采样和 Adam 矩演化；证明范围是所保存行为、概率、值函数、更新证据与计数的一致性。','',
    '## 结论边界','',
    '这是同一批预录数据上的离线数字对照。8 个数字种子不是 8 个独立类器官；真实信号也不是当前游戏动作触发的神经反馈。即使达到本轮阳性判据，也只能支持这批数据在此软件任务中的条件性收益，不能证明湿件因果计算收益或跨样本泛化。','',
    '完整远程结果：`192.0.2.15:/home/researcher/mea-long-ablation-20261116/experiment01`。本地同目录下 `delivery/experiment01` 保留完整检查点、轨迹、验证记录和统计结果。','',
    f'数据封存指纹：`{PIN}`。','',
    '实现裁决：采用固定 8 个种子和 600 决策评估，计算需数小时且仍受单次生物记录限制；工作区无 Git，因此以隔离目录、冻结源码和哈希清单恢复及追溯；保留所有审查与运行记录。无遗留次要审查问题。','']
lines+=['训练决策的平均输入 token：'+ '；'.join(f'{labels[a]} {mean(token_counts[a]):.1f}' for a in ARMS)+'。','']
lines+=['## 结果文件','',
    f'- [统一统计 JSON]({(campaign/"analysis.json").as_posix()})',
    f'- [本地结果复核]({(root/"local_verification.json").as_posix()})',
    f'- [传输与逐文件校验]({(root/"transport_verification.json").as_posix()})',
    f'- [完整轨迹与检查点归档]({(root/"campaign_delivery.zip").as_posix()})','']
(root/'ABLATION_28080_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(receipt))
