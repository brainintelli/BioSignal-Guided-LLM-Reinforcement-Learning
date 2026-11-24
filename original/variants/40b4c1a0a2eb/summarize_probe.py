"""Validate downloaded evidence and report input participation, not efficacy."""
import csv,hashlib,json,math
from pathlib import Path
r=Path(__file__).resolve().parent
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
delivery=read('gpu_delivery.json');assert delivery['passed']
for n,h in delivery['files'].items():assert sha(r/n)==h
source=read('source_manifest.json')
for n,q in source['files'].items():assert sha(r/n)==q['sha256']
audit=read('audit.json');data=read('probe_input.json');probe=read('model_probe.json');receipt=read('model_probe_verification.json')
old=json.loads((r.parent/'mea_stim_offline_20261117/registration.json').read_text(encoding='utf-8'))
assert probe['initial']==probe['final']==old['common_initial']
assert receipt['passed'] and receipt['exact_outputs_recomputed'] and receipt['parameters_unchanged']
assert probe['optimizer_updates']==probe['device_commands_sent']==probe['environment_actions_executed']==0
assert probe['formal_training_admitted'] is False and probe['critic_evaluated'] is False and probe['alpha']==1 and not probe['r2_reward_enabled'] and not probe['realtime_training']
rows=[];codes=['移动','前往','采集','吃饭','喝水','休息','建造','等待'];features=data['window']['features']
assert len(probe['rows'])==16
for b in range(8):
    pair=[q for q in probe['rows'] if q['block']==b];a=next(x for x in pair if x['condition']=='none');z=next(x for x in pair if x['condition']=='provisional_state2')
    assert a['observation']==z['observation']
    assert a['prepared']['provisional_state2'] is None
    assert z['prepared']['provisional_state2']==dict(zip(data['window']['feature_names'],features))
    assert a['prepared']['input_ids']!=z['prepared']['input_ids']
    for q in pair:
        assert q['prepared']['contract']=='mea.external_spikes.actor_inference.v1' and not q['prepared']['training_admitted']
        assert not q['prepared']['external_source_quality']['raw_quality_verified']
        assert 'wetware_snapshot' not in q['prepared']
        for k in ['verb_probabilities','conditional_target_probabilities']:
            assert all(math.isfinite(v) and 0<=v<=1 for v in q[k]) and abs(sum(q[k])-1)<1e-6
        assert q['proposed_game_action']==q['prepared']['branches'][q['selected_verb']][q['selected_target']]
    delta=[abs(x-y) for x,y in zip(a['verb_probabilities'],z['verb_probabilities'])]
    rows.append(dict(context=b,none_greedy_verb=codes[a['selected_verb']],state2_greedy_verb=codes[z['selected_verb']],
        max_verb_probability_change=max(delta),total_variation_distance=sum(delta)/2,
        proposed_action_changed=a['proposed_game_action']!=z['proposed_game_action'],none_tokens=len(a['prepared']['input_ids']),state2_tokens=len(z['prepared']['input_ids'])))
with (r/'model_input_comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(passed=True,contexts=8,unique_neural_windows=1,model_inputs=16,
    changed_probability_contexts=sum(x['max_verb_probability_change']>1e-8 for x in rows),changed_proposed_actions=sum(x['proposed_action_changed'] for x in rows),
    max_probability_change=max(x['max_verb_probability_change'] for x in rows),weights_unchanged=True,training_performed=False,
    performance_improvement_evaluated=False,raw_quality_verified=False,formal_training_admitted=False)
with (r/'probe_summary.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
text=['# 新增22021刺激文件：离线模型接入结果','',
 '已将本文件导出的数值state2输入原始Qwen3.5-0.8B＋LoRA，完成实际actor前向推理。当前完成的是接入检查，未进行训练更新、critic评估、游戏执行或性能优劣评估。R2只记录、alpha=1，实时训练关闭。','',
 f"文件记录时间为{audit['start_beijing']}至{audit['stop_beijing']}，共{audit['wall_duration_s']:.3f}秒。H5芯片编号22021已由用户确认；类器官身份未提供。此文件包含202组、606条刺激标记，标记中的amplitude/phase均为200；这些是记录的参数，不能当作实测输出电压或波形。",'',
 f"尖峰共{audit['spikes_total']:,}条，原始映射746行对应744个不同通道；93341条尖峰落在映射外。刺激通道807、823各对应两个电极，存在身份歧义。本次统一排除全部10个刺激通道（包括这两路）及映射外事件，仅保留734个无内部映射冲突的通道。没有原始波形或连续帧号，不能验证刺激伪迹、ADC轨限或数据完整性。",'',
 '最后刺激标记后可观察到约62.007秒尖峰。沿用60秒保护间隔，再增加1秒保守余量后，选取61–62秒的单个1秒窗口作为临时推理输入。窗口被已有尖峰时间边界包围，不代表这1秒已获原始信号完整性或无伪迹认证。','',
 '| state2特征 | 数值 |','|---|---:|',
 f'| 尖峰数 | {features[0]:.0f} |',f'| 每通道平均放电率 | {features[1]:.6f} Hz |',
 f'| 活跃通道比例 | {features[2]:.6f} |',f'| 平均尖峰绝对幅值 | {features[3]:.6f} μV |','',
 '特征从源H5以另一种计算路径独立重建，158条事件完全一致，数值误差小于1e-10。使用H5的lsb转换到微伏；没有用留存结果拟合归一化参数。源H5已从微信临时目录复制并校验。','',
 '在8个固定游戏初始状态上，各计算不接入state2和接入该窗口两种输入，共16个输入。两种输入共享相同的数据质量提示；只有数值state2有无不同。没有将数据包装成audited_finalized_h5快照，也没有送进正式训练数据入口。','',
 f"8个状态中，{summary['changed_probability_contexts']}个状态的动作类别概率发生数值变化，{summary['changed_proposed_actions']}个状态的贪心建议动作不同；最大概率绝对变化为{summary['max_probability_change']:.6f}。这些结果只说明模型接收了输入并产生不同输出，不代表游戏得分、探索能力、存活时长或生物计算效果改善。8个状态重复使用同一个1秒窗口，不是8份独立神经数据。",'',
 '| 游戏状态 | 无state2建议类别 | 接入state2建议类别 | 最大概率变化 |','|---|---|---|---:|']
for q in rows:text.append(f"| {q['context']} | {q['none_greedy_verb']} | {q['state2_greedy_verb']} | {q['max_verb_probability_change']:.6f} |")
text+=['','模型使用与上一轮相同的原始actor/critic/base/reference四组参数哈希。两次独立进程得到完全一致的输入词元、概率及建议动作；运行前后四组参数均未改变，梯度为空，优化器更新为0。','',
 '本文件不满足上一轮384步训练＋60秒隔离＋150步留存评估的长度，也缺少原始波形和同一类器官的匹配基线。用户已确认暂无其他文件，因此本轮保留为独立推理接入结果，不与28080的结论合并。后续需同一类器官至少10分钟刺激前基线、停止刺激后至少12分钟连续记录，均保存原始波形及刺激事件，才继续同预算训练对照。','',
 f"[数值state2输入]({(r/'probe_input.json').as_posix()}) · [模型输入对比]({(r/'model_input_comparison.csv').as_posix()}) · [独立进程核验]({(r/'model_probe_verification.json').as_posix()})",'',
 f"源SHA256：{data['source_sha256']}。原始文件备份source.raw.h5，审计audit.json，完整模型输入/输出model_probe.json，均保存在本目录。"]
with (r/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
print(json.dumps(summary,indent=2));print('REPORT',sha(r/'REPORT.md'))
