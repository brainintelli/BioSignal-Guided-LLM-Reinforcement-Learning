"""Render the verified fixed-budget comparison, including negative results."""
import datetime,json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence/experiment01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def f(x):return f'{x:.6f}'
def interval(xs):return '['+', '.join(f(x) for x in xs)+']'
def main():
    a=read(E/'analysis.json');s=read(R/'descriptive_results.json');v=read(R/'download_verification.json');t=read(R/'transport_verification.json');reg=read(E/'registration.json')
    assert a['status']=='complete' and v['passed'] and t['passed'] and read(E/'independent_analysis.json')['passed']
    events=[json.loads(line) for line in (E/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    timezone=datetime.timezone(datetime.timedelta(hours=8))
    period=' — '.join(datetime.datetime.fromtimestamp(events[i]['time'],timezone).strftime('%Y-%m-%d %H:%M:%S') for i in (0,-1))
    labels=dict(baseline='原 R1',state='仅 state2',reward='仅辅助奖励',both='state2＋辅助奖励',shuffled='state2＋辅助奖励打乱对照')
    lines=['# 22554 离线 state 与辅助奖励探索性对照', '', f'正式训练及复核：北京时间 {period}；服务器 `192.0.2.16`。', '',
        '**结论：本轮没有证明 state2＋辅助奖励组合能可靠提升模型成绩。仅辅助奖励组出现了值得独立复验的正向信号。**', '',
        f"组合组训练后原 R1 比基线高 {f(a['primary']['both-baseline']['post']['mean_difference'])}，95% 配对 bootstrap 区间 {interval(a['primary']['both-baseline']['post']['bootstrap_95'])}，双侧精确检验经两项主要比较校正后 p＝{a['primary']['both-baseline']['post']['p_adjusted_primary']}。组合组相对打乱组的差值为 {f(a['primary']['both-shuffled']['post']['mean_difference'])}，校正后 p＝{a['primary']['both-shuffled']['post']['p_adjusted_primary']}。因此没有证据将收益归因于保留神经记录的原始顺序。", '',
        f"仅辅助奖励相对基线的训练后差值为 {f(a['secondary']['reward-baseline']['post']['mean_difference'])}，95% 区间 {interval(a['secondary']['reward-baseline']['post']['bootstrap_95'])}，未校正 p＝{a['secondary']['reward-baseline']['post']['p_two_sided']}。这是预先列出的次要比较，包含多项探索性比较，没有作为确认性检验控制其多重性；不能据此宣称神经辅助奖励已验证有效。", '',
        '## 五组成绩', '',
        '主指标是每组 8 个配对种子块、每块 2 个留存世界的平均折扣原 R1。每条评估轨迹最多 96 次决策，遇到自然死亡才提前结束。数值越高越好；辅助奖励不计入这个分数。', '',
        '| 条件 | 训练前原 R1 | 训练后原 R1 | 前后变化 |', '| --- | ---: | ---: | ---: |']
    for arm,label in labels.items():
        x=s['arms'][arm];lines.append(f"| {label} | {f(x['pre']['discounted_r1'])} | {f(x['post']['discounted_r1'])} | {f(x['gain'])} |")
    lines+=['', '小样本 percentile bootstrap 区间与精确检验来自不同方法，可能不完全对应；本轮按预登记的精确检验判断主要比较，不能只凭区间不跨零认定效果。新增输入会立即改变预训练模型的行为，因此训练后的组间差值不全是学习效果。仅 state2 和组合组的前测相同；原 R1 和仅奖励组的前测也相同。', '',
        '| 比较（前者减后者） | 训练后差值 | 训练后 95% 区间 | 双侧 p（原始） | 主要比较校正 p | 前后增益差 |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    pairs=[('both-baseline','组合－原 R1',True),('both-shuffled','组合－打乱',True),('state-baseline','仅 state2－原 R1',False),('reward-baseline','仅奖励－原 R1',False),('both-state','组合－仅 state2',False)]
    for key,label,primary in pairs:
        x=a['primary' if primary else 'secondary'][key];p=x['post'];lines.append(f"| {label} | {f(p['mean_difference'])} | {interval(p['bootstrap_95'])} | {p['p_two_sided']} | {p.get('p_adjusted_primary','—')} | {f(x['gain']['mean_difference'])} |")
    lines+=['', f"组合相对基线的前后增益差为 {f(a['primary']['both-baseline']['gain']['mean_difference'])}，95% 区间 {interval(a['primary']['both-baseline']['gain']['bootstrap_95'])}，未校正 p＝{a['primary']['both-baseline']['gain']['p_two_sided']}。这也不支持本轮组合训练带来更大的学习增益。全部逐种子差值、增益区间及原始 p 值见 `evidence/experiment01/analysis.json`。", '',
        '## 短程行为与训练量', '',
        '| 条件 | 后测原 R1 未折扣和 | 后测观察 tick | 后测决策数 | 后测自然死亡 / 16 | 后测末健康 | 饥饿资源值 | 口渴资源值 | 等待动作比例 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,label in labels.items():
        x=s['arms'][arm]['post'];lines.append(f"| {label} | {f(x['raw_r1'])} | {x['ticks']:.2f} | {x['decisions']:.2f} | {x['deaths']}/16 | {x['final_health']:.2f} | {x['final_hunger']:.2f} | {x['final_thirst']:.2f} | {x['wait_fraction']:.2%} |")
    lines+=['', 'tick 是这段固定决策预算内实际观察到的跨度，不能直接当作完整寿命。hunger/thirst 是游戏原始资源字段；库存和所有动作计数保存在 `descriptive_results.json`，不将字段名称解释成主观感受。']
    if sum(s['arms'][arm]['post']['deaths']+s['arms'][arm]['pre']['deaths'] for arm in labels)==0:
        lines+=['', '**160 条前后测轨迹都到达 96 次决策上限，没有观察到自然死亡；本实验未验证长期生存能力。**']
    lines+=['', '| 条件 | 8 个种子块实际 actor 步数 | 每块 critic 步数 | 每块训练原 R1 和（均值） | 每块实际优化奖励和（均值） |', '| --- | --- | --- | ---: | ---: |']
    for arm,label in labels.items():
        x=s['arms'][arm]['train'];lines.append(f"| {label} | {', '.join(map(str,x['actor_steps']))} | {', '.join(map(str,x['critic_steps']))} | {f(x['original_r1_mean_sum'])} | {f(x['optimized_mean_sum'])} |")
    lines+=['', f"40/40 单元完成并通过新进程复核；每个模型独立从原始 checkpoint_0064 开始，训练 128 条决策、进行一次 PPO 调用。正式训练决策总数为 {v['train_decisions']:,}，含前后测的正式新决策共 {v['collection_decisions']:,}；总计 {s['runtime']['actor_steps_total']} 次 actor、{s['runtime']['critic_steps_total']} 次 critic 优化步骤。另有 24 条只使用校准段的接口短测决策，不计入效果样本。没有为追求结果增加预算、替换种子或剔除失败条件。", '',
        '## 实验定义与限制', '',
        '- 使用用户确认的芯片 22554 五分钟无刺激记录。原 H5 严格全文件 QC 失败：133 条未映射事件、2 条记录起点前事件；本次沿用明确的过滤范围，保留 57,517 条事件，并保留全部 135 条排除索引。不能把这个子集重新标为全文件 QC 通过。',
        '- 去掉首尾边缘后构造 298 个不重叠的一秒窗口。仅窗口 0–59 拟合标准化和奖励参考；训练使用 60–187，188 为 bootstrap；189–198 隔离；评估使用 199–294，295 为 bootstrap；296–297 不参与正式训练或评估。',
        '- state2 包含计数、每通道放电率、活跃通道比例和平均绝对幅度，用校准段 median/MAD 标准化并裁剪。五组 critic 都为 112 维，新增四列从零开始；关闭 state2 的组使用原 state1 文本及零神经列。',
        '- 辅助分数采用“基线接近程度”，不是先前对随机噪声反而升高的新颖度。依据 log1p 放电率、活跃比例、10 ms 跨通道共现比例计算接近概率 p，并按校准段均值中心化为 u∈[-1,1]。启用辅助奖励的三组采用 R1＋0.05u，其余两组只用 R1；R1 系数始终为 1。这是独立实验性的加法奖励入口。',
        '- u 来自模型采取动作之前的预录当前窗口，不是该动作诱发的组织响应。一秒记录窗口与一次游戏决策的人为对应不等于真实时间同步；R1、折扣和 GAE 仍使用原生游戏 tick。不能据此证明生物闭环、组织舒适度、组织学习或实时 R2 的效力。',
        '- 打乱组在训练、评估各自范围内同时打乱完整 state2 特征和辅助分数，保留配对关系及边际分布，不跨分段；每块的置换在训练前冻结，前后测及两个世界复用。打乱组后测均值最高，但也不能把未登记的事后比较包装成确认性发现。本轮没有“仅奖励”的打乱或常数辅助奖励对照，不能认定该次要信号来自神经特异信息。',
        '- 8 张 RTX4090 各承载一个配对种子块，块内五组共享 GPU、世界与动作种子并轮换运行顺序；没有把五个条件的梯度混在一起。这是 8 次数字种子重复，生物记录仍只有 1 份，不是 8 个独立类器官。',
        '- 原始 R1 的主要比较只有“组合－基线”和“组合－打乱”，采用 256 种符号翻转的双侧精确检验及 Bonferroni 两项校正；检验依赖配对差值可交换/对称的零假设。20,000 次配对块 bootstrap 的 95% 区间是小样本条件性描述区间，不是类器官总体区间。次要比较和增益检验均为未校正的探索结果。',
        '- 本轮只覆盖一批 128 条训练决策及短程评估。不能从未显著推出永远无效，也不能从仅奖励组的次要信号直接启用实时生物奖励。独立新种子和新记录的预登记复验才会提供新增证据。', '',
        '统计方法定义参考 [SciPy paired permutation 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) 和 [paired bootstrap 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。本任务用标准库实现，并由另一套实现独立复算。', '',
        '## 核验与可追溯性', '',
        f"25 项接口/协议/GPU测试通过。40 个独立新进程在同一物理 GPU 上重放保存动作，重建原生状态、输入、奖励、GAE，并核对初始/更新检查点及优化器内容；全部旧/新策略统计和 bootstrap 的最大误差为 {v['maximum_statistic_error']}。复核不重新抽样动作，也不重复训练，不宣称训练过程已位级重现。", '',
        f"正式收集及复核耗时 {s['runtime']['seconds']/60:.2f} 分钟；单进程最大 CUDA allocated 峰值 {s['runtime']['max_peak_allocated_gib']:.2f} GiB、reserved 峰值 {s['runtime']['max_peak_reserved_gib']:.2f} GiB、RSS 峰值 {s['runtime']['max_process_peak_rss_gib']:.2f} GiB。GPU 监测有 {v['telemetry_samples']} 个样本，错误样本 {v['telemetry_error_samples']}；所有收集/复核 PID 均在实际 GPU 使用记录中出现。结束后无本任务训练或游戏子进程、无 GPU 计算进程。", '',
        f"运行环境 24,551 个文件、5,823,470,108 字节重新核验一致。完整归档 {t['archive_bytes']:,} 字节，本地核验 {t['files']:,} 项，原始 H5 也重新校验。原始归档 SHA256：`{t['archive_sha256']}`。慢速原始传输改用分块无损压缩；恢复后的原始归档哈希完全一致，未改变内容。", '',
        f"注册 SHA256：`{v['registration_sha256']}`。数据 SHA256：`{reg['source_identity']['dataset_sha256']}`。共同起点 actor 语义哈希：`{reg['common_initial']['actor']}`。源码与分析代码在结果产生前已冻结。", '',
        '首次数据审计因 Windows/Linux exp 的约 3.8×10⁻¹⁶ 舍入差异失败；失败日志保留，数据字节与冻结分数未改。仅独立重算分数的比较允许 10⁻¹² 舍入，模型统计的 10⁻⁶ 门槛未放宽。正式 40 组没有失败或替换。', '',
        '- 主分析：`evidence/experiment01/analysis.json`；独立统计复核：`evidence/experiment01/independent_analysis.json`。',
        '- 行为明细：`descriptive_results.json`；本地交付核验：`download_verification.json`；传输核验：`compressed_transport_verification.json`、`transport_verification.json`。',
        '- 每组轨迹及检查点：`evidence/experiment01/b00_baseline` 等全部 40 个目录。检查点使用本实验的 112 维 critic 和独立 `sr_policy.py` 加载入口，不直接替换旧 108 维 critic 服务。',
        '- 服务器目录：`/home/researcher/mea-r2-22554/state_reward_20261116_01/experiment01`；各条件检查点为 `bXX_<arm>/checkpoint`。',
        '- 设计与执行取舍、TDD 失败/通过记录见 `progress.md` 及设计/计划文档。当前入口不进行刺激或实时采集，未改动共享 P0 奖励约束。', '',
        '## 实施取舍记录', '',
        '- 按用户确认使用 22554，避免混用另一条线的 28080 数据；代价是只有这一份短记录，不能推广到其他组织。',
        '- 在独立入口采用 R1 系数固定为 1 的加法辅助奖励；代价是未检验其他权重与候选，不能推广到正式 alpha 混合或在线 R2。',
        '- 选用一秒窗口和校准/训练/留存隔离；代价是短行为时域，以及人为离线关联。',
        '- 八卡作为八个配对种子块，冻结两项主要比较；代价是不回答八卡同步单模型训练的吞吐或更长训练效果。',
        '- 每块固定一个训练/评估置换对；代价是对置换的平均只覆盖这八组预注册排序。复核只重放保存动作而不重训；代价是不提供完整训练位级复现。',
        '- 当前目录没有 Git，以隔离目录和哈希清单保存完整成功/失败证据；未删除实验工作区，避免失去原始记录。用户已明确授权离线探索，无重复审批门槛。', '',
        '独立最终审查通过，无 Critical、Important 或 Minor 发现；审查记录见 [final_review.md](final_review.md)。', '']
    (R/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(R/'report.md'))
if __name__=='__main__':main()
