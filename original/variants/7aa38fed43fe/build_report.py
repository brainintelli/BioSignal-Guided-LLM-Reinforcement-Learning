"""Render only verified measured values into the Chinese delivery report."""
import json
from pathlib import Path
import statistics as stats

folder=Path(__file__).resolve().parent
def read(name):return json.loads((folder/name).read_text(encoding='utf-8'))
r=read('comparison.json');a=read('independent_audit.json');reg=read('registration.json')
policy=read('optimized_policy.json')
replay=read('replay_verification.json');tokens=read('token_verification.json');engine=read('engine_checks.json')
assert replay['passed'] and tokens['passed'] and all(x['passed'] for x in engine)
assert a['evidence_consistent'] and r['passed']==a['passed']
names={'baseline':'原实现 B4×1536','compact':'仅去填充 B1×实际长度','optimized':'去填充＋KV＋合并投影'}
lines=['# 强化学习去填充与 KV cache 深度优化', '',
'通过 Netcatty 在集群 1 节点 5、RTX 6000 Ada 单卡上执行。初始化按用户选择：checkpoint_0032 的 actor 与 critic，重新创建优化器、重新采样。正式训练未启动，诊断权重未保存，没有切换服务。', '',
'## 验收结果', '',
f"本次登记验证{'通过' if r['passed'] else '未通过'}：包含三轮首次开销的采样＋PPO＋critic 总时间加速 **{a['total_speedup']:.4f} 倍**；逐轮总时间中位数加速 **{a['median_speedup']:.4f} 倍**。两项门槛均为严格大于 2.0，相对原固定填充实现计算。相对仅去填充 compact 的总时间加速为 **{a['speedup_vs_compact']:.4f} 倍**。", '',
'| 实现 | 三轮采样秒 | 三轮更新秒 | 合计秒 | 逐轮中位数秒 | 峰值已分配显存 GiB |', '|---|---:|---:|---:|---:|---:|']
for mode in names:
    lines.append(f"| {names[mode]} | {a['sampling'][mode]:.3f} | {a['updating'][mode]:.3f} | {a['totals'][mode]:.3f} | {a['medians'][mode]:.3f} | {a['peak_GiB'][mode]:.2f} |")
lines+=['', '每臂 384 次真实决策、24 次实际 actor 更新；三臂总计 1,152 次真实决策。PPO epochs=2、minibatch=32、microbatch=4、critic epochs=8 均保持不变，没有依赖提前停止或少更新取得速度收益。', '',
'时间包括输入准备、actor 和参考模型采样、Godot、奖励、轨迹写入、GAE、更新前概率检查、PPO 反传、critic 和缓存刷新。模型加载、独立概率检查及固定前向预热在主计时外；测量内新形状的首次开销保留。无 checkpoint 写盘或完整选择/最终评估。因此这是已验证的短程采样与更新吞吐，不能直接当作完整 98,304 决策训练任务或长期收敛保证。', '',
'三臂从相同权重和相同世界种子开始，但采用各自 on-policy 动作，轨迹和 target 分支数量可不同；决策数与更新数相同不代表每轮执行完全相同的 token 工作量。另提供相同 64 状态的固定前向对照。测试按 baseline→compact→optimized 顺序，未做随机交错或跨日重复，不对该单次短程测量给出统计置信区间。', '',
'## 算法与执行路径', '',
'算法仍为 LoRA actor＋独立 97 维事实特征 critic 的 PPO，使用按实际动作 tick 折扣的 GAE、冻结 M6 父模型 KL 正则及条件分支重要性修正。策略分两阶段：P(verb|state) × P(target|state,verb)，在完整合法候选集合上读单 token 概率；只有一个合法目标时目标概率为 1。', '',
'![优化后的游戏强化学习路径](kv_pipeline.svg)', '',
'1. 每个真实输入采用 B1 精确长度，不填充到 1536、不添加虚拟批次行；1536 保留为输入上限，超限报错。读取骨干输出的末位隐状态，直接投影到候选 token，避免计算随后丢弃的全词表输出与返回全部中间隐状态。',
'2. 完整 verb 提示只 prefill 一次，直接得到 verb 概率，并在同一次调用中把 KV/线性注意力状态传给 target 续算。复制缓存容器，共享只读张量，保留反向图；调用结束释放，不能跨 optimizer step 复用。',
'3. LoRA 有效投影按参数版本缓存，冻结基座不改写。MLP gate/up、线性注意力 QKV/z/b/a、全注意力 Q/K/V 分组执行；FP32 分块解析梯度返回各个原 LoRA A/B 参数。',
'4. 融合冻结零中心 RMSNorm 与 MLP SiLU 门控的前后向。权重版本检查集中在公开分布/采样调用入口，省去逐层、逐续算重复检查。', '',
'并行投影和归一化的实际启用数量见 `optimized_policy.json`。新实现位于 `agent/m8x_grouped_lora.py`，继承完整 verb KV 路径；原 72 个冻结核心文件和第一版 92 个注册文件均保持原字节。正式入口没有自动切换。', '',
'## 新数值布局与动作概率', '',
'**新布局不能称为与旧策略逐位等价。** 为使 target 可以承接 verb KV，新 target 的前一个 assistant 消息保留 4-token 空 thinking 标记；观察、动词和合法目标内容不变。FP32 LoRA 合成后舍入为 BF16、分组 GEMM、融合算子和分段 prefill 也改变浮点运算顺序。参考权重仍冻结，但改变计算布局后，其输出概率也可能不同。', '',
'| 实现 / 模型 | 最大联合概率差 | 平均 KL（旧→新） | 最大 KL | 贪心动作变化 / 64 |','|---|---:|---:|---:|---:|']
for mode in ('compact','optimized'):
    for key,label in [('actor_vs_baseline','actor'),('reference_vs_baseline','reference')]:
        p=r['performance'][mode][key]
        lines.append(f"| {mode} / {label} | {p['max_probability_difference']:.8f} | {p['mean_kl']:.8f} | {p['max_kl']:.8f} | {p['greedy_changes']} |")
lines+=['', '比较覆盖 64 个真实既有诊断状态的全部合法联合动作。相对旧策略的差异仅报告，未用事后阈值把它宣称为等价；PPO 的复算门槛适用于新布局自己采集的 old log-prob。', '',
'| 实现 | 逆序与 microbatch=1/3/4 最大 log-prob 差 | no_grad 与梯度模式最大差 |','|---|---:|---:|']
for mode in names:
    p=r['performance'][mode];lines.append(f"| {mode} | {p['zero_update_reordered_max']:.9g} | {p['zero_update_grad_mode_max']:.9g} |")
lines+=['', '逆序检查覆盖 64 个状态，梯度模式检查覆盖其中 16 个状态。每轮实际更新前另外检查最初 16 条新采样的概率复算。门槛均为 1e-6。新实验重置优化器，并重新采集 old log-prob、参考分布和优势，不拼接旧 rollout。', '',
'## 实际更新稳定性', '',
'| 实现 | 轮次 | 采样秒 | 更新秒 | actor steps | 最大近似 KL | actor 梯度范数均值 | 缓存刷新误差 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for mode in names:
    for row in r['stability'][mode]['updates']:
        lines.append(f"| {mode} | {row['update']} | {row['rollout_seconds']:.3f} | {row['update_seconds']:.3f} | {row['optimizer_steps']} | {row['approx_kl_max']:.6f} | {row['actor_grad_norm']:.5f} | {row['weight_cache_refresh_max']:.9g} |")
lines+=['', '表中梯度范数为裁剪前返回值。所有轮次损失、梯度和 KL 有限，zero-update 检查通过，未触发 KL 提前停止；actor 权重确实变化，参考权重未变。每轮优化后清除合成权重与分组权重缓存，强制重建的概率与自动按版本刷新结果一致；KV 调用域已释放，合并投影无未消费分支。短期稳定不证明长期生存率或收敛提高。', '',
'## 固定前向与 max token 判断', '',
'| 实现 | 64 状态三次前向秒 | 前向中位数秒 | 单独首次预热秒 |','|---|---|---:|---:|']
for mode in names:
    p=r['performance'][mode];times=p['fixed_forward_seconds']
    lines.append(f"| {mode} | {', '.join(f'{v:.3f}' for v in times)} | {stats.median(times):.3f} | {p['warmup_seconds']:.3f} |")
lines+=['', '固定前向包括 actor/reference 两阶段概率与 critic，使用相同 64 个状态和已记录 verb，无环境交互或 optimizer step。主时间表则使用各臂自己的新真实轨迹。', '',
'| 实现 | verb token 最小 / 均值 / 最大 | target token 最小 / 均值 / 最大 | 达到输入上限次数 |','|---|---|---|---:|']
for mode in names:
    q=a['lengths'][mode]
    lines.append(f"| {mode} | {q['verb_min']} / {q['verb_mean']:.1f} / {q['verb_max']} | {q['target_min']} / {q['target_mean']:.1f} / {q['target_max']} | {q['input_cap_hits']} |")
lines+=['', '当前路径没有自由文本逐 token 生成循环，`max_tokens=1536` 是输入限制及旧填充长度，不能解释为模型经常生成满 1536 token。旧实现的主要浪费是固定物理形状、两阶段重复前向及大量小算子/投影调用。本次没有通过缩短生成上限提速。', '',
'## 新实验登记与训练参数', '',
f"登记 SHA-256：`{reg['registration_sha256']}`。绑定 {len(reg['source_hashes'])} 个源文件和 {len(reg['input_files'])} 个模型输入文件；源码归档、初始 actor/critic、基座/tokenizer、父参考模型、运行包与实际 GPU 内核均有身份记录。", '',
'checkpoint_0032 是最好已观测的研究候选，旧筛选为 11/16，尚未通过原生存目标验收。', '',
'登记目录：`/home/researcher/meadowmind-gpu/m8x-kv-runs/kv_attempt2`。`registration.json` 与 `preregistration.json` 一致，源码包为 `frozen_source.zip`。登记后对比前后重新验证源码与模型输入。', '',
'| 项目 | 登记值 |','|---|---|',
f"| 基座 / actor 可训练 / critic 参数数 | {policy['base_parameters']:,} / {policy['actor_trainable_parameters']:,} / {policy['critic_parameters']:,} |",
'| 初始模型 | exploration_attempt1/checkpoint_0032，actor + critic |',
'| 冻结参考 | M6 small_v2_stable/checkpoint_0016 |',
'| 优化器 | 新 AdamW，eps=1e-5、weight_decay=0 |',
'| actor / critic lr | 1e-5 / 3e-4 |',
'| PPO epochs / minibatch / microbatch | 2 / 32 / 4 |',
'| critic epochs / minibatch | 8 / 128 |',
'| clip / target KL / reference KL 系数 | 0.15 / 0.03 / 0.005 |',
'| gamma_tick / GAE lambda | 0.9995 / 0.999 |',
'| 梯度裁剪 actor / critic | 1 / 1 |',
'| 正式计划 | 48×4×512 = 98,304 个新决策，尚未执行 |',
'| 正式熵系数计划 | 0.002→0.0002，98,304 决策退火；本次短诊断固定 0.002 |',
'| 新训练 / 选择 / 最终测试种子 | 120000..124095 / 943000..943015 / 953000..953099，均未用于本次性能验证 |',
'| 性能验证世界 / RNG | 883810..883813，自然死亡同槽 seed+4；RNG=20261109 |',
'| 正式训练 / 新 checkpoint / 发布 | 未启动 / 0 / 未授权 |', '',
'这些文件登记了新布局与未来计划；此次交付为独立资格验证入口，没有把新布局接入原冻结的正式保存/评估流程，也没有产生可发布的新训练模型。', '',
'## 真实 state / action / reward', '']
traces=[json.loads(x) for x in (folder/'optimized_transitions.jsonl').read_text(encoding='utf-8').splitlines()]
sample=next(x for x in traces if x['executed_target_token_ids'] is not None)
(folder/'real_example.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2),encoding='utf-8')
excerpt=dict(source=sample['source'],seed=sample['seed'],update=sample['update'],step=sample['step'],env=sample['env'],
             state_text=sample['model_input'][-1]['content'],action=sample['action'],reward=sample['reward'],
             log_probability=sample['log_probability'],verb_input_tokens=len(sample['executed_verb_token_ids']),
             target_input_tokens=len(sample['executed_target_token_ids']))
lines+=['以下摘自本次 optimized 真实轨迹，完整观察、真实执行 token、转移与下一状态见 `real_example.json`。', '',
        '```json',json.dumps(excerpt,ensure_ascii=False,indent=2),'```','',
'## 失败路线与工程验证', '',
'第一版登记不可覆盖，三轮总耗时 250.486→131.405 秒，仅 1.9062 倍，虽然中位数达到 2.0062 倍，仍判为失败。原始登记、日志、三臂 1,152 条轨迹及独立重放证据保留在 `../rl_kv_20261107/`。本次修订使用重新测量的基线，不借用第一版较慢时间。', '',
'开发中保留但未采用：精确观察前缀 KV（额外短前向更慢）；共享系统前缀（小内核开销抵消收益）；TF32（无明显提速）；通用 FLA RMSNorm（更慢）；区域 torch.compile（no_grad/grad log-prob 差 0.050144，超过 1e-6，拒绝采用）。未修改安装的库文件。', '',
'172 项节点 5 Python 回归中，169 项通过、3 项按条件跳过；Linux Godot 导入和 10 组引擎测试均通过。首次整套回归漏传 Linux Godot 路径而失败，修正环境后重跑，原失败日志保留。新增独立 float64 梯度参考验证了每个 LoRA A/B 和输入梯度，AdamW 测试验证缓存更新且基座未变；CUDA 测试覆盖 RMSNorm、SiLU 及分组输出的非连续视图。', '',
f"本次 {replay['real_godot_replayed_steps']} 条轨迹已重新运行真实 Godot，逐条核对状态、合法动作、转移、奖励与终止；{tokens['verb_inputs_verified']} 份 verb 输入与 {tokens['target_inputs_verified']} 份 target 输入已用固定 tokenizer 重算真实 token ID。各臂自然死亡数：{replay['natural_deaths']}。", '',
'证据：`comparison.json`、`independent_audit.json`、`replay_verification.json`、`token_verification.json`、`engine_checks.json`、`regression_preseal.log`。原始 `*_transitions.jsonl`、`*_probabilities.json` 与 `*_stability.json` 均保留。', '',
'前置资料：[原算法与游戏机制调查](../rl_report_20261107/report.html)、[仅去填充对照报告](../rl_speed_20261107/report.html)。', '']
(folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(report_created=True,passed=r['passed'],total_speedup=a['total_speedup'])))
