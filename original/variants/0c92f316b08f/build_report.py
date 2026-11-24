"""Build the report tables from measured JSON, never from hand-entered scores."""
import json
import statistics as stats
from pathlib import Path

folder = Path(__file__).resolve().parent
result = json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
p = result['performance']
s = result['stability']
names = dict(baseline='原实现 B4×1536',exact='仅去除多余输出计算',compact='去填充 B1×真实长度',prefix='去填充＋系统前缀复用')
lines = ['# 强化学习速度优化与节点 5 验证', '',
         '测量日期：2026-11-07。通过 Netcatty 在集群 1 节点 5 执行；这是独立的限额诊断，没有启动正式训练，没有保存新检查点。', '',
         '## 结论', '',
         '固定长度填充是已验证的主要优化点。当前策略直接读取候选 token 的概率，没有自由文本生成循环，`max_tokens=1536` 是输入限制与旧实现的填充长度；本次没有通过缩短输出上限提速。', '',
         '前置调查：[算法、训练参数、真实 state/action/reward 与示意图](../rl_report_20261107/report.html)。', '',
         '本轮优先候选是显式指定 `FastPolicy(..., optimization="compact")`。前缀复用继续保留为显存较紧时的实验选项：本次降低了显存，但采样和后续更新均慢于 compact。该类构造参数默认仍为实验性的 prefix；调用时必须按报告明确指定所选模式。正式训练与服务入口仍使用原 Policy，本次没有自动切换。', '',
         '## 实现范围', '',
         '- 新增 `agent/m8x_fast_policy.py` 与 `agent/m8x_fast_layout.py`，通过独立 `FastPolicy` 接入原有动作分布、critic 和 PPO 更新函数。原 72 个冻结核心文件保持不变。',
         '- `exact`：仍使用 B4×1536，但直接调用 backbone，只投影候选 token；不再额外执行最后位置的全词表输出头，也不返回所有层的 hidden states。原实现已使用 `logits_to_keep=1`，因此这项节省本来就有限。',
         '- `compact`：每个真实输入使用 B1×自身 token 长度。去掉 1536 左填充和补足 B4 的重复行；同一个输入的物理形状不受同批其他样本影响。输入超过 1536 仍明确报错，不截断状态或合法动作。',
         '- `prefix`：在 compact 上复用精确相同的 337-token 系统消息。推理缓存按参数对象、存储与版本失效；参考模型与 actor 缓存分离。训练时仅在一次前向图内部共享可微前缀，clone 分支保留梯度，禁止复用过期训练图。',
         '- 没有把 verb 的完整 KV 直接交给 target：两种聊天模板的 assistant 尾部不同，这样复用会改变条件概率。当前仅复用逐 token 相同的系统前缀。',
         '- `FastPolicy.save()` 明确拒绝作为旧登记模型保存。正式启用需要新的策略布局身份、实验登记与输出目录，本次未执行该步骤。', '',
         '## 性能对比', '',
         '同一张 RTX 6000 Ada、同一 BF16 基座、同一 checkpoint_0032 和父模型。PyTorch 2.6.0+cu124，Transformers 5.22.1；CPU 线程数 8。64 个真实状态等距抽自前一报告的 288 条诊断记录。每种模式先预热，再重复测量 3 次；下表取中位数。', '',
         '| 模式 | 固定工作量 / 64 状态（秒） | 实际 act / 64 状态（秒） | 决策/秒 | act 加速比 |',
         '|---|---:|---:|---:|---:|']
for mode,row in p.items():
    lines.append(f"| {names[mode]} | {row['fixed_workload_median_seconds']:.3f} | {row['act_median_seconds']:.3f} | {row['act_decisions_per_second']:.3f} | {p['baseline']['act_median_seconds']/row['act_median_seconds']:.2f}× |")
lines += ['', '固定工作量使用相同记录中的 verb，包含 actor、冻结参考模型两阶段概率及 critic；是严格同任务比较。`act` 包含 categorical 采样，随机种子固定，但不同数值布局可能选到不同分支，因此另列。计时执行 CUDA synchronize。准备输入与 Godot 环境耗时不在该表内；下面短更新表包含它们。', '',
          '## 动作概率验证', '',
          '对全部 64 个状态枚举所有合法 verb/target 联合动作，同时测 actor 与参考模型。概率表保存在各模式的 `*_probabilities.json`；不是仅检查已采样动作。', '',
          '| 模式 / 模型 | 最大动作概率绝对差 | 平均联合 KL | 最大联合 KL | 最高概率动作改变的状态数 |',
          '|---|---:|---:|---:|---:|']
for mode in ('exact','compact','prefix'):
    for key,label in [('actor_vs_baseline','actor'),('reference_vs_baseline','reference')]:
        r=p[mode][key]
        lines.append(f"| {mode} / {label} | {r['max_action_probability_difference']:.8f} | {r['mean_joint_kl']:.8f} | {r['max_joint_kl']:.8f} | {r['greedy_action_changed_states']}/64 |")
lines += ['', 'KL 方向是旧布局到新布局。`exact` 的概率一致性门槛为 1e-6；`compact` 与 `prefix` 明确按新数值布局处理，未宣称与旧布局完全等价。参考权重冻结不意味着更换计算布局后其概率逐位不变。', '',
          '| 模式 | 逆序及 microbatch=1/3/4 最大 log-prob 差 | no_grad 与有梯度最大差 |',
          '|---|---:|---:|']
for mode,r in p.items():
    lines.append(f"| {mode} | {r['reordered_zero_update_logp_max']:.9g} | {r['grad_mode_logp_max']:.9g} |")
lines += ['', '复算使用各布局自己采集的概率，要求误差 ≤1e-6。完整重分组检查覆盖 64 个状态，有梯度比较覆盖其中 16 个状态。新布局不能接续旧布局的 rollout 或复用旧 log-prob/reference 概率。', '',
          '## 短更新稳定性', '',
          '三组均从同一 checkpoint_0032 和 critic 开始；各自重新采集 4 个真实 Godot 世界（种子 882710–882713），每轮 16×4=64 个决策，共 3 轮、192 个决策/组。只在内存中执行 PPO 优化器更新，随后丢弃。每组最多 12 个 actor optimizer step；共 576 条新真实交互。', '',
          '沿用 actor lr=1e-5、critic lr=3e-4、PPO clip=0.15、target KL=0.03、reference KL 系数=0.005、2 epochs、minibatch=32、microbatch=4、critic epochs=8、gamma_tick=0.9995、GAE lambda=0.999。短诊断熵系数固定为 0.002，没有执行完整训练的熵退火。', '',
          '| 模式 | 轮次 | rollout 秒 | PPO+critic 秒 | actor steps | 最大近似 KL | zero-update log-prob 差 | 峰值已分配显存 GiB |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
for mode,arm in s.items():
    for r in arm['updates']:
        lines.append(f"| {mode} | {r['update']} | {r['rollout_seconds']:.3f} | {r['update_seconds']:.3f} | {r['optimizer_steps']} | {r['approx_kl_max']:.6f} | {r['zero_update_logp_max']:.9g} | {r['cuda_peak_bytes']/2**30:.2f} |")
lines += ['', '该表 rollout 包含输入准备、真实环境、奖励计算及 GAE；PPO 时间包含概率复算、actor 反传和 critic。新形状首次遇到时的编译与首次优化器状态分配没有人为扣除。三条 on-policy 轨迹可能不同，不能把它们当作严格同一工作量。短更新能够发现数值失稳与缓存错误，不能证明长期收敛或生存率不下降。', '',
          '| 模式 | 三轮总耗时（秒） | actor 确实更新 | 参考权重保持冻结 |',
          '|---|---:|---|---|']
for mode,arm in s.items():
    total=sum(r['rollout_seconds']+r['update_seconds'] for r in arm['updates'])
    lines.append(f"| {mode} | {total:.3f} | {arm['actor_changed']} | {arm['reference_unchanged']} |")
warm_base=stats.mean(r['update_seconds'] for r in s['baseline']['updates'][1:])
warm_compact=stats.mean(r['update_seconds'] for r in s['compact']['updates'][1:])
lines += ['', f'以第 2、3 轮均值描述后续更新：baseline 为 {warm_base:.3f} 秒，compact 为 {warm_compact:.3f} 秒，约 {warm_base/warm_compact:.2f}×。这是短诊断中的观察值，不是完整 2048 决策 rollout 的训练速度承诺。包含首次开销的三轮总耗时应同时参考上表。', '',
          '各轮损失、梯度范数、KL 等完整指标在 `*_stability.json`。prefix 每轮更新后会比较自动失效重建的缓存与手动清空缓存后的概率，并检查缓存参数版本；缓存误差与有限数值门槛结果见 `comparison.json`。', '',
          '## 证据与复现', '',
          f"诊断总耗时 {result['seconds']:.1f} 秒。检查 {result['original_files_checked']} 个原始核心或保留权重相关文件；变更清单 `{result['source_and_weight_mismatches']}`。这不包含对巨大基座文件的全量重哈希。未启动正式训练，保存检查点数量为 {result['checkpoints_saved']}。", '',
          '| 检查门槛 | 结果 |', '|---|---|']
for key,value in result['gates'].items():lines.append(f'| {key} | {value} |')
lines += ['', '独立复核：576 条新交互已在本地 Godot 全量重放，状态、输入、合法动作、下一状态和奖励全部匹配；本地 72 个冻结文件与基准新增源码的 SHA256 均匹配，`baseline` 和 `exact` 的完整概率 JSON 相等。见 `replay_verification.json`。', '',
          '工程回归：节点 5 运行 147 项 Python 测试，144 通过、3 跳过；5 项新增优化测试全部通过。跳过项为 2 项 Windows PowerShell 测试和 1 项缺少已删除历史训练证据的测试；两个 Windows 测试已在本地通过。节点 5 的 Godot 导入与 10 套引擎测试均通过。见 `remote_checks.json` 和 `remote_python.log`。', '',
          '过程中的失败未隐藏：首次前缀提取被 Qwen 模板拒绝，增加用户边界探针后修正；v1 基线检查因诊断脚本跨批保留计算图而 OOM，明确释放每批图后以新目录重跑。v2 Netcatty 前台连接中断，远程进程继续完成；因此以落盘结果、独立复核与进程退出确认完成，不宣称观察到了已丢失的前台退出码。远程回归首轮缺少测试临时目录，补齐后原测试重跑通过。本地整套验证受沙箱回环 HTTP 和用户目录限制，日志保留；相关检查已在远端通过。', '',
          '远程结果目录：`/home/researcher/meadowmind-gpu/diagnostics/rl_speed_20261107_v2`。本地镜像：`output/rl_speed_20261107`。`diagnostic_manifest.json` 在运行前写入预算、样本索引、容差和源码 SHA256；`comparison.json` 汇总实测结果；失败尝试保留在 `attempts.json`。', '',
          '仅复现本次限额诊断的命令如下（必须换成尚不存在的输出目录；GPU 有其他任务时脚本拒绝运行）：', '',
          '```bash', 'cd /home/researcher/meadowmind-gpu/m8b-project',
          'env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \\',
          '  PYTHONPATH=/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site:/home/researcher/meadowmind-gpu/m8b-project \\',
          '  /home/researcher/meadowmind-gpu/m6-venv/bin/python -u tools/benchmark_m8x_fast_policy.py \\',
          '  --dataset /tmp/meadowmind_rl_profile_20261107/fresh_transitions.jsonl \\',
          '  --output /home/researcher/meadowmind-gpu/diagnostics/rl_speed_NEW', '```', '']
(folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'report_created':True,'passed':result['passed']}))
