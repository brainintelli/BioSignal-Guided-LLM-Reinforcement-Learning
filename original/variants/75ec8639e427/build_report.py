"""Build a measured report from the final default-layout probe and preserved trial."""
from pathlib import Path
import json
import math
import hashlib

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))

p = read('remote_evidence/probe_default.json')
r = read('remote_evidence/reload_default.json')
integrity = read('remote_evidence/final_integrity.json')
scene = read('remote_evidence/probe.json')
view = read('game_view.json')
replay = read('replay_audit.json')
assert r['logp_error_max'] == 0 and p['actor_unchanged']
assert p['optimizer_steps'] == 0 and not p['formal_training_started']
assert p['metadata']['inference_layout'] == 'semantic_v3_system_only'
assert not p['metadata']['scene_cache_active']
assert view['failures'] == 0
assert replay['passed'] and replay['max_float_difference'] == 0
sample = p['rollout'][0]
timings = p['fixed_sequence_forward_timings']
labels = {'stationary':'静止：等待三次', 'moving':'移动：连续三步'}
rows = []
for name, values in timings.items():
    a, b = values['system_only'], values['system_scene']
    counts = b['counters']
    hit = counts['scene_hits'] / (counts['scene_hits'] + counts['scene_misses'])
    rows.append(f"|{labels[name]}|{a['median_seconds']*1000:.2f}|{b['median_seconds']*1000:.2f}|{hit:.0%}|{a['counters']['forward_calls']}→{b['counters']['forward_calls']}|")
timing_table = '\n'.join(rows)
audit = p['text_audit']
probability_table = '\n'.join(f"|{name}|{p['probability'][key]:.9g}|" for name,key in [
    ('冷热缓存','cold_error'),('重复执行','repeat_error'),('重排','reorder_error'),
    ('单样本／分组','single_error'),('有／无梯度','grad_mode_error'),('参数版本刷新','parameter_version_refresh_error')])
rules_review = (PROJECT/'docs/SURVIVAL_V2_STATE_REVIEW.md').read_text(encoding='utf-8').split('## 当前规则',1)[1]
rules_review = '## 当前规则' + rules_review.rsplit('实测、真实输入与渲染图见',1)[0]
report = f'''# 三项生存游戏与中文 state 更新报告

2026-11-08 · 本地 Godot 4.7.2 / Netcatty 集群1节点5 · 新规则 open4/world6

已更新当前游戏、HUD、环境传输、state、system prompt、critic 输入与奖励结算接口。**没有启动正式训练或模型服务，没有优化器更新或新 checkpoint。** 当前默认缓存为实测更快的固定前缀模式；场景缓存虽能提高命中率，但在本次短序列上更慢，因此仅保留为显式实验选项。

{rules_review}

## 节点5实测：排序与缓存

固定规则及观察标题为 **{audit['stationary']['fixed_system_prefix_tokens']} token**，首个完整动作输入为 **{audit['stationary']['input_lengths'][0]} token**。新 system prompt 为 **{len(sample['state'][0]['content'])} 字符**。长度包含模型真实 chat template；不是估算字符数代替 token 数。

在同一观察序列中，仅将慢变场景移到动态数值之前，静止相邻观察的最长共同前缀由 `{audit['stationary']['dynamic_first_common_prefix_tokens']}` 增至 `{audit['stationary']['reordered_common_prefix_tokens']}` token；移动序列只由 `{audit['moving']['dynamic_first_common_prefix_tokens']}` 增至 `{audit['moving']['reordered_common_prefix_tokens']}`。移动导致相对位置变化，不能承诺整段地标始终命中。

每个序列4个真实观察，先预热，测量3次，每次从空前缀缓存开始；表中时间是4次动词前向的中位数，调用次数是3次测量累计，**不是完整RL训练时间**。

|场景|固定前缀：毫秒|两级前缀：毫秒|场景查询命中率|累计模型调用|
|---|---:|---:|---:|---:|
{timing_table}

静止场景下，计算 token 从4665降至2982（少36.1%），但新增调用抵消了收益；移动场景两级缓存0命中，调用从15增至27。首轮实验得到相同方向的结果，原始记录一并保留。最终默认 `system_only`，不为提高命中率而牺牲实际延迟。前一版的2.95倍整体加速不能直接用于新规则。

## 真实模型概率与缓存校验

从 checkpoint_0032 验证并加载 actor，独立冻结 reference 使用 checkpoint_0016；**没有读取旧97维 critic**，显式初始化新版96维 critic。GPU探针不创建优化器；有限非零梯度检查及参数零值版本递增，只验证缓存失效，不改变权重数值。

|同一默认布局比较|最大动作概率差|
|---|---:|
{probability_table}

全合法联合动作概率归一化误差最大 **{p['probability']['normalized_max_error']:.3g}**，梯度均有限，范数 **{p['probability']['gradient_norm']:.5f}**。新进程逐一重算全部 **{r['actions_checked']}** 个实际动作，最大 log-prob 误差 **{r['logp_error_max']:g}**。这些检查验证同一布局的一致性；不同BF16布局不宣称逐位等价。

最终默认布局采样16个真实动作，并独立验证4个“强制前往动词、模型选择完整目标”的导航样例；后者不是自然采样选择率。CPU审计重新运行实际Godot：16条原始观察、20条模型输入和完整转移/reward、27个逐刻snapshot均一致，最大浮点差0（阈值1e−12）。其中12次自然移动和4次导航均一格一刻；另重放1次采集、3次休息。导航原记录未保存原始observation/seed，按生成脚本的固定seed重建，只声称比较了已保存的输入和转移。详见[重放审计](replay_audit.json)。这是一轮有界推理验证，不代表训练收敛、长期生存或新的两倍整体吞吐验收。

## 实际 system prompt

下面完整复制最终默认布局送入模型的规则，无人工改写。

```text
{sample['state'][0]['content']}
```

## 真实 state、action、reward

最终默认布局第一条模型采样记录：

```text
{sample['state'][1]['content']}
```

动作：**{sample['semantic_target']}一格**，引擎绑定 `{json.dumps(sample['action'],ensure_ascii=False)}`。联合概率 **{math.exp(sample['log_probability']):.8f}**，log-prob **{sample['log_probability']:.9f}**。reward **{sample['reward']['reward']:.12f}**，分项 `{json.dumps(sample['reward']['components'],ensure_ascii=False)}`。reward在动作完成后由训练端计算，上面的模型输入不含它。

[完整真实记录](real_sample.json)包含前后状态、每刻trace与奖励分项；[纯state文本](state_example.txt)、[纯system文本](system_prompt.txt)可直接查看。

## 实际游戏与回归

![新规则游戏画面](game_view.png)

这是 Godot 自身 Viewport 导出的1440×900真实渲染图，不是示意图。窗口模式运行，自动执行一次合法移动，13项渲染／HUD／动作不变量全部通过；三项身体数值来自实际环境。桌面窗口读取曾因授权请求超时未完成，随后通过应用自身渲染导出完成视觉核对。

- 本地完整回归192项：171通过、21条件跳过；11个Godot套件全部通过。
- 节点完整回归192项：189通过、3条件跳过；包括本地缺依赖而跳过的4项缓存/梯度/critic测试。
- 新规则独立主场景44项不变量通过；内存中故意重新加入夜间伤害，真实触发3项失败、退出码1，证明对应断言能检测回归。正常实现重跑通过。
- 新节点副本258个源码/资源哈希核验；旧实验副本124个冻结文件未变；checkpoint_0032文件哈希未变。GPU验证已结束，没有新checkpoint。
- 新目录首次回归因临时目录和旧测试数据缺失失败；补齐测试运行环境后原测试全部重跑，未修改断言掩盖失败。首次失败日志保留。

## 文件与证据

- [输入与规则审查](../../docs/SURVIVAL_V2_STATE_REVIEW.md)、[最终源码清单](source_manifest.json)、[本地完整验证](project_verification/checks.json)。
- [最终GPU探针](remote_evidence/probe_default.json)、[独立进程概率复算](remote_evidence/reload_default.json)、[首轮场景缓存实验](remote_evidence/probe.json)。
- [节点测试日志](remote_evidence/unit_tests_final.log)、[最终完整性](remote_evidence/final_integrity.json)、[原始证据包](remote_evidence.zip)。
- [场景测试](qa_engine.log)、[负例验证](qa_engine_negative.log)、[渲染验证](game_view.json)。
- 节点新项目：`/home/researcher/meadowmind-gpu/m8s3-project`；原始结果：`/home/researcher/meadowmind-gpu/m8s3-diagnostics`。
- 原始节点证据包 SHA-256：`{hashlib.sha256((ROOT/'remote_evidence.zip').read_bytes()).hexdigest()}`。

协作采用用户指定的 [Godot 开发助手](plugin://godot-dev-assistant@workbuddy-experts)，分工覆盖Godot规则、状态契约和独立QA；当前无Godot MCP，验证使用实际引擎CLI与Netcatty。
'''
(ROOT/'report.md').write_text(report,encoding='utf-8')
summary = dict(world=p['metadata']['environment_version'], layout=p['metadata']['inference_layout'],
               probability=p['probability'], reload_logp_max=r['logp_error_max'], text_audit=audit,
               timings=timings, replay=replay, integrity=integrity, formal_training_started=False,
               optimizer_steps=0, checkpoint_writes=0)
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'characters':len(report),'reload_logp_max':r['logp_error_max'],'default_layout':p['metadata']['inference_layout']}))
