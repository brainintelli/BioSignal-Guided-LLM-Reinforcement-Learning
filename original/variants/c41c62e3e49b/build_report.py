"""Render the delivery report from downloaded, independently audited evidence."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence' / 'remote'
read = lambda name: json.loads((E / name).read_text(encoding='utf-8'))
reg = read('registration_final.json')
result = read('equal_update2/comparison.json')
audit = read('equal_update2/independent_audit.json')
control = read('control2/comparison.json')
control_audit = read('control2/independent_audit.json')
extra = read('extended_probe2.json')
gate = result['speed_gate']
assert gate['passed'] and result['probability_consistency_passed']
assert audit['reload_consistency_passed'] and control_audit['reload_consistency_passed']

def totals(value):
    return {mode: dict(rollout=sum(x['rollout_seconds'] for x in s['updates']),
                      update=sum(x['update_seconds'] for x in s['updates']),
                      total=sum(x['rollout_seconds'] + x['update_seconds'] for x in s['updates']),
                      peak=max(x['peak_allocated_bytes'] for x in s['updates']) / 2**30,
                      kl=max(x['approx_kl_max'] for x in s['updates']))
            for mode, s in value['stability'].items()}

t, ct = totals(result), totals(control)
names = {'baseline': '固定填充＋原生 PEFT', 'optimized': '规范分桶＋融合投影'}
lines = [
    '# 新规则强化学习速度优化：规范分桶与等更新复测',
    '',
    '2026-11-08，北京时间 · Netcatty · 集群1节点5 `researcher-local-5` · RTX 6000 Ada 48GB',
    '',
    f"**主对照通过：采样＋更新 {t['baseline']['total']:.3f} → {t['optimized']['total']:.3f} 秒，整体 {gate['all_three_rounds_speedup']:.3f} 倍；逐轮中位数比 {gate['median_speedup']:.3f} 倍。正式训练未启动。**",
    '',
    '最终实验 `survival_batch_diagnostic2`，严格消融 `survival_batch_control2`。每组从同一初始 actor、新 critic 和空优化器重新开始，各完成384次真实决策、24次actor与24次critic更新。最终主对照与消融合计1536次决策、96次actor与96次critic更新，分属四个独立模型，不能串接为正式训练。',
    '',
    '本次另有被替代初版的768次决策、48次actor及48次critic更新；含初版的真实模型诊断累计2304次决策、144次actor及144次critic更新。各GPU剖析、梯度和边界探针均为零优化器更新。初版严格对照在执行前被替代，未计入工作量。',
    '',
    '时间包括真实Godot交互、中文state与token构造、转移日志写入、全部样本更新前log-prob检查、PPO与critic更新；不含模型加载、固定样例概率/预热检查、独立审计和模型保存。计时前清空/重新加载各组模型，首轮新投影缓存的构造计入采样。没有checkpoint写盘、正式评估、模型服务切换或后台训练。',
    '',
    '## 1. 最新进展与本次修复',
    '',
    '上一轮新规则对照只有1.631倍；同融合投影的严格消融为0.9755倍。问题在于逐条B1 state加固定前缀KV使主干调用过碎：token减少却损失批量效率。此次保留新版state和语义动作定义，改为完整state的规范B4分桶，复用图内目标KV。',
    '',
    '- 每个梯度图中，主干每四条普通state由5次前向降到1次。旧优化布局同一四条state前向＋反向均值284.04毫秒，初版分桶中位数约171.38毫秒；这只是固定微批证据，最终整体收益见下表。',
    '- 剖析中`cudaLaunchKernel`调用16568→3196，`aten::mm`调用2876→591。两个剖析的record_shapes设置不同，因此只用调用数量定位，不把剖析墙钟差作速度结论。',
    '- 初版分桶在真实样本上通过，但长输入恰好填满桶时会出现全1 attention mask；Transformers SDPA省略mask后选取不同执行路径，896 token样例换批log-prob偏差0.050341606。已让每条输入至少保留一个屏蔽的前置填充token，复测895、896、897、1279、1280、1281、1500、1536 token换批误差为0。',
    '- 修复前登记、源码快照、2.547倍旧结果和失败探针均保留，状态已标记被替代；最终结果来自修订版重新登记和重新采样。原固定填充基线与前轮源代码保持不变。',
    '- Windows本地HTTP测试首次被沙箱禁止回环套接字，按允许回环的执行方式重新运行后通过；未修改测试来绕过失败。',
    '',
    '## 2. 登记与初始化',
    '',
    f"登记SHA-256：`{reg['registration_sha256']}`；绑定275个源码/配置和16个输入，执行、独立审计及交付时复核。源码项目 `/home/researcher/meadowmind-gpu/m8s3-batch2-project`，最终登记 `/home/researcher/meadowmind-gpu/m8s3-runs/survival_batch_diagnostic2`。",
    '',
    f"actor只加载checkpoint_0032 LoRA：`{result['stability']['baseline']['initial_actor']}`。新critic为96→128→128→1，末层为零；两组初始化哈希均为`{result['stability']['baseline']['initial_critic']}`。旧97维critic、optimizer和rollout均不加载。reference固定为checkpoint_0016。",
    '',
    '地图seeds：[884010,884011,884012,884013]；随机种子20261112。4环境×32决策×3轮；仅自然死亡后换种子，非终止段末正常bootstrap。两组AdamW初始状态均为空，actor与critic实际发生更新，reference哈希保持不变。',
    '',
    '规则仍为open4/world6/m8_survival_v2：生命、水分、饱食三项；取消体力和夜间环境扣血，保留饥饿、缺水与敌人伤害。中文上下左右移动和地标导航每次一格。地标无路程/可前往提示，地形保留通行信息；固定规则和场景在动态身体、时间、历史之前。actor/critic仅接收白名单事实，不含reward、return、advantage、未来、推荐动作或训练端路径距离。',
    '',
    '| 固定参数 | 值 |',
    '| --- | --- |',
    '| PPO epochs / minibatch / microbatch | 2 / 32 / 4 |',
    '| critic epochs / minibatch | 8 / 128 |',
    '| actor / critic学习率 | 5e-7 / 3e-4 |',
    '| AdamW eps / weight decay | 1e-5 / 0 |',
    '| clip / target KL | 0.15 / 0.03 |',
    '| reference KL / entropy | 0.005 / 0.002 |',
    '| gamma / GAE lambda（每刻） | 0.9995 / 0.999 |',
    '| actor / critic梯度裁剪 | 1 / 1 |',
    '| 精度 / 最大逻辑输入长度 | BF16 / 1536，超长拒绝，不截断 |',
    '| 梯度检查点 / 温度 | 关闭 / 1.0 |',
    '',
    '## 3. 执行布局与速度',
    '',
    '基线为原生PEFT、完整B4×1536 state；不足四条重复补物理行，但虚拟行不计loss。候选使用融合投影、B4×896/1280/1537三个物理桶，选最小严格大于单条逻辑长度的桶。桶宽不依赖同批其它观察；1537包含一个屏蔽填充位，逻辑输入上限仍为1536。完整state左填充，不能在循环状态前缀与后缀之间插入零来假装支持变长缓存。候选不启用system或scene持久KV，两组均复用图内目标KV及相同短前缀批量算法。',
    '',
    '| 组别 | 决策 | actor/critic步数 | 采样秒 | 更新秒 | 合计秒 | 峰值GiB |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for mode in ('baseline', 'optimized'):
    x = t[mode]
    lines.append(f"| {names[mode]} | 384 | 24 / 24 | {x['rollout']:.3f} | {x['update']:.3f} | {x['total']:.3f} | {x['peak']:.3f} |")
lines += ['', '| 组别 | 轮次 | 采样秒 | 更新秒 | 合计秒 | actor/critic步数 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
for mode, s in result['stability'].items():
    for x in s['updates']:
        lines.append(f"| {names[mode]} | {x['update']} | {x['rollout_seconds']:.3f} | {x['update_seconds']:.3f} | {x['rollout_seconds']+x['update_seconds']:.3f} | {x['optimizer_steps']} / {x['critic_optimizer_steps']} |")
fixed = statistics.median(result['performance']['baseline']['fixed_forward_seconds']) / statistics.median(result['performance']['optimized']['fixed_forward_seconds'])
lines += ['', f'相同32条新规则观察和固定动作分支的前向中位数加速{fixed:.3f}倍。实际rollout独立采样，BF16布局和后续更新会使动作轨迹不同；等决策与等优化器更新不代表逐轨迹完全相同。', '',
          f"严格消融两组共用相同融合投影：固定1536为{ct['baseline']['total']:.3f}秒，规范分桶为{ct['optimized']['total']:.3f}秒，比值{control['speed_gate']['all_three_rounds_speedup']:.3f}，中位数比{control['speed_gate']['median_speedup']:.3f}。该值单独列出，主对照的两倍以上收益是融合投影与规范分桶的组合，不能全部归给去填充。", '',
          '## 4. 动作概率、梯度与短程稳定性', '',
          '策略定义保持：π(动作,目标|state)=合法中文动词softmax × 完整合法中文目标树路径概率；动词通过目标合法集条件化，并与目标树根共享state表示。', '',
          '| 组别 | 重排/换批 | 梯度模式 | 每轮全128更新前 | 更新后缓存刷新 | 首轮128跨进程 |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for mode, s in result['stability'].items():
    m = result['performance'][mode]
    lines.append(f"| {names[mode]} | {m['reorder_max']} | {m['grad_mode_max']} | {max(x['pre_update_all_logp_max'] for x in s['updates'])} | {max(x['refresh_error'] for x in s['updates'])} | {audit['arms'][mode]['first_round_logp_error_max']} |")
p = result['performance']['optimized']['actor_vs_baseline']
r = result['performance']['optimized']['reference_vs_baseline']
lines += ['', f"跨布局联合动作分布最大绝对差{p['max_abs']:.8f}，平均KL {p['mean_kl']:.8f}，最大KL {p['max_kl']:.8f}，32个state中argmax变化{p['argmax_changes']}个；reference最大绝对差{r['max_abs']:.8f}。这是已测得的BF16算术差异，未宣称与原生布局逐位等价，不能复用旧rollout。", '',
          f"含导航分支的批量/逐条梯度余弦{extra['gradient']['cosine']:.8f}，相对L2差{extra['gradient']['relative_l2']:.3%}；导航缓存与独立完整前缀概率最大差{extra['navigation_cached_full_probability_max']:.8f}。这些跨执行路径差异不作为1e-6同布局复算通过的证据。", '',
          '| 组别 | 轮 | max approx KL | policy loss | critic loss | actor裁剪前norm均值 | critic裁剪前norm均值 | critic更新后解释方差 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for mode, s in result['stability'].items():
    for x in s['updates']:
        lines.append(f"| {names[mode]} | {x['update']} | {x['approx_kl_max']:.7f} | {x['policy_loss']:.6f} | {x['critic_loss']:.6f} | {x['actor_grad_norm']:.4f} | {x['critic_grad_norm']:.4f} | {x['explained_variance_after']:.5f} |")
lines += ['', '全部loss/梯度有限，无KL早停，每组24/24实际更新。裁剪前norm可大于1；实际裁剪阈值为1。新critic的解释方差仍可能接近零或为负，短程数值稳定不代表critic已学好、生存能力改善或长期收敛。后两轮actor没有保存，因此跨进程权重重载只覆盖首轮每组128条；每轮全部128条均完成进程内更新前复算。', '',
          '## 5. 独立审计与真实采样', '',
          '最终主对照768条转移、中文state/token、动作绑定和reward均经独立Godot重放核验；严格消融另768条也通过。动作计数与输入长度如下：', '',
          '| 组别 | 动作计数 | 世界刻数 | 自然死亡 | 输入token均值/范围 |', '| --- | --- | ---: | ---: | --- |']
for mode, s in result['stability'].items():
    tok = audit['arms'][mode]['input_tokens']
    lines.append(f"| {names[mode]} | {json.dumps(s['action_counts'], ensure_ascii=False)} | {sum(x['world_ticks'] for x in s['updates'])} | {s['natural_deaths']} | {tok['mean']:.2f} / {tok['min']}–{tok['max']} |")
sample = json.loads((E / 'equal_update2/optimized_transitions.jsonl').read_text(encoding='utf-8').splitlines()[0])
(ROOT / 'real_sample.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
lines += ['', '首条真实模型观察：', '', '```text', sample['model_input'][-1]['content'], '```', '',
          f"实际动作：`{json.dumps(sample['action'], ensure_ascii=False)}`；语义目标：`{sample['semantic_target']}`；joint log-prob：{sample['log_probability']:.9f}。完整转移和训练端reward见real_sample.json，reward不进入state。", '',
          '## 6. 工程验证与交付', '',
          '- 最终节点202项Python测试：199通过、3项条件跳过；本地202项：181通过、21项条件跳过，无失败。Godot导入及11个引擎套件通过。',
          '- 新增五项回归覆盖样本重排/分组、物理补齐、禁止截断、桶边界mask稳定性和登记拒绝旧状态/预算变化。GPU探针先在旧实现复现5次前向与长输入概率偏差，再验证修复。',
          '- 最终登记、源文件快照、配置、comparison、independent_audit、稳定性、概率、真实轨迹、剖析及失败日志均在evidence/remote。final_integrity.json记录最终哈希、checkpoint扫描和进程检查。',
          '- 本地交付再次验证了证据包SHA-256、73个证据文件、快照内275个源码和本地9个新增源码/配置文件；记录在evidence/local_delivery_verification.json。含中文清单统一以UTF-8读取。',
          '- 旧诊断的已登记源码留在旧项目；新源码在m8s3-batch2-project。登记目录status.json区分主速度门槛与严格消融结果。正式训练状态持续为not_started；无新模型或优化器checkpoint，无服务启动。', '',
          '实现入口：agent/m8s3_batch_policy.py；布局规则：agent/m8s3_batch_layout.py；登记：tools/register_m8s3_batch.py；有界对照：tools/benchmark_m8s3_batch.py；独立审计：tools/audit_m8s3_batch.py。正式训练入口未自动切换到本诊断实现。', '',
          '浮点批处理可能改变舍入、不能假设逐位一致，参见[PyTorch数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)；剖析本身也有开销，参见[PyTorch Profiler说明](https://docs.pytorch.org/tutorials/beginner/profiler.html)。本报告速度和概率数值均来自本机实测证据。', '']
(ROOT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(dict(speedup=gate['all_three_rounds_speedup'], totals=t, control=ct), ensure_ascii=False, indent=2))
