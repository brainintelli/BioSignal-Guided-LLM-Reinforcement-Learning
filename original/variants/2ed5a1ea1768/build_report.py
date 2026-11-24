"""Render only measured artifacts; no fabricated benchmark defaults."""
import html,json,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence'
def read(name):return json.loads((E/name).read_text(encoding='utf-8'))
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def main():
    r=read('comparison.json');audit=read('independent_audit.json');reg=json.loads((ROOT/'registration.json').read_text(encoding='utf-8'))
    gate=r['speed_gate'];passed=gate['passed'] and audit['reload_consistency_passed'] and r['probability_consistency_passed']
    arm_names={'baseline':'固定填充 B4×1536','optimized':'去填充＋固定规则 KV＋批量语义目标'}
    totals={k:{part:sum(x[part+'_seconds'] for x in v['updates']) for part in ('rollout','update')} for k,v in r['stability'].items()}
    for t in totals.values():t['total']=sum(t.values())
    samples=[json.loads(s) for s in (E/'optimized_transitions.jsonl').read_text(encoding='utf-8').splitlines()]
    chosen=samples[0]
    dump(ROOT/'real_sample.json',chosen)
    rows=[]
    for mode,name in arm_names.items():
        s=r['stability'][mode];t=totals[mode]
        rows.append([name,sum(x['decisions'] for x in s['updates']),s['actual_actor_steps'],s['actual_critic_steps'],f"{t['rollout']:.3f}",f"{t['update']:.3f}",f"{t['total']:.3f}"])
    roundrows=[];stable=[]
    for mode,name in arm_names.items():
        for x in r['stability'][mode]['updates']:
            roundrows.append([name,x['update'],f"{x['rollout_seconds']:.3f}",f"{x['update_seconds']:.3f}",f"{x['rollout_seconds']+x['update_seconds']:.3f}",x['optimizer_steps'],x['critic_optimizer_steps']])
            stable.append([name,x['update'],f"{x['approx_kl_max']:.7f}",f"{x['policy_loss']:.6f}",f"{x['critic_loss']:.6f}",f"{x['actor_grad_norm']:.4f}",f"{x['critic_grad_norm']:.4f}",str(x['kl_early_stop'])])
    probs=[]
    for mode,name in arm_names.items():
        m=r['performance'][mode];u=r['stability'][mode]['updates'];a=audit['arms'][mode]
        probs.append([name,m['reorder_max'],m['grad_mode_max'],max(x['pre_update_all_logp_max'] for x in u),max(x['refresh_error'] for x in u),a['first_round_logp_error_max'],a['first_round_all_logps_checked']])
    p=r['performance']['optimized']['actor_vs_baseline'];pr=r['performance']['optimized']['reference_vs_baseline']
    fixed=statistics.median(r['performance']['baseline']['fixed_forward_seconds'])/statistics.median(r['performance']['optimized']['fixed_forward_seconds'])
    stats=r['stability']['optimized']['updates'][-1]['cache_stats'];hits=stats['system_hits'];misses=stats['system_misses']
    c=reg['config'];init=r['stability']['optimized']
    text=f'''# 新规则、新 state：相同更新次数的强化学习速度复测

2026-11-08 · Netcatty · 集群1节点5 `researcher-local-5` · 主实验 `survival_diagnostic3`，额外严格对照 `survival_diagnostic2`

**组合门槛：{'通过' if passed else '未通过'}。**采样＋更新合计由 {totals['baseline']['total']:.3f} 秒变为 {totals['optimized']['total']:.3f} 秒，整体 **{gate['all_three_rounds_speedup']:.3f} 倍**；逐轮耗时中位数比 **{gate['median_speedup']:.3f} 倍**。两组均使用新规则、新 state 和同一语义动作分布定义。

没有启动正式训练，没有保存模型或优化器 checkpoint，没有切换模型服务。主实验和严格对照各2组，每组从初始actor和新critic开始，执行24次actor与24次critic更新；两次实验合计96次actor、96次critic更新，不能串接为同一个训练模型。每组24次短程更新不能证明长期收敛或生存能力。速度口径包含真实 Godot 采样、state 构造、日志写入、全部样本更新前概率检查、PPO 和 critic 更新；不含加载、独立审计、正式保存和完整评估。

## 1. 登记与初始化

登记 SHA-256：`{reg['registration_sha256']}`。绑定 {len(reg['source_hashes'])} 个源码/配置和 {len(reg['input_files'])} 个模型输入。执行前后逐个校验哈希。

实测硬件：NVIDIA RTX 6000 Ada Generation，49140 MiB 显存，驱动595.91.07；单GPU串行执行各组，无其他GPU任务。模型为Qwen3.5-0.8B，使用本机已固定的离线运行环境。

- actor：只加载 checkpoint_0032 的 LoRA，初始哈希 `{init['initial_actor']}`。
- critic：重新随机初始化 96→128→128→1 MLP，末层权重和偏置为零；两组初值哈希均为 `{init['initial_critic']}`。不读取旧 97 维 critic。
- actor 与 critic AdamW 初始状态均为空；不读取历史 optimizer 或 rollout。冻结 reference 为 checkpoint_0016。
- 新地图 seeds：{c['seeds']}；随机种子 {c['rng_seed']}；每组 4 环境×32 决策×3 轮。只在自然死亡后换种子，活着的采样段末端正确 bootstrap。
- 正式训练状态始终为 `not_started`；诊断输出没有模型或优化器checkpoint。登记目录的 status.json 已记录完成诊断、速度门槛未通过。

## 2. 共同规则与概率定义

`open4 / world6 / m8_survival_v2`：只保留生命、水分、饱食；取消体力和夜间环境扣血。饥饿、缺水、敌人伤害保留。中文上下左右移动及地标导航每次一格。地标不含“路程”“可前往”，周围地形保留通行信息；固定规则与较稳定场景在动态身体、时间、历史前。

actor 与 critic 使用同一事实白名单，不含 reward、return、advantage、未来轨迹、最优动作或训练端路径距离。奖励仅由训练端基于真实转移计算。

`π(动作,目标|state)=合法中文动词 softmax × 完整合法中文目标树路径概率`。动词通过候选目标集合条件化；动词和目标树根共享 state 表示。固定填充组也共享这个根，避免额外重复前向夸大差距。两个执行布局数值未必逐位相同，跨布局差异单独列出。

## 3. 参数和对照边界

{table(['参数','固定值'],[['PPO epochs / minibatch / microbatch','2 / 32 / 4'],['critic epochs / minibatch','8 / 128'],['actor / critic 学习率','5e-7 / 3e-4'],['AdamW eps / weight decay','1e-5 / 0'],['clip / target KL','0.15 / 0.03'],['reference KL / entropy 系数','0.005 / 0.002'],['gamma / GAE lambda（每刻）','0.9995 / 0.999'],['actor / critic 梯度裁剪','1 / 1'],['精度 / 最大输入长度','BF16 / 1536；超长报错，不截断'],['梯度检查点 / 温度','关闭 / 1.0']])}

主实验固定填充组：原生 PEFT 投影，每次 state 物理 B4×1536 完整前向；不足4行重复补齐但丢弃虚拟行结果；采用与优化组相同的图内目标 KV/短前缀批量算法，但不缓存固定 system 前缀。优化组：实际长度 state、参数版本控制的固定 system KV、分组/融合投影与相同目标分支算法；梯度缓存仅在当前计算图共享，更新后失效。场景 LRU 默认关闭，延续先前实测较快的 system_only。

主实验衡量 state 去填充、固定规则 KV 与融合投影的组合收益，不是“只改变 padding”的消融；不使用旧 open3 的 2.950 倍结果作为本次证据。另做了两组共用相同融合投影的严格对照：固定填充94.313秒，去填充96.680秒，加速0.9755倍，中位数比0.9999，数值检查通过但速度未提升。这说明投影实现与GPU批量效率必须分别考察，不能把全部收益归给去填充或KV。

第一次登记因完整前缀语义分支梯度图超出48GB而失败，尚未执行PPO更新；原始登记、源码快照和失败日志保留，不计入速度。修复检查循环的图生命周期后，多分支完整前缀图仍超显存，单独加融合投影也未解决，因此让两组共用图内目标KV，保留state的B4×1536填充作为可执行基线。这个目标共享机制也用于主实验原生PEFT基线，不故意让它重复计算所有目标的长前缀。没有降低microbatch、更新次数、跳过目标梯度或启用激活重算。

## 4. 实际速度

{table(['组别','真实决策','actor实际步数','critic实际步数','采样秒','更新秒','合计秒'],rows)}

{table(['组别','轮','采样秒','更新秒','合计秒','actor步数','critic步数'],roundrows)}

本轮峰值 PyTorch 已分配显存：

{table(['组别','峰值 GiB'],[[arm_names[m],f"{max(x['peak_allocated_bytes'] for x in s['updates'])/1024**3:.3f}"] for m,s in r['stability'].items()])}

相同32个新规则观察和固定动作分支，3次固定前向耗时的中位数比为 **{fixed:.3f} 倍**。实际采样的动作组成如下；由于 BF16 布局差异和独立参数更新，轨迹可能不同，不能把这解释为逐轨迹严格相同的工作负载。

严格对照提供了瓶颈线索：固定填充与去填充的累计forward token分别为3,059,847和866,551（减少71.68%），但前向调用由508升到2,214（4.36倍）；相同固定32-state前向中位数由1.649秒变为1.810秒。优化版按state逐条B1计算，固定填充能用B4并行；这些观测支持“短前向调用开销与批量效率抵消token节省”的解释，但本轮没有CUDA逐kernel剖析，不能把它当成已测得的kernel耗时分解。下一步应验证多个state动态后缀的联合计算，并继续复用本轮的概率一致性与实际更新次数门槛。

{table(['组别','实际动作计数','世界刻数','自然死亡'],[[arm_names[m],json.dumps(s['action_counts'],ensure_ascii=False),sum(x['world_ticks'] for x in s['updates']),s['natural_deaths']] for m,s in r['stability'].items()])}

优化组最终累计固定前缀查询命中 {hits}/{hits+misses}（{100*hits/(hits+misses):.2f}%）；这是包含校验调用的累计计数，不是独立计时区间的缓存命中率。输入 token 实测：{json.dumps({m:audit['arms'][m]['input_tokens'] for m in arm_names},ensure_ascii=False)}。这里没有自由生成直到 max_new_tokens 的采样循环，1536 是固定填充基线的计算长度上限，不能把填充浪费归因为模型输出到 max token。

## 5. 概率与短程稳定性

{table(['组别','重排/换批logp差','启用梯度logp差','每轮全部128更新前差','更新后缓存概率差','独立进程logp差','独立重载条数'],probs)}

更新前检查覆盖每组3×128条；独立进程重载检查每组首轮全部128条（此时采样 actor 仍为 checkpoint_0032）。后两轮已改变的 actor 不存盘，因此未宣称后两轮有独立进程权重重载验证。全部768条真实转移、中文输入/token、动作和 reward 使用独立 Godot 重放核验。

跨布局 actor 联合动作概率：最大绝对差 {p['max_abs']:.8f}，平均 KL {p['mean_kl']:.8f}，最大 KL {p['max_kl']:.8f}，32个固定 state 中 argmax 变化 {p['argmax_changes']} 个；reference 最大绝对差 {pr['max_abs']:.8f}。同布局复算通过不等于跨布局逐位一致。

额外导航预检中，同一个state的目标树使用缓存续算与独立完整前缀评分，最大概率差为原生PEFT 0.00479835、融合投影0.00146733；两种执行布局存在BF16数值差异，未宣称它们满足1e-6逐位等价。PPO采样和更新均固定各自执行布局，实际同布局logp重算及跨进程误差按上表判断。

{table(['组别','轮','max approx KL','policy loss','critic loss','actor裁剪前norm均值','critic裁剪前norm均值','KL早停'],stable)}

各项 loss/梯度均有限、参考模型冻结情况及 actor/critic 实际变化见 comparison.json。裁剪前范数大于1不表示未裁剪；登记中阈值为1。上述结果仅支持短程数值稳定性，不证明长期训练质量。

## 6. 真实样例

以下摘自优化组首条实际采样记录，完整原始观察和转移在 real_sample.json。

```text
{chosen['model_input'][1]['content']}
```

语义目标：`{chosen['semantic_target']}`；环境动作：`{json.dumps(chosen['action'],ensure_ascii=False)}`；joint log-prob：{chosen['log_probability']:.9f}；reward：{chosen['reward']['reward']:.9f}。奖励分项 `{json.dumps(chosen['reward']['components'],ensure_ascii=False)}` 仅记录在训练日志，不进入模型 state。

## 7. 工程验证与证据

新增5项测试覆盖物理填充、相同实际工作量、数值失败/早停拒绝、固定参数与登记哈希。服务器完整197项测试194通过、3条件跳过；本地197项中175通过、21跳过，1项HTTP测试因沙箱禁止套接字访问失败，服务器复核通过。预检曾发现Windows路径分隔符污染远程配置，新增回归断言后修复；该次失败发生在模型加载前，未作优化器更新。服务器首次完整测试未指定Linux Godot路径导致3项失败，补齐MEADOW_GODOT后通过，失败日志均保留。

原始证据：registration.json；evidence/comparison.json；evidence/independent_audit.json；两组 stability.json、probabilities.json、policy.json、transitions.jsonl；controlled_states.json；equal_update3.log；smoke和单元测试日志。主实验位于 `/home/researcher/meadowmind-gpu/m8s3-runs/survival_diagnostic3`，性能证据位于 `/home/researcher/meadowmind-gpu/m8s3-diagnostics/equal_update3`。严格对照另见 registration_control2.json、comparison_control2.json、audit_control2.json 和服务器 equal_update2；全部768条转移重放、首轮每组128条跨进程logp检查也通过（误差0）。源码快照用于保留不同登记版本，旧登记不能用新的源码冒充复现。
'''
    (ROOT/'report.md').write_text(text,encoding='utf-8')
    # Small self-contained HTML rendering with explicit tables, fenced samples and links.
    parts=[];lines=text.splitlines();in_code=False;code=[];tab=[]
    def flush():
        if not tab:return
        rows=[s.strip('|').split('|') for s in tab];parts.append('<table><thead><tr>'+''.join('<th>'+html.escape(x.strip())+'</th>' for x in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in row)+'</tr>' for row in rows[2:])+'</tbody></table>');tab.clear()
    for line in lines:
        if line.startswith('```'):
            flush()
            if in_code:parts.append('<pre>'+html.escape('\n'.join(code))+'</pre>');code=[]
            in_code=not in_code;continue
        if in_code:code.append(line);continue
        if line.startswith('|'):tab.append(line);continue
        flush()
        if not line:continue
        if line.startswith('# '):parts.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):parts.append('<h2>'+html.escape(line[3:])+'</h2>')
        else:
            escaped=html.escape(line)
            escaped=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',escaped)
            escaped=re.sub(r'`([^`]+)`',r'<code>\1</code>',escaped)
            parts.append('<p>'+escaped+'</p>')
    flush()
    bars=''.join(f'<div class="bar-label">{name} · {totals[m]["total"]:.2f} 秒</div><div class="bar" style="width:{100*totals[m]["total"]/totals["baseline"]["total"]:.2f}%"></div>' for m,name in arm_names.items())
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>新规则强化学习速度复测</title><style>body{font:16px/1.75 system-ui,"Microsoft Yahei",sans-serif;color:#19323d;background:#f4f6f5;max-width:1180px;margin:40px auto;padding:0 28px}h1{font-size:30px}h2{margin-top:48px;border-top:1px solid #bac9c8;padding-top:20px}table{border-collapse:collapse;width:100%;font-size:14px;background:white;margin:20px 0;display:block;overflow:auto}th,td{padding:9px 13px;border:1px solid #dce4e2;text-align:left}th{background:#e2eeeb}pre{white-space:pre-wrap;background:#e6edeb;padding:22px;overflow-wrap:anywhere}p{overflow-wrap:anywhere}.bar{height:21px;background:#13756e;margin:3px 0 18px}.bar-label{font-size:14px}a{color:#08716a}</style><main><a href="registration.json">登记</a> · <a href="evidence/comparison.json">性能原始数据</a> · <a href="evidence/independent_audit.json">独立审计</a> · <a href="real_sample.json">真实样例</a>'+bars+''.join(parts)+'</main></html>'
    (ROOT/'report.html').write_text(document,encoding='utf-8')
    print(json.dumps(dict(passed=passed,speedup=gate['all_three_rounds_speedup'],totals=totals,report=str(ROOT/'report.html')),ensure_ascii=False))
if __name__=='__main__':main()
