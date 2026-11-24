"""Render measured semantic-v2 results without substituting pilot timings."""
from pathlib import Path
import hashlib,json,math,statistics,shutil
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'evidence/m8s2-diagnostics/compare1'
def read(n):return json.loads((BASE/n).read_text(encoding='utf-8'))
def save(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
c=read('comparison.json');audit=read('independent_audit.json');reg=json.loads((ROOT/'registration.json').read_text())
diagnostics=ROOT/'evidence/m8s2-diagnostics'
paired=json.loads((diagnostics/'token_audit.json').read_text())['summary']
gradient=json.loads((diagnostics/'batch_gradient.json').read_text())
integrity=json.loads((diagnostics/'final_integrity.json').read_text())
evidence_sha=hashlib.sha256((ROOT/'evidence.zip').read_bytes()).hexdigest()
names={'baseline':'原协议固定填充','old_optimized':'原协议去填充＋KV','semantic':'中文语义第二版'}
totals={}
for mode,data in c['stability'].items():
    rows=data['updates']
    totals[mode]={'rollout':sum(r['rollout_seconds'] for r in rows),'update':sum(r['update_seconds'] for r in rows),
        'total':sum(r['rollout_seconds']+r['update_seconds'] for r in rows),
        'steps':sum(r['optimizer_steps'] for r in rows),'ticks':sum(r['world_ticks'] for r in rows),
        'peak_gib':max(r['peak_allocated_bytes'] for r in rows)/2**30,
        'fixed_forward':statistics.median(c['performance'][mode]['fixed_forward_seconds'])}
table='\n'.join(f"|{names[m]}|{v['rollout']:.2f}|{v['update']:.2f}|{v['total']:.2f}|{v['steps']}|{v['ticks']}|{v['peak_gib']:.2f}|" for m,v in totals.items())
rounds='\n'.join('|'+str(i+1)+'|'+'|'.join(f"{c['stability'][m]['updates'][i]['rollout_seconds']+c['stability'][m]['updates'][i]['update_seconds']:.2f}" for m in names)+'|' for i in range(3))
new=c['stability']['semantic'];cache=new['updates'][-1]['cache_stats'];hit=cache['system_hits']/(cache['system_hits']+cache['system_misses'])
speed=c['speed_gate']['all_three_rounds_speedup'];median=c['speed_gate']['median_speedup'];vsold=totals['old_optimized']['total']/totals['semantic']['total']
fixedspeed=totals['baseline']['fixed_forward']/totals['semantic']['fixed_forward']
under=c['understanding'];numeric=c['performance']['semantic']
records=[json.loads(s) for s in (BASE/'semantic_transitions.jsonl').read_text().splitlines()]
first=records[0]
sample={k:first[k] for k in ('seed','update','step','env','model_input','semantic_target','action','transition','reward','log_probability')}
sample['joint_probability']=math.exp(sample['log_probability']);save('real_sample.json',sample)
verbs=['move','navigate','gather','eat','drink','rest','build_camp','wait']
counts='\n'.join('|'+v+'|'+'|'.join(str(c['stability'][m]['action_counts'].get(v,0)) for m in names)+'|' for v in verbs)
stable='\n'.join(f"|{r['update']}|{r['optimizer_steps']}|{r['approx_kl_max']:.6f}|{r['actor_grad_norm']:.3f}|{r['zero_update_logp_max']:g}|{'是' if r['kl_early_stop'] else '否'}|" for r in new['updates'])
token_rows='\n'.join(f"|{names[m]}|{x['input_tokens']['mean']:.2f}|{x['input_tokens']['max']}|" for m,x in audit['arms'].items())
reload_rows='\n'.join(f"|{names[m]}|{x['first_round_all_logps_checked']}|{x['first_round_logp_error_max']:.9g}|{'通过' if x['reload_logp_within_1e_6'] else '未通过'}|" for m,x in audit['arms'].items())
summary={'registration_sha256':reg['registration_sha256'],'timings':totals,'speed_gate':c['speed_gate'],
    'cross_process_reload_consistency_passed':audit['reload_consistency_passed'],'system_prefix_hit_rate':hit,
    'fixed_workload_forward_speedup':fixedspeed,'semantic_vs_old_kv':vsold,
    'paired_token_audit':paired,'batch_gradient_check':gradient,'final_integrity':integrity,
    'formal_training_started':False,'checkpoint_writes':0}
save('analysis_summary.json',summary)
passed=c['speed_gate']['passed'] and audit['reload_consistency_passed']
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 930 300" role="img" aria-labelledby="title"><title id="title">相同384次决策、24次更新的端到端耗时</title><rect width="930" height="300" fill="#f5f8fa"/><g font-family="Microsoft YaHei,sans-serif" fill="#20384a"><text x="24" y="30" font-size="19">相同384次决策、24次actor更新：耗时（秒）</text>']
for i,(m,t) in enumerate(totals.items()):
    y=65+i*60;x=225;scale=2.1;a=t['rollout']*scale;b=t['update']*scale
    svg += [f'<text x="24" y="{y+22}" font-size="14">{names[m]}</text><rect x="{x}" y="{y}" width="{a}" height="32" fill="#207b82"/><rect x="{x+a}" y="{y}" width="{b}" height="32" fill="#91bcc2"/><text x="{x+a+b+12}" y="{y+22}" font-size="14">{t["total"]:.2f}s</text>']
svg += [f'<text x="225" y="260" font-size="14">深色：采样与环境　浅色：PPO与critic更新</text><text x="24" y="286" font-size="14">合计加速 {speed:.3f}×；三轮中位数比 {median:.3f}×；包含首轮开销。</text></g></svg>']
(ROOT/'timing.svg').write_text('\n'.join(svg),encoding='utf-8')
report=f'''# 中文语义协议第二轮：速度、概率与稳定性实测

2026-11-08 · Netcatty / 集群1节点5 `researcher-local-5` · checkpoint_0032 · 仅有界诊断

**本次组合门槛：{'通过' if passed else '未通过'}。**原协议固定填充三轮采样＋更新 {totals['baseline']['total']:.2f} 秒，新中文语义协议 {totals['semantic']['total']:.2f} 秒，合计加速 **{speed:.3f}倍**，逐轮中位数比 **{median:.3f}倍**。三臂实际更新次数与首轮重载数值误差见下表，不能用预试验早停时的耗时替代本次结果。

没有启动正式训练、没有保存诊断模型、没有切换服务。模型加载、独立题集、概率审计、checkpoint写盘和完整评估不在这个短程采样＋更新速度口径内；长期收敛与生存率尚未验证。

## 1. 实际修改

1. **简短中文 state**：按地形归类位置，明确“其余地形”；将地标反复出现的“就绪、再生0刻”合并为固定默认说明；压缩历史和规则文字。保留所有可见事实、合法动作和97维critic特征，不裁剪候选。
2. **共享规则 KV**：固定system规则跨采样调用复用；参数版本变化立即重建。训练时只在当前forward图内共享，保留梯度；动态state不跨时刻复用。
3. **批量目标分支**：一个state的全部非根中文目标决策前缀组成一次batch，计算完整合法名称概率。短前缀允许少量右填充，state仍按实际长度计算。
4. **移除重复目标前向**：中文动作评分与目标树根共用state表示。向上/下/左/右均为单token，移动目标不再多跑一遍模型。多token地标继续计算所有决策分支。
5. **固定 FLA 数值配置**：消除由进程预热顺序决定的自动调优配置；缓存分支保留梯度并避免单token卷积原地修改共享根。

引擎规则不变：移动按地图上下左右，每次一格；前往选定地标只避障走一格，不自动采集或喝水。模型选择真实中文token及完整地标名，引擎ID仅用于执行绑定。

这不是声称新策略与旧自回归接口概率相同。新定义是：

`π(v,t|s) = softmax(合法中文动词 logits) × P(完整中文目标 | state, 动词对应的合法目标集合)`。

动词与目标树根共享隐藏表示；动词通过目标候选集合施加条件，不再另加动词文本前向。目标叶子概率是完整路径上的合法分支概率之积，唯一后继概率1，EOS处理前缀重名。PPO仍使用完整分布的熵与KL。这个变化已单独登记，所有对照重新采样，不复用旧log-prob或优化器。

## 2. 跨进程偏差的定位与修复

复现时，直接加载旧KV策略的前4个动作log-prob与“先运行B4×1536、再加载同一策略”最大相差约 **0.04831**；后一条路径完全复现上轮历史log-prob。有效合并权重哈希相同，逐层输出在早期层开始分歧，运行时记录到FLA选择的配置不同。

固定已登记的内核配置后，两条独立进程路径的log-prob差为 **0**，捕获的逐层输出哈希一致。前向配置不再随之前的请求长度/预热负载选择。反向配置还针对Ada共享内存上限选择了可执行的两阶段dhu内核；最初三阶段配置超出共享内存的失败日志保留。

因此本次有证据把已复现的概率漂移定位到自动调优执行配置，而非 checkpoint 变化。最终又使用新登记和全新进程，对每臂首轮全部128个实际采样动作重算：

|策略|检查动作数|最大 log-prob 误差|1e−6门槛|
|---|---:|---:|---|
{reload_rows}

固定配置定义了新的数值执行约定，不能要求它重现旧的未固定配置历史概率；要求的是新约定在重复执行、进程重启及不同预热历史中保持一致。不会放宽容差来制造通过结果。

## 3. 相同更新次数的速度对照

每臂384次真实决策，3轮×4环境×32步；2个PPO epoch，minibatch32、microbatch4；每轮8次actor更新。critic每轮8个epoch。三臂统一actor学习率 **5e−7**、KL阈值 **0.03**。这是诊断用稳定学习率，不代表已经确定正式训练最优学习率。

较高学习率预试验曾早停，失败结果保留。最终三臂使用同一学习率、相同初始权重、相同seed与更新预算；降低学习率不减少前后向次数，未降低KL门槛。

![相同工作量耗时](timing.svg)

|方案|采样秒|更新秒|总计秒|actor更新次数|世界刻数|峰值GiB|
|---|---:|---:|---:|---:|---:|---:|
{table}

|轮次|原固定填充：秒|原去填充KV：秒|中文语义第二版：秒|
|---|---:|---:|---:|
{rounds}

相对本轮原协议去填充KV路径，新协议加速 **{vsold:.3f}倍**。两倍主要门槛的分母是原B4×1536协议，不把这两个比较混用。计时包含采样、实际环境交互、日志、bootstrap、PPO与critic更新，包含首轮开销，没有挑选最快一轮。

同一32个真实state、同一动作分支的固定actor＋reference＋critic前向，三次中位数分别为：原固定填充 **{totals['baseline']['fixed_forward']:.3f}秒**、原KV **{totals['old_optimized']['fixed_forward']:.3f}秒**、新语义 **{totals['semantic']['fixed_forward']:.3f}秒**。对应固定前向加速 **{fixedspeed:.3f}倍**，该项不含反向，不能直接替代整体训练指标。

## 4. 文本长度与缓存复用

384条相同历史观察的单元检查确认：可见事实结构、合法引擎动作、目标名称和critic特征等价；规则与state字符总量至少减少15%。另外对这384条观察实际分词：上一版中文协议平均 **{paired['previous_semantic']['mean']:.2f} token**，本版 **{paired['compact_semantic']['mean']:.2f} token**，配对减少 **{paired['paired_mean_reduction']:.2%}**。字符数与token数分别统计。

以下token长度来自各臂实际轨迹，轨迹不同，不当作配对文本消融实验：

|方案|平均动词输入token|最长输入token|
|---|---:|---:|
{token_rows}

新语义计数器累计规则前缀命中 **{cache['system_hits']}**、构建 **{cache['system_misses']}**，命中率 **{hit:.2%}**；包括采样、PPO及更新后复核。训练中的规则KV只在同一计算图共享，不为了提高命中率而跨更新使用旧图。state作用域复用命中 {cache['state_hits']} 次，非根目标分支batch {cache['branch_batches']} 次、分支行 {cache['branch_rows']} 行。命中率的单位是缓存查询次数，不是token占比。

反向前的实模型探针确认：批量导航目标有非零有限梯度；冷热前缀、样本逆序、单样本与分组、梯度/无梯度的概率误差为0。额外的缓存单测验证了分支梯度累加、卷积状态不污染父分支。

独立逐分支实现与批量目标实现的梯度余弦相似度 **{gradient['gradient_cosine']:.8f}**、相对L2差 **{gradient['gradient_relative_l2']:.2%}**，最大概率差 **{gradient['max_probability_difference']:.6f}**；约541万梯度元素参与比较。两者因BF16执行布局不同并非逐位相等。该探针不执行优化器更新，权重保持不变。

不存在自由生成直到1536输出token的路径。1536仍是输入上限；语义方向是一个token，地标是有限候选名称。当前优化直接减少前向次数、规则前缀重算和目标分支串行调用。

## 5. 概率与短更新稳定性

运行内重新排序/分组最大log-prob误差 **{numeric['reorder_max']:g}**，梯度模式切换误差 **{numeric['grad_mode_max']:g}**。每轮更新前沿用原训练器对前16条样本的重算检查，更新后强制刷新缓存检查；另外独立重载审计覆盖首轮全部128条，不能把前16条检查夸大成每轮全量审计。

新语义缓存/批量路径对独立完整前缀评分（4个导航state）：最大动作概率差 **{numeric['cached_vs_full_prefill']['max_abs']:.6f}**，平均KL **{numeric['cached_vs_full_prefill']['mean_kl']:.6f}**，argmax变化 **{numeric['cached_vs_full_prefill']['argmax_changes']}/4**。BF16分块与完整前缀执行不是逐位相同；运行内与跨进程检验固定的是登记的同一执行布局。

|新协议轮次|actor更新次数|最大approx KL|平均梯度范数（裁剪前）|更新前log-prob误差|早停|
|---|---:|---:|---:|---:|---|
{stable}

所有指标有限、reference冻结状态和权重保持不变，actor确有诊断更新；没有保存更新后的权重。短更新的稳定性不等于长期收敛、生存率达标或最终学习率已确定。

## 6. 环境理解与实际动作构成

沿用固定96道均衡是/否题和32道合法方向题；每种事实类别正负各6道，每个方向8道。正常RL采样不包含题目指令。

|指标|原协议KV|新中文语义|
|---|---:|---:|
|事实题正确数|{under['old_optimized']['facts_correct']}/96|{under['semantic']['facts_correct']}/96|
|方向执行正确数|{under['old_optimized']['grounding_correct']}/32|{under['semantic']['grounding_correct']}/32|

state与动作接口同时变化，不能把差异归因于文字长度单一因素。题集有限，没有据此宣称长期游戏能力提高。协议切换也会显著改变策略的动作偏好，以下为全部实际决策：

|引擎动作|原固定填充|原KV|新语义|
|---|---:|---:|---:|
{counts}

新语义自然采样中有322次移动、1次采集，**没有选中前往**；唯一采集目标无须非根分支评分，因此本段累计目标分支batch为0。不能将2.95倍整体收益归因于批量导航分支。另行强制选择“前往”动词、由模型选择完整目标的4个真实Godot探针，均恰好移动一格、经过一刻，实际state/action/reward另附。批量目标分支的前向、反向与性能验证独立于自然采样轨迹。

因此真实轨迹时间差包含执行优化和动作负载变化；固定state前向测试用于补充控制后者。保留一步导航能力不等于当前checkpoint已经频繁或合理地使用导航；正式训练前仍需做更长的独立行为与生存评估。

## 7. 真实样例与引擎重放

下面直接取新语义轨迹第一条，seed={first['seed']}，update={first['update']}，step={first['step']}，env={first['env']}。完整system规则、前后观察与token ID都在原始记录中。

```text
{first['model_input'][1]['content']}
```

实际引擎动作 `{json.dumps(first['action'],ensure_ascii=False)}`，语义目标 **{first['semantic_target'] or '无额外目标'}**。联合log-prob **{first['log_probability']:.9f}**，概率 **{math.exp(first['log_probability']):.6f}**。经过 {first['transition']['elapsed_ticks']} 刻，reward **{first['reward']['reward']:.12f}**，分项为 `{json.dumps(first['reward']['components'],ensure_ascii=False)}`。这是模型采样记录，没有用人工规则替代选择。

独立Godot重放覆盖三臂全部 **{sum(x['engine_replay']['records'] for x in audit['arms'].values())}条**转移，逐条核对原观察、state、实际token、语义/引擎动作、reward和终止标志；move/navigate均核对恰好一格且一刻。

## 8. 验证、登记与材料

本地完整回归180项：163通过、17条件跳过；10个Godot套件通过。节点180项：177通过、3条件跳过（2项仅限Windows、1项缺少已删除的旧训练证据）。原有112个冻结文件不改，本次登记共124个源码/配置、19个模型与数据输入；所有哈希在运行前后核验。诊断结束后GPU工作结束、没有新checkpoint、正式训练状态为not_started。

运行环境：RTX 6000 Ada 48GB，驱动595.91.07，PyTorch 2.6.0+cu124、Transformers 5.22.1、PEFT 0.21.2、Triton 3.2.0、fla-core 0.3.2。计时在同一节点顺序执行，三轮是同一次有界诊断的连续更新，未宣称为多次独立重复实验的统计置信区间。

- [原始对照](evidence/m8s2-diagnostics/compare1/comparison.json)、[独立重放和跨进程审计](evidence/m8s2-diagnostics/compare1/independent_audit.json)、[汇总数据](analysis_summary.json)、[真实样例](real_sample.json)。
- [节点测试日志](gpu_unit_tests.log)、[本地工程验证](project_verification/checks.json)、[登记](registration.json)。
- [配对token审计](evidence/m8s2-diagnostics/token_audit.json)、[批量与串行梯度](evidence/m8s2-diagnostics/batch_gradient.json)、[四个真实一步导航样例](evidence/m8s2-diagnostics/one_step_navigation_probe.json)、[最终完整性核验](evidence/m8s2-diagnostics/final_integrity.json)。
- [原始证据包](evidence.zip)，SHA-256：`{evidence_sha}`；内部58个文件的哈希已逐一验证，包含预试验失败记录。
- 登记SHA-256：`{reg['registration_sha256']}`。
- 节点登记：`/home/researcher/meadowmind-gpu/m8s2-runs/semantic_diagnostic2/registration.json`；测量与原始轨迹：`/home/researcher/meadowmind-gpu/m8s2-diagnostics/compare1`。
- 源码入口：`agent/m8s2_contract.py`、`agent/m8s2_cache.py`、`agent/m8s2_policy.py`、`agent/m8s2_kernels.py`；诊断与审计为 `tools/benchmark_m8s2.py`、`tools/audit_m8s2_diagnostic.py`。
'''
(ROOT/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({'characters':len(report),'combined_gate':passed,'speedup':speed,'median_speedup':median,'fixed_forward_speedup':fixedspeed,'cache_hit_rate':hit},ensure_ascii=False))
