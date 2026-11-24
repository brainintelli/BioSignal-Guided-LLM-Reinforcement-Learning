"""Build the report from immutable measured artifacts, including failed gates."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'evidence/m8s-diagnostics/semantic_compare1'
def read(name): return json.loads((BASE / name).read_text(encoding='utf-8'))
def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
c = read('comparison.json')
audit = read('independent_audit.json')
reg = json.loads((ROOT / 'registration.json').read_text(encoding='utf-8'))
names = {'baseline':'原协议：固定填充', 'old_optimized':'原协议：去填充＋KV', 'semantic':'中文语义：去填充＋KV'}
totals = {}
for mode, value in c['stability'].items():
    rows = value['updates']
    totals[mode] = {
        'rollout_seconds':sum(r['rollout_seconds'] for r in rows),
        'update_seconds':sum(r['update_seconds'] for r in rows),
        'total_seconds':sum(r['rollout_seconds']+r['update_seconds'] for r in rows),
        'actor_steps':sum(r['optimizer_steps'] for r in rows),
        'decisions':sum(r['decisions'] for r in rows),
        'world_ticks':sum(r['world_ticks'] for r in rows),
        'peak_gib':max(r['peak_allocated_bytes'] for r in rows)/2**30,
        'fixed_forward_median':statistics.median(c['performance'][mode]['fixed_forward_seconds']),
    }
auc = {}
for mode in ('old_optimized','semantic'):
    answers = read(mode+'_comprehension.json')['answers']
    by_category = {}
    for cat in sorted({x['category'] for x in answers}):
        pos = [x['probability_yes'] for x in answers if x['category']==cat and x['answer']]
        neg = [x['probability_yes'] for x in answers if x['category']==cat and not x['answer']]
        assert len(pos)==len(neg)==6
        by_category[cat] = sum(float(p>n)+0.5*float(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
    auc[mode] = {'by_category':by_category,'macro_auc':statistics.mean(by_category.values()),
                 'yes_predictions':sum(x['prediction'] for x in answers), 'post_hoc_exploratory':True}
records = [json.loads(line) for line in (BASE/'semantic_transitions.jsonl').read_text(encoding='utf-8').splitlines()]
samples = []
for verb in ('move','navigate','gather'):
    i,r = next((i,r) for i,r in enumerate(records,1) if r['action']['verb']==verb)
    sample = {k:r[k] for k in ('seed','update','step','env','model_input','semantic_target','action','reward','log_probability','value','transition')}
    sample.update(source_line=i, joint_probability=math.exp(r['log_probability']),
                  semantic_verb={'move':'移动','navigate':'前往','gather':'采集'}[verb])
    samples.append(sample)
dump('real_samples.json',samples)
summary = {'registration_sha256':reg['registration_sha256'], 'timings':totals,
           'primary_comprehension':c['understanding'], 'exploratory_auc':auc,
           'speed_gate_passed':False, 'in_process_consistency_passed':c['probability_consistency_passed'],
           'cross_process_reload_consistency_passed':audit['reload_consistency_passed'],
           'formal_training_started':False,'checkpoint_writes':0}
dump('analysis_summary.json',summary)

table = '\n'.join(f"|{names[m]}|{v['rollout_seconds']:.2f}|{v['update_seconds']:.2f}|{v['total_seconds']:.2f}|{v['actor_steps']}|{v['world_ticks']}|{v['peak_gib']:.2f}|" for m,v in totals.items())
rounds = '\n'.join('|'+str(i+1)+'|'+'|'.join(f"{c['stability'][m]['updates'][i]['rollout_seconds']+c['stability'][m]['updates'][i]['update_seconds']:.2f}" for m in names)+'|' for i in range(3))
fact_rows='\n'.join(f"|{cat}|{c['understanding']['old_optimized']['categories'][cat]:.1%}|{c['understanding']['semantic']['categories'][cat]:.1%}|{auc['old_optimized']['by_category'][cat]:.3f}|{auc['semantic']['by_category'][cat]:.3f}|" for cat in auc['semantic']['by_category'])
sample_rows='\n'.join(f"|{s['source_line']}|{s['seed']} / {s['step']}|{s['semantic_verb']} → {s['semantic_target']}|{s['joint_probability']:.6f}|{s['reward']['reward']:.12f}|{s['transition']['elapsed_ticks']}|" for s in samples)
profiles = audit['arms']
ratio_target=profiles['semantic']['component_profile']['navigation_target_seconds']/profiles['old_optimized']['component_profile']['navigation_target_seconds']
protocolratio=totals['semantic']['total_seconds']/totals['old_optimized']['total_seconds']
fixedratio=totals['semantic']['fixed_forward_median']/totals['old_optimized']['fixed_forward_median']

# A standalone, data-derived vector chart; no remote resources or scripts.
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 300" role="img" aria-labelledby="t d">',
     '<title id="t">384 次真实决策的采样与更新耗时</title><desc id="d">中文语义臂仅完成 21 次 actor 更新，两个旧协议臂各 24 次，不能宣称等工作量加速。</desc>',
     '<rect width="920" height="300" fill="#f5f8fa"/><g font-family="Microsoft YaHei, sans-serif" fill="#263b4a">',
     '<text x="24" y="30" font-size="19">384 次真实决策：采样与更新耗时（秒）</text>']
for i,(mode,t) in enumerate(totals.items()):
    y=65+i*60; x=215; scale=2.2; a=t['rollout_seconds']*scale; b=t['update_seconds']*scale
    svg += [f'<text x="24" y="{y+21}" font-size="14">{names[mode]}</text>',
            f'<rect x="{x}" y="{y}" width="{a}" height="30" fill="#297a83"/>',
            f'<rect x="{x+a}" y="{y}" width="{b}" height="30" fill="#90b8bf"/>',
            f'<text x="{x+a+b+12}" y="{y+21}" font-size="14">{t["total_seconds"]:.2f}s</text>']
svg += ['<rect x="215" y="244" width="16" height="13" fill="#297a83"/><text x="238" y="256" font-size="13">采样与环境交互</text>',
        '<rect x="390" y="244" width="16" height="13" fill="#90b8bf"/><text x="413" y="256" font-size="13">PPO / critic 更新</text>',
        '<text x="24" y="283" font-size="13" fill="#8b3f23">语义臂：21 次 actor 更新；其余：24 次。两倍速度门槛未通过。</text></g></svg>']
(ROOT/'timing.svg').write_text('\n'.join(svg),encoding='utf-8')

report=f'''# 简短中文 state 与语义动作：节点 5 实测报告

2026-11-08 · 集群 1 节点 5（`researcher-local-5`）· 初始权重 `checkpoint_0032` · 仅有界诊断

已接入简短中文 state、上下左右每次一格、中文动作和一步导航，并完成三臂实测。**目前不能保持两倍以上整体加速，也未达到正式训练前的全部验证条件。**方向执行题由 11/32 提升到 32/32；主要事实理解题由 54/96 变为 53/96。运行内 PPO 概率重算一致，但跨进程重载检查出现偏差。新协议首轮触发 KL 早停，需继续修复和复测。

本轮没有启动正式训练、没有保存诊断 checkpoint、没有切换游戏服务。三臂均重新加载同一初始权重，重置优化器并重新采样。诊断中的短 PPO 更新仅用于测量训练耗时和稳定性。

## 1. 已采用的 state 和动作协议

|项目|当前实现|
|---|---|
|环境名|移出 actor 输入，保留在运行元数据和身份校验中。|
|日内步|改为“清晨/白天/黄昏/夜间＋距入夜或天亮多少刻”，时间精度不丢失。|
|局部地图|保留以自身为中心的 25 格，用中文地形、相对位置及默认地形＋例外位置表达。未观察、水面、实体阻挡明确区分。|
|身体与背包|生命、饱食、水分、体力，以及木材、石头、食物；单位和数值方向由固定规则说明。|
|可用信息|保留合法方向、地标种类与相对位置、已知路径长度、可执行动作、再生时间、访问次数、最近 4 次动作、营地和威胁。|
|移动|模型选择“移动”，再选择“向上/向右/向下/向左”；方向固定为地图方向，每次恰好一格。|
|一步导航|模型选择“前往”，再选择完整中文目标，例如“蘑菇（左5格）”；引擎避障推进一格，不自动采集或喝水。|
|其他动词|采集、吃饭、喝水、休息、建造、等待。|

语义文本与引擎 action 保持一一对应。引擎仍需要 `verb` 和目标 ID 执行动作，但模型不再选择任意字母或 `tN` 别名。合法候选约束决定哪些动作可选；不会在非法输出后用规则代替模型挑选另一个动作。

新增入口为 `agent/m8s_contract.py`、`agent/m8s_policy.py`，状态格式在 `agent/m8s_state.py`，完整目标概率在 `agent/m8s_trie.py`。旧协议及前次登记的 102 个文件保持不变；本次登记绑定 112 个源码/配置文件及 19 个输入文件。critic 沿用信息等价的 97 维特征，不把隐藏事实额外提供给 actor。

“简短中文”是表达方式的变化，**不等于 tokenizer 更短**。本轮实际 rollout 的平均动词输入长度为原协议优化臂 912.80 token、新协议 955.83 token；二者轨迹不同，此数值只描述实际负载。先前同观察配对的候选统计同样显示约 5.5% 的 token 增加。

## 2. 真实 state、action、reward

以下 state 是新协议真实模型轨迹第 1 行的完整 user 消息，seed=883810，update=1，step=0，env=0。固定 system 规则、实际输入 token ID、前后观察保存在原始 JSONL；不是人工编造的展示场景。

```text
{samples[0]['model_input'][1]['content']}
```

模型本次选择 `移动 → 向上`，绑定引擎动作 `{{"verb":"move","target_id":"tile_47_46"}}`。角色从 (47,47) 移到 (47,46)，经过 1 刻。饱食 82→81.8，水分 78→77.92，体力 92→91.65，生命保持 100。

动作联合 log-prob = **{samples[0]['log_probability']:.9f}**，联合概率 **{samples[0]['joint_probability']:.6f}**。reward = **0.008875**，来自存活项 0.01、身体势能差 −0.000625、饮水势能差 −0.0005；折扣为 0.9995。发现新格子 15 个记录在环境转移中，本次奖励没有额外按发现数量加分。

另抽取每种动作首次出现的真实记录，选择规则固定为“首次出现”，没有按奖励挑选：

|JSONL 行号|seed / step|语义动作|联合概率|reward|经过刻数|
|---|---|---|---:|---:|---:|
{sample_rows}

一步导航样例从 (47,47) 到 (46,47)，仍只走一格；目标在左 5 格不意味着一次走 5 格。采集样例食物由 2 增至 5，耗时 2 刻；“每次一格”约束只适用于移动和导航。

完整样例见 [real_samples.json](real_samples.json)；原始轨迹见 [semantic_transitions.jsonl](evidence/m8s-diagnostics/semantic_compare1/semantic_transitions.jsonl)。三臂合计 1,152 条真实转移已由独立 Godot 进程逐条重放，观察、输入文本与 token、引擎动作、reward、终止标志均匹配。所有 872 次 move/navigate 的曼哈顿位移均为 1、耗时均为 1 刻；新协议占其中 292 次。

## 3. 算法、概率定义与参数

沿用分层离散动作 PPO：先选动词，再选该动词合法目标；使用 critic、按世界刻折扣的 GAE、PPO clipping、reference KL 和熵正则。语义动词及四个方向已用实际 tokenizer 确认各为一个 token。多 token 地标名称使用有限合法目标前缀树。

`π(动作|state) = π(中文动词|state) × π(中文目标|state, 动词)`。

目标概率为沿完整名称路径的各分支 softmax 概率之积。唯一合法后继是强制延续，概率为 1；仍保留其上下文供后续分支使用。EOS 区分前缀相同的长短名称。PPO 使用全部合法目标的归一化分布，而非首 token 概率或平均 token log-prob。该定义属于受约束语法策略，与把无约束语言模型整句概率归一化的定义不同。

KV 缓存复用 state 和已有目标前缀，包含模型所需的注意力、卷积和循环状态；梯度路径保留。缓存不跨 optimizer 更新存活，合并投影在权重更新后刷新。新旧协议的策略分布不同，因此旧 rollout 和旧 log-prob 不可继续使用。reference 仍是冻结的 checkpoint_0016 权重，但语义臂也采用新的输入和动作定义。

|参数|本次实测|
|---|---|
|模型 / 硬件|Qwen3.5-0.8B；RTX 6000 Ada 48GB|
|初始 actor|checkpoint_0032；每臂重新载入；优化器重置|
|环境 / seed|open3，world_version=5；883810–883813|
|轮次 / 采样|每臂 3 轮；4 环境 × 32 决策/轮；384 决策/臂|
|PPO / minibatch / microbatch|2 epoch；32；4|
|actor LR / clip / 梯度裁剪|1e−5；0.15；1.0|
|target KL / reference KL 系数|0.03；0.005|
|熵系数|初始 0.002，登记的衰减终值 0.0002，周期 98,304 决策|
|critic|8 epoch；minibatch 128；LR 3e−4；梯度裁剪 1.0|
|每刻折扣 / GAE λ|0.9995 / 0.999|
|输入上限 / 梯度检查点|1536 token / 关闭|
|随机种子|20261110；新采样，不复用历史优化器或 rollout|

完整配置见 [registration.json](registration.json)。原固定长度路径 B4×1536 是输入填充；该数值不是输出 1536 token 的生成预算。动词/方向单 token；此前实际目标名称长度上限为 11 token，当前有限目标语法也不存在无限自由生成直到 max token 的路径。

## 4. 整体速度：两倍门槛未通过

计时包括真实采样、环境交互、日志、bootstrap、PPO 和 critic 更新；不含模型加载、单独理解题、单独概率检查和之后的审计。各臂顺序运行、无并发 GPU 工作。决策数相等，但策略不同导致动作构成和世界刻数不同；另提供相同状态的固定前向测试。

![三臂采样与更新耗时](timing.svg)

|方案（每臂384决策）|采样秒|更新秒|合计秒|actor 更新次数|世界刻数|峰值 GiB|
|---|---:|---:|---:|---:|---:|---:|
{table}

|轮次|原协议固定填充：秒|原协议优化：秒|中文语义优化：秒|
|---|---:|---:|---:|
{rounds}

原协议优化本轮总耗时加速 **1.658×**，逐轮中位数比 **1.830×**，均未超过 2。不能将此前另一组较短诊断的 2.060× 当成本次结果。第一轮与后续轮存在明显耗时差异；本轮未进一步证明具体是编译、缓存还是工作负载差异，不剔除第一轮来宣称达标。

新协议 244.14 秒相对固定填充的原始时间比为 **1.031×**，但首轮 KL 早停导致只完成 **21 次** actor 更新，而对照为 **24 次**。因此这不是有效的等工作量加速结论，登记中的 `equal_work=false`、`speed_gate.passed=false`。新协议实际耗时为原协议优化臂的 **{protocolratio:.3f} 倍**，即使少做 3 次 actor 更新仍更慢。

相同 32 个真实 state、固定目标分支的 actor＋reference＋critic 前向测试，三次重复中位数依次为 **4.687、1.887、2.671 秒**。新协议相对原协议优化路径增加 **{(fixedratio-1)*100:.1f}%**，说明成本增加不只是采样到了不同动作。

## 5. 当前主要瓶颈

独立组件测量采用相同 4 个 state、导航分支，每状态先预热一次，再重复 3 次，CUDA 同步计时。此项只测推理组件，不等于完整训练耗时分解。

|组件中位数|原协议优化|中文语义优化|
|---|---:|---:|
|动词阶段|23.06 ms|23.13 ms|
|导航目标阶段|23.16 ms|137.39 ms|

完整中文目标评分约为旧目标评分的 **{ratio_target:.2f} 倍**。当前实现虽复用 KV，但仍依次计算多个目标前缀树分支；新语义轨迹中 296 次多目标动作共涉及 936 个分支节点，平均 3.16 个、最多 6 个。评分为了 PPO 熵与 KL 会计算全部合法目标，成本高于只生成选中的短文本。

此外，新协议选择导航＋采集 192/384 次（50.0%），原协议优化为 128/384 次（33.3%），更频繁进入较昂贵的多 token 目标路径；新 state 的实际 token 数也略增。固定填充和重复 state 前向已被优化，但节省被新的目标评分开销明显抵消。**这次慢的直接证据指向目标树的串行评分及负载变化，不支持“模型频繁输出到 max token”这一解释。**

## 6. 环境理解：方向改善，整体事实正确率没有改善

96 道事先固定的是/否题，每类 12 道、正负各 6，道题来源为真实观察；32 道方向题每方向 8 道，期望动作均在当前合法候选内。两种 state 的事实题使用相同诊断规则。数据含 384 条已有模型轨迹观察及 52 条真实 Godot 时间快照，独立标签审计没有发现错误。

|主要指标|原协议优化|中文语义|
|---|---:|---:|
|事实题正确数|54/96|53/96|
|事实题准确率|56.25%|55.21%|
|方向执行题|11/32（34.38%）|32/32（100%）|
|回答“是”的次数|6/96|5/96|

方向测试证明该模型在新接口下能正确把“向上/下/左/右”对应到合法一步动作；同时修改了 state 和 action，不能把增益全部归因于其中一个。事实题存在明显“否”偏置，不能据此宣称新 state 已使模型全面理解环境，也不能只看单例解释。

为区分“事实区分能力”和“是/否阈值偏置”，结果产生后补算了按类别的 AUROC，不调整阈值、不增加训练、不将其代替主要正确率。**这是探索性指标，未预先登记**：宏平均由 {auc['old_optimized']['macro_auc']:.4f} 提升至 {auc['semantic']['macro_auc']:.4f}，提示部分事实的概率排序有改善，仍需独立题集复核。

|类别|旧准确率|新准确率|旧探索性 AUC|新探索性 AUC|
|---|---:|---:|---:|---:|
{fact_rows}

这些是短题集上的接口诊断，没有多 seed 统计显著性结论；没有用当前题集拟合回答阈值。

## 7. 动作概率：运行内通过，跨进程严格重载未通过

|检查|结果和范围|
|---|---|
|概率归一化|32 个固定 state 的全部合法动作、actor/reference 均在 1e−5 容差内归一化。|
|运行内重算|样本逆序、batch 1/3/4、梯度/无梯度、PPO 更新前 log-prob、更新后强制刷新投影缓存：测得最大误差均为 0。|
|语义 KV 对完整前缀重算|4 个导航 state，最大动作概率差 0.028940，平均 KL 0.000899，最大 KL 0.003174；argmax 0/4 改变。不能称逐位相等。|
|原协议优化对固定填充|32 个 state，最大动作概率差 0.040940，平均 KL 0.006113，argmax 5/32 改变。|
|新语义协议对旧固定填充|最大动作概率差 0.440376，平均 KL 1.322162，argmax 18/32 改变；接口改变后本就不要求保持旧策略概率。|
|独立进程重新加载 checkpoint_0032|各臂首轮 128 个动作：固定填充最大 log-prob 差 0；旧优化 0.046612；新语义 0.053862。后两项未通过 1e−6 容差。|

运行内一致性与跨进程重载是不同检查。原始 `comparison.json` 中 `probability_consistency_passed=true` **只表示它包含的运行内检查**；补充的 `independent_audit.json` 明确记录 `reload_consistency_passed=false`。不能只引用前者来宣布全部概率验证通过。

重载探针确认权重未改变，并核对了输入 token；reference 开/关、模型 metadata 读取、创建优化器、重置采样 seed、act/distributions 调用方式，均未消除偏差。原因还没有被隔离到具体内核或计算路径，不能武断归因于 BF16 或 Triton 自动调优。应在恢复训练、重用采样或跨进程部署之前解决或建立有证据支持的数值容差；本次不放宽门槛。

## 8. 短更新稳定性与验证状态

|新协议轮次|actor 更新次数|最大 approx KL|KL 早停|平均 actor 梯度范数（裁剪前）|
|---|---:|---:|---|---:|
|1|5|0.035766|是，超过 0.03|8.806|
|2|8|0.024294|否|7.720|
|3|8|0.016037|否|9.384|

损失、梯度和回报均为有限值，reference 权重保持冻结，actor 在诊断更新中确有变化；没有 NaN/Inf。首轮早停保护正常触发，但意味着沿用旧协议学习率并不能认定为新协议已经稳定。三臂短轨迹均无自然死亡，这不等于通过长期生存或收敛验收。

新增 5 项语义测试经历失败→实现→通过，覆盖真实观察动作映射、事实保持、中文目标、前缀 EOS、分布归一化和输入顺序不变性。本地完整 Python 套件 177 项：161 通过、16 跳过；节点 5 套件 177 项：174 通过、3 跳过。节点跳过项为两个 Windows PowerShell 专用用例和一个缺少历史证据的测试。官方本地验证脚本的 10 个 Godot 套件全部通过。

保留初次测试失败日志：本地 sandbox 限制 HTTP 测试监听，获得工具自动审核批准后正常通过；节点初次未设置 `MEADOW_GODOT` 导致两个用例使用 Windows 路径，设置真实 Linux Godot 路径后完整重跑通过。没有改断言或跳过失败用例来得到通过结果。

最终核验：112 个登记源码/配置、19 个输入哈希均匹配；诊断目录无 checkpoint；GPU 测试及诊断进程结束。单元测试通过不覆盖上述性能和跨进程概率的失败门槛。

## 9. 下一阶段应先解决的工作

1. 保留中文语义和完整目标概率定义，把相同深度的前缀树分支合并成批处理，减少串行小 kernel 和重复前缀调度；缓存仍要覆盖 Qwen 混合注意力/循环状态并保持梯度。先对同一 state 的全部动作概率、梯度和跨进程重载做对照，再测端到端。
2. 对新协议单独登记较小 actor 学习率的短诊断，保留 KL 阈值 0.03，保证各臂实际完成相同 optimizer 步数；不能靠少更新、放宽 KL 或缩减目标候选取得两倍数字。
3. 对事实题增加独立验证集及答案顺序/表述对照，区分“否”输出偏置与真正理解错误。需要语义适配时必须单独登记，不能把当前小题集训练后再当测试集使用。
4. 对跨进程数值偏差逐层定位，记录实际执行内核及配置，比较 state prefill、分支 KV、合并 LoRA 投影等路径。当前排查证据不足以指定唯一根因。

本次结束时保留诊断实现与全部证据。正式训练仍为 **not_started**。

## 10. 证据与登记

- [对照原始结果](evidence/m8s-diagnostics/semantic_compare1/comparison.json)、[独立重载与引擎重放审计](evidence/m8s-diagnostics/semantic_compare1/independent_audit.json)、[汇总机器可读数据](analysis_summary.json)。
- [新协议事实题](evidence/m8s-diagnostics/semantic_compare1/semantic_comprehension.json)、[旧协议事实题](evidence/m8s-diagnostics/semantic_compare1/old_optimized_comprehension.json)、[方向执行题](evidence/m8s-diagnostics/semantic_compare1/semantic_grounding.json)。
- [节点完整测试日志](gpu_unit_tests.log)、[本地测试日志](unit_tests_unsandboxed.log)、[本地官方验证记录](project_verification/checks.json)。
- [原始证据包](evidence.zip)：SHA-256 `39284e564e4d4e375e002c7abd1debbf9defac5ade9a178f7563fd2911320b77`，包含 30 个带逐文件哈希的原始产物；之后的最终完整性检查和测试日志单独保存。
- 登记 SHA-256：`{reg['registration_sha256']}`。模型参数 digest 为 `21f5287ea68726a4f5769b826db29f52a8e2d8664dcbb3e6ad68500c357ccf66`；adapter 文件 SHA-256 为 `4a73bfa50295d2351f054440f3d9b534c511d388729e0816995701f02a99022e`，二者是不同校验对象。
- 远程登记：`/home/researcher/meadowmind-gpu/m8s-runs/semantic_diagnostic1/registration.json`。原始结果：`/home/researcher/meadowmind-gpu/m8s-diagnostics/semantic_compare1`。
- 登记配置的继承元数据 `config.previous_registration_sha256` 残留了更早 attempt 的值；实际登记脚本核验的顶层 `previous_registration_sha256` 是正确的 `58c000843defbf3b8631ad0afd05d934db49e15af4768c6f51b483fb106e6c60`。本轮运行使用明确的三臂列表及 `SemanticPolicy`，没有用该继承字段选择算法；原登记保持不改，后续登记应清理这些继承元数据。
'''
(ROOT/'report.md').write_text(report,encoding='utf-8')
shutil.copytree(ROOT.parent/'current/verification',ROOT/'project_verification',dirs_exist_ok=True)
print(json.dumps({'report_characters':len(report),'real_samples':len(samples),'auc':auc,'timings':totals},ensure_ascii=False))
