"""Build a Chinese evidence report and matching SVG/PNG algorithm diagram."""
import hashlib
import html
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
profile = json.loads((HERE/'profile.json').read_text(encoding='utf-8'))
rows = [json.loads(x) for x in (HERE/'fresh_transitions.jsonl').read_text(encoding='utf-8').splitlines()]
chosen = {}
for index, row in enumerate(rows, 1):
    chosen.setdefault(row['action']['verb'], {**row, 'jsonl_line':index})
(HERE/'real_examples.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'state_example.json').write_text(json.dumps(rows[3],ensure_ascii=False,indent=2),encoding='utf-8')

# Diagram: one coordinate system renders both portable SVG and raster preview.
W,H = 1640,1120
canvas = Image.new('RGB',(W,H),'#f4f7fb')
d = ImageDraw.Draw(canvas)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="100%" height="100%" fill="#f4f7fb"/>']
fontfile='C:/Windows/Fonts/msyh.ttc'
boldfile='C:/Windows/Fonts/msyhbd.ttc'
def text(x,y,value,size=24,color='#24354b',bold=False):
    font=ImageFont.truetype(boldfile if bold else fontfile,size)
    d.text((x,y),value,font=font,fill=color)
    svg.append(f'<text x="{x}" y="{y+size}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>')
def box(x,y,w,h,title,lines,color='#ffffff',accent='#325ba4'):
    d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=color,outline='#cbd6e4',width=2)
    d.rectangle((x+1,y+20,x+6,y+h-20),fill=accent)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{color}" stroke="#cbd6e4" stroke-width="2"/>')
    svg.append(f'<rect x="{x+1}" y="{y+20}" width="5" height="{h-40}" fill="{accent}"/>')
    text(x+22,y+16,title,27,accent,True)
    for i,line in enumerate(lines):text(x+22,y+64+i*35,line,22)
def arrow(points,color='#718299',width=3):
    d.line(points,fill=color,width=width,joint='curve')
    x,y=points[-1]; ax,ay=points[-2]; theta=math.atan2(y-ay,x-ax)
    tip=[(x,y),(x-13*math.cos(theta)+6*math.sin(theta),y-13*math.sin(theta)-6*math.cos(theta)),
         (x-13*math.cos(theta)-6*math.sin(theta),y-13*math.sin(theta)+6*math.cos(theta))]
    d.polygon(tip,fill=color)
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}"/>')
    svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in tip)+f'" fill="{color}"/>')
text(52,35,'湿件生存游戏 · 当前强化学习闭环',39,bold=True)
text(55,96,'open3 / world 5  ·  Qwen3.5-0.8B  ·  两阶段离散动作 PPO + LoRA',24,'#53667e')
text(55,143,'01  采样：4 个真实 Godot 环境同步收集，每轮 512 步，共 2048 次决策',23,'#325ba4',True)
box(50,195,390,180,'真实游戏环境', ['生命 / 饱食 / 水分 / 体力','已知地标、局部地图、动作历史','完整合法动作；仅自然死亡重置'])
box(515,195,600,180,'策略：编码观察 → 动词 → 按需选目标', ['中文事实 + 规则 → 固定输入 B4 × 1536','Actor 前向：动词约 168 ms，目标约 169 ms','读取单位置 logits；没有自回归生成循环'])
box(1190,195,400,180,'执行模型选择的动作', ['如 go + A → navigate(target_id)','移动 1 tick / 采集 2 / 休息 3','每次只执行所选动作，不代做补给'])
arrow([(440,282),(515,282)]);arrow([(1115,282),(1190,282)])
arrow([(1390,375),(1390,425),(245,425),(245,375)])
text(535,389,'后继观察 + 精确逐 tick 转移 → 下一次决策',22)
text(55,462,'02  记账：奖励、价值估计与旧策略概率进入同一条真实转移',23,'#325ba4',True)
box(50,515,390,210,'环境奖励 r', ['每个存活 tick +0.01（带折扣）','生命 / 储备势能差 + 水源势能差','r = 存活 + γ^Δt Φ(s′) − Φ(s)','无动作奖励、探索奖励或达标奖励'],accent='#247769')
box(515,515,600,210,'Rollout buffer：2048 条新决策', ['(观察, 动词, 目标, r, Δt, V, old log π, πref)','按真实持续时间计算 GAE / return','512 步尾部仍活着：V(s′) 自举，不重开世界','自然死亡：后继价值与势能归零'],accent='#247769')
box(1190,515,400,210,'辅助计算', ['冻结 M6 父策略：参考 KL 分布','参考动词 + 目标：约 169 +169 ms','独立 critic：97 → 128 → 128 → 1','critic 不替模型选择动作'],accent='#247769')
arrow([(245,425),(35,425),(35,610),(50,610)],'#247769');arrow([(440,610),(515,610)],'#247769')
arrow([(1190,610),(1115,610)],'#247769')
arrow([(1115,330),(1145,330),(1145,480),(1390,480),(1390,515)],'#247769')
text(55,753,'03  更新：独立优化 Actor 与 Critic；然后用新 Actor 采下一轮',23,'#325ba4',True)
box(515,805,600,200,'PPO 更新 + 保存 + 每 16 轮筛选', ['Actor：minibatch 32 / microbatch 4，至多 2 epochs','clip 0.15；参考 KL 0.005；熵 0.002 → 0.0002','old-policy KL > 0.03：停止当前 Actor 更新','Critic：8 epochs；筛选只评估，不反传'],accent='#8857a8')
arrow([(815,725),(815,805)],'#8857a8')
arrow([(815,1005),(815,1025),(20,1025),(20,175),(815,175),(815,195)],'#8857a8')
box(50,815,390,180,'主要耗时（本次短探针）', ['策略计算 96.5%  /  环境 2.3%','冻结参考前向占总采样约 48.1%','瓶颈在长输入反复前向与反传'],color='#edf4ff',accent='#325ba4')
box(1190,815,400,180,'max_tokens 的真实含义', ['1536 = 输入上限 + 固定填充长度','不是输出长度预算','288 次新决策：generate 调用 0'],color='#edf4ff',accent='#325ba4')
text(55,1050,'证据：2026-11-07 节点 5 短探针；零优化器更新，未重启正式训练。历史试验停于 40/48 轮。',21,'#53667e')
svg.append('</svg>')
(HERE/'rl_algorithm.svg').write_text('\n'.join(svg),encoding='utf-8')
canvas.save(HERE/'rl_algorithm.png')

def dump(value):return json.dumps(value,ensure_ascii=False,indent=2)
def code_link(rel,line):return f'[{rel}]({(ROOT/rel).as_posix()}:{line})'
timing=profile['timing']; total=profile['measured_seconds']
time_rows='\n'.join(f'| {label} | {timing[key]["n"]} | {timing[key]["mean"]*1000:.3f} | {timing[key]["sum"]:.3f} | {timing[key]["sum"]/total*100:.2f}% |' for key,label in
    [('actor_verb','Actor 动词前向'),('actor_target','Actor 目标前向'),('reference_verb','父参考动词前向'),('reference_target','父参考目标前向'),('prepare','观察编码与 tokenize'),('env_step','Godot RPC step'),('reward','奖励计算'),('critic','Critic 价值估计')])
sample_rows=[]
for verb in ['navigate','move','eat','gather','rest','drink','wait']:
    r=chosen[verb]; tr=r['transition']; b,e=tr['before'],tr['after']; comp=r['reward']['components']
    action=json.dumps(r['action'],ensure_ascii=False,separators=(',',':'))
    sample_rows.append(f'| {r["jsonl_line"]} | {r["seed"]} | {b["tick"]}→{e["tick"]} | `{action}` | {comp["survival"]:.12f} | {comp["potential"]:.12f} | {comp["water_potential"]:.12f} | **{r["reward"]["reward"]:.12f}** |')
nav=rows[3]
state_summary={k:nav['facts'][k] for k in ['环境','日内步','剩余数值','背包','营地相对位置','局部5x5','危险','合法动作代码']}
state_summary['地标条目节选']=nav['facts']['地标[别名,种类,dx,dy,路程,可用,再生步]'][7:8]
state_summary['go目标条目节选']=nav['facts']['各动作的合法目标']['go'][:1]
report=f'''# 当前强化学习算法与训练速度调查报告

取证日期：2026-11-07，北京时间。对象：`open3 / world_version=5 / m8_exploration_v1`，已停止的 `exploration_attempt1`。通过 Netcatty MCP 复用“集群1节点5”连接，核实主机 `researcher-local-5`、实际权重与运行依赖。报告、图和原始证据均保存在本目录。

## 1. 结论与证据边界

**当前慢的主要机制是：固定长输入的多次模型前向，以及 PPO 对这些长输入重复做反向传播。不是模型频繁生成到 max token。** 当前策略没有 `generate()` 解码循环；`max_tokens=1536` 同时约束输入长度和物理 padding 长度。每次读取最后位置的候选 token logits，从动词/目标离散分布采样。

本次用真实候选 `checkpoint_0032`、真实 GPU 和真实 Godot 采集 288 次新决策，其中 32 次用于预热，256 次计时。计时段 **44.627 秒、5.736 决策/秒**；策略计算占 **96.46%**，Godot 步进占 **2.30%**。冻结父参考前向占总采样时间 **48.10%**。模型输入平均约 927 token（动词阶段）、989 token（目标阶段），却都补齐到 1536。

本次另外测了 Actor 前向/反向的单个 32 样本 minibatch，预热后约 **7.062 秒**；没有调用优化器 step，没有保存检查点。模型和参考权重哈希前后完全一致。全部 288 条转移已在本地真实 Godot 中重放，并重算奖励、核对输入与动作。

**限制：旧 `training.jsonl`、逐步轨迹和作业训练日志已按此前清理要求删除。** 保留的进度摘要、注册文件和筛选报告可以证明预算、配置与结果，但不能恢复原 16.24 小时中每个阶段的精确占比。下文把“历史事实”“代码确定行为”“本次短探针测量”“估算/待验证建议”分开；新样例不冒充旧训练轨迹，也不作为生存验收结果。

## 2. 当前项目进度

游戏原型、碰撞、动作协议、持久 Godot 训练桥接、GPU 策略与回放工程已经建立。M6 的短窗口延寿曾通过当时验收；当前 M8B 的目标是独立测试至少 90/100 局在 1800 tick 结束仍存活。一天为 180 tick。M9 的 95% 超过 30 天尚未开始。

最近一轮于 **2026-11-07 20:23:26** 主动停止，退出码 `-15 / requested_stop`。作业墙钟 58,466.386 秒，即 **16.241 小时**。完成 40/48 轮更新；完整更新使用 81,920 决策、95,908 世界 tick。实际采样 83,968 决策，另 2,048 条属于未完成更新。不能把它算成第 41 轮完成。

| 同一组 16 个筛选世界 | 1800 tick 结束仍活着 | RMST1800 / tick |
|---|---:|---:|
| 原规则 M6 父模型 | 0/16 | 459.875 |
| open3 新规则 M6 父模型 | 9/16 | 1464.5625 |
| open3 第 16 轮候选 | 9/16 | 1401.375 |
| open3 第 32 轮候选 | 11/16 | 1598.7500 |

第 32 轮相对同规则父模型的观察差为 +2 局、+134.1875 tick；它是筛选结果，不能宣称总体成功率已提升或最终验收通过。规则调整带来的收益也不能归为 RL 收益。第 48 轮、完整候选选择和独立 100 局测试均未完成。保留的第 40 轮仅为最后完整更新，没有独立评估。

本次连接前，远端无训练/服务/GPU 计算进程，GPU 利用率 0%。短探针结束后 GPU 计算进程为空。正式训练未恢复，部署未改变。

依据：{code_link('docs/CURRENT_STATE.md',1)}、{code_link('output/current/training_status.json',1)}、本目录 `registration.json`、`validation_0032.json`、`balanced_parent.json` 与 `remote_snapshot.txt`。

## 3. 实际算法：两阶段离散 PPO + LoRA + 独立 Critic

训练入口为 `agent/m8x_train.py`，策略为 `agent/m8x_policy.py`，PPO 更新复用 `agent/m8_train.py:update`。当前没有使用 GRPO、DQN、离线旧轨迹重放或 SIL；文件名中的 M8/M8B/M6 是依赖历史，不应据名称推断当前算法。`m8b_water_potential.py` 的旧模块说明声称尚未接入，但实际已由当前 `m8x_contract.py` 调用。

PPO 交替收集新策略交互数据和优化裁剪目标；GAE 使用价值估计构造优势；LoRA 冻结基座、仅更新低秩参数。理论来源分别为 [PPO 原论文](https://arxiv.org/abs/1707.06347)、[GAE 原论文](https://arxiv.org/abs/1506.02438)、[LoRA 原论文](https://arxiv.org/abs/2106.09685)。下面的精确形式以本项目实现为准。

### 3.1 状态与策略概率

Actor 读取中文规则和序列化观察事实，而不是图像或全部隐藏地图。事实包括四项生存数值、背包、日内时间、营地相对位置、已知地标及距离、5×5 局部地图、局部访问次数、威胁、最近最多四个动作及变化，以及完整合法动作和目标列表。

先在八个动词中施加合法性 mask，采样动词 v；若该动词有多个合法目标，再把已选动词和目标菜单接到输入中，采样目标 t。目标按稳定内容哈希排序，用 `A–Z、a–z、0–9、!、?` 共 64 个单 token 标签映射回真实动作。

```text
πθ(a | o) = πθ(v | o) × πθ(t | o, v)
log πθ(a | o) = log πθ(v | o) + log πθ(t | o, v)
```

单一合法目标不执行目标模型前向，其概率为 1；采样实现仍会对退化目标分布调用采样，从而保持既定 RNG 调用习惯。非法动作不参与选择；执行器只执行选中的动作。`navigate` 只是朝所选已知地标走一格，并不自动采集、吃喝或执行整条路线。

每次完整模型前向为 `B=4, L=1536`；不足四个真实目标样本时复制最后一行补位，再丢弃补位输出。`enable_thinking=False`、`use_cache=False`、`logits_to_keep=1`。代码取最后层最后位置 hidden state，与候选 token 的输出嵌入权重点积得到分类 logits；不存在等待 EOS 的循环。

### 3.2 优势、损失与参考策略

每轮 4 个环境各走 512 个决策，得到 2048 个新样本。动作持续时间 Δt 不固定，折扣和 GAE 都按真实 tick 计算：

```text
δt = rt + γ^Δt × (1 − terminal) × V(st+1) − V(st)
At = δt + γ^Δt × λ^Δt × (1 − boundary) × At+1
return = At + V(st)
```

自然死亡时 next value 为 0。rollout 尾部仍活着则使用真实后继状态的 Critic 值自举，只截断本批优势递推，不重置环境。1800 tick 是评估观察窗，训练没有到 10 天自动重开。

Actor 使用标准化优势和联合动作概率比 `ρ=exp(logπnew−logπold)`。代码最小化：

```text
Lactor = mean[max(−A×ρ, −A×clip(ρ, 0.85, 1.15))]
         + 0.005 × KLreference − entropy_coef × H
```

参考分布来自独立加载、完全冻结的 M6 父策略，沿用当前 open3 观察接口。条件目标 KL/熵由旧动词策略抽样，因此实现用 `w=πnew(v|o)/πold(v|o)` 作条件分支重要性修正：`KLreference = KLverb + w×KLtarget`，熵同理。该 w 不裁剪，也不是联合 PPO ratio。

old-policy 近似 KL 为 `mean[(ρ−1)−logρ]`。一个 minibatch 完成前向/反向后，若它超过 0.03，就跳过该 minibatch 的 optimizer step 并停止本轮 Actor 更新。因此 **epochs=2 是上限，实际优化器步数不必等于 128**。这项检查在反向传播之后，触发早停的 minibatch 仍已付出计算成本。

Critic 是独立的 `97→128→128→1` Tanh MLP，输入只从 Actor 可见事实提取，损失为 `0.5×mean[(V−return)²]`；梯度不会进入语言模型。

依据：{code_link('agent/m8x_policy.py',103)}、{code_link('agent/m8_train.py',49)}、{code_link('agent/m8x_train.py',37)}、{code_link('agent/rl_contract.py',159)}、{code_link('agent/m6_contract.py',82)}。

## 4. 冻结训练参数

下表来自远端保留的 `registration.json.config`，与本地冻结配置一致；不是建议值。

| 项目 | 实际值 |
|---|---|
| 基座 | Qwen3.5-0.8B；加载模块总参数 852,985,920；不使用视觉输入 |
| Actor 初始化 | M6 `small_v2_stable/checkpoint_0016` |
| 父 Actor SHA256 | `02111ea2ce34f61647713d348099de4fc577763bc2423e2a1078097e36dff211` |
| LoRA | r=8，alpha=16，dropout=0，bias=none |
| 可训练 Actor 参数 | 5,411,328，约为所加载基座参数的 0.6344% |
| LoRA 目标模块后缀 | down_proj, gate_proj, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, k_proj, o_proj, out_proj, q_proj, up_proj, v_proj |
| Critic | 97→128→128→1；29,185 参数；加载父 critic |
| 权重/推理 | 基座 BF16，SDPA；固定 B4×1536，左 padding |
| 优化器 | Actor、Critic 各自 AdamW；eps=1e-5，weight_decay=0；未覆盖默认 betas=(0.9,0.999) |
| actor_lr / critic_lr | 1e-5 / 3e-4 |
| envs / rollout_steps / updates | 4 / 512 / 48 |
| 新决策预算 | 98,304；无旧回放、无示范动作 |
| Actor epochs / minibatch / microbatch | 2 / 32 / 4，8 次微批梯度累积构成一个 minibatch |
| Critic epochs / minibatch | 8 / 128 |
| PPO clip / target_kl | 0.15 / 0.03 |
| 冻结父参考 KL 系数 | 0.005 |
| 熵系数 | 从 0.002 按新采样决策预算线性降至 0.0002；每轮取该轮采样开始时对应值 |
| Actor / Critic grad clip | 1 / 1 |
| gamma_tick / gae_lambda | 0.9995 / 0.999；二者都按持续 tick 指数折扣 |
| gradient_checkpointing | false |
| zero-update logp 容差 | 1e-6；更新前检查前 16 条样本 |
| max_tokens | 1536：输入上限与固定填充长度；超限抛错，不静默截断 |
| temperature | 1；当前实际使用未额外缩放的 Categorical 分布 |
| 初始化随机种子 | 98413 |
| 训练世界种子 | 100000…104095，顺序取用，死亡时推进 |
| 筛选世界种子 | 941000…941015，replicate=0 |
| 最终测试世界种子 | 951000…951099，replicate=0；本轮未开启 |
| 筛选频率 / batch | 每 16 轮 / 4 个物理槽，结束槽位动态补新世界 |
| 训练时限 / 作业上限 | 129600 秒（36 h）/ 172800 秒（48 h） |
| horizon / training_horizon | 1800 / null |
| bootstrap 统计参数 | seed=912800，samples=10000 |
| GPU | NVIDIA RTX 6000 Ada Generation，49,140 MiB |
| 实际运行依赖 | Python 3.12.16，PyTorch 2.6.0+cu124，Transformers 5.22.1，PEFT 0.21.2 |
| 加速内核 | causal-conv1d 1.7.0 隔离构建；fla-core 0.3.2；执行函数与哈希匹配登记 |

训练时限在 `train()` 内计时，包含其间筛选；作业计时还包括开头的两个父策略对照评估等工作。二者不能混为纯优化时间。

当前世界：一天 180 tick；饱食每 tick −0.20，水分 −0.08；饮水 +50，进食 +18；采集食物一次 +3；再生 90 tick；移动体力 −0.35；采集耗 2 tick/2 体力；休息耗 3 tick、+14 体力。营地内且动作开始时饱食和水分均≥20，休息另恢复 2 生命。夜间营地外伤害 0.10/tick，近身威胁伤害 1.20/tick。

## 5. state、action、reward 的真实样例

以下是本次 **2026-11-07 21:16–21:18** 在节点 5 重新采集的诊断数据。使用候选第 32 轮，Actor 哈希为 `21f5287ea68726a4f5769b826db29f52a8e2d8664dcbb3e6ad68500c357ccf66`。世界种子 882610…882613，采样种子 20261107；它们不属于该试验的训练/筛选/测试种子集合。没有训练更新、没有脚本代选动作。

### 5.1 一个完整闭环：选择 go，再选择 A

原始文件 `fresh_transitions.jsonl` **第 4 行**：seed=882613，env=3，tick=0。下列为真实 Actor facts 的节选，地标/目标字段明确标为节选；完整状态、完整提示词、97 维 critic 特征、动作前后快照和逐 tick trace 见 `state_example.json`。

```json
{dump(state_summary)}
```

`t7` 是已观察到的 rock 地标；`go` 分支的 `A` 映射到它。模型实际选择：动词索引 1、代码 `go`；目标索引 0、代码 `A`。传给 Godot 的动作是：

```json
{dump(nav['action'])}
```

真实状态变化：位置 `(47,47)→(48,47)`；tick `0→1`；生命 `100→100`，饱食 `82→81.8`，水分 `78→77.92`，体力 `92→91.65`。这次只向地标走一格，未采石。动词提示输入 820 token、目标提示输入 903 token，物理张量各补齐至 1536。联合动作 log probability 为 −1.8019237518310547，Critic 值为 6.299812316894531。

真实 reward 对象的数值部分为：

```json
{dump({k:nav['reward'][k] for k in ['reward','components','discount','elapsed_ticks']})}
```

### 5.2 实际奖励公式与数值复算

令 H/F/W/E 分别为生命、饱食、水分、体力，γ=0.9995，Δt 为动作实际持续 tick。死亡后的所有势能取 0。

```text
Φbase(s) = 0.5×H/100 + 0.25×[min(1,F/60)+min(1,W/60)+min(1,E/60)]
u(s) = max(0, 1−W/60)
d(s) = min(32, 可达且已知水源的最短路程)；未知/不可达按 32
Φwater(s) = 1 − u(s)×d(s)/32
r = Σ(j=0…Δt−1) γ^j×0.01×该 tick 结束仍存活
    + γ^Δt×Φbase(s′)−Φbase(s)
    + γ^Δt×Φwater(s′)−Φwater(s)
```

储备值限制在 0…100；base potential 使用引擎精确 snapshot，water potential 使用 Actor 序列化事实，因此水分等事实可能已四舍五入为一位小数。死亡 tick 不领存活奖励。没有额外“死亡 −5”“探索 +0.02”“指定吃喝奖励”或到达 10 天奖励；那些旧版参数不属于当前路径。

上例动作前后 H=100 且三项储备均≥60，所以 `Φbase=1.25`，水分≥60 所以 `Φwater=1`。因此 `r=0.01+(0.9995×1.25−1.25)+(0.9995×1−1)=0.008875`。

### 5.3 更多真实动作与奖励

下表行号对应 `fresh_transitions.jsonl`，均为一基行号。`real_examples.json` 保留每种动作首个真实样例的完整 state/action/reward 和行号。

| 行号 | 世界 seed | tick | action | 存活项 | 基础势能差 | 水源势能差 | reward |
|---:|---:|---|---|---:|---:|---:|---:|
{chr(10).join(sample_rows)}

例如第 24 行采集使食物 `2→5`、体力 `90.6→88.6`，经过 2 tick；第 42 行休息使体力 `87.55→100`，经过 3 tick；第 73 行饮水使水分 `76.48→99.92`，经过 1 tick。多 tick 奖励更大是累计存活时间，不能据此认定存在独立“休息奖励”。

当前动词与实际执行的映射：

| 模型代码 | Godot verb | 目标 | 常规 tick |
|---|---|---|---:|
| move | move | 相邻合法 tile | 1 |
| go | navigate | 已知地标；仅走一格 | 1 |
| take | gather | 邻近资源 | 2 |
| eat | eat | 无 | 1 |
| drink | drink | 无，岸边才合法 | 1 |
| rest | rest | 无 | 3 |
| camp | build_camp | 无，消耗 4 木 2 石 | 4 |
| wait | wait | 无 | 1 |

死亡可能提前终止多 tick 动作，实际 Δt 以引擎 trace 为准。288 条短探针中没有抽到 camp；报告不伪造它的模型样例。

## 6. 游戏中的算法示意图

![当前强化学习闭环](rl_algorithm.png)

可缩放矢量版为 `rl_algorithm.svg`。图中的毫秒数来自本次短探针，不代表旧运行每一时刻的速度。

## 7. 速度调查：max token 假说与实际瓶颈

### 7.1 “频繁输出到 max token”被当前实现排除

四项证据相互对应：

1. `_tokens()` 把 `max_tokens` 用于检查**输入**长度，超限会抛出异常。
2. `_logits()` 用 `padding="max_length", max_length=1536`，每次都处理固定长输入。
3. 模型直接 forward，`logits_to_keep=1` 后从候选 logits 采样；不存在 `max_new_tokens`、生成循环或 EOS/finish_reason 分支。
4. 探针给 Actor 和参考模型的 `generate()` 加入一旦调用就报错的守卫；288 次真实动作成功完成，调用次数为 0。

因此不存在“1536 token 输出截断比例”。实际是每步 1 个动词选择，加上按需 1 个目标选择，而不是生成 1536 token 的回复。即使系统提示写“只输出代码”，真正保证短决策的是分类式调用路径，不靠提示词自律。

### 7.2 分阶段实测

计时范围：预热 32 决策之后的 256 决策，4 个 Godot 环境，候选 0032，带父参考策略，与训练采样路径一致。每个 GPU 段前后同步计时，包含该段 Python/传输开销；这是阶段墙钟，不是纯 CUDA kernel 时间。

| 阶段 | 调用次数 | 平均每调用 / ms | 总秒数 | 占计时段 |
|---|---:|---:|---:|---:|
{time_rows}

上表未单列的调度、分布采样、诊断记录构造等约占其余 0.50%。`act_total=43.049 秒 / 96.46%` 是包住四类模型前向和 critic 的父计时项，**不能再与它们相加**。日志写入在本探针计时段之后；另做相同 open/append/close 模式测试，288 行约 0.0275 秒、0.096 ms/行（页缓存写入，非 fsync 持久落盘），不足以解释当前秒级模型开销。

### 7.3 瓶颈一：两阶段 × Actor/Reference = 大量完整前向

64 批动作中 63 批至少有一个动作需要多目标选择，因而实际做了 **254 次 B4×1536 模型前向**：Actor 动词 64、Actor 目标 63、参考动词 64、参考目标 63。参考模型两项合计 21.465 秒，占总采样 **48.10%**。

只有 175/256=68.36% 的真实动作需要目标选择，但只要同批有一个需要，就运行完整目标前向。目标阶段 63×4=252 个物理槽中只有 175 个真实目标，77 个是复制补位，**30.56% 的目标物理槽无对应环境决策**。补位仍消耗模型计算，不能计入样本量。

冻结父参考是训练目标的一部分。把它直接删掉会改变算法，不能当成无损加速。未来可评估两个适配器共用基座、前缀复用或更高效的参考计算；需测量实际收益并保持概率、KL 和参数冻结语义。

### 7.4 瓶颈二：每次重复编码并计算固定长输入

| 输入阶段 | 均值 | P95 | 最大 | 物理长度 |
|---|---:|---:|---:|---:|
| 动词 | 926.668 | 955 | 965 | 1536 |
| 需要多目标的目标阶段 | 988.817 | 1028 | 1046 | 1536 |

仅看真实行，动词 padding 约 39.67%，目标 padding 约 35.62%；再加复制行，两者合计实际有效输入 token 约占总物理输入槽位的 52.58%。这只是槽位利用率，**不等于 GPU FLOP 浪费的精确比例**。

`use_cache=False` 使动词和目标阶段都重新计算规则与观察的大部分前缀。训练反向也要重复做全序列计算。目标 prompt 是原 messages 加已选动词与目标问题，需要精确确认 tokenizer/模板前缀及模型缓存语义，才能设计合法缓存；不能直接跨不断变化的状态重用整段 KV。

本次在同一组真实输入上做了隔离 forward 对照，输出未用于游戏动作：

| 物理形状 | 5 次预热后中位数 | 相对 B4×1536 | 未 mask 的候选 log-softmax 最大差 |
|---|---:|---:|---:|
| B4×1536 | 174.954 ms | 基线 | 0 |
| B4×965（该组输入最长 965） | 108.055 ms | 单次快 38.24% | 0.0710373 |
| B1×1536（只算第一行） | 36.748 ms | **工作量不同，不可直接作整批提速比** | 0.1259723 |

每种形状预热 2 次，再取 5 次测量的中位数。数值差针对未 mask 的八个动词候选 log-softmax，不等同于项目的已选合法联合动作 `zero_update_logp` 指标。它至少说明更改物理形状并非已证实概率等价；正式优化仍需合法分布、已选联合 logp、批次组成和 CUDA RNG 的专门回归。

当前固定形状原本就是为了控制 BF16 kernel 随长度/批次变化带来的概率漂移。直接把 1536 改小既会触发注册形状校验，也可能让更长的合法观察溢出。建议在**新协议**中设计统一且可验证的形状策略，或先压缩事实表达，再验证采样与更新同形状；不改变旧冻结配置。

### 7.5 瓶颈三：PPO 的重复反向计算

一个完整无早停更新上限为：2048/32×2=128 个 Actor minibatch；每个 8 个 microbatch，每个 microbatch 至多有动词和目标两次模型前向及对应反传。因此最多 1024 个微批、2048 次 Actor 前向加反传。此外还有更新前数值一致性检查、Critic 更新、权重/参考哈希检查和检查点写盘。

本次使用采集的真实 rows、GAE 和 PPO 目标，做 3 次“32 样本前向+反向+梯度裁剪但不 step”的计时：13.478 秒（首次反向含冷启动）、7.056 秒、7.069 秒。后两次中位数 7.062 秒。若假设完整执行 128 minibatch，粗估 Actor 更新为 **904 秒/轮（约 15.1 分钟）**；按短探针速度粗估 2048 条 rollout 为 **357 秒/轮（约 5.95 分钟）**。这是工作量估算，**不是恢复出的历史单轮耗时**，未覆盖 optimizer.step、检查点和筛选，且 KL 早停可显著缩短实际更新。

所以即使进一步优化采样，PPO 更新也可能成为总时间的大头。LoRA 的 0.63% 可训练参数降低了参数/优化器负担，但反向仍需经过参与适配的语言模型计算图，不意味着只计算 0.63% 的网络。

### 7.6 瓶颈四：筛选本身占用了大量决策预算

已保留四组筛选报告的决策数为 5921、18132、18388、23112，合计 **65,553 次**；相当于实际训练采样 83,968 次的 **78.07%**。评估不算 PPO 新训练样本，不带参考前向，也不反传，但仍需要 Actor 两阶段推理和环境步进。

将总作业 58,466 秒直接除以 40 轮，会混入两个训练前父对照评估和中途两次筛选。`83968/58466≈1.436 决策/秒` 只能叫“总作业时间口径下的训练采样数/秒”，不能当作纯采样性能。当前筛选已有动态补槽，不应再把“没有补槽”列作已知瓶颈。

### 7.7 已排查但没有证据支持的主因

| 怀疑 | 当前证据 |
|---|---|
| 自回归长输出达到 max token | 当前路径没有 generate；排除 |
| GPU 被其他训练抢占 | 探针前后无其他 GPU 计算进程；无法追溯旧 16 h 的所有瞬间 |
| Godot 按真实时间等待 tick | headless 同步逻辑推进；实测每动作 RPC 约 4.01 ms，非主瓶颈 |
| 每一步重新启动 Godot | 每环境持久进程；仅 reset 世界，不重启引擎 |
| 每一步经 HTTP/SSH 远程采样 | 训练模型和 Godot 同节点，通过本地 stdin/stdout 通信；Netcatty 是本次管理通道 |
| 纯 Python 慢卷积 fallback | 当前执行内核身份与冻结配置一致，已接入 causal-conv1d 和 FLA；不能按包名猜测其缺失 |
| JSONL 写盘是主要耗时 | 本次页缓存 append 约 0.096 ms/行；未发现足以解释模型开销的量级 |
| Critic 太大 | 29,185 参数，采样价值计算均值 0.109 ms/批；不是采样主因 |

## 8. 计算速度与学习效率要分开

短探针 288 条全部处于开局高储备区，奖励只有三种：268 条单 tick 为 0.008875，7 条双 tick 为 0.0177455625，13 条三 tick 为 0.02661168971875。吃、喝、移动、等待在三项储备均≥60 且健康相同时，当前即时奖励可以完全相同；有用行为需要通过后续存活和 Critic/GAE 传回差异。这是当前公式和真实开局样本支持的事实，不表示全部历史训练样本都如此。

第 32 轮保留筛选报告中有 5 个死亡世界：4 个最终水分为 0、1 个最终饱食为 0。仅根据终态可以说补给失败值得关注；终态没有足够信息恢复每次路径选择或确定全部死因归属。下一次开发诊断应跟踪缺水/缺食前若干百 tick 的可见目标、动作分布、优势和价值误差。

当前 γ×λ≈0.9985005，对 GAE 递推的名义时间尺度约 667 tick；同时 rollout 每环境最多保留 512 个决策的显式递推，其尾部依赖 Critic 自举。这些与长期补给信用分配相关，但尚无消融证据证明它们是未达标的决定性原因。不能把修改奖励或增加学习率直接包装成性能修复。

## 9. 建议优先级与可验证的下一步

| 顺序 | 建议 | 验证要求 / 预期作用 |
|---|---|---|
| P0 | 后续新实验保留分阶段耗时：prepare、Actor/Reference 的 verb/target、env、reward/log、PPO/critic/save/validation | 保留真实输入长度、物理槽数、优化器实际步数和 KL 早停；不再只记录总墙钟 |
| P1 | 做输入长度/填充优化的独立开发试验 | 本次单 forward 缩短输入已有约 38% 测量收益；须覆盖最长合法观察、零更新概率一致性和新训练数值基线，不能直接复用旧 ratios |
| P1 | 优化两阶段重复前缀和参考计算 | 用确切 token 前缀、Qwen 混合架构缓存与概率对照验证；保持冻结父分布和 KL 定义，不把删除参考当无损优化 |
| P1 | 专门 profile PPO 前向/反向 | 统计实际 minibatch 数、KL 早停位置、backward 内核时间；检查是否可避免未使用的全词表 lm_head 结果和全部层 hidden_states 留存 |
| P2 | 对 batch/目标调度做数值安全的微基准 | 先测有效样本/秒与显存，再决定固定形状方案；不能只凭显存空闲增加 batch。本文约 3.97 GiB 是采样段峰值，不是训练反向峰值 |
| P2 | 保留持久 Godot；按证据再考虑异步环境与缓冲日志 | 当前两项只占小部分，优先收益明显低于模型计算 |
| P2 | 将训练、筛选、最终测试预算分开显示 | 正式协议的筛选不可中途偷偷删减；新实验可预先规划开发/筛选开销 |
| 独立课题 | 排查长期补给与奖励饱和区的信用分配 | 使用新的开发种子做可归因消融，分别报告规则变化收益和 RL 收益 |

本次仅进行了调查、短探针和报告生成；未修改冻结训练源码、世界规则或候选权重。后续若开始新训练，需要新名称、新登记与输出目录，不能把本次探针或新训练拼接成旧 `exploration_attempt1` 的后续完成轮次。

## 10. 证据清单与复核方法

| 文件 | 用途 |
|---|---|
| `report.html` / `report.md` | 本报告的离线阅读版 / 源文本 |
| `rl_algorithm.svg` / `rl_algorithm.png` | 可缩放示意图 / 图片预览 |
| `registration.json` | 远端原试验注册配置和 72 个冻结源文件哈希 |
| `validation_0032.json` / `balanced_parent.json` | 保留的候选/同规则父模型筛选报告 |
| `remote_snapshot.txt` / `remote_details.txt` | 实际远端进程、GPU、权重 metadata 与运行环境 |
| `profile_probe.py` / `profile.json` | 本次探针源码、计时、物理前向记录、权重前后哈希 |
| `fresh_transitions.jsonl` | 288 条新采集的完整真实转移，约 4.69 MB |
| `state_example.json` / `real_examples.json` | 一个完整闭环和七类真实动作例子 |
| `postcheck.json` | 远端 72 源文件一致性、日志微基准和结束后 GPU 状态 |
| `verify_evidence.py` / `verification.json` | 本地真实 Godot 重演 288 条、输入动作和奖励复算验证 |

本地与远端当前 72 个核心文件均匹配旧登记 SHA256。探针 Actor 与参考权重前后相同，optimizer_steps=0。复核脚本成功退出，288/288 条真实 Godot 重演一致；这不替代已经删除的历史轨迹或完整 CUDA RNG 生存验收审计。

诊断命令以 `profile_probe.py --output <新的目录>` 运行，需使用登记 Python 和 PYTHONPATH 中的卷积 overlay。脚本拒绝复用输出目录，且 GPU 有计算工作时会退出。正式训练入口并未调用。
'''
# Correct the table wording to state the exact warm-up procedure in one place.
report=report.replace('5 次预热后中位数','预热后 5 次测量中位数')
(HERE/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({'report_characters':len(report),'diagram_dimensions':[W,H],'examples':len(chosen)},ensure_ascii=False))
