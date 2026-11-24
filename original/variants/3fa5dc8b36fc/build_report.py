"""Build a self-contained Chinese report from frozen registration and checked evidence."""
from pathlib import Path
import hashlib
import html
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from agent.m9_contract import encode, potential
from agent.m9_metrics import summarize, select_arm

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def esc(value):
    return html.escape(str(value), quote=True)

REG_PATH = "output/m9_repair_20261112/evidence/registration.json"
SNAP_PATH = "output/m9_followup_20261113_1052/evidence/snapshot.json"
reg, snap = read(REG_PATH), read(SNAP_PATH)
config = reg["config"]
assert sha(SNAP_PATH) == "c87d3d5cd8829243b3771e221a2c997981f6b589c0dcdc8a0da9c5abbf2bf0b6"
assert config == read("configs/m9_research_resume5.json")
assert snap["registration_sha256"] == reg["registration_sha256"]
SOURCE_PATHS = [
    "configs/m9_research_resume5.json", "agent/m9_contract.py", "agent/m9_policy.py",
    "agent/m9_run.py", "agent/m9_metrics.py", "agent/m9_context.py",
    "agent/m8s3_contract.py", "agent/m8s3_supply.py", "agent/m8b_water_potential.py",
    "scripts/m9_world.gd", "scripts/survival_world.gd", "tools/m9_campaign.py",
    "tools/m9_resume.py", "tools/m8b_metrics.py", "tests/fixtures/m9_token_overflow.json",
]
verified = {}
for path in SOURCE_PATHS:
    digest = sha(path)
    assert reg["source_hashes"][path] == digest, f"Frozen source mismatch: {path}"
    verified[path] = digest

META = {
    "control": dict(title="原规则控制组", category="基准", short="控制组", question="在相同起点上继续短程训练，本身能带来多大改善？",
        mechanism="保留96×96有限地图、原中文观察、每次导航1格、supply补给奖励，以及每刻0.9995折扣。它是其他单因素组的共同参照。", expected="为所有干预提供基线，区分‘改变规则的作用’与‘继续训练的作用’。",
        reading="8轮后30天存活由4/16降至3/16，RMST也下降。当前短程训练并未稳定改善原方案，不能把继续PPO训练本身视为必然增益。", caveat="这是两次各8轮的早期筛选。下降可能与采样和优化波动有关；仅凭这些终点不能诊断训练崩溃。"),
    "state_reserves": dict(title="显式补给余量", category="State", short="补给余量", question="把饥渴算术直接写出来，是否更容易学会提前补给？",
        mechanism="在原观察前增加：不补充时饱食耗尽约ceil(饱食/0.2)刻、水分耗尽约ceil(水分/0.08)刻，以及背包食物可补18×库存的饱食。地图、动作、奖励和折扣保持控制组设置。", expected="减少模型自行换算剩余补给时间的负担，促使其及早处理紧迫需求。",
        reading="单独添加余量信息后，30天存活3/16→1/16，10天仅8/16。当前证据不支持把这项表示变化独立视为有效改进。", caveat="耗尽刻数不是死亡倒计时，库存营养也不是已经吃进身体的储备。它没有给出推荐动作；组合组较好不能反证这一项单独有效。"),
    "state_compact": dict(title="紧凑JSON观察", category="State", short="JSON状态", question="相同事实换成结构化JSON，会更容易理解或更节省计算吗？",
        mechanism="用无缩进、无额外分隔空格的JSON替换观察正文，保留身体、背包、时间、地形、地标、营地、威胁、合法动作、踏入次数和最近动作。规则提示仍保留。", expected="检验结构化字段与原中文叙述的区别，而不是添加信息或更换模型。",
        reading="30天存活2/16→4/16，RMST15.31→16.86天，但仍低于多个候选。实际观察并不一定短：本报告样例中正文字符数明显多于原中文。", caveat="研究4在本组r1的训练后评估遇到1541-token观察，超过旧1536上限；研究5扩展容量后复用已训练权重补完评估。‘compact’是配置名称，不是更少token或更快的保证。"),
    "action_macro4": dict(title="最多四格目标导航", category="Action", short="四格导航", question="减少走向同一地标时的重复决策，能否减少中途偏离目标？",
        mechanism="仅把navigate改为最多连续避障走4格。模型仍先选择一个合法地标；到达、动作不再合法或死亡时停止。move仍是1格，其他动作的时长和效果不变。", expected="让一次导航意图保持得更久，提高到达资源点的机会，降低每一小步重新决策造成的漂移。",
        reading="30天存活7/16→9/16，RMST21.92→24.12天，筛选排名第3。它在有限图上已有较好的筛选表现。", caveat="不会自动采集、吃饭或喝水。最多4格不代表速度变快：每格仍消耗1刻，饥渴、威胁和奖励逐刻结算；同等决策数会产生不同世界tick。"),
    "reward_bottleneck": dict(title="食水短板奖励", category="Reward", short="短板奖励", question="把食物和水中较弱的一侧放大，能否减少‘一项充足、另一项耗尽’？",
        mechanism="存活每刻0.01，加包含min(N,W)的势函数差分；N由饱食与最多6份库存食物计算，W由水分计算。当前可见可用资源的距离也进入势函数。去掉supply的低储备持续代价，改用新的生命与储备权重。", expected="提供更强调食水均衡的训练信号，并在低储备时体现接近补给的价值。",
        reading="30天存活4/16→5/16，但RMST18.85→18.51天，两个指标并未同步提高。短板奖励单独使用尚未表现出明显优势。", caveat="这不是单纯给旧reward加一个min项：水距离由旧路径长度改为曼哈顿距离，尺度、权重和风险项也一起变化。它是一个完整奖励方案对比，不能单独归因于min(N,W)。"),
    "reward_survival": dict(title="纯存活奖励", category="Reward", short="纯存活", question="移除补给塑形后，仅靠活着的回报能否学得更好？",
        mechanism="每个实际存活tick给0.01，死亡tick为0，势函数为0，没有采集、探索、建造、移动或吃喝的单独奖励。多刻动作仍逐刻折扣累加。", expected="提供最直接的存活目标对照，检查复杂奖励是否可能引入短期偏好或额外学习难度。",
        reading="30天存活4/16→7/16，10天16/16，RMST18.85→22.37天。它是本轮两个单独reward干预中表现更好的一个，整体排名第4。", caveat="没有固定死亡惩罚并不表示死亡无损失：死亡终止后续存活回报。结果只是短程筛选，不足以证明纯存活奖励对所有训练预算都最好。"),
    "long_credit": dict(title="延长回报折扣跨度", category="Discount", short="长折扣", question="让更远的未来回报保留更大权重，能否改善长期补给规划？",
        mechanism="仅将gamma_tick从0.9995提高到0.99985；GAE lambda仍是0.999。实际动作持续d刻时使用gamma^d，奖励势函数的差分也使用新的gamma。", expected="降低远期生存回报被折扣的速度，为延迟后果保留更多训练权重。",
        reading="30天存活4/16→4/16，RMST18.85→18.90天，变化很小。单独改变折扣在当前8轮预算下未显示明确收益。", caveat="折扣半衰期变长并不等于模型已经能规划那么远；优势估计、128步rollout、critic质量及优化噪声仍会影响学习。"),
    "infinite_freeze": dict(title="扩展地图＋离区冻结", category="Map", short="分区冻结", question="跨越原地图边界，并保留离开区域的资源状态，能否延长生存？",
        mechanism="启用64×64区块生成，玩家所在区块及周围8块为活跃区。活跃区域资源时钟推进；其余区块暂停。初次访问生成，返回使用保存的原状态。其余训练设置与控制组相同。", expected="减少受原96×96边界限制的资源可达性问题，并检验资源不能在远处自动再生的玩法。",
        reading="30天存活2/16→10/16，RMST17.80→24.65天，排名第2，入围64轮确认。但10天为14/16，仍有较短生存失败局。", caveat="原型同时引入新地形、资源分布与更低的扩展区敌人密度。因此相对有限图的收益不能仅归因于消除边界；冻结机制本身应与infinite_global比较。"),
    "infinite_global": dict(title="扩展地图＋全局再生", category="Map", short="全局再生", question="同样的扩展地图中，让远处资源继续再生，会比冻结更好吗？",
        mechanism="使用与冻结组相同的生成器、区块范围和逐格导航，但资源可用时间以全局tick计。离开后仍经过再生时间，返回时按全局时钟判断是否恢复。", expected="作为地图配对对照，隔离‘资源局部冻结’与‘资源全局再生’这套时钟规则的影响。",
        reading="30天存活7/16→6/16，RMST23.12→21.62天；10天16/16。当前30天筛选落后冻结组，但10天指标反而更好。", caveat="资源理论上更容易恢复，不代表模型就会更好地使用它。策略可能很少返回旧资源，或在训练中形成不同轨迹；需要轨迹分析才能解释差异，不能只由成绩推断原因。"),
    "combined": dict(title="五项组合方案", category="Combined", short="五项组合", question="将候选机制组合后，能否相互配合，形成更可靠的长期生存策略？",
        mechanism="同时启用补给余量state、最多4格导航、bottleneck奖励、gamma=0.99985和infinite_freeze地图；其余PPO和共同初始化不变。", expected="测试组合效果：观察突出需求，导航保持目标，奖励强调食水短板，折扣看得更远，地图允许继续探索。上述配合作用是设计假设。",
        reading="30天存活2/16→11/16，10天16/16，RMST18.66→26.37天，筛选排名第1，已入围64轮确认。它是这轮筛选最值得继续投入的组合。", caveat="这不是完整因子实验；只有一个组合组，没有逐项删除的组合消融。不能据此确认每一项都有效，也不能宣称已找到全局最佳state/action/reward。"),
}

rows, rank_rows = [], []
for arm, override in config["arms"].items():
    runs = sorted([r for r in snap["runs"] if r["phase"] == "explore" and r["name"].rsplit("_r", 1)[0] == arm], key=lambda x:x["name"])
    assert len(runs) == 2 and all(r["updates"] == 8 and r["complete"] for r in runs)
    summaries = {}
    for phase in ["initial", "evaluation"]:
        episodes = []
        for run in runs:
            report = run["evaluations"][phase]["report"]
            summary = summarize(report["episodes"])
            assert summary == report["summary"]
            episodes.extend(report["episodes"])
            if phase == "evaluation":
                rank_rows.append(dict(arm=arm, replicate=int(run["name"][-1]), summary=summary))
        summaries[phase] = summarize(episodes)
    timings = [m["rollout_seconds"] + m["update_seconds"] for r in runs for m in r["metrics"]]
    rows.append(dict(id=arm, **META[arm], config={**config["common"], **override}, override=override,
        initial=summaries["initial"], after=summaries["evaluation"],
        replicas=[dict(name=r["name"], summary=r["evaluations"]["evaluation"]["report"]["summary"]) for r in runs],
        ticks=sum(r["ticks"] for r in runs), decisions=sum(r["decisions"] for r in runs),
        average_update_seconds=sum(timings)/len(timings)))
ranking = select_arm(rank_rows, list(config["arms"]))
assert ranking == snap["ranking"]["ranking"]
for row in rows:
    row["rank"] = ranking.index(row["id"]) + 1
ranked = sorted(rows, key=lambda x:x["rank"])
fixture = read("tests/fixtures/m9_token_overflow.json")
samples = {variant:encode(fixture["observation"], variant)["messages"][1]["content"] for variant in ("legacy", "reserves", "compact")}
data = dict(as_of=snap["checked_utc"], as_of_cst="2026-11-13 10:52:03", arms=rows, ranking=ranking,
    common=config["common"], registration_sha256=reg["registration_sha256"], snapshot_sha256=sha(SNAP_PATH),
    sources=verified, samples=samples)
(OUT / "report_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

STATE_LABEL = {"legacy":"原中文", "reserves":"中文＋余量", "compact":"紧凑JSON"}
MAP_LABEL = {"finite":"96×96有限", "infinite_freeze":"扩展／冻结", "infinite_global":"扩展／全局"}
REWARD_LABEL = {"supply":"补给supply", "bottleneck":"短板", "survival":"纯存活"}
def source_link(path, label=None):
    return f'<a href="../../{esc(path)}">{esc(label or path)}</a>'

matrix = []
cards = []
for i, row in enumerate(rows, 1):
    c, a, b = row["config"], row["initial"], row["after"]
    matrix.append(f'''<tr data-category="{row['category']}" data-arm="{row['id']}"><th scope="row"><a href="#arm-{row['id']}"><span class="num">{i:02}</span>{row['title']}</a><small>{row['id']}</small></th><td>{STATE_LABEL[c['state_variant']]}</td><td>{c['navigation_steps']}格</td><td>{REWARD_LABEL[c['reward_variant']]}</td><td>{c['gamma_tick']}</td><td>{MAP_LABEL[c['map_mode']]}</td><td><strong>{b['successes']}/16</strong></td><td>{b['rmst_days']:.2f}天</td></tr>''')
    diff = b['rmst_days'] - a['rmst_days']
    badge = '<span class="pill selected">入围确认</span>' if row['id'] in ranking[:2] else '<span class="pill">探索完成</span>'
    rep_text = ''.join(f"<tr><td>{esc(x['name'])}</td><td>{x['summary']['successes']}/8</td><td>{x['summary']['successes_10day']}/8</td><td>{x['summary']['rmst_days']:.2f}天</td><td>{x['summary']['cvar20']/180:.2f}天</td></tr>" for x in row['replicas'])
    cards.append(f'''<article class="arm-card" id="arm-{row['id']}"><div class="arm-heading"><span class="arm-index">{i:02}</span><div><div class="eyebrow">{row['category']} / {row['id']}</div><h3>{row['title']}</h3></div>{badge}</div><p class="question">{row['question']}</p><div class="arm-grid"><div><h4>实际改了什么</h4><p>{row['mechanism']}</p><h4>想验证的假设</h4><p>{row['expected']}</p></div><div><h4>已有结果如何理解</h4><p>{row['reading']}</p><h4>解释边界</h4><p>{row['caveat']}</p></div></div><div class="arm-stats"><div><small>30天存活 · 训练前 → 后</small><b>{a['successes']}/16 → {b['successes']}/16</b></div><div><small>平均存活RMST · 前 → 后</small><b>{a['rmst_days']:.2f} → {b['rmst_days']:.2f}天</b></div><div><small>RMST变化</small><b class="{'positive' if diff>=0 else 'negative'}">{diff:+.2f}天</b></div><div><small>冻结规则下的筛选排名</small><b>{row['rank']} / 10</b></div></div><details><summary>查看精确配置与两次独立训练结果</summary><div class="detail-grid"><div><h4>相对common的覆盖项</h4><pre>{esc(json.dumps(row['override'],ensure_ascii=False,indent=2))}</pre><p class="fine">空对象表示完全使用控制配置。每次训练8轮、16,384次决策。表中CVaR20是最短20%生存时长的加权均值。</p></div><div class="table-wrap"><table><thead><tr><th>训练重复</th><th>30天</th><th>10天</th><th>RMST</th><th>CVaR20</th></tr></thead><tbody>{rep_text}</tbody></table></div></div></details></article>''')

results_rows = ''.join(f'''<tr><td>{r['rank']:02}</td><th scope="row"><a href="#arm-{r['id']}">{r['title']}</a></th><td>{r['initial']['successes']}/16 → <strong>{r['after']['successes']}/16</strong></td><td>{r['initial']['rmst_days']:.2f} → <strong>{r['after']['rmst_days']:.2f}</strong></td><td>{r['after']['successes_10day']}/16</td><td>{min(x['summary']['cvar20'] for x in r['replicas'])/180:.2f}</td></tr>''' for r in ranked)
efficiency_rows = ''.join(f'''<tr><th scope="row">{r['title']}</th><td>{r['decisions']:,}</td><td>{r['ticks']:,}</td><td>{r['ticks']/r['decisions']:.3f}</td><td>{r['average_update_seconds']/60:.2f}分钟</td></tr>''' for r in rows)
sample_panels = ''.join(f'''<details class="sample" {'open' if key=='legacy' else ''}><summary>{STATE_LABEL[key]} · 观察正文{len(samples[key]):,}字符</summary><pre>{esc(samples[key])}</pre></details>''' for key in samples)
source_rows = ''.join(f'<tr><td>{source_link(path)}</td><td class="hash">{digest}</td></tr>' for path, digest in verified.items())
discount_rows = ''.join(f'<tr><th>{g}</th><td>{g**1800:.4f}</td><td>{g**5400:.4f}</td><td>{math.log(.5)/math.log(g):.0f}刻 / {math.log(.5)/math.log(g)/180:.2f}天</td></tr>' for g in (.9995,.99985))

# A transparent hypothetical drinking example, calculated by the actual reward potential.
from copy import deepcopy
before = deepcopy(fixture['observation'])
before['player'].update(x=0,y=0,health=100,hunger=40,thirst=20)
before['inventory']['berries']=2
before['nearby']=[dict(kind='berry',x=5,y=0,available=True),dict(kind='water',x=0,y=0,available=True)]
after=deepcopy(before);after['player'].update(hunger=39.8,thirst=69.92)
phi_before=potential(before,'bottleneck');phi_after=potential(after,'bottleneck')
reward_example=.01+.9995*phi_after-phi_before

CSS = r'''
h1{text-wrap:balance}.title-tail{display:inline-block}
:root{--paper:#f4f3ed;--card:#fffefa;--ink:#172c2b;--muted:#566661;--line:#d9dfd7;--green:#1b6a57;--mint:#e3eee4;--coral:#b35132;--gold:#e5bc66;--font:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font);font-size:16px;line-height:1.8}a{color:var(--green);text-decoration-thickness:1px;text-underline-offset:4px}a:hover{color:var(--coral)}button,select,input{font:inherit}button,select{border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);padding:8px 14px}button{cursor:pointer}button:hover{border-color:var(--green)}button:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--gold);outline-offset:3px}button.active{background:var(--ink);color:white;border-color:var(--ink)}p{margin:0 0 16px}h1,h2,h3,h4{line-height:1.35;letter-spacing:-.025em}h1{font-size:clamp(38px,4.7vw,66px);margin:18px 0 24px;font-weight:750}h2{font-size:30px;margin:0 0 20px}h3{font-size:25px;margin:3px 0 12px}h4{font-size:15px;margin:18px 0 8px;color:var(--green);letter-spacing:0}small,.fine{font-size:13px;color:var(--muted)}code,pre,.hash{font-family:Consolas,"Cascadia Mono",monospace}code{font-size:.88em;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1eb;border:1px solid var(--line);border-radius:6px;padding:18px;line-height:1.65;font-size:13px;margin:12px 0}strong{font-weight:700}ul,ol{padding-left:24px}li{margin:8px 0}svg text{font-family:var(--font)}.skip{position:fixed;left:10px;top:-100px;background:white;padding:10px;z-index:9}.skip:focus{top:10px}
.shell{max-width:1540px;margin:auto;display:grid;grid-template-columns:225px minmax(0,1fr)}aside{padding:36px 24px;border-right:1px solid var(--line);position:sticky;top:0;align-self:start;height:100vh;overflow:auto}.brand{font-weight:800;font-size:17px;letter-spacing:.08em}.brand span{display:block;font-weight:400;font-size:11px;letter-spacing:.15em;color:var(--muted)}nav{margin:34px 0}nav a{display:block;color:var(--muted);font-size:13px;text-decoration:none;border-left:2px solid transparent;padding:9px 0 9px 12px}nav a:hover{border-left-color:var(--green);color:var(--green)}aside .fine{line-height:1.65}main{padding:45px 56px 64px;min-width:0}.topline{display:flex;justify-content:space-between;align-items:center;gap:20px;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.09em;color:var(--green)}.hero{padding-bottom:42px;border-bottom:1px solid var(--ink)}.hero .lead{font-size:19px;max-width:830px;line-height:1.9}.hero .sub{color:var(--muted);max-width:850px}.hero strong{color:var(--green)}.status{border-left:4px solid var(--green);padding:14px 20px;background:var(--mint);margin-top:26px}.status p{margin:0}.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:28px 0 0}.metric-strip>div{border-top:1px solid var(--line);padding-top:18px}.metric-strip b{display:block;font-size:34px;line-height:1.4}.metric-strip small{display:block}.metric-strip .unit{font-size:16px}.section{padding:48px 0 14px;border-bottom:1px solid var(--line)}.section-head{display:flex;gap:18px;align-items:baseline}.section-no{font-family:Consolas,monospace;font-size:13px;color:var(--coral)}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px}.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:24px;margin:16px 0}.panel h3{font-size:20px}.note{border-left:3px solid var(--gold);background:#fbf1d9;padding:16px 20px;margin:20px 0;font-size:14px}.note p:last-child{margin:0}.pill{display:inline-block;font-size:11px;border:1px solid var(--line);padding:3px 9px;border-radius:30px;color:var(--muted);white-space:nowrap}.selected{background:var(--mint);color:var(--green);border-color:#c4d9cc}.table-wrap{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65;background:var(--card)}th,td{text-align:left;padding:12px 12px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:#e9eee7;font-weight:600;white-space:nowrap}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}tbody th{font-weight:500}tbody tr:hover{background:#f0f4ed}th small{display:block;font-size:10px;font-family:Consolas,monospace;font-weight:400}.num{color:var(--coral);font-family:Consolas,monospace;margin-right:8px;font-size:11px}.matrix{min-width:860px}.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:20px 0 8px}.toolbar button{font-size:12px;padding:5px 11px}.count{margin-left:auto;font-size:12px;color:var(--muted)}.arm-card{padding:30px 30px 22px;background:var(--card);border:1px solid var(--line);border-radius:8px;margin:20px 0;scroll-margin-top:25px}.arm-heading{display:flex;gap:18px;align-items:center}.arm-heading .pill{margin-left:auto}.arm-index{font-family:Consolas,monospace;color:#73958b;font-size:36px;line-height:1}.question{font-size:18px;line-height:1.7;border-bottom:1px solid var(--line);padding:8px 0 17px;margin-bottom:6px}.arm-grid{display:grid;grid-template-columns:1fr 1fr;gap:30px}.arm-grid p{font-size:14px;line-height:1.9}.arm-stats{display:grid;grid-template-columns:1.1fr 1.3fr .8fr .8fr;gap:15px;background:#f0f3ed;padding:18px;margin:16px 0}.arm-stats small{font-size:11px;display:block}.arm-stats b{font-size:17px;display:block;margin-top:5px}.positive{color:var(--green)}.negative{color:var(--coral)}details{border-top:1px solid var(--line);padding:12px 0 4px}summary{cursor:pointer;font-size:13px;font-weight:600}details[open] summary{margin-bottom:16px}.detail-grid{display:grid;grid-template-columns:1fr 1.3fr;gap:22px}.detail-grid .table-wrap{align-self:start}.sample pre{font-size:12px;max-height:460px;overflow:auto}.fact-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.fact-grid div{background:#edf1eb;padding:15px 18px;border-radius:5px;font-size:14px}.fact-grid b{display:block}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.flow div{background:var(--ink);color:#f7f5ec;border-radius:6px;padding:20px;font-size:13px}.flow strong{display:block;color:#cde3d0;font-size:16px}.formula{background:var(--ink);color:#e6f1df;padding:22px;border-radius:6px;font-family:Consolas,"Microsoft YaHei",monospace;font-size:14px;line-height:1.9;white-space:pre-wrap;overflow-wrap:anywhere;margin:18px 0}.formula .soft{color:#aebeb5}.legend{display:flex;gap:20px;font-size:12px;color:var(--muted);margin:10px 0}.dot{display:inline-block;width:9px;height:9px;background:var(--green);border-radius:50%;margin-right:6px}.dot.before{background:#c7d1c6}.chart{background:var(--card);padding:20px 10px;border:1px solid var(--line);border-radius:6px}.chart svg{width:100%;height:auto;display:block}.chart-top{display:flex;align-items:center;justify-content:space-between;gap:20px}.chart-top label{font-size:13px}.chart-top select{margin-left:6px;font-size:13px}.map-demo{display:grid;grid-template-columns:240px 1fr;gap:26px;align-items:center}.map-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;max-width:240px}.chunk{aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#e7e7de;border:1px solid #d4d8ce;border-radius:4px;font-size:11px;color:#687369}.chunk.active{background:#cde1d0;color:#17523e;border-color:#8eb49a}.chunk.center{background:var(--green);color:white}.chunk.resource{border:2px solid var(--coral);color:var(--coral);font-weight:700}.clock-display{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.clock-display div{padding:12px;background:#edf1eb;border-radius:4px}.clock-display b{display:block;font-size:22px}.clock-display small{display:block;font-size:11px}.timeline{margin:24px 0;border-left:2px solid #b8cebd;padding-left:26px}.timeline .step{position:relative;padding:0 0 22px}.timeline .step:before{content:'';position:absolute;left:-33px;top:7px;width:12px;height:12px;background:var(--green);border:2px solid var(--paper);border-radius:50%}.timeline h3{font-size:20px}.timeline p{font-size:14px}.gate{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.gate>div{border:1px solid var(--line);background:var(--card);padding:20px;border-radius:6px}.gate b{font-size:27px;display:block;color:var(--green)}.gate p{font-size:13px;margin:5px 0 0}.hash{font-size:11px;word-break:break-all}.references table{font-size:12px}.references td:first-child{min-width:260px}.footer{padding-top:30px;color:var(--muted);font-size:12px}.hidden,[hidden]{display:none!important}.nojs{background:#fbf1d9;padding:16px}.source-note{font-size:12px;color:var(--muted);margin-top:14px}.print-button{font-size:12px;padding:5px 10px;white-space:nowrap}.mobile-nav{display:none}.scroll-hint{display:none}.nowrap{white-space:nowrap}
@media(min-width:1400px){main{padding-left:65px;padding-right:65px}}@media(max-width:1190px){.shell{grid-template-columns:180px minmax(0,1fr)}aside{padding:30px 16px}main{padding:35px 30px}.arm-card{padding:25px 22px}.arm-stats{grid-template-columns:1fr 1fr}.map-demo{grid-template-columns:200px 1fr}.flow{grid-template-columns:1fr 1fr}}
@media(max-width:850px){.shell{display:block}aside{display:none}main{padding:24px 22px}.mobile-nav{display:block;margin:18px 0;padding:10px 0;border-bottom:1px solid var(--line);font-size:13px}.mobile-nav a{margin-right:18px;white-space:nowrap}.topline{font-size:10px}.hero .lead{font-size:17px}.two-col,.arm-grid,.detail-grid{grid-template-columns:1fr;gap:4px}.metric-strip{grid-template-columns:1fr 1fr}.metric-strip b{font-size:30px}.section{padding-top:36px}h2{font-size:26px}.arm-card{padding:22px 18px}.arm-heading{gap:12px}.arm-index{font-size:28px}.arm-heading h3{font-size:21px}.arm-heading .pill{font-size:10px}.question{font-size:16px}.arm-stats{padding:14px;gap:14px}.arm-stats b{font-size:16px}.map-demo{grid-template-columns:1fr}.map-grid{margin:auto}.gate{grid-template-columns:1fr}.chart-top{align-items:flex-start;flex-direction:column;gap:3px}.chart-top select{margin-left:0;margin-top:8px;max-width:100%}.chart{padding:10px 0;overflow:auto}.chart svg{min-width:620px}.scroll-hint{display:block}.fact-grid{grid-template-columns:1fr}.clock-display b{font-size:19px}.count{width:100%;margin-left:0}.formula{font-size:12px;padding:16px}.flow div{padding:16px}.sample pre{max-height:360px}.topline span{max-width:220px}.status{padding:13px 16px}.references td:first-child{min-width:190px}}
@media print{@page{size:A4;margin:15mm}body{background:white;font-size:10pt;color:#182a26}.shell{display:block}aside,.toolbar,.print-button,.mobile-nav,.chart-top select,.chart-top label,.map-controls,.scroll-hint{display:none!important}main{padding:0}.hero{padding-bottom:18px}h1{font-size:30pt}h2{font-size:20pt}h3{font-size:16pt}.section{padding:24px 0 8px}.arm-card{break-inside:avoid;padding:18px;margin:15px 0}.table-wrap{overflow:visible}.matrix{min-width:0}table{font-size:8pt}th,td{padding:7px}.arm-grid,.two-col{grid-template-columns:1fr 1fr}.arm-stats{grid-template-columns:1fr 1fr}.chart svg{min-width:0}.chart{overflow:visible}.sample pre{max-height:none;overflow:visible}.hash{font-size:7pt}.hero .lead{font-size:13pt}.formula{background:#edf1eb;color:#182a26;border:1px solid #bbb}.formula .soft{color:#53645a}.flow div{background:#edf1eb;color:#182a26;border:1px solid #bbb}.flow strong{color:#164f3e}.metric-strip b{font-size:25pt}.panel{padding:15px}a{color:inherit}.source-note{font-size:8pt}.references{break-before:page}.gate{grid-template-columns:1fr 1fr 1fr}.hidden,tbody tr[hidden]{display:table-row!important}}
'''

HTML = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>research5｜10组生存训练方案详解</title><meta name="description" content="research5十组强化学习实验的冻结配置、状态、动作、奖励、分区地图与已验证探索结果。"><style>__CSS__</style></head><body>
<a class="skip" href="#main">跳转到报告正文</a><div class="shell"><aside><div class="brand">MEADOWMIND<span>RESEARCH NOTE / 05</span></div><nav aria-label="报告目录"><a href="#overview">01　先看整体设计</a><a href="#matrix">02　十组对照矩阵</a><a href="#arms">03　逐组方案详解</a><a href="#state">04　State：看见什么</a><a href="#action">05　Action：如何执行</a><a href="#reward">06　Reward：怎样评分</a><a href="#map">07　地图：生成与冻结</a><a href="#results">08　已有结果与局限</a><a href="#milestone">09　训练阶段与验收</a><a href="#sources">10　证据与代码索引</a></nav><p class="fine">依据冻结登记与实际代码编写。<br>成绩快照：<br><strong>2026-11-13 10:52</strong><br>北京时间</p><button class="print-button" id="print-side">打印 / 保存为PDF</button><p class="fine" style="margin-top:22px">离线单文件报告<br>所有图表和数据已内嵌</p></aside>
<main id="main"><header class="hero"><div class="topline"><span>强化学习研究 / 第二个长期生存里程碑 M9</span><button class="print-button" id="print-top">打印报告</button></div><h1>research5<br>10组生存训练方案详解</h1><p class="lead">同一模型、同一起点，逐项比较<strong>观察表达、目标导航、奖励规则、回报折扣与地图机制</strong>，寻找更能活过30个游戏日的策略。</p><p class="sub">这10组共用PPO/LoRA训练框架。research5延续research4的预登记对照实验，修复长观察导致的token超限，复用已完成训练；并非另起10种强化学习算法。</p><div class="metric-strip"><div><small>对照方案</small><b>10<span class="unit"> 组</span></b></div><div><small>每组独立探索训练</small><b>2 × 8<span class="unit"> 轮</span></b></div><div><small>筛选后进入确认</small><b>2<span class="unit"> 组</span></b></div><div><small>目标观察窗</small><b>30<span class="unit"> 游戏日</span></b></div></div><div class="status"><p><strong>这份报告的结果基于2026-11-13 10:52北京时间快照。</strong>20项探索训练已完成，combined与infinite_freeze入围；当时combined确认训练27/64轮。最终100局尚未开始，不能认定M9已通过。本报告未重新查询节点的实时状态。</p></div></header>
<div class="mobile-nav"><a href="#matrix">对照矩阵</a><a href="#arms">逐组详解</a><a href="#reward">奖励公式</a><a href="#results">结果</a></div><noscript><p class="nojs">JavaScript未启用：正文、完整数据表和逐组说明仍可阅读；交互图表与筛选不可用。</p></noscript>
<section id="overview" class="section"><div class="section-head"><span class="section-no">01</span><h2>整体设计：只改明确登记的变量</h2></div><p>共同基础是Qwen3.5-0.8B与supply3第48轮研究checkpoint。所有组加载同一actor和108维critic，重新创建优化器、训练世界和随机数状态。这个共同起点是尚未发布的研究权重：上一阶段supply3最终10天验收为88/100，仍未达标。</p><div class="two-col"><div class="panel"><h3>为什么关注补给与边界？</h3><p>上一阶段100局中，失败局包括8局缺水、4局食物耗尽。失败局最后最多200次决策的记录中，约95%处于距边界4格内；低水分时也经常看不到水源。</p><p class="fine">这提示需要比较补给可见性、目标保持与地图扩展，但位置相关性本身不能证明边界是唯一死因。</p></div><div class="panel"><h3>三个需要分清的层次</h3><p><strong>游戏规则：</strong>有限图、扩展图以及资源再生时钟。<br><strong>策略接口：</strong>传给模型的state，及一次action能执行几格。<br><strong>学习信号：</strong>reward和每刻回报折扣gamma。</p><p class="fine">改变state、action或地图后，即使权重完全相同，训练前行为和成绩也可能不同。</p></div></div><div class="fact-grid"><div><b>16个持续世界</b>每轮每世界128次决策，共2,048次；自然死亡后重置，10天或30天不强制重置训练世界。</div><div><b>同等模型决策预算</b>探索每次8轮＝16,384决策；每组两次共32,768决策。世界tick和墙钟时间并不相等。</div><div><b>相同策略框架</b>先选合法动作类型，再选语义目标；温度1，合法动作类型混合10%均匀探索。</div><div><b>持续估值</b>rollout到128步仍存活时，用critic继续估值；只有自然死亡才把后续价值置零。</div></div><details><summary>共同PPO与执行参数</summary><div class="table-wrap"><table><thead><tr><th>项目</th><th>冻结设置</th><th>作用</th></tr></thead><tbody><tr><td>Actor / Critic学习率</td><td>2×10⁻⁶ / 1×10⁻⁴</td><td>分别更新策略与价值函数</td></tr><tr><td>Actor训练</td><td>2遍，minibatch128，microbatch4</td><td>每轮轨迹的PPO优化预算</td></tr><tr><td>Critic训练</td><td>16遍，minibatch256</td><td>学习存活回报估值</td></tr><tr><td>PPO clip / target KL</td><td>0.1 / 0.015</td><td>约束更新幅度与提前停止</td></tr><tr><td>Reference KL / entropy</td><td>0.02 / 0.002</td><td>保持与共同初始策略的联系，保留探索</td></tr><tr><td>梯度裁剪</td><td>Actor1，Critic1</td><td>限制梯度范数</td></tr><tr><td>GAE lambda</td><td>0.999，全部相同</td><td>优势估计参数</td></tr><tr><td>输入容量</td><td>4096 tokens，其中64预留动作续写</td><td>观察上限4032；只扩展溢出容量，不截断旧观察</td></tr><tr><td>固定参考模型</td><td>supply3 checkpoint_0048</td><td>KL锚与最终同规则未训练基线</td></tr></tbody></table></div></details><p class="source-note">依据：冻结配置、m9_policy.py、m9_run.py与M9研究协议。当前用户所指的第二个长期生存里程碑在仓库中记作M9。</p></section>

<section id="matrix" class="section"><div class="section-head"><span class="section-no">02</span><h2>10组配置，一张表看清</h2></div><p>除列出的变化外，全部继承共同配置。第2至9组各比较一类干预，第10组叠加五项变化。点击名称跳到详细解释。</p><div class="toolbar" role="group" aria-label="按干预类型筛选"><button class="filter active" data-filter="全部" aria-pressed="true">全部</button><button class="filter" data-filter="State" aria-pressed="false">State</button><button class="filter" data-filter="Action" aria-pressed="false">Action</button><button class="filter" data-filter="Reward" aria-pressed="false">Reward</button><button class="filter" data-filter="Discount" aria-pressed="false">折扣</button><button class="filter" data-filter="Map" aria-pressed="false">地图</button><button class="filter" data-filter="Combined" aria-pressed="false">组合</button><span class="count" id="matrix-count" aria-live="polite">显示10组</span></div><p class="fine scroll-hint">宽表可横向滑动。</p><div class="table-wrap"><table class="matrix" id="matrix-table"><caption class="fine" style="text-align:left;padding:12px">配置按预登记顺序；成绩为两次训练后各8局的探索汇总</caption><thead><tr><th>方案</th><th>State</th><th>navigate上限</th><th>Reward</th><th>gamma / 刻</th><th>地图</th><th>活到30天</th><th>RMST</th></tr></thead><tbody>__MATRIX__</tbody></table></div><p class="fine">“1格/4格”只指navigate；采集、休息、建造等本来就可能持续多刻。RMST为30天观察窗内的平均存活，超过30天的部分未观测。</p></section>

<section id="arms" class="section"><div class="section-head"><span class="section-no">03</span><h2>逐组详解：改动、假设与实测</h2></div><p>以下“想验证的假设”是设计动机；“已有结果”来自冻结的探索评估，不能把二者混为已证明的机制。</p>__CARDS__</section>

<section id="state" class="section"><div class="section-head"><span class="section-no">04</span><h2>State：相同事实，三种表达</h2></div><p>模型不是直接读取整张地图或内部引擎对象，而是接收经过字段白名单构造的中文规则与观察。actor决定行动；critic从同一批事实提取固定108维数值特征来估计价值。</p><div class="two-col"><div class="panel"><h3>模型实际能看到</h3><ul><li>生命、饱食、水分、木材、石头、食物库存。</li><li>昼夜状态、距昼夜切换剩余刻数。</li><li>周围5×5地形、局部踏入次数。</li><li>观察到的地标、相对位置、可用性、再生倒计时及合法交互标记。</li><li>营地与威胁位置、合法方向及动作、最近4次行动与数值变化。</li></ul></div><div class="panel"><h3>这次状态实验没有改变</h3><ul><li>没有给actor未来事件、完整隐藏地图、reward数值或推荐行动。</li><li>三种表达使用相同事实来源；reserves增加的只是当前事实算术。</li><li>critic仍为96维原特征＋12维浆果/蘑菇就绪特征，没有为reserves或compact另换网络输入维数。</li><li>系统规则会与4格导航或地图时钟机制一致，避免对模型描述错误规则。</li></ul></div></div><h3>余量信息的直观例子</h3><p>若饱食40、水分24、食物2份，就添加：“不补充时饱食耗尽约200刻，水分耗尽约300刻；背包食物共可补36饱食。”其意义是把算术显式化。食水耗尽后还会扣生命，所以这些数值不是剩余寿命。</p><div class="note"><p><strong>“紧凑JSON”不等于更省token。</strong>无空格JSON仍会重复字段名、位置和布尔值。下面同一真实观察的正文长度为：原中文__LEGACY_CHARS__字符、余量中文__RESERVES_CHARS__字符、JSON __COMPACT_CHARS__字符。字符数不是token数，也不是整段输入长度。</p></div><h3>同一条真实观察的三种编码</h3><p class="fine">来源：token超限回归fixture，世界seed3301001、tick5085。下面直接调用冻结encoder输出，仅展示user观察正文；系统规则与动作续写未计入字符数。折叠项可展开查看完整文字。</p>__SAMPLES__<p class="source-note">注意奖励通道与state通道的区别：旧supply水势函数会读取训练端的已观察水地标路径步数；这些路径步数不会传入actor或critic。新bottleneck改用相对坐标的曼哈顿距离。详见下一节公式。</p></section>

<section id="action" class="section"><div class="section-head"><span class="section-no">05</span><h2>Action：模型选目标，执行器走路径</h2></div><div class="flow"><div><strong>1 / 观察</strong>规则、当前事实与合法动作</div><div><strong>2 / 选择动作</strong>模型选择移动、前往、采集等类型</div><div><strong>3 / 选择目标</strong>模型选合法方向或地标</div><div><strong>4 / 引擎执行</strong>避障与逐刻生存结算，返回新观察</div></div><div class="table-wrap"><table><thead><tr><th>动作</th><th>耗时</th><th>共同规则</th></tr></thead><tbody><tr><td>move / 移动</td><td>1刻</td><td>合法方向走1格，macro4组也相同</td></tr><tr><td>navigate / 前往</td><td>1格1刻；上限1或4格</td><td>只去模型已选定的目标；不自动互动</td></tr><tr><td>gather / 采集</td><td>2刻</td><td>树得2木材、岩石得2石头、浆果/蘑菇得3食物；资源90刻再生</td></tr><tr><td>eat / 吃饭</td><td>1刻</td><td>消耗1食物，恢复18饱食，上限100</td></tr><tr><td>drink / 喝水</td><td>1刻</td><td>合法饮水位置恢复50水分，上限100</td></tr><tr><td>rest / 休息</td><td>3刻</td><td>距营地≤2.5格、行动前饱食与水分均≥20时恢复2生命</td></tr><tr><td>build_camp / 建营地</td><td>4刻</td><td>消耗4木材、2石头，需满足可建造条件</td></tr><tr><td>wait / 等待</td><td>1刻</td><td>仅推进世界</td></tr></tbody></table></div><p>每刻饱食−0.2、水分−0.08；饱食为0扣1.2生命，水分为0扣1.8生命。每天180刻，第140刻进入夜间。夜间本身不扣血；营地外敌人距离&lt;1.6格时每刻扣1.2生命。</p><div class="note"><p><strong>macro4是持续目标，不是加速时间或自动生存。</strong>到浆果旁边之后，需要模型下一次自己选择采集，再自己选择吃饭。一次四格导航期间，角色仍连续承受四刻消耗。所有组都已有基础避障导航，因此实验比较的是导航持续步数。</p></div></section>

<section id="reward" class="section"><div class="section-head"><span class="section-no">06</span><h2>Reward：三套规则的精确区别</h2></div><p>奖励只用于训练评分，不会直接替模型决定动作。统一按动作实际持续时间计算，而不是每次模型开口就给固定奖励。</p><div class="formula">r(s → s′, d) = Σᵢ₌₀ᵈ⁻¹ γⁱ uᵢ + γᵈ Φ(s′) − Φ(s)
后续回报的折扣 = γᵈ
<span class="soft">d = 这次动作实际推进的世界tick；自然死亡时Φ=0。</span></div><p>势函数Φ给状态打分，但实际加入reward的是相邻状态的折扣差分。不能把“当前Φ很高”理解为每刻都获得同样高的奖励；多次重复某个动作也没有独立次数奖金。</p><div class="panel"><h3>A. supply：维持储备，并小幅惩罚低食水</h3><p>用于control、两种state组、macro4、long_credit与两种地图组。设H为饱食、T为水分、L为生命、F为食物库存：</p><div class="formula">N = min(1, (H + 18 × min(F, 6)) / 120)
Φ_vitals = 0.5 × L/100 + 0.25 × [min(1,H/60) + min(1,T/60)]
Φ_water  = 1 − max(0,1−T/60) × D_path/32
Φ_food   = 0.5N + 1 − (1−N) × D_food/16
Φ_supply = Φ_vitals + Φ_water + Φ_food

u_alive = 0.01 − 0.0035 × [max(0,1−H/40)² + max(0,1−T/40)²]
u_dead = 0</div><p class="fine">D_path是观察到的水地标中可用路径步数的最小值，截断到32；无可达已知水源时取32。D_food只考虑观察到且可用的浆果/蘑菇，取曼哈顿距离减相邻交互半径1、下限0、上限16；缺失取16。生命与基础储备势使用transition的精确数值，水与食物势使用观察值，代码按这两部分分别计算差分。</p><p>食水低于40时持续效用略低，促使策略保持安全储备。这个风险代价直接改变了训练偏好，不能把整套supply都说成“只加不改变目标的势函数”。</p></div>
<div class="panel"><h3>B. bottleneck：突出食水短板，按紧迫性考虑距离</h3><div class="formula">N = min(1, (H + 18 × min(F,6)) / 120)
W = min(1, T/80)
Φ_bottleneck = 0.25 × L/100
             + 0.75 × min(N,W) + 0.25N + 0.25W
             + 0.25 × (1−N) × (1−D_food/16)
             + 0.25 × (1−W) × (1−D_water/16)
u_alive = 0.01；u_dead = 0</div><p>min(N,W)代表较弱的一侧；食物非常多而水分很低时，短板项仍低。食物库存最多按6份参与N，超过营养上限后不再给囤积增加势值。N、W充足时，相应的资源距离项减弱。</p><p class="fine">D_food与上面相同；D_water改用当前观察到且可用的水岸曼哈顿距离，不减交互半径。两者均上限16，缺失取16。这些距离不保证障碍物后的真实路程短。</p><h4>数值例子：一次喝水如何评分</h4><p>假设生命100、饱食40、水分20、食物2份，食物交互距离4，水岸就在脚下。执行喝水1刻后，饱食39.8、水分69.92。按gamma=0.9995计算：</p><pre>Φ_before = __PHI_BEFORE__
Φ_after  = __PHI_AFTER__
r = 0.01 + 0.9995 × Φ_after − Φ_before = __EXAMPLE_REWARD__</pre><p class="fine">这是用于解释公式的假设状态，由实际potential函数计算；并非训练成绩。吃饭、喝水没有固定奖励：最终值取决于身体、库存、资源距离和动作耗时。</p></div>
<div class="panel"><h3>C. survival：只有实际活着的时间</h3><div class="formula">Φ_survival = 0
u_alive = 0.01；u_dead = 0
r = Σᵢ₌₀ᵈ⁻¹ γⁱ × 0.01 × 1[该tick结束仍存活]</div><p>它不包含补给距离、储备偏好或固定死亡扣分。活得更久才有更多未来存活回报。当前纯生存组在探索中优于单独短板组，但不能因此断言它在更长训练后仍最佳。</p></div>
<h3>gamma到底把未来看多远？</h3><p>long_credit和combined使用0.99985，其余使用0.9995。下面仅展示数学权重，不是模型实际规划能力：</p><div class="table-wrap"><table><thead><tr><th>每刻gamma</th><th>10天后回报权重γ¹⁸⁰⁰</th><th>30天后回报权重γ⁵⁴⁰⁰</th><th>权重降为一半的时间</th></tr></thead><tbody>__DISCOUNT__</tbody></table></div><p class="source-note">代码依据：m9_contract.Reward / potential、m8s3_contract.potential / _training_water、m8s3_supply.food_potential / living_utility。所有方案均不按贴边、跨区、发现新格或采集次数发额外奖金。</p></section>

<section id="map" class="section"><div class="section-head"><span class="section-no">07</span><h2>“无限地图”具体做到了什么？</h2></div><p>这里实现的是可跨越原96×96外边界的分区扩展研究原型。每块64×64，玩家所在区块与相邻8块构成3×3活跃范围；不是只有角色脚下的一格或单个房间在运行。</p><div class="panel"><div class="map-demo"><div><div class="map-grid" aria-label="5乘5区块示意，中心3乘3活跃">__CHUNKS__</div><p class="fine" style="margin-top:10px">绿色：活跃3×3区块<br>灰色：范围外已有区块<br>橙框R：离开的资源区块</p></div><div><h3>离开后100刻，再回来</h3><p>假设离开活跃范围时，R区资源还需90刻恢复。玩家继续在别处活动100刻，两种规则的差别如下：</p><div class="map-controls toolbar" role="group" aria-label="地图时钟规则"><button class="map-mode active" data-mode="freeze" aria-pressed="true">离区冻结</button><button class="map-mode" data-mode="global" aria-pressed="false">全局再生</button></div><div class="clock-display" aria-live="polite"><div><small>世界时间</small><b>+100刻</b></div><div><small>R区资源剩余</small><b id="resource-clock">90刻</b></div><div><small>角色仍需补给</small><b>持续消耗</b></div></div><p id="map-explanation" class="fine">冻结模式：R区局部时钟不动，返回后还要在活跃范围内推进90刻。玩家并未随资源一起暂停。</p></div></div></div><div class="two-col"><div class="panel"><h3>按坐标生成，返回不重刷</h3><p>旧0..95区域保留原地形与初始资源。扩展区域由世界seed与整数坐标决定：每32格单元生成确定性小湖，非水格有8.5%的资源候选概率。首次需要时生成区块，之后保留已生成地形和耗尽状态。</p><p>生成顺序不会改变同一坐标的世界；离开再进入不会凭空补满资源。freeze是暂停再生倒计时，global是按世界tick判断再生。</p></div><div class="panel"><h3>“不会撞旧边界”有适用范围</h3><p>扩展模式允许穿过原外边界，但水面、树木、岩石、营地等碰撞规则仍存在。坐标设有±1,000,000边界，并非数学无穷。</p><p>导航只使用已发现地形，在玩家附近16格范围进行有界局部BFS；不会为了找到远方食物读取全图或自动替模型选目的地。</p></div></div><div class="note"><p><strong>地图对照必须披露的限制：</strong>扩展区目前不新增敌人，威胁仍主要来自原出生区固定位置；新增资源和地形分布也不同。有限图vs扩展图比较包含多项环境变化。freeze与global共享生成器和威胁设置，是更直接的时钟机制对照，但模型规则文字也会随机制调整。</p></div><p>当前原型把访问过的区块留在内存，没有完成磁盘LRU流送、完整跨区存档和可见游戏渲染集成。因此这轮训练可以回答“扩展地图机制是否值得继续研究”，不能单凭训练成绩宣称已交付可发布的无限世界游戏。</p></section>

<section id="results" class="section"><div class="section-head"><span class="section-no">08</span><h2>已有结果：谁值得继续训练？</h2></div><p>每组结果来自<strong>两份独立优化得到的策略，各评估8个世界</strong>。训练前后使用配对世界；同一重复的世界种子也在方案之间复用。每组16局是探索汇总，不是一份最终策略的16局，更不是最终100局。</p><div class="chart-top"><div><h3>训练前 → 训练后</h3><p class="fine">按冻结的筛选排名排列。默认显示30天窗内平均存活。</p></div><label for="chart-metric">图表指标 <select id="chart-metric"><option value="rmst">平均存活RMST（天）</option><option value="success">30天存活（局 / 16）</option><option value="ten">10天存活（局 / 16）</option></select></label></div><div class="legend"><span><i class="dot before"></i>共同起点在该组规则下的表现</span><span><i class="dot"></i>训练8轮后的表现</span></div><div id="results-chart" class="chart" aria-live="polite"></div><p class="fine scroll-hint">图表可横向滑动，完整数值见下表。</p><div class="table-wrap"><table><thead><tr><th>排名</th><th>方案</th><th>30天存活 前→后</th><th>RMST天 前→后</th><th>训练后10天</th><th>较差重复CVaR20 / 天</th></tr></thead><tbody>__RESULTS__</tbody></table></div><div class="two-col"><div class="panel"><h3>当前可以作出的判断</h3><ul><li>组合组11/16、RMST26.37天，探索排名第一。</li><li>冻结地图10/16、RMST24.65天，说明扩展地图值得进一步确认。</li><li>四格导航和纯存活奖励也有较好的探索表现。</li><li>余量state、长折扣、短板reward单独变化没有显示一致的明显改善。</li></ul></div><div class="panel"><h3>当前还不能作出的判断</h3><ul><li>不能说五个组件均有效或组合已达到最优。</li><li>不能把冻结地图的收益全归因于边界消失。</li><li>不能将两次训练的16局成功率直接当作最终模型可靠性。</li><li>不能把RMST26.37天说成自然寿命：观察窗到30天停止。</li></ul></div></div><p><strong>为什么有的排名与平均存活不完全一致？</strong>冻结规则先按两次重复的30天成功总数，再比较两次中较差的CVaR20，再比较RMST总和，最后按登记顺序打破平局。CVaR20衡量最短20%寿命的平均值，尾部边界局按比例加权。所有探索训练后评估的360刻前早死数均为0。</p><details><summary>查看计算成本与世界时间：相同决策预算并不等于相同训练量</summary><div class="table-wrap"><table><thead><tr><th>方案</th><th>两次训练决策数</th><th>训练世界tick合计</th><th>tick / 决策</th><th>平均每轮用时</th></tr></thead><tbody>__EFFICIENCY__</tbody></table></div><p class="fine">平均每轮用时为16条更新日志中rollout_seconds＋update_seconds的均值；不含初始/最终评估、模型加载、审计或排队时间。这是这次执行实测，不是严格硬件基准。总世界tick含采集、休息等多刻动作，因此控制组也会大于决策数。</p></details></section>

<section id="milestone" class="section"><div class="section-head"><span class="section-no">09</span><h2>从探索到M9，通过哪些关卡？</h2></div><div class="timeline"><div class="step"><h3>① 工程检查与GPU计时</h3><p>短smoke验证权重更新、重载、轨迹和执行正确性；完整批量校准测吞吐。短测不参与生存质量排名。</p></div><div class="step"><h3>② 10组 × 2次 × 8轮探索 <span class="pill selected">已完成</span></h3><p>共160轮、327,680次决策；每次训练前后各8局，观察5400刻。筛选出combined与infinite_freeze。</p></div><div class="step"><h3>③ 前两组分别重新训练64轮 <span class="pill">快照时进行中</span></h3><p>各131,072次决策，从共同supply3起点重新训练，不沿用探索第8轮权重。使用新的24局确认集，固定评估第64轮模型，不逐轮挑最好成绩。快照时combined为27/64轮，另一组尚未开始确认。</p><p class="fine">combined确认阶段训练前基线为7/24活到30天、23/24活到10天、RMST22.38天；这些不是训练27轮后的成绩。</p></div><div class="step"><h3>④ 确认过门槛，才解锁全新100局</h3><p>按确认结果的30天存活、CVaR20、RMST等冻结顺序选唯一候选。至少23/24活到30天、23/24通过10天且无早死，才进入100局候选与同规则初始模型的配对评估。</p></div><div class="step"><h3>⑤ 数值通过＋独立审计＋真实游戏验收</h3><p>完成轨迹/随机数审计、checkpoint重载、真实HTTP与可见游戏验收，才具备里程碑完成证据。自动研究流程不会自行发布模型。</p></div></div><h3>最终100局的核心门槛</h3><div class="gate"><div><b>≥95 / 100</b><p>在5400刻结束时仍存活。恰好第5400刻死亡算失败。</p></div><div><b>≥90 / 100</b><p>保留1800刻（10天）的生存回归门槛。</p></div><div><b>≤5 / 100</b><p>360刻之前自然死亡的早死局数。</p></div></div><p style="margin-top:18px">还须相对同规则共同起点满足非劣性：配对RMST差异95%区间下界&gt;−270刻（−1.5天），成功率差异下界&gt;−0.05，候选CVaR20≥基线的90%。bootstrap固定10,000次、随机种子912800。95/100是样本验收门槛，不代表已证明总体成功率的95%置信下界达到95%。</p><details><summary>数据隔离、research5续跑与退出清理</summary><p>探索评估世界：3300000…3300007、3301000…3301007；确认：3342000…3342023；最终：3400000…3400099。不同用途分开，最终测试不能反复曝光后继续挑选方案。</p><p>research4完成104轮后在compact r1评估遇到输入超限。research5保持state/action/reward、采样与种子不变，将容量1536扩展到4096，保留旧分桶，并新增溢出分桶。12组完整结果复用；第13组复用最终权重，在原冻结8局从头补评估，保留旧失败轨迹，不拼接有利片段。</p><p>登记使用独立v5项目和research_attempt5输出。快照核验了277项源码、629项输入；20份探索checkpoint重载与41份完整初始/训练后报告核查通过。此次快照检查没有重跑全部原生轨迹审计，后续确认与最终流程仍需执行。</p><p>监督器按正常结束、异常、停止或硬超时清理本任务进程组并记录GPU计算进程。快照时GPU仍在执行训练；真实释放须在任务结束后核验。这份方案报告不额外启动训练或定时监控。</p></details></section>

<section id="sources" class="section references"><div class="section-head"><span class="section-no">10</span><h2>证据与代码索引</h2></div><p>配置以research5冻结登记为准，结果由完整episode终点重新汇总，并重算筛选排名。报告构建时逐一比对下列源文件与登记哈希，全部一致。内容和图表已内嵌，单个HTML文件可离线阅读；以下相对链接用于在原项目中追溯证据。</p><ul><li>__REG_LINK__：冻结配置、源码与输入文件登记。</li><li>__SNAP_LINK__：2026-11-13 10:52节点检查原始快照。</li><li>__STATUS_LINK__：当时的进度、效果、时间估算与GPU检查记录。</li><li>__PROTOCOL_LINK__：预登记研究设计；__RESUME_LINK__：token修复与旧结果复用规则。</li><li><a href="report_data.json">report_data.json</a>：本报告的结构化结果、三种state样例与源文件哈希。</li></ul><div class="panel"><h4>登记摘要</h4><p class="hash">__REG_SHA__</p><h4>快照文件SHA-256</h4><p class="hash">__SNAP_SHA__</p><h4>共同初始actor SHA-256</h4><p class="hash">__WARM_SHA__</p></div><details><summary>查看本报告核验的15项源码与fixture哈希</summary><div class="table-wrap"><table><thead><tr><th>文件</th><th>SHA-256</th></tr></thead><tbody>__SOURCES__</tbody></table></div></details><div class="note"><p>结果支持的结论是：组合方案和分区冻结方案值得进入更长的独立确认训练。是否完成第二个长期生存里程碑，仍应以最终100局、审计及真实游戏验收为准。</p></div></section><footer class="footer">Meadowmind · research5研究说明 · 报告日期2026-11-13 · 证据时间2026-11-13 10:52:03（北京时间）<br>游戏日＝180 tick；所有生存成绩均截断于5400 tick。单文件离线HTML，无外部字体、脚本或图表依赖。</footer></main></div>
<script id="report-data" type="application/json">__DATA__</script><script>__JS__</script></body></html>'''

JS = r'''
"use strict";
const report = JSON.parse(document.getElementById('report-data').textContent);
const ranked = report.arms.slice().sort((a,b)=>a.rank-b.rank);
document.querySelectorAll('.filter').forEach(button=>button.addEventListener('click',()=>{
  const filter=button.dataset.filter;
  document.querySelectorAll('.filter').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  let count=0;document.querySelectorAll('#matrix-table tbody tr').forEach(row=>{row.hidden=filter!=='全部' && row.dataset.category!==filter;if(!row.hidden) count++;});
  document.getElementById('matrix-count').textContent=`显示${count}组`;
}));
const NS='http://www.w3.org/2000/svg';
function svgel(tag,attrs,text){const el=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs||{}))el.setAttribute(key,value);if(text!==undefined)el.textContent=text;return el;}
function renderChart(){
  const metric=document.getElementById('chart-metric').value;
  const key=metric==='rmst'?'rmst_days':metric==='success'?'successes':'successes_10day';
  const max=metric==='rmst'?30:16, suffix=metric==='rmst'?'天':'局',W=920,H=605,left=130,right=90,plot=W-left-right;
  const root=svgel('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-labelledby':'chart-title chart-desc'});
  root.append(svgel('title',{id:'chart-title'},metric==='rmst'?'10组方案训练前后平均存活时间':`${metric==='success'?'30':'10'}天存活局数对比`));
  root.append(svgel('desc',{id:'chart-desc'},'按冻结筛选排名排列，灰色为训练前，绿色为训练8轮后。完整数值可在后续数据表中读取。'));
  for(let n=0;n<=4;n++){const value=max*n/4,x=left+plot*n/4;root.append(svgel('line',{x1:x,y1:36,x2:x,y2:H-30,stroke:'#dce2d7','stroke-dasharray':n?'3 5':'0'}));root.append(svgel('text',{x,y:23,'text-anchor':'middle',fill:'#657269','font-size':12},String(value)));}
  ranked.forEach((arm,index)=>{const y=55+index*54;
    root.append(svgel('text',{x:left-15,y:y+15,'text-anchor':'end','font-size':14,fill:'#203a32'},`${arm.rank}. ${arm.short}`));
    [['initial','#c7d1c6',0],['after','#1b6a57',19]].forEach(([phase,color,offset])=>{
      const value=arm[phase][key],width=plot*value/max;
      const rect=svgel('rect',{x:left,y:y+offset,width,height:13,rx:2,fill:color});rect.append(svgel('title',{},`${arm.title} ${phase==='initial'?'训练前':'训练后'}：${metric==='rmst'?value.toFixed(2):value}${suffix}`));root.append(rect);
      root.append(svgel('text',{x:left+width+7,y:y+offset+11,'font-size':12,fill:phase==='after'?'#1b6a57':'#657269'},metric==='rmst'?value.toFixed(2):`${value}/16`));
    });
  });
  document.getElementById('results-chart').replaceChildren(root);
}
document.getElementById('chart-metric').addEventListener('change',renderChart);renderChart();
document.querySelectorAll('.map-mode').forEach(button=>button.addEventListener('click',()=>{
  const freeze=button.dataset.mode==='freeze';document.querySelectorAll('.map-mode').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  document.getElementById('resource-clock').textContent=freeze?'90刻':'0刻／已恢复';
  document.getElementById('map-explanation').textContent=freeze?'冻结模式：R区局部时钟不动，返回后还要在活跃范围内推进90刻。玩家并未随资源一起暂停。':'全局模式：世界已经过去100刻，超过剩余90刻，返回时资源已恢复。角色的饥渴与生命照常结算。';
}));
let printDetails=[];
window.addEventListener('beforeprint',()=>{printDetails=[...document.querySelectorAll('details')].map(el=>[el,el.open]);printDetails.forEach(([el])=>el.open=true);});
window.addEventListener('afterprint',()=>{printDetails.forEach(([el,open])=>el.open=open);});
document.querySelectorAll('.print-button').forEach(button=>button.addEventListener('click',()=>window.print()));
document.documentElement.dataset.reportReady='true';
'''

chunks = ''.join(f'<span class="chunk {"active" if 1<=y<=3 and 1<=x<=3 else ""} {"center" if x==2 and y==2 else ""} {"resource" if x==0 and y==1 else ""}">{"角色" if x==2 and y==2 else "R" if x==0 and y==1 else "64²"}</span>' for y in range(5) for x in range(5))
substitutions = {
    'CSS':CSS, 'JS':JS, 'MATRIX':''.join(matrix), 'CARDS':''.join(cards), 'RESULTS':results_rows,
    'EFFICIENCY':efficiency_rows, 'SAMPLES':sample_panels, 'DISCOUNT':discount_rows, 'CHUNKS':chunks,
    'LEGACY_CHARS':str(len(samples['legacy'])), 'RESERVES_CHARS':str(len(samples['reserves'])), 'COMPACT_CHARS':str(len(samples['compact'])),
    'PHI_BEFORE':f'{phi_before:.6f}', 'PHI_AFTER':f'{phi_after:.6f}', 'EXAMPLE_REWARD':f'{reward_example:.6f}',
    'SOURCES':source_rows, 'REG_LINK':source_link(REG_PATH,'research5冻结登记'), 'SNAP_LINK':source_link(SNAP_PATH,'原始检查快照'),
    'STATUS_LINK':source_link('output/m9_followup_20261113_1052/report.md','最近一次训练进度报告'),
    'PROTOCOL_LINK':source_link('docs/M9_RESEARCH_PROTOCOL.md','M9研究协议'),
    'RESUME_LINK':source_link('docs/M9_RESEARCH_RESUME5.md','research5技术续跑说明'),
    'REG_SHA':reg['registration_sha256'], 'SNAP_SHA':sha(SNAP_PATH), 'WARM_SHA':reg['warm_actor_sha256'],
    'DATA':json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'),
}
HTML = HTML.replace('10组生存训练方案详解</h1>', '10组生存训练<span class="title-tail">方案详解</span></h1>')
for key, value in substitutions.items():
    HTML=HTML.replace('__'+key+'__',value)
(OUT/'report.html').write_text(HTML,encoding='utf-8')
(OUT/'report.js').write_text(JS,encoding='utf-8')
print(json.dumps({'report':str(OUT/'report.html'),'bytes':len(HTML.encode('utf-8')),'arms':len(rows),'verified_sources':len(verified),'ranking':ranking,'reward_example':reward_example},ensure_ascii=True))
