from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib,json,csv,re

root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
s=read(folder/"evidence/progress_snapshot.json")
v=read(folder/"evidence/final_verification.json")
g=read(folder/"evidence/gpu_release_verification.json")
for name,digest in v["report_sha256"].items():
    assert hashlib.sha256((folder/"evidence"/(name+".json")).read_bytes()).hexdigest()==digest
candidate=read(folder/"evidence/test_candidate.json")
assert len(candidate["episodes"])==100 and sum(e["alive_at_horizon"] for e in candidate["episodes"])==88
assert g["compute"]["exit_code"]==0 and not g["compute"]["stdout"]
assert not g["matching_processes"] and g["owned_pids"]["exit_code"]==1
assert v["computed_gate_matches_saved"] and not v["gate"]["numerical_gate_passed"]
def cst(t):
    return datetime.fromisoformat(t).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
finished=s["job"]["finished.json"]
hours=finished["seconds"]/3600
summary=v["gate"]["candidate"]
with (folder/"final100.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["seed","replicate","survival_ticks","survival_days_capped10","alive_at_horizon","terminal","hunger","thirst","berries","x","y"])
    for e in candidate["episodes"]:
        z=e["final"]
        writer.writerow([e["seed"],e["replicate"],e["survival_ticks"],e["survival_ticks"]/180,e["alive_at_horizon"],e["terminal"],z["hunger"],z["thirst"],z["inventory"]["berries"],z["player"]["x"],z["player"]["y"]])
failed_table="\n".join(f"| {e['seed']} | {e['ticks']} | {'食物耗尽' if e['hunger']==0 else '水分耗尽'} | {e['berries']} | ({e['player']['x']},{e['player']['y']}) |" for e in v["failures"])
report=f"""# supply3最终结果与GPU释放核验

核查时间：{cst(g['observed_utc'])} 北京时间。通过Netcatty MCP连接集群1节点5，实际主机researcher-local-5，执行只读检查。

**任务已结束；最终100局88/100存活，未达到90/100门槛，当前M8B里程碑未完成。GPU已释放。**

## 训练与剩余时间

- 结束时间：{cst(finished['finished_utc'])} 北京时间；自提交至全部离线阶段结束共约35小时27分（{hours:.4f}小时）。
- 完成216轮、442368条训练决策、517929世界tick。第208和216轮连续两次16/16，按预先登记规则提前停止；原最大预算256轮。
- 第216轮筛选16/16、10天窗口内平均存活10天。按“存活数、CVaR20、RMST、并列取较早轮次”的冻结排名，最终100局测试候选为 **checkpoint_0048**，不是最后的216轮。
- 训练、独立重载、训练回放、最终测试、测试回放、旧规则描述性对照6阶段退出码均为0；最终状态numerical_gate_failed。
- 本轮任务剩余时间 **0**。里程碑仍未完成；下一轮达标需要的训练时间未知。

## 最终100局复核

三个评估分支均恰好包含登记的100个独立种子2200000–2200099，每个种子replicate=0，无重复或缺失。重新计算全部统计和10000次配对bootstrap（固定seed=912800），与保存的数值门槛报告完全相同。

| 验收项目 | 实测 | 门槛 | 结果 |
|---|---:|---:|---|
| 第1800 tick结束仍存活 | 88/100（88%） | 至少90/100 | **未通过，差2局** |
| 第360 tick前死亡 | 0/100 | 不超过5/100 | 通过 |
| 平均受限存活时间RMST | 1761.89 tick，约9.7883游戏天 | 评估窗1800 tick | 描述性指标 |
| 同规则父模型RMST差异95%区间 | [570.3095, 714.02075] tick | 下界大于-90 | 通过 |
| 同规则父模型存活率差异95%区间 | [0.73, 0.90] | 下界大于-0.05 | 通过 |
| 最差20%平均存活CVaR20 | 1609.45 tick | 至少0.9×735.95=662.355 | 通过 |

同规则M6父模型为6/100、RMST1118.01 tick（约6.2112天）。本轮相对父模型已有明确改善，但绝对存活目标未过。88局在10天处右删失，平均9.79天是**10天观察窗口内的受限均值**，不是无限时长的自然平均寿命。

种子2200022恰在1800 tick死亡，terminal=true、alive_at_horizon=false，正确计为失败；仅统计survival_ticks≥1800会错计成89/100。存活率双侧95%精确区间约79.98%–93.64%，不构成总体90%存活保证。

## 失败记录

12局失败的终态中，8局水分为0，4局饱食和食物库存为0，无360 tick前早死。失败终点集中在右下侧（x=80–94、y=87–94），提示持续补给与位置偏移需要进一步诊断；终点分布本身不能证明边界碰撞是死亡原因。

| 世界种子 | 存活tick | 终态资源问题 | 剩余浆果 | 终点 |
|---|---:|---|---:|---|
{failed_table}

筛选集曾16/16，独立测试仅88/100，说明筛选满分不能替代100局泛化验收。后续若优化，应新登记训练与筛选方案、保留失败证据，使用新的独立验收种子；不能反复测试这100局来挑选模型。本次检查未提交下一轮训练。

## 审计与证据一致性

- 登记摘要通过；286项冻结源码、24项输入哈希复核通过。
- checkpoint_0048的6项文件哈希通过，actor摘要{v['actor_sha256']}与测试claim、筛选、最终评估一致。
- 原任务独立重载16个动作的log-prob误差为0。
- 原任务独立Godot状态/转移/reward及CUDA RNG回放：训练与筛选1,119,494条，最终候选与同规则父对照229,160条，均通过，最大浮点误差0。本次重新核验全部247份被审计轨迹的哈希及审计程序身份。
- 本次读取最终候选154,519条、父对照74,641条原始轨迹，逐条检查模型身份和动作来源，重新提取全部200局终点、死亡标志、决策数、最终状态，与逐局报告全部相符。
- 旧规则对照100局仅用于描述性比较；上述229,160条测试回放范围不包括旧规则分支。
- 本次数值与轨迹复核命令退出码0，仅CPU文件读取和统计，没有重跑最终100局或加载GPU模型。

## GPU释放核验

{cst(g['observed_utc'])} 实查：

- RTX6000 Ada利用率 **0%**，显存 **19 MiB / 49140 MiB**。
- nvidia-smi计算进程查询退出码0，输出为空。
- 监督1574503、训练1574511、最后评估2816678均不存在；未发现supply3训练、审计或模型服务进程。
- 监督结束记录owned_process_groups_cleaned=true，与实时检查一致。19 MiB为设备空闲基线，不是本任务模型残留。

M8B仍未完成，未发布模型，真实HTTP与渲染游戏验收尚未完成。本次为用户发起的一次检查，不建立周期检查。

## 本地证据

- [实时快照](evidence/progress_snapshot.json)
- [统计与轨迹复核](evidence/final_verification.json)
- [只读核验命令](evidence/verification_command.txt)
- [GPU释放核验](evidence/gpu_release_verification.json)
- [冻结登记](evidence/registration.json)
- [原始最终候选100局](evidence/test_candidate.json)
- [原始同规则父对照100局](evidence/test_parent.json)
- [原始旧规则对照100局](evidence/test_original_v2.json)
- [原始数值门槛报告](evidence/numerical_gate.json)
- [最终100局CSV](final100.csv)
"""
(folder/"report.md").write_text(report,encoding="utf-8")
status_path=root/"output/current/training_status.json"
previous=read(status_path)
(folder/"evidence/previous_status.json").write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding="utf-8")
status=dict(previous)
status.update(status="numerical_gate_failed",milestone_achieved=False,training_running=False,formal_submitted=True,
    service_running=False,periodic_check_active=False,active_followup_waiting=False,observed_utc=g["observed_utc"],
    phase_observed_utc=g["observed_utc"],finished_utc=finished["finished_utc"],completed_updates=216,completed_decisions=442368,
    completed_ticks=517929,stopping_reason="registered_consecutive_perfect",gpu_processes="",gpu_snapshot=g["gpu"]["stdout"],
    gpu_released=True,final_test_started=True,final_test_completed=True,numerical_gate_passed=False,
    formal_independent_audit_completed=True,online_acceptance_completed=False,pending_screen=None,elapsed_hours=hours,
    remaining_estimate=dict(total_remaining_hours=0,status="job_finished",milestone_remaining_time_unknown=True),
    report="output/rl_supply_followup_20261111_1227/report.md",final_test=summary,final_comparison=v["gate"],
    failure_counts=v["failure_counts"],current_worker_pid=None,live_owned_processes=[],
    latest_screen=dict(update=216,summary=s["run"]["training_complete.json"]["final_evaluation"],
        eligible_by_screen_rule=True,actor_sha256=s["run"]["training_complete.json"]["final_actor"]),
    best_screen=dict(update=48,summary=s["run"]["selection.json"]["summary"],eligible_by_screen_rule=True,
        actor_sha256=v["actor_sha256"],checkpoint_files_verified=6,independent_reload=v["independent_reload"]),
    budget_scope="Original forecast retained for history; actual completed duration is elapsed_hours",
    followup_policy="User-triggered only; job finished, no periodic or active waiting")
status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
path=root/"README.md"
lines=path.read_text(encoding="utf-8").splitlines()
i=next(i for i,line in enumerate(lines) if line.startswith("**当前："))
lines[i]="**当前：2026-11-11 12:30通过Netcatty核验，节点5supply3已于06:37结束，最终100局88/100存活，未达到90/100门槛，M8B尚未完成；GPU已释放。** 训练在216轮连续两次16/16后提前停止，最终候选为按并列规则选出的48轮。10天窗口平均存活9.79游戏天；12局失败中8局水分耗尽、4局食物耗尽。独立审计和轨迹复核通过，GPU0%、19MiB、无计算进程。本轮剩余时间0；下一轮达标时间未知。见[最终报告](output/rl_supply_followup_20261111_1227/report.md)、[训练协议](docs/M8S3_SUPPLY_PROTOCOL.md)。"
path.write_text("\n".join(lines)+"\n",encoding="utf-8")
note="""最新核验：2026-11-11 12:30北京时间，supply3已于06:37结束，完整离线任务约35小时27分。216轮、442368新决策后，208/216两次16/16触发登记的提前停止；最终测试按冻结排名选择48轮。最终100局88/100存活，未达到90/100，M8B未完成。10天窗口内RMST1761.89 tick（9.7883游戏天），早死0/100；同规则M6父对照6/100，三项非劣比较通过，绝对存活目标差2局。

12局失败为8局水分耗尽、4局食物耗尽；其中1局恰在1800 tick死亡，仍算失败。登记、286源码、24输入、候选6文件、247份审计轨迹哈希复核通过；最终候选与父模型200局的229160条原始转移终点重算一致。原任务独立回放累计1348654条最大误差0，独立重载误差0。GPU0%、19MiB、无计算进程，自有训练/监督/评估均退出，已确认释放。本轮剩余时间0，下一轮达标时间未知；不自动续训、不定时检查。见[最终报告](../output/rl_supply_followup_20261111_1227/report.md)。

以下为此前检查和历史记录。

"""
path=root/"docs/CURRENT_STATE.md"
text=path.read_text(encoding="utf-8")
assert "# 当前状态\n\n" in text
path.write_text(text.replace("# 当前状态\n\n","# 当前状态\n\n"+note,1),encoding="utf-8")
path=root/"docs/MILESTONE_ROADMAP.md"
lines=path.read_text(encoding="utf-8").splitlines()
i=next(i for i,line in enumerate(lines) if line.startswith("更新："))
lines[i]="更新：2026-11-11 12:30。supply3在216轮按连续满分规则提前停止，全部离线阶段06:37结束；最终100局88/100，低于90/100，M8B仍未完成。独立审计与逐局复核通过，GPU已释放。本轮剩余时间0，下一轮达标时间未知。"
i=next(i for i,line in enumerate(lines) if line.startswith("| **M8B 当前**"))
lines[i]="| **M8B 当前** | **90% 对局超过 10 天**，水耗低于饥饿、稳定RL | 新测试至少90/100在1800 tick结束仍活着；早死、同规则父非劣、证据及线上一致性 | **supply3已结束，最终88/100未达标；审计通过，GPU已释放** |"
text="\n".join(lines)+"\n"
text=text.replace("## M8B 执行与验收\n\n","## M8B 执行与验收\n\n本轮最终88/100；原始轨迹、独立审计与统计均已复核，见[最终报告](../output/rl_supply_followup_20261111_1227/report.md)。下述预算为提交时历史预测，实际全流程35小时27分，未达标且没有自动重启。\n\n",1)
path.write_text(text,encoding="utf-8")
path=root/"docs/WORKFLOW.md"
text=path.read_text(encoding="utf-8")
note="最新检查2026-11-11 12:30：supply3已于06:37结束；216轮按登记条件提前停止，最终测试48轮候选88/100，未达到90/100。独立审计、100局轨迹和统计已复核，GPU0%、19MiB且无计算进程，已释放。最终测试claim已使用，不重跑这100局挑选模型；后续优化另行登记。本次仅检查，不提交新任务或建立定时轮询。见[最终报告](../output/rl_supply_followup_20261111_1227/report.md)。以下为此前记录。\n\n"
path.write_text(text.replace("## 远端\n\n","## 远端\n\n"+note,1),encoding="utf-8")
for link in re.findall(r"\]\(([^)]+)\)",report):
    assert (folder/link).exists(),link
print(json.dumps(dict(report=str(folder/"report.md"),original_report_hashes_verified=3,final100_successes=88,
    mean_days=summary["rmst"]/180,total_hours=hours,gpu_released=True,report_links_valid=True),ensure_ascii=False))

