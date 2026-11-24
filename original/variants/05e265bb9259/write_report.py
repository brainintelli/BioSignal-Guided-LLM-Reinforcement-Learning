"""Render an exact-scope report only after downloaded evidence passes."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
receipt=read(R/'download_verification.json');assert receipt['terminal']['passed'] and receipt['stability']['passed']
t=read(E/'terminal01/report.json');s=read(E/'stability01/report.json');v=read(E/'stability01/independent_verification.json')
rr=[read(E/f'stability01/round_{i:02d}/report.json') for i in range(1,9)]
first=json.loads((E/'terminal01/round_01/rollout.jsonl').read_text(encoding='utf-8').splitlines()[0]);tr=first['training']
GiB=1024**3;MiB=1024**2
lines=['# 22554 B线：真实终止分支与 R1-only 连续训练测试','',
'本轮仅验证 `.16` 的软件训练稳定性。无刺激，R2、state2 均关闭；alpha=1，108维 critic，使用原生 M9 游戏和原有 PPO 超参数。没有访问 MaxOne，也未新增生物学记录。','',
'## 验收结果','',
f'- 终止测试：32次新模型决策、1次原生死亡、1次 PPO 调用；训练、保存重载和独立进程重放通过。',
f'- 连续长测：8轮×128次决策，共1024次新决策、{receipt["stability"]["ticks"]} tick、{s["terminals"]}次原生死亡；累计{sum(x["applied_steps"]["actor"] for x in rr)}次 actor、{sum(x["applied_steps"]["critic"] for x in rr)}次 critic 优化步骤。采样／更新／保存及核验阶段用时{s["seconds"]/60:.2f}分钟（不含随后独立进程验证）。',
'- 每轮旧策略统计复算、保存重载、独立进程旧／新检查点复算均通过1e-6门槛；全部逐条原生转移、R1、输入身份、GAE与优化器步骤连续性通过。',
'- 全程未出现 OOM、NaN 或训练崩溃；每轮 LoRA/critic 均改变，冻结基座及参考模型保持不变。','',
'## 真实终止证据','',
'固定种子4126101从原生初始世界执行492次合法等待，在生命值0.4时交给GPU模型；没有注入生命值或死亡标记。准备阶段的脚本动作不进入PPO数据。',
f'模型首个动作是 `{first["action"]}`，在tick493触发真实死亡。终止行 reward={tr["reward"]:.12f}，value={tr["value"]:.12f}，advantage={tr["advantage"]:.12f}，return={tr["return"]:.12f}；next value为0。之后重置种子4126102，另外31次模型决策纳入同批更新。',
'独立进程重放完整准备前缀与32次模型转移；将死亡后奖励增加100万后，死亡行的GAE/return保持不变。本批死亡发生在第0行，没有死亡前的模型采样行；独立的“最后一行恰好终止”分支尚未动态覆盖。采样末尾仍存活的截断使用critic bootstrap，不伪造死亡。终止观测没有送入下一次模型决策。','',
'## 长测逐轮结果','',
'长测重新从原始 checkpoint_0064 开始；不接续终止测试或上一轮短测权重。优化器对象和状态持续保留，游戏只在原生死亡后重置，更新边界不重置世界。','',
'| 轮次 | tick | 死亡次数 | actor步骤 | critic步骤 | KL提前停止 | 秒 | GPU峰值分配GiB | 轮末GPU分配GiB | 轮末RSS GiB |',
'|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|']
for x in rr:
    m=x['resources'];lines.append(f'| {x["round"]} | {x["ticks"]} | {len(x["terminal_indices"])} | {x["applied_steps"]["actor"]} | {x["applied_steps"]["critic"]} | {x["metrics"]["kl_early_stop"]} | {x["seconds"]:.1f} | {m["peak_allocated"]/GiB:.3f} | {m["allocated"]/GiB:.3f} | {m["rss"]/GiB:.3f} |')
g=s['resource_gate'];errors={key:max(r[key] for r in v['rounds']) for key in ['old_error','new_error','bootstrap_error']}
lines+=['',f'独立进程全部8轮最大误差：{errors}。',
f'预先登记的资源筛查通过：以第2轮为基线，第3–8轮结束时GPU分配最大增长{g["gpu_growth_bytes"]/MiB:.3f} MiB（上限512 MiB）；RSS最大增长{g["rss_growth_bytes"]/GiB:.3f} GiB（上限4 GiB）。测量前统一清除前缀缓存、GC并同步CUDA。这里的GPU分配量不是nvidia-smi的完整进程占用；RSS包含CPU暂存及分配器缓存。',
'独立审查发现一项轻微证据偏差：记录了peak allocated和轮末reserved，但未记录计划要求的peak reserved。无法补造历史峰值；不影响上述allocated/RSS筛查，保留为下轮采集改进项。','',
'## 模型、来源与交付','',
'- 机器：192.0.2.16 / researcher-local-6；物理GPU1，RTX4090，UUID `GPU-e9739564-e2ec-59c3-10f7-096d780abcbc`。',
'- 原始权重：`/home/researcher/mea-r2-22554/import_20261116_01/assets/m9-runs/research_attempt5/confirm/combined_r2/checkpoint_0064`。',
'- 最终长测检查点：`/home/researcher/mea-r2-22554/stability_20261116_01/stability01/round_08/checkpoint`。',
f'- 最终actor语义SHA256：`{s["final_actor"]}`。',
'- 每轮检查点均保存LoRA、critic、优化器与RNG；本任务验证了单进程连续优化器状态和权重重载，没有实现崩溃后恢复世界及优化器的自动续训。',
f'- 下载验证通过，核验{receipt["verified_manifest_files"]}个campaign清单文件；完整传输清单另覆盖日志、验证结果和脚本。运行时24551文件、5823470108字节重新逐一核验；原始模型、项目和既有运行时保持一致。',
'- 本轮仍使用上轮验证过的CPU暂存实现，保持张量布局；没有修改共享agent源码、模型算法或超参数。','',
'## 结论范围与保留问题','',
'这些结果说明该固定预算下的终止处理、连续R1训练、保存和重载链路可用；不是长期无泄漏保证，不证明策略生存能力提升，也不证明state2/R2或类器官效果。正式增益需独立种子、匹配对照与评估预算。',
f'历史 `.15`→`.16` probe差异仍为{s["cross_host_probe_error"]:.17g}，跨机probe仍判失败，没有放宽1e-6门槛；当前PPO数据全部在`.16`新采样，旧机概率未进入更新。',
'本工作区无Git仓库，按登记采用独立目录、只增不覆盖的检查点和SHA256清单留档；没有Git历史回退。','',
'## 证据索引','',
'- `download_verification.json`：本地接收与全量门槛核验。',
'- `evidence/terminal01/`：真实终止轨迹、检查点与独立重放。',
'- `evidence/stability01/`：8轮逐条轨迹、统计、模型与优化器状态。',
'- `evidence/runtime_audit.json`、`execution_environment.json`、`evidence_manifest.json`：环境、进程身份与文件清单。',
'- `progress.md`、`final_review.md`：执行记录与独立代码/证据评审。','']
with (R/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
print('report.md written')
