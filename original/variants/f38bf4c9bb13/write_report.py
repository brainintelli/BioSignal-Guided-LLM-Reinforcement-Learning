"""Render exact-scope report from locally verified immutable evidence."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence';O=E/'joint03';G=2**30
def read(p):return json.loads(p.read_text(encoding='utf-8'))
v=read(R/'download_verification.json');assert v['passed']
root=read(O/'report.json');ind=read(O/'independent_verification.json');tail=read(E/'tail01/report.json');fixture=read(E/'tail_fixture.json')
ranks=[[read(O/f'rank_{r:02d}/round_{n:02d}/report.json') for n in range(1,9)] for r in range(8)]
iv=[r for rank in ind['ranks'] for r in rank['rounds']]
errors={k:max(r[k] for r in iv) for k in ['old_error','new_error','bootstrap_error']}
summary=[]
for n in range(8):
    ss=[rs[n] for rs in ranks];s=ss[0]
    summary.append(f"| {n+1} | {sum(x['ticks'] for x in ss)} | {sum(len(x['terminal_indices']) for x in ss)} | {s['metrics']['optimizer_steps']} | {s['metrics']['critic_optimizer_steps']} | {max(x['seconds'] for x in ss):.1f} | {max(x['resources']['peak_allocated'] for x in ss)/G:.3f} | {max(x['resources']['peak_reserved'] for x in ss)/G:.3f} |")
memory=[]
for rank,rs in enumerate(ranks):
    gate=read(O/f'rank_{rank:02d}/report.json')['resource_gate']
    memory.append(f"| {rank} | {max(x['resources']['rss_before_host_cache_release'] for x in rs)/G:.3f} | {max(x['resources']['rss'] for x in rs)/G:.3f} | {max(x['resources']['peak_rss_process'] for x in rs)/G:.3f} | {gate['rss_growth_bytes']/G:.3f} | {gate['gpu_growth_bytes']/2**20:.3f} |")
ee=[json.loads(s) for s in (E/'tail01/rollout.jsonl').read_text(encoding='utf-8').splitlines()];last=ee[-1]['training']
telemetry=[json.loads(s) for s in (E/'joint03_gpu_telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
text=f'''# 22554 B线：批末实际终止与8卡联合训练验证

本轮在192.0.2.16完成批末恰好终止的实际验证，以及8张RTX4090上的同步R1-only训练。无刺激，R2、state2均关闭，alpha=1；使用原M9 state1、108维critic和原始checkpoint_0064。没有连接MaxOne或新增类器官记录。

## 验收结果

- 批末验证：{tail['decisions']}条新模型动作，只有最后一条自然死亡；训练、保存重载和独立进程原生回放均通过。
- 8卡联合训练：8轮×128条全局新决策，共{root['global_decisions']}条、{root['global_ticks']} tick、{root['global_terminals']}次原生死亡。累计{root['joint_actor_steps']}次共同actor、{root['joint_critic_steps']}次共同critic步骤；不能再乘以8。
- 采样、更新、逐轮保存与收集检查合计{root['seconds']/60:.2f}分钟，不含随后独立进程验证。8个rank每轮模型、critic与优化器状态哈希完全一致。
- 8个新进程分别在原采样GPU上核验全部64个rank/轮；逐条原生转移、R1、输入身份、GAE、检查点及优化器连续性通过。最大统计误差为{errors}，门槛仍为1e-6。
- 最终验收运行未出现OOM、NaN或崩溃；LoRA与critic每轮更新，冻结基座及参考模型不变。全部8个rank通过预先登记的资源筛查。

## 批末实际终止

固定原生世界种子{fixture['world_seed']}，先执行{fixture['prefix_steps']}次合法等待构造临近死亡状态；没有注入生命值或终止标志。准备前缀不进入PPO。探索阶段选定并冻结4条长度、模型初始化后的CPU/CUDA RNG和预期动作后，正式采集重新运行；正式批次没有遇到死亡就事后缩短。

前3条模型动作未终止，第4条触发原生死亡，批内共{tail['ticks']} tick。最后一条reward={last['reward']:.12f}，value={last['value']:.12f}，advantage={last['advantage']:.12f}，return={last['return']:.12f}。
终止bootstrap为0，prepared为None；用会立即报错的模型回调验证了终止分支不会准备或估值死亡/重置观测。将外部bootstrap换成1e9，全部4条GAE/return保持一致，包括死亡前3条。之后实际reset并采样1条新动作，单独留档且不进入该PPO批次。独立进程重放前缀、4条动作及reset，旧/新模型统计误差均为0。

## 8卡实现及逐轮结果

每张卡运行独立原生世界、每轮采16条；全局128条，优势在全体128条上归一化。每个共同minibatch由各rank的4条构成，共32条；NCCL将梯度求均值后统一裁剪和更新，全局KL控制共同提前停止。采样行保留在原采样GPU计算旧概率。所有8个物理GPU UUID、PID与nvidia-smi实占用均留证，监测从本轮启动覆盖至所有收集进程结束，共{len(telemetry)}次采样。

8rank实际数学校验覆盖全局归一化、均值梯度、裁剪AdamW、局部缺失梯度、共同KL停止及集体拒绝；均值梯度与集中式参考最大误差2.842170943040401e-14。每轮只有rank0保存一份共同检查点，其余rank重载并复核本卡统计。世界跨更新继续，仅原生终止后重置；优化器对象跨8轮保留。

| 轮次 | 全局tick | 死亡 | actor步骤 | critic步骤 | 最慢rank秒 | 最大peak allocated GiB | 最大peak reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary)}

8个世界和每世界16条的GAE段改变了样本分布与截断跨度。本轮没有测试与单卡长轨迹的逐位等价或匹配吞吐基准，不能把耗时差直接报告为加速比，也不说明策略能力提升。

## 内存与失败证据

资源门槛始终为：相对第2轮，第3–8轮轮末GPU allocated增长不超过512 MiB、RSS增长不超过4 GiB。下表分别列出清理前、清理后的轮末RSS以及进程历史峰值；后者含瞬时CPU暂存。

| rank/GPU | 最大清理前RSS GiB | 最大清理后RSS GiB | 进程峰值RSS GiB | 清理后RSS最大增长 GiB | GPU allocated最大增长 MiB |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(memory)}

最终运行在每轮CUDA工作结束并释放张量后，调用经过实际回归验证的固定版本锁页缓存释放接口。独立回归中，释放192 MiB锁页张量后残留RSS从约192 MiB降至约0.55 MiB，仍存活张量和异步复制正确；联合回归8/8通过。实际训练记录保留清理前/后RSS及进程peak RSS，没有改低原始采样值或放宽门槛。
该接口为私有torch._C._host_emptyCache，精确绑定已核验的Torch2.6.0+cu124；升级运行时必须重新验证。它释放空闲缓存，代价是减少后续分配复用。对应实现见[PyTorch v2.6.0官方CUDA源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/csrc/cuda/Module.cpp)；NCCL后端参见[官方分布式文档](https://docs.pytorch.org/docs/2.6/distributed.html)。

保留的失败尝试：scout01因模型先吃食物而未找到指定终止夹具；joint01在首次更新后散列AdamW标量step时报错，未验收检查点；joint02完成全部8轮且数值/同步检查通过，但CPU锁页缓存保留导致RSS超出4 GiB增长门槛，因此整体判失败。分别保留当时源码、日志、轨迹和joint02全部检查点。scalar哈希回归2/2通过后重新启动；锁页缓存回归2/2通过后，joint03再次从原始权重、全新优化器和世界启动，才得到本报告的验收结果。两个早期监控尝试的失败日志也保留。

GPU0上仍有其他7个worker各约408 MiB的辅助CUDA上下文；更早选择device没有消除它们。主模型确实分别运行在8张卡并共同更新。按卡的全量nvidia-smi占用与每rank分配器峰值均有留档，两者不可混为一项。

## 模型与交付

- 服务器工作区：`/home/researcher/mea-r2-22554/eightgpu_20261116_01`。
- 最终共同检查点：`/home/researcher/mea-r2-22554/eightgpu_20261116_01/joint03/checkpoint_08`。
- 最终actor语义SHA256：`{root['final_actor']}`。
- 原始共同起点保持不变，原actor语义SHA256为`94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0`；本轮不接续终止测试或失败尝试的权重。
- 本地文件清单核验{v['verified_files']}项通过，完整归档及传输SHA256均核验。运行时24551文件、5823470108字节重新逐一校验。
- 逐轮共同检查点保存LoRA、critic、优化器和rank0 RNG；每rank额外保存各轮RNG。已验证权重重载、落盘优化器与连续状态的一致性；没有实现崩溃后自动恢复全部世界并续训。
- 原始模型、既有运行时、共享agent源码及A线工作均未修改。工作区无Git，使用独立目录与SHA256清单留档。

## 结论范围

本轮补齐的是批末实际终止处理、8卡共同更新、固定1024条预算下的资源稳定性和独立保存重载验证。它不证明无限期无泄漏、学习增益或类器官/R2效果。历史`.15`到`.16`的probe差异0.074654579162597656仍判失败，未放宽1e-6；本轮PPO全部使用`.16`新采样概率。22554既有基线严格质控失败和R2候选限制不因本次软件测试而改变。

独立审查通过，无需修复的Critical、Important或Minor问题。审查者重新核验1119项文件和全部轨迹/资源检查、归档SHA256，并运行边界测试3/3通过。joint03本身未发生自然死亡或KL提前停止；自然死亡由单独的实际批末测试覆盖，KL协调由实际8rank数学测试及控制流审查支持，不能声称本次训练触发过这两个分支。详见[独立审查](final_review.md)。

## 证据索引

- `download_verification.json`、`transport_receipt.json`：本地全量检查与传输收据。
- `evidence/tail01/`、`evidence/tail_fixture.json`：批末终止夹具、正式轨迹、独立回放。
- `evidence/joint03/`：8卡、8轮完整证据与共同检查点。
- `evidence/joint03_gpu_telemetry.jsonl`：全部卡的进程、利用率与占用采样。
- `evidence/joint01/`、`evidence/joint02/`及根目录日志/历史源码：失败证据。
- `evidence/runtime_audit.json`、`evidence/evidence_manifest.json`：运行时与交付文件核验。
- `progress.md`、`final_review.md`：执行记录与独立审查。
'''
with (R/'report.md').open('x',encoding='utf-8') as f:f.write(text)
print('report.md written from verified evidence')
