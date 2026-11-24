"""Publish a measured status only after all independent evidence is present."""
import json
from pathlib import Path
folder=Path(__file__).resolve().parent;root=folder.parents[1]
a=json.loads((folder/'independent_audit.json').read_text())
r=json.loads((folder/'registration.json').read_text())
assert a['passed']
for name in ('replay_verification.json','token_verification.json','render_verification.json'):
    assert json.loads((folder/name).read_text())['passed'],name
assert 'README.md' not in r['source_hashes'] and 'docs/CURRENT_STATE.md' not in r['source_hashes']
entry=f'''## 新登记：去填充与 KV cache 深度优化

已按用户选择从 `checkpoint_0032` 权重初始化，登记独立 `kv_attempt2`；重置优化器并使用新 rollout。登记绑定 102 个源文件、18 个模型输入文件，SHA-256 为 `{r['registration_sha256']}`。原 72 个冻结核心及第一版 92 个绑定文件保持原字节。

节点 5 限额对比中，原 B4×1536 三轮采样加 PPO/critic 总时间为 {a['totals']['baseline']:.3f} 秒，新候选为 {a['totals']['optimized']:.3f} 秒，合计加速 **{a['total_speedup']:.3f} 倍**，逐轮中位数加速 **{a['median_speedup']:.3f} 倍**；相对仅去填充 compact 额外加速 {a['speedup_vs_compact']:.3f} 倍。包含测量内首轮开销，各臂均完成 384 个真实决策、24 次 actor 更新；未减少 epochs 或触发早停。该值是短程采样与更新吞吐，不含模型加载、单独概率检查、正式 checkpoint 写盘及完整评估。

实现合并完整 verb 的 KV 与 target 续算、LoRA 投影缓存、同输入并行投影及融合 RMSNorm/SiLU。新 target 提示格式和浮点顺序会改变旧布局的动作概率，因此不能复用旧 rollout。新布局自己的逆序、分组、梯度模式及更新前概率复算误差为 0；更新后缓存重建一致。1,152 条新轨迹已独立 Godot 重放并重算实际 token。172 项 Python 回归中 169 项通过、3 项条件跳过，Godot 导入及 10 组测试通过。

第一版因合计只有 1.906 倍而未通过，失败结果完整保留。修订版详细结果、真实样例与示意图见[深度优化报告](../output/rl_kv2_20261107/report.html)，协议见 [M8X_KV2_PROTOCOL](M8X_KV2_PROTOCOL.md)。**正式训练仍未启动，没有新 checkpoint，未切换服务。** 短更新稳定不代表长期收敛或生存率达标；原冻结的正式保存/评估入口未自动切换到新布局。

'''
p=root/'docs/CURRENT_STATE.md';s=p.read_text(encoding='utf-8')
assert '## 新登记：去填充与 KV cache 深度优化' not in s
s=s.replace('## 速度优化诊断',entry+'## 先期速度优化诊断',1);p.write_text(s,encoding='utf-8')
p=root/'README.md';s=p.read_text(encoding='utf-8')
old=next(line for line in s.splitlines() if line.startswith('- 本次研究：'))
s=s.replace(old,f'- 新实验：[去填充与 KV cache 深度优化报告](output/rl_kv2_20261107/report.html)，节点 5 短程采样＋更新合计加速 {a["total_speedup"]:.3f} 倍；从 checkpoint_0032 初始化，新建优化器及 rollout，尚未启动正式训练或切换服务。前置资料：[算法与真实样例](output/rl_report_20261107/report.html)、[仅去填充诊断](output/rl_speed_20261107/report.html)。')
s=s.replace('`output/rl_report_20261107` 和 `output/rl_speed_20261107` 保存之后新做的诊断证据','`output/rl_report_20261107`、`output/rl_speed_20261107`、`output/rl_kv_20261107` 与 `output/rl_kv2_20261107` 保存之后的新诊断、登记及性能证据')
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(updated=['README.md','docs/CURRENT_STATE.md'],formal_training_started=False)))
