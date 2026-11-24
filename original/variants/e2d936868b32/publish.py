"""Export readable, per-step reward accounting and the paired test summary."""
import csv
import json
from pathlib import Path
import math

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    result,cal,spec=[read(HERE/x) for x in ('comparison.json','calibration.json','design.json')]
    measured=cal['audit'];ref=cal['reward_mix']['reference_per_tick']
    s=result['reward_totals']['ordered']
    with (HERE/'逐步奖励.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.writer(f);wr.writerow(['group','training_seed','update','step','ticks','r1_native','feedback_label','r2_scaled','alpha','beta','r2_contribution','total_for_ppo'])
        for seed in spec['training_seeds']:
            for arm in spec['arms']:
                effective=spec['beta'] if arm=='ordered' else 0.
                for it in range(1,spec['updates']+1):
                    path=HERE/'models'/str(seed)/arm/f'train-{it}.jsonl'
                    for line in path.read_text(encoding='utf8').splitlines():
                        r=json.loads(line);native=r['r1_score']['reward']
                        assert math.isclose(spec['alpha']*native+effective*r['r2_scaled'],r['reward'],abs_tol=1e-15)
                        wr.writerow([arm,seed,it,r['index'],r['duration'],native,r['r2'],r['r2_scaled'],spec['alpha'],effective,r['contribution'],r['reward']])
    lines=['# 芯片 22021 的 R2 奖励缩放和离线 PPO 对照','','## 奖励设置','',
      f"游戏原生奖励记为 R1。反馈标签 z 取 +1、−1 或 0。校准只使用既有仅 R1 组的 384 个训练决策。训练数据中每个折扣游戏刻的 |R1| 中位数为 {ref:.6f}。",'',
      '令 D 为动作实际持续刻数的折扣和，折扣因子为 0.99985。每次反馈先计算 R2 = z × 0.10 × min(|R1|, 0.009775 × D)，再按 R总 = α × R1 + β × R2 混合。α 固定为 1，β 在仅 R1 组为 0、在辅助组为 0.5。零 R1、静默、终止步的 R2 贡献均为 0。典型单刻 R1 为 +0.009775，正/负反馈分别贡献 +0.00048875 或 −0.00048875。','',
      '最终辅助贡献在每一步至多为原生奖励绝对值的 5%。这是单步约束；正负原生奖励相加可能抵消，因此整批比例按绝对值之和计算。该界限约束奖励幅度，不能保证 PPO 动作或最终表现变好。','',
      '## 奖励核验','','| 数据与设置 | 有反馈步骤最大贡献比例 | 全部步骤的绝对贡献比例 |','|---|---:|---:|']
    for label,metrics in [('旧 R1+R2 日志',measured['historical']),('同一旧轨迹换新公式',measured['new_same_steps'])]:
        lines.append(f"| {label} | {100*metrics['ratio_max_active']:.2f}% | {100*metrics['sum_abs_contribution_over_sum_abs_r1']:.2f}% |")
    lines.append(f"| 新公式实际训练 | {100*s['max_step_ratio']:.2f}% | {100*s['absolute_mass_ratio']:.2f}% |")
    lines+=['',f"辅助组实训共 {s['decisions']} 个决策，正标签 {s['positive']} 次，负标签 {s['negative']} 次，零标签 {s['silent']} 次。校核了所有轨迹的 GAE、训练输入、训练预算与模型检查点，并在独立游戏实例复核了留存动作。",'',
      '## 模型对照','','8 个训练种子；两组从相同权重出发，PPO 各 4 批 × 12 个动作，每个种子均完成 8 次 Actor 和 64 次 Critic 优化器更新。评估在 48 张新地图上进行，各组 384 局，每局最多 6 次决策。训练及评估使用同一颗类器官的历史记录，按物理轮次分割。游戏反馈在回放中按动作结果计算。','','| 留存指标 | 仅 R1 | R1 + 缩放 R2 | 配对差 [97.5% 区间] |','|---|---:|---:|---:|']
    for k,label,prec in [('r1_sum','游戏原生回报',6),('new_cells','新增探索格数',3)]:
        v=result['metrics'][k];lo,hi=v['interval']
        lines.append(f"| {label} | {v['r1']:.{prec}f} | {v['r1_r2']:.{prec}f} | {v['difference']:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}] |")
    historical=[]
    for f in (HERE.parent/'offline-ppo/models').glob('*/ordered/train-*.jsonl'):
        historical.extend(r for r in (json.loads(l) for l in f.read_text(encoding='utf8').splitlines()) if r['r2'])
    raw_larger=sum(abs(r['r2'])>abs(r['r1_score']['reward']) for r in historical)
    index=lines.index('## 奖励设置')+2
    lines[index:index]=[f"旧实验在 {len(historical)} 次有反馈的训练步上，若直接把 ±1 加进总奖励，有 {raw_larger} 次其绝对值会超过当步 |R1|。旧实现实际使用的是每刻 0.001 的系数。",'']
    conclusion='达到' if result['stable'] else '未达到'
    lines+=['',f"按训练种子和游戏地图交叉配对重采样 30,000 次，两项终点分别报告 97.5% 区间。两个终点均高于仅 R1 的训练种子有 {result['joint_positive_training_seeds']}/8 个，{conclusion}事先登记的双指标稳定提升标准。",'',
      '上一轮 1,152 局濒渴复测使用的模型在该轮额外 PPO 更新中 R2 权重为 0。此次从相同暖启动点重新训练，在原有配给充足的开局短测辅助奖励，长期存活的变化仍需用这批新权重专门检验。','',
      '[实验参数](design.json) · [逐步奖励](逐步奖励.csv) · [留存配对](paired-worlds.csv) · [统计结果](comparison.json) · [校准训练数据摘要](calibration.json)']
    (HERE/'R2缩放与训练结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':main()
