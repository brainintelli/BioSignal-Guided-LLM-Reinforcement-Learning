from pathlib import Path
import json, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parent
d=root/'multipoint_evidence'
s=json.loads((d/'summary.json').read_text())
q=json.loads((d/'point_01_train.qc.json').read_text())
state=json.loads((d/'state.json').read_text())
z=np.load(d/'traces.npz')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
 'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(3,2,figsize=(14,11),layout='constrained')
blue='#147d92';red='#b74848';orange='#d58a36'
a=ax[0,0];a.plot(z['trace_t']*1000,z['trace_uv']/1000,lw=.7,color=blue)
for t in z['event_t']:a.axvline(t*1000,color=orange,lw=.7,alpha=.65)
a.set(title='A｜实际记录波形：刺激电极 14220',xlabel='相对首个事件（ms）',ylabel='记录端信号（mV）')
a.text(.98,.98,'橙线为事件；记录端饱和不能用于校准输出电压',ha='right',va='top',transform=a.transAxes,fontsize=8)
a=ax[0,1];actual=np.diff(z['event_t'])*1000;expected=np.array([10 if i%3!=2 else 180 for i in range(14)])
a.plot(np.arange(1,15),actual,'o',color=blue,label='实际相邻事件间隔')
a.plot(np.arange(1,15),expected,'_',color=orange,ms=13,label='设定间隔')
a.set(title='B｜15 个事件时序符合设定',xlabel='相邻事件序号',ylabel='间隔（ms）');a.legend(frameon=False,fontsize=8)
a.text(.02,.78,'组起点相隔 200 ms；\n上一组末脉冲到下一组首脉冲相隔 180 ms',transform=a.transAxes,fontsize=9)
a=ax[1,0];vals=[x['fraction_channels_passed']*100 for x in q['recovery']]+[q['final_recovery']['fraction_channels_passed']*100]
display_vals=[s['posthoc_stability']['pre_vs_pre']['fraction_channels_passed']*100]+vals
bars=a.bar(['刺激前互比\n（事后诊断）','末脉冲后 2 s','末脉冲后 10 s','文件结束前'],display_vals,color=['#86929a',red,red,red],width=.6)
a.axhline(95,color=orange,ls='--',label='预设门槛 95%');a.set_ylim(0,105)
for b,v in zip(bars,display_vals):a.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}%',ha='center')
a.set(title='C｜恢复判据未通过；基线互比也未达标',ylabel='共同通过全部条件的通道（%）');a.legend(frameon=False,fontsize=8)
a=ax[1,1];trial=q['trials'][0];v=[trial['pre_rate_hz'],trial['post_rate_hz']]
b=a.bar(['首脉冲前 −5 至 −2 s','末脉冲后 +2 至 +5 s'],v,color=[blue,orange],width=.55)
for bar,val in zip(b,v):a.text(bar.get_x()+bar.get_width()/2,val+6,f'{val:.0f}',ha='center')
a.set_ylim(0,max(v)*1.28);a.set(title='D｜短窗口放电变化仅作描述',ylabel='固定 ROI 总放电率（spikes/s）')
a.text(.5,.98,f"{(v[1]/v[0]-1)*100:+.1f}%｜单串刺激、无假刺激对照、QC 未通过",ha='center',va='top',transform=a.transAxes,fontsize=9)
a=ax[2,0];a.plot(z['train_rate_t'],z['train_rate_hz'],color=blue,lw=1)
a.axvspan(0,.8204+2,color=orange,alpha=.18,label='刺激与早期伪迹区间')
a.axhline(q['reference_rate'],color='gray',ls='--',label='先前基线参考 191.25')
a.set(title='E｜设备检测计数（刺激窗口含伪迹）',xlabel='相对首个事件（s）',ylabel='检测计数/s（1 秒分箱）');a.legend(frameon=False,fontsize=8)
a=ax[2,1];r=s['recovery_30s_rates'];a.plot(np.arange(len(r))*30+15,r,'o-',ms=3,color=blue)
a.axhline(q['reference_rate'],color='gray',ls='--',label='先前基线参考 191.25')
a.set(title='F｜追加无刺激恢复记录',xlabel='恢复文件内时间（s；与前文件有间隙）',ylabel='spikes/s（30 秒分箱）');a.legend(frameon=False,fontsize=8)
fig.suptitle('28073｜真实刺激已执行；首点质控停止后续刺激',fontsize=19,weight='bold')
fig.supxlabel('1/20 个点 · 15/300 个脉冲 · 无假刺激 · R2 关闭 · 未进入模型训练\n描述性记录，不构成指标改善或生物计算效果证据',fontsize=11)
fig.savefig(root/'multipoint_results.png',dpi=160)
fig.savefig(root/'multipoint_results.svg')

def cst(t):return datetime.datetime.fromtimestamp(t,datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
recovery=state['qc']['recovery_recovery']
rows=[]
for name,v in s['posthoc_stability'].items():
 rows.append(f"| {name} | {100*v['fraction_channels_passed']:.1f}% |")
record_rows=[]
for v in s['records']:
 record_rows.append(f"| `{Path(v['path']).name}` | {v['size_bytes']:,} | `{v['sha256']}` |")
report=f'''# 28073 真实刺激能力测试记录

已执行第 1 个点，15 个真实脉冲；随后恢复质控失败，剩余 19 个点没有执行。未进行假刺激，R2 始终关闭，模型训练未启动。

![实际对比图]({root.as_posix()}/multipoint_results.png)

## 执行与停止

- 芯片 28073；采集机 192.0.2.21；临时类器官标识 `TEMP-28073-20261117T061300Z`（未取得独立生物样本编号）。
- 本枚类器官计时起点 2026-11-17 14:13:00 +08:00，绝对截止 16:13:00；本次程序 {cst(s['started_wall'])} 至 {cst(s['finished_wall'])}，未重置累计计时。
- 第 1 点：电极 14220、通道 831。3 脉冲 × 5 组；脉冲起点间隔 10 ms，组起点间隔 200 ms。末脉冲结束距首脉冲起点 820.4 ms。
- 命令幅值每相 298.1901 mV（目标 300 mV 的 DAC 量化值），每相 200 μs，双相后回零。命令值不是电极负载下实测电压。
- 累计保守占用 15 个脉冲；实际 H5 有 15 个事件，全部时间偏移符合设定。存在刺激锁定伪迹；密集串内伪迹重叠，不据此逐个独立校准脉冲电压。
- 选中通道最后一次 ADC 触顶位于末脉冲结束后 {q['selected_last_rail_relative_final_pulse_end_s']*1000:.2f} ms。该项通过，固定读出通道的整体恢复项未通过。
- 刺激文件实际长 {q['duration_seconds']:.2f} s，覆盖末脉冲后 {q['actual_post_end_seconds']:.4f} s；另存无刺激恢复 {s['recovery_metadata']['duration_seconds']:.2f} s。后者覆盖检查状态 `{recovery['status']}` 仅证明连续时长/元数据/无事件，不等同于噪声或生物状态恢复通过。
- 控制程序最终状态 `{s['status']}`；清理错误 `{s['cleanup_errors']}`。关停后独立检查确认 32 路刺激单元关闭命令均收到回执、DAC 归零命令收到回执、进程已退出、MaxLab 服务无 H5 文件句柄。完整原始文件与设备调用审计保留于采集机。

## 结果怎么解释

固定 ROI 为预先选定的 372 通道，排除所有 20 个候选刺激通道。预设恢复判据为至少 95% 通道同时满足：MAD 比 0.67–1.5、基线偏移 ≤5 ADC、非零 MAD、前后均无 ADC 触顶。

| 检查窗口 | 通过通道 | 通过比例 |
|---|---:|---:|
| 末脉冲后 +2 s | {sum(q['recovery'][0]['channel_pass'])}/372 | {vals[0]:.1f}% |
| 末脉冲后 +10 s | {sum(q['recovery'][1]['channel_pass'])}/372 | {vals[1]:.1f}% |
| 文件末端 | {sum(q['final_recovery']['channel_pass'])}/372 | {vals[2]:.1f}% |

上述检查的 ROI 窗口均未见 ADC 触顶。失败主要涉及 MAD 比变化和基线偏移；单凭这次结果不能区分自然波动、采集漂移和刺激影响，也不能推出组织损伤。原判据和停止结果保留，没有通过调阈值继续刺激。

3 秒诊断窗口放电率从 {trial['pre_rate_hz']:.0f} 升至 {trial['post_rate_hz']:.0f} spikes/s（{(trial['post_rate_hz']/trial['pre_rate_hz']-1)*100:+.1f}%）。但末脉冲后两个 30 秒窗为 {q['post_end_30s_rates'][0]:.2f}、{q['post_end_30s_rates'][1]:.2f} spikes/s；先前基线参考 {q['reference_rate']:.2f}。这些窗口位置和时长不同，不应把它们当作重复试验。只有 1 串刺激、1 枚样本、无假刺激对照，且质控失败，因此不判定弱阳性、持续改善、探索欲或恐惧等效应。

## 事后诊断（不改变本次停止判定）

| 对比 | 通过比例 |
|---|---:|
{chr(10).join(rows)}

刺激之前两个 1 秒窗口互比，通过率也只有 {100*s['posthoc_stability']['pre_vs_pre']['fraction_channels_passed']:.1f}%，低于同一 95% 判据。首个恢复窗中有 {int(np.isclose(np.array(q['recovery'][0]['mad_ratios']),2/3).sum())} 个通道的 MAD 比为 2/3，略低于固定下限 0.67；ADC 离散量化会影响此判定。这提示本次门槛对自身基线也不稳定，不能把“质控失败”直接解释成“刺激造成损伤”。需要在新的刺激前独立校准基线窗口与噪声判据；没有事后放宽阈值或补发剩余脉冲。

以上按相同数值规则检查采样窗口，属于事后定位，未进行新的刺激或参数修改。恢复文件与刺激文件之间存在真实的关档/分析/重开间隙。

## 保存与训练

原始文件根目录：`/home/researcher/mea_28073_live_20261117_1413/multipoint_run/point_01/`。本地保存完整 JSON 质控、计划、事件、调用审计、源文件哈希和图表所用波形摘录；未把波形摘录冒充完整 raw 文件。

| 文件 | 字节 | SHA256 |
|---|---:|---|
{chr(10).join(record_rows)}

本次记录进入保留待审状态，标记 `quarantine_failed_recovery_QC`，不作为合格 state 或正向 reward 自动进入模型训练。R2、辅助生物奖励和在线学习均未开启。本轮没有模型提升结果。
'''
(root/'28073_MULTIPOINT_REPORT.md').write_text(report,encoding='utf-8')
(root/'training_exclusion.json').write_text(json.dumps({'schema':'mea.training.disposition.v1','session_id':s['session_id'],
 'status':'quarantine_failed_recovery_QC','training_started':False,'r2_enabled':False,
 'reason':q['errors'],'records':s['records']},indent=2),encoding='utf-8')
print('Wrote figure, report and explicit training exclusion')
