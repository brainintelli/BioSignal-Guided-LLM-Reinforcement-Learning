from pathlib import Path
import csv,json,math
from datetime import datetime,timedelta,timezone
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from livepair.figures import _plot_style
import numpy as np

def run(p):
 r=p/'records'
 read=lambda f:json.loads(f.read_bytes())
 a=read(r/'factorial-results.json');s=read(p/'protocol.json');reg=read(p/'session-route.json')
 warm=read(r/'warmup.json');end=read(r/'end.json')
 labels={'anchor_regular':'8418 · 等间隔','anchor_jitter':'8418 · 变间隔','new_regular':'11972 · 等间隔','new_jitter':'11972 · 变间隔','sham':'零输出'}
 effects={'position':'位点（11972 减 8418）','rhythm':'节律（变间隔减等间隔）','interaction':'两点的节律差之差'}
 colors=['#226880','#7ba8b4','#a25034','#ce936e']
 g=a['groups'];eff=a['effects']
 def clock(t):return datetime.fromtimestamp(t,timezone(timedelta(hours=8))).strftime('%H:%M:%S')
 with _plot_style():
  fig=Figure(figsize=(11,5.1),dpi=155,layout='constrained')
  ax,bx=fig.subplots(1,2)
  for k,color in zip(list(labels)[:4],colors):
   ax.plot(range(1,5),g[k]['cpp']['vs_sham'],label=labels[k],color=color,marker='o',lw=1.6)
  ax.axhline(0,c='gray',ls=':',lw=1);ax.set(xticks=range(1,5),xlabel='随机轮次',ylabel='相对同轮零输出的尖峰数差（个/5 秒）',title='每轮的刺激响应')
  ax.legend(fontsize=8,ncol=2)
  for j,(src,name,color) in enumerate([('cpp','C++ 返回流','#226880'),('h5','H5 设备帧','#bd7954')]):
   y=[eff[k][src]['mean'] for k in effects]
   bx.bar(np.arange(3)+(j-.5)*.34,y,width=.34,label=name,color=color)
  bx.axhline(0,c='gray',lw=1);bx.set(xticks=range(3),xticklabels=['新点减原点','变间隔减等间隔','交互作用'],ylabel='平均配对差（个/5 秒）',title='位置、节律与交互作用');bx.legend(fontsize=8)
  fig.suptitle('芯片 28080 · 同一新样本的四轮位点与节律对照');FigureCanvasAgg(fig).print_png(str(p/'位点与节律响应.png'))
 lines=['# 芯片 28080 · 位点与节律对照结果','',
 f"同一颗样本在 {clock(end['started'])}–{clock(end['finished'])} 完成四轮、20 次随机试验。每轮各包含两电极 × 两节律和一次零输出。原始波形复核了 16 次非零序列共 48 个脉冲，另有 4 次零输出。连同此前六轮，该样本共记录 28 次非零序列、84 个脉冲。结束后刺激单元关闭，主服务无打开的 H5。",
 '',f"芯片由新静默记录确认为 28080。8418 是原有刺激点，在此前六轮直接接受过 24 个脉冲。11972 此前在本样本的实验记录中没有直接接受过刺激。本轮两点各接受 24 个脉冲。两点相距约 658 µm。11972 在两个静默段中均通过筛选，且离已用输出点较远。16859 作为额外监测点，本次脉冲数为 0。每相设定约 199.79 mV、200 µs。等间隔条件在 0、200、400 ms 输出。变间隔条件保留首末起点，每轮两个位点共用同一中间起点。四轮中间起点依次为 "+'、'.join(map(str,s['jitter_mid_ms']))+' ms。',
 '',f"本次路由经静默 H5 核对，在 {a['roi']} 个上一轮也曾记录的非刺激活跃电极上比较响应。C++ 共记录 {a['cpp_rows']:,} 行尖峰。指标为首个事件后 2–7 秒减去事件前 6–1 秒的尖峰数，以同轮零输出衡量自发波动。",
 '','| 条件 | C++ 事件前 | C++ 事件后 | C++ 相对零输出 | 四轮中高于零输出 | H5 相对零输出 |','|---|---:|---:|---:|---:|---:|']
 for k,l in labels.items():
  c=g[k]['cpp'];h=g[k]['h5'];direction='—' if k=='sham' else f"{sum(x>0 for x in c['vs_sham'])}/4"
  lines.append(f"| {l} | {c['pre']:.1f} | {c['post']:.1f} | {c['mean_vs_sham']:+.1f} | {direction} | {h['mean_vs_sham']:+.1f} |")
 lines+=['','## 哪个因素与响应有关','',
 '| 比较 | C++ 四轮配对差（个/5 秒） | C++ 均差 | H5 均差 | C++ 校正 p |','|---|---|---:|---:|---:|']
 for k,l in effects.items():
  v=eff[k];series='、'.join(f'{x:+g}' for x in v['cpp']['blocks'])
  lines.append(f"| {l} | {series} | {v['cpp']['mean']:+.1f} | {v['h5']['mean']:+.1f} | {v['cpp']['p_holm']:.3f} |")
 same=[effects[k] for k in effects if eff[k]['consistent_direction']]
 lines+=['',('两种记录方式在四轮中方向均一致的比较为'+'、'.join(same)+'。' if same else '三项主比较均未达到两条记录流四轮全部同向的预设探索标准。'),
 '位点效应对两种节律取平均，节律效应对两个位点取平均。交互作用衡量新点的节律差减去原点的节律差。四轮来自同一颗类器官，C++ 和 H5 是同一批事件的两种读数。双侧符号检验即使四轮全同向，未校正 p 值最低也只有 0.125。三项主比较采用 Holm 校正。不同位点先前的直接刺激经历和对应刺激通道不同。',
 '', f"11972 使用变间隔时，C++ 相对同轮零输出的四次差值依次为 {'、'.join(f'{x:+g}' for x in g['new_jitter']['cpp']['vs_sham'])} 个尖峰/5 秒。平均为 {g['new_jitter']['cpp']['mean_vs_sham']:+.2f}，H5 复核平均为 {g['new_jitter']['h5']['mean_vs_sham']:+.2f}。两个记录方式都只有两轮高于零输出。这个组合可供另一颗样本复测，目前没有跨轮复现的新弱阳性。",
 '', '此前六轮中，8418 等间隔刺激相对零输出平均为 +54.3 个尖峰/5 秒，4/6 轮更高。本轮同一条件为 −5.75，1/4 轮更高。前后两次路由及纳入统计的通道数量不同（235 和 216），分别报告。两次结果合看，已有记录也未显示这一条件能稳定增加尖峰。[此前六轮报告](../本轮采集刺激结果.md)',
 '','![四轮波形后尖峰响应](位点与节律响应.png)','',
 f"采集机共保存 {a['raw_count']} 份本次的原始 H5，合计 {a['raw_bytes']:,} 字节。已逐份核对 SHA-256、设备事件和波形起点、分析窗尖峰数及 C++ 回执。" + ("工作区副本已通过同一套复算。[工作区复核](workspace-readback.json) " if (p/'workspace-readback.json').is_file() else "") + "R2 只记录，α=1。本轮测量神经响应，游戏表现仍需单独评估。",
 '','[实验登记](protocol.json) · [逐试验计数](records/response-by-trial.csv) · [全部比较](records/factorial-results.json) · [原始文件校验值](records/raw-hashes.json) · [结束回执](records/end.json)']
 (p/'本轮结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':
 import sys
 run(Path(sys.argv[1]))
