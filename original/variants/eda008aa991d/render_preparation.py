"""Render the unarmed nominal waveform and candidate recording schedule."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'.tools/mea_baseline_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

out=Path(__file__).resolve().parent/'hardware_preparation'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['axes.unicode_minus']=False
signal=json.loads((out/'waveforms.json').read_text())['H']
fig,(ax,cycle)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[1.3,1]})
times=[-1.];values=[0.]
for seg in signal['segments']:
    start=seg['start_frame']/20
    times.extend([start,start+seg['duration_frames']/20]);values.extend([seg['voltage_mv']]*2)
times.append(23.);values.append(0.)
ax.step(times,values,where='post',color='#3464d9',linewidth=2)
ax.axhline(0,color='#8893a5',lw=.8)
ax.set_xlim(-1,23);ax.set_ylim(-260,260)
ax.set_title('候选刺激波形：每组3脉冲，脉冲起点间隔10ms',fontproperties=font,fontsize=15,pad=12)
ax.set_xlabel('时间（ms）',fontproperties=font);ax.set_ylabel('标称电压（mV）',fontproperties=font)
ax.text(.98,.94,'每相200μs；H/W/F逻辑位点使用同一波形\n实际电极、DAC码值、极性和输出尚未绑定',ha='right',va='top',transform=ax.transAxes,fontproperties=font,color='#5a6474')
ax.grid(alpha=.15)
cycle.broken_barh([(0,20)],(.1,.55),facecolors='#b8c6dd')
cycle.broken_barh([(12,1)],(.85,.55),facecolors='#3a997d')
cycle.scatter([.0204],[1.75],marker='v',color='#d17b36',s=90)
cycle.scatter([2.0204,10.0204],[1.75,1.75],color='#7955b9',s=45)
cycle.text(.2,1.82,'刺激结束 20.4ms',fontproperties=font,fontsize=10)
cycle.text(2.2,1.58,'恢复检查',fontproperties=font,fontsize=10)
cycle.text(10.2,1.58,'恢复检查',fontproperties=font,fontsize=10)
cycle.text(12.5,1.12,'state2\n12–13s',ha='center',va='center',fontproperties=font,color='white',fontsize=9)
cycle.text(10,.38,'连续采集 raw + spikes + events + 设备帧号',ha='center',va='center',fontproperties=font)
cycle.set_xlim(0,20);cycle.set_ylim(0,2.2);cycle.set_yticks([])
cycle.set_xlabel('一个候选反馈周期内的时间（s）',fontproperties=font)
cycle.set_title('候选周期至少20秒；无刺激轮采用同位置的采集窗口',fontproperties=font,fontsize=13,pad=12)
fig.text(.5,.015,'准备图，不是实测波形。12–13秒窗口仍需当前样本恢复质控；本次未发送任何真实刺激。',ha='center',fontproperties=font,color='#7c4c31')
fig.tight_layout(rect=[0,.045,1,1])
fig.savefig(out/'stimulus_and_capture.png',dpi=170)
fig.savefig(out/'stimulus_and_capture.svg')
print(str(out/'stimulus_and_capture.png'))
