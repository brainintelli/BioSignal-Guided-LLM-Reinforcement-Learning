"""Describe sealed model inputs only; never inspect interim model outcomes."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parent
source=root.parent/'mea_stim_offline_20261117/data'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'
if sha(source/'manifest.json')!=pin:raise ValueError('Data manifest mismatch')
manifest=read(source/'manifest.json')
for name in ('metadata.json','before.json','after.json'):
    if sha(source/name)!=manifest['files'][name]['sha256']:raise ValueError('Source file mismatch')
data={p:read(source/(p+'.json')) for p in ('before','after')}
prior=read(source.parent/'input_summary.json');summary={}
for phase,rows in data.items():
    if len(rows)!=600:raise ValueError('Incomplete input')
    summary[phase]={}
    for name,lo,hi in [('train',0,384),('heldout',445,595)]:
        values=[mean(row['features'][j] for row in rows[lo:hi]) for j in range(4)]
        if any(not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip(values,prior[phase][name]['mean_features'])):
            raise ValueError('Independent input mean differs')
        summary[phase][name]=dict(windows=hi-lo,mean_features=values)
summary['source_manifest_sha256']=pin
summary['model_results_included']=False
summary['interpretation']='Input distribution only; no stimulus-causal or model-benefit inference'
(root/'input_distribution.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
labels=['事件数 / 秒','事件率 / 通道 / 秒','活跃通道比例（%）','平均绝对幅值（μV）']
for j,ax in enumerate(axes.flat):
    ax.axvspan(385,445,color='#ABB5C0',alpha=.18)
    ax.axvspan(445,595,color='#74B99E',alpha=.12)
    for phase,label,color in [('before','刺激前','#2878B5'),('after','刺激后','#D97721')]:
        vals=[r['features'][j]*(100 if j==2 else 1) for r in data[phase]]
        ax.plot(range(600),vals,color=color,alpha=.17,linewidth=.65)
        ax.plot([i+14.5 for i in range(0,600,30)],[mean(vals[i:i+30]) for i in range(0,600,30)],color=color,label=label,linewidth=2)
    ax.axvline(384,color='#777',linestyle=':',linewidth=.8)
    ax.axvline(445,color='#777',linestyle=':',linewidth=.8)
    ax.set_title(labels[j]);ax.grid(alpha=.15);ax.set_xlim(0,599)
    ax.set_xlabel('各阶段内窗口编号（每窗 1 秒）')
axes[0,0].legend(loc='upper right')
fig.suptitle('28080：训练输入信号的分布\n淡线为 1 秒原始特征，实线为仅用于展示的 30 秒均值',fontsize=15)
fig.text(.5,.01,'训练 0–383；384 为 bootstrap；灰区为隔离段；绿区为共同留存评估。刺激后从末脉冲结束 60 秒后取样。\n这是输入分布描述，不是模型收益或刺激因果效果。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.065,1,.93))
for ext in ('png','svg'):fig.savefig(root/('input_distribution.'+ext),dpi=180)
plt.close(fig)
print('Sealed inputs verified; model outcomes not loaded; plot written')
