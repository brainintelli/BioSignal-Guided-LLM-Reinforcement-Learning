import json,sys,math,hashlib
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt,font_manager
from PIL import Image
root=Path(__file__).resolve().parent
r=json.loads((root/'forward-result.json').read_text(encoding='utf8'))
s=r['summary'];data=r['rows'];p=np.array([a['verbs_prob'] for a in data]).reshape(2,16,8)
assert np.allclose(p.sum(axis=2),1,atol=1e-6) and r['same_input_repeat_exact'] and len({x['context_id'] for x in data})==1
assert all(q['value_error']<.0001 and q['abs_logprob_error']<.0001 for q in r['archived_first_replay'].values())
change=(p[1]-p[0])*100;tv=abs(change).sum(axis=1)/2
assert abs(float(tv.mean())-100*s['total_variation_mean'])<1e-8 and int(np.sum(p[0].argmax(axis=1)!=p[1].argmax(axis=1)))==s['greedy_verb_changed']
assert r['after_seconds_from_last_tail']==[505,521] and r['real_stimuli_this_run']==0
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.2,4.4),dpi=150,gridspec_kw={'width_ratios':[1.05,1]})
fig.subplots_adjust(top=.73,bottom=.21,left=.085,right=.97,wspace=.39)
fig.text(.065,.92,'芯片 28080  |  换入刺激后信号，模型的动作概率变了多少？',fontsize=16,weight='bold',color='#173f4a')
fig.text(.065,.83,'同一游戏状态与同一组模型权重；刺激前后各 16 个一秒留出窗口，按序号配对。',size=10,color='#48636d')
ok=p[0].mean(axis=0)>0
names=np.array(['移动','导航','采集','进食','饮水','休息','搭营','等待'])[ok]
vals=change.mean(axis=0)[ok];y=np.arange(len(vals))
ax.axvline(0,c='#84929b',lw=.85);ax.barh(y,vals,color=['#257872' if v>0 else '#b1684d' for v in vals],height=.52)
for i,v in enumerate(vals): ax.text(v+(.012 if v>0 else -.012),i,f'{v:+.3f}',va='center',ha='left' if v>0 else 'right',size=9)
ax.set(yticks=y,yticklabels=names,xlim=(-.48,.62),ylim=(len(vals)-.5,-.5),xlabel='平均概率变化（百分点，后减前）')
bx.bar(np.arange(1,17),tv,color='#487385');bx.axhline(tv.mean(),color='#a45f38',ls='--',lw=1)
bx.set(xlim=(.3,16.7),xticks=[1,4,8,12,16],xlabel='配对窗口序号',ylabel='动作概率分布距离（总变差，%）')
bx.text(.98,.92,f'平均 {tv.mean():.2f}%',transform=bx.transAxes,ha='right',size=9,color='#884d29',bbox=dict(facecolor='white',edgecolor='none'))
for k in (ax,bx):k.spines[['right','top']].set_visible(False)
fig.text(.065,.048,'刺激后窗口距末脉冲结束 505–521 秒。所有窗口概率最高的动作均为“移动”。',size=9,color='#48636d')
path=root/'action-probability-comparison.png';fig.savefig(path)
with Image.open(path) as im:im.verify()
print('validated',len(data),'rows',r['after_seconds_from_last_tail'],s['greedy_verb_changed'],hashlib.sha256((root/'forward-result.json').read_bytes()).hexdigest())
