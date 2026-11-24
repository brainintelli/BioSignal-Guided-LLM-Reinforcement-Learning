"""Display verified primary and descriptive results without new inference."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
audit_path=root/'final_audit.json'
receipt=json.loads((root/'completion_receipt.json').read_text(encoding='utf-8'))
if hashlib.sha256(audit_path.read_bytes()).hexdigest()!=receipt['files']['final_audit.json']:
    raise ValueError('Audited results changed')
a=json.loads(audit_path.read_text(encoding='utf-8'))
if not a['passed'] or a['cells']!=24:raise ValueError('Complete audit required')
arms=('none','before','after');labels=('不加信号','刺激前信号','刺激后信号')
metrics=[('总体 R1','主指标：未检出额外收益',[a['arms'][x]['post'] for x in arms],'.4f'),
         ('探索格数','次指标：均值增加',[a['secondary'][x]['visited_count'] for x in arms],'.1f'),
         ('水分储备','次指标：均值下降',[a['secondary'][x]['final_thirst'] for x in arms],'.2f')]
changes={name:dict(before=values[1],after=values[2],none=values[0],
                  after_vs_none_pct=100*(values[2]/values[0]-1),
                  after_vs_before_pct=100*(values[2]/values[1]-1)) for name,title,values,fmt in metrics}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams['axes.unicode_minus']=False
colors=['#7E8995','#377CB3','#D98225']
fig,axes=plt.subplots(1,3,figsize=(13.4,5.3))
for ax,(name,title,values,fmt) in zip(axes,metrics):
    bars=ax.bar(range(3),values,color=colors,width=.62,zorder=3)
    for bar,value in zip(bars,values):
        ax.annotate(format(value,fmt),(bar.get_x()+bar.get_width()/2,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=12,weight='bold')
    ax.set_title(title,pad=14,fontsize=13);ax.set_xticks(range(3),labels,fontsize=11)
    ax.set_ylabel(name,fontsize=12);ax.set_ylim(0,max(values)*1.22)
    ax.grid(axis='y',alpha=.18,zorder=0);ax.spines[['top','right']].set_visible(False)
    delta=changes[name]['after_vs_none_pct']
    ax.text(.5,-.19,f'刺激后组相对不加信号组：{delta:+.2f}%',ha='center',transform=ax.transAxes,fontsize=11)
fig.suptitle('28080：探索格数有上升线索，整体收益未确认弱阳性',fontsize=17,y=.98)
fig.text(.5,.91,'24 个模型 / 8 个配对数字种子 / 共同留存评估 / R2 关闭',ha='center',fontsize=11,color='#555')
fig.text(.5,.04,'柱形表示均值；探索与水分为描述性次指标，不能替代 R1 的预定检验。\nR1 两项对照的 95% 区间均跨零，校正后 p 均为 1.00；单枚类器官、无假刺激生物对照。',ha='center',fontsize=10)
fig.subplots_adjust(left=.06,right=.99,top=.76,bottom=.25,wspace=.3)
for ext in ('png','svg'):fig.savefig(root/('effect_overview.'+ext),dpi=180)
plt.close(fig)
result=dict(source_sha256=receipt['files']['final_audit.json'],cells=24,positive_evidence=a['positive_evidence'],
            metric_changes=changes,new_training=False,new_stimulation=False)
(root/'effect_overview.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
