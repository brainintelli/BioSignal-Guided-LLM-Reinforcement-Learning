"""Plot frozen, independently verified experiment results; no new inference."""
import hashlib,json,os,sys,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'mea_22554_state_reward_20261116';P=R.parent.parent
sys.path.insert(0,str(R/'plot_deps'));os.environ['MPLCONFIGDIR']=str(R/'mpl_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

index=read(S/'delivery_index.json');manifest=read(S/'evidence/evidence_manifest.json')
for name in ('descriptive_results.json','evidence/evidence_manifest.json'):
    path=S/name;entry=index['artifacts'][path.relative_to(P).as_posix()]
    assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
analysis_path=S/'evidence/experiment01/analysis.json'
assert sha(analysis_path)==manifest['experiment01/analysis.json']['sha256']
A=read(analysis_path);D=read(S/'descriptive_results.json');assert A['status']=='complete'
ARMS=['baseline','state','reward','both','shuffled']
LABELS=['原 R1 基线','仅 state','仅 reward','state + reward','打乱对照']
SHORT=['基线','state','reward','组合','打乱']
COLORS=['#64748B','#3974BC','#148574','#D07836','#8064A2']
CONTRASTS=[('both','baseline','组合－基线',True),('both','shuffled','组合－打乱',True),('state','baseline','state－基线',False),('reward','baseline','reward－基线',False),('both','state','组合－state',False)]
B={(x['block'],x['arm']):x for x in D['blocks']};assert len(B)==40
for arm in ARMS:
    for phase in ('pre','post'):
        assert abs(statistics.mean(B[b,arm][phase] for b in range(8))-A['arms'][arm][phase])<1e-12
    assert abs(D['arms'][arm]['gain']-(A['arms'][arm]['post']-A['arms'][arm]['pre']))<1e-12
    for phase in ('pre','post'):
        x=D['arms'][arm][phase];assert sum(x['action_counts'].values())==16*x['decisions']
        assert abs(x['wait_fraction']-x['action_counts']['wait']/sum(x['action_counts'].values()))<1e-12
for a,b,_,primary in CONTRASTS:
    c=A['primary' if primary else 'secondary'][a+'-'+b]
    for phase in ('post','gain'):
        actual=[B[i,a]['post']-B[i,b]['post']-(B[i,a]['pre']-B[i,b]['pre'] if phase=='gain' else 0) for i in range(8)]
        assert max(abs(x-y) for x,y in zip(actual,c[phase]['differences']))<1e-12
        assert abs(statistics.mean(actual)-c[phase]['mean_difference'])<1e-12

for name in ('msyh.ttc','msyhbd.ttc'):fm.fontManager.addfont('C:/Windows/Fonts/'+name)
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':12.5,'axes.unicode_minus':False,
    'figure.facecolor':'white','axes.facecolor':'white','text.color':'#182A3B','axes.labelcolor':'#35485A',
    'xtick.color':'#526373','ytick.color':'#35485A','axes.edgecolor':'#BEC9D2','grid.color':'#E7ECF0',
    'svg.fonttype':'path','savefig.facecolor':'white','axes.titleweight':'bold','axes.titlepad':16})
generated=[]
def style(ax,grid='x'):
    for side in ('top','right','left'):ax.spines[side].set_visible(False)
    ax.tick_params(axis='both',length=0,pad=9);ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,lw=.8)
def title(fig,heading,sub):
    fig.text(.055,.947,heading,size=23,weight='bold',va='top')
    fig.text(.055,.884,sub,size=12.5,color='#526373',va='top')
def save(fig,name):
    for ext in ('png','svg'):
        path=R/(name+'.'+ext);fig.savefig(path,dpi=170);generated.append(path.name)
    plt.close(fig)
def signed(x):return f'{x:+.4f}'

# 1. Score levels are point charts (zoomed numeric axis, no truncated bars).
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8.3),gridspec_kw={'width_ratios':[1.05,1]})
fig.subplots_adjust(left=.145,right=.935,top=.755,bottom=.23,wspace=.34)
title(fig,'22554：训练后的成绩，与训练本身的变化','五组、8 个配对数字种子块；每块 2 个评估世界。所有分数均为原 R1，不含辅助 reward。')
y=np.arange(5)
for i,arm in enumerate(ARMS):
    pre=A['arms'][arm]['pre'];post=A['arms'][arm]['post']
    ax.plot([pre,post],[i,i],lw=3,color=COLORS[i],alpha=.7,zorder=2)
    ax.scatter([pre],[i],s=92,facecolor='white',edgecolor='#768595',linewidth=1.7,zorder=4)
    ax.scatter([post],[i],s=92,color=COLORS[i],zorder=5)
    ax.annotate(f'{pre:.4f}',(pre,i),xytext=(0,-22),textcoords='offset points',ha='center',fontsize=11,color='#687888')
    ax.annotate(f'{post:.4f}',(post,i),xytext=(0,13),textcoords='offset points',ha='center',fontsize=12,weight='bold')
ax.set(yticks=y,yticklabels=LABELS,xlim=(1.19,1.36),ylim=(4.62,-.62),xlabel='平均折扣原 R1（数值轴局部放大）')
ax.set_xticks([1.20,1.25,1.30,1.35]);ax.set_title('A  训练前 vs 训练后',loc='left',fontsize=15)
ax.legend(handles=[Line2D([],[],marker='o',linestyle='',mfc='white',mec='#768595',label='训练前'),Line2D([],[],marker='o',linestyle='',color='#35485A',label='训练后')],loc='lower left',bbox_to_anchor=(0,-.29),frameon=False,ncol=2,fontsize=11.5)
style(ax)
all_gains=[]
for i,arm in enumerate(ARMS):
    gains=np.array([B[b,arm]['post']-B[b,arm]['pre'] for b in range(8)]);all_gains.extend(gains)
    mean=float(gains.mean());bx.barh(i,mean,height=.42,color=COLORS[i],alpha=.62,zorder=2)
    bx.scatter(gains,i+np.linspace(-.16,.16,8),s=32,color='#344656',edgecolors='white',linewidth=.6,zorder=3)
    bx.text(1.035,i,signed(mean),transform=bx.get_yaxis_transform(),va='center',ha='left',fontsize=12,weight='bold')
lo=min(-.05,min(all_gains)-.025);hi=max(.10,max(all_gains)+.025)
bx.axvline(0,color='#536575',lw=1.3);bx.set(yticks=y,yticklabels=[],ylim=(4.62,-.62),xlim=(lo,hi),xlabel='后测 R1 − 前测 R1')
bx.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:+.2f}'));bx.set_title('B  每组的训练前后增益',loc='left',fontsize=15)
bx.text(1.035,1.05,'均值',transform=bx.transAxes,size=11.5,ha='left')
style(bx);bx.text(0,-.24,'细点：8 个种子块的前后变化；色条：均值',transform=bx.transAxes,fontsize=11.5,color='#526373')
fig.text(.055,.075,'组合组自身变化为 −0.0086；其“前后增益”比基线低 0.0147，差异没有可靠证据。',fontsize=13,weight='bold')
fig.text(.055,.035,'注意：新增输入会立即改变前测表现。训练后排名不能直接解释为学习效果；数字种子重复不等于独立类器官重复。',fontsize=11,color='#526373')
save(fig,'01_score_and_learning')

# 2. Reuse registered intervals and exact p-values without new tests or resampling.
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8.3),sharey=True)
fig.subplots_adjust(left=.17,right=.955,top=.755,bottom=.23,wspace=.15)
title(fig,'效果是否稳定：配对差值与不确定性','点为平均差，线为原报告的 95% 配对 bootstrap 区间。越向右，越支持比较名称中的前者。')
for panel,phase in ((ax,'post'),(bx,'gain')):
    for i,(a,b,label,primary) in enumerate(CONTRASTS):
        c=A['primary' if primary else 'secondary'][a+'-'+b][phase];m=c['mean_difference'];l,h=c['bootstrap_95'];color=COLORS[ARMS.index(a)]
        if primary:panel.axhspan(i-.46,i+.46,color='#F1F4F7',zorder=0)
        panel.errorbar(m,i,xerr=[[m-l],[h-m]],fmt='D',markersize=7,color=color,elinewidth=2.2,capsize=5,zorder=3)
        p=c.get('p_adjusted_primary',c['p_two_sided']);kind='校正' if 'p_adjusted_primary' in c else '未校正'
        panel.text(.194,i-.29,f'Δ {signed(m)}  |  p{kind}={p:.4f}',ha='right',va='center',fontsize=10.5)
    panel.axvline(0,color='#536575',lw=1.2,zorder=1);panel.axhline(1.5,color='#B9C5CE',lw=1)
    panel.set(yticks=y,ylim=(4.6,-.65),xlim=(-.15,.20));panel.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:+.2f}'));style(panel)
ax.set_yticklabels([x[2]+(' 〔主要〕' if x[3] else ' 〔次要〕') for x in CONTRASTS],fontsize=12)
ax.set_title('A  训练后的组间差',loc='left',fontsize=15);ax.set_xlabel('前者后测 − 后者后测')
bx.set_title('B  训练前后增益的组间差',loc='left',fontsize=15);bx.set_xlabel('前者(后−前) − 后者(后−前)')
fig.text(.055,.11,'组合两项主要比较均未显示可靠提升。仅 reward 的正向结果属于次要、未校正的探索信号。',fontsize=13,weight='bold')
fig.text(.055,.061,'后测的两项主要 p 值按 Bonferroni 校正；其余 p 值未校正。区间是描述性区间，不能代替预登记检验。',fontsize=11.2,color='#526373')
fig.text(.055,.027,'n = 8 个配对数字种子块；1 份生物记录。小样本 bootstrap 区间与精确符号检验的 p 值不一定完全对应。',fontsize=11.2,color='#526373')
save(fig,'02_paired_intervals')

# 3. Every registered paired seed difference, including negative outcomes.
matrix=np.array([A['primary' if p else 'secondary'][a+'-'+b]['post']['differences'] for a,b,_,p in CONTRASTS])
fig,ax=plt.subplots(figsize=(15,7.5));fig.subplots_adjust(left=.17,right=.96,top=.72,bottom=.25)
title(fig,'逐种子检查：平均值背后，有多少次方向一致','每格是同一数字种子块内的后测原 R1 差值；蓝色为前者更高，橙红色为前者更低。')
im=ax.imshow(matrix,cmap='RdBu',vmin=-.30,vmax=.30,aspect='auto',interpolation='nearest')
for i in range(5):
    for j in range(8):
        value=matrix[i,j];rgba=im.cmap(im.norm(value));lum=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
        ax.text(j,i,f'{value:+.3f}',ha='center',va='center',fontsize=13,color='white' if lum<.52 else '#152B3C')
    ax.text(8.18,i,f'{int((matrix[i]>0).sum())}/8',ha='center',va='center',fontsize=15,weight='bold')
ax.set(xticks=np.arange(8),xticklabels=[f'块 {i}' for i in range(8)],yticks=y,yticklabels=[x[2] for x in CONTRASTS],xlim=(-.5,8.8),ylim=(4.5,-.5))
ax.xaxis.tick_top();ax.tick_params(length=0,pad=12);ax.text(8.18,-.85,'前者更高',ha='center',va='center',fontsize=12)
for sp in ax.spines.values():sp.set_visible(False)
ax.axhline(1.5,color='white',lw=4);ax.axvline(7.5,color='white',lw=8)
cax=fig.add_axes([.17,.145,.61,.027]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.3,-.15,0,.15,.3]);cb.outline.set_visible(False);cb.ax.tick_params(length=0,labelsize=11)
fig.text(.055,.061,'仅 reward：7/8 个块高于基线；组合：5/8 个块高于基线，但仅 1/8 个块高于打乱对照。',fontsize=13,weight='bold')
fig.text(.055,.025,'方向一致次数是描述统计，不是另一个显著性检验。全部八个块均保留；这不是八个独立类器官。',fontsize=11.2,color='#526373')
save(fig,'03_seed_differences')

# 4. Behavioral endpoints are descriptive, censored at a decision budget.
fig,axs=plt.subplots(2,2,figsize=(15,10.4));fig.subplots_adjust(left=.085,right=.965,top=.79,bottom=.15,wspace=.25,hspace=.52)
title(fig,'行为层面：短程得分变化，不等于寿命变长','后测每组 16 条轨迹；全部达到 96 次决策上限，全部无自然死亡。下图仅展示截断时点的描述均值。')
x=np.arange(5)
for ax,key,heading,limit,fmt in [(axs[0,0],'ticks','A  固定决策预算内观察到的 tick',160,'{:.2f}'),(axs[0,1],'final_health','B  截断时健康值',110,'{:.2f}'),(axs[1,1],'wait_fraction','D  等待动作比例（%）',12,'{:.2f}%')]:
    values=[D['arms'][a]['post'][key]*(100 if key=='wait_fraction' else 1) for a in ARMS]
    ax.bar(x,values,color=COLORS,width=.55,alpha=.85)
    for j,val in enumerate(values):ax.text(j,val+limit*.025,fmt.format(val),ha='center',fontsize=11.5)
    ax.set(xticks=x,xticklabels=SHORT,ylim=(0,limit));ax.set_title(heading,loc='left',fontsize=14);style(ax,'y')
ax=axs[1,0]
for offset,key,color,label in [(-.18,'final_hunger','#D59A55','hunger 资源值'),(.18,'final_thirst','#4B85AE','thirst 资源值')]:
    values=[D['arms'][a]['post'][key] for a in ARMS];ax.bar(x+offset,values,width=.32,color=color,label=label,alpha=.9)
    for j,val in enumerate(values):ax.text(j+offset,val+2,f'{val:.1f}',ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=SHORT,ylim=(0,115));ax.set_title('C  截断时的游戏原生资源值',loc='left',fontsize=14);style(ax,'y');ax.legend(frameon=False,fontsize=10.5,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.03))
fig.text(.055,.074,'观察到的 tick 是各自动作轨迹所经过的时间，不是完整生存时长，也不是相同 tick 下的横向评测。',fontsize=12.5,weight='bold')
fig.text(.055,.032,'资源字段保留游戏原义；等待比例下降或单项资源增加，都不能单独证明策略总体更好。生物记录 n = 1。',fontsize=11.2,color='#526373')
save(fig,'04_behavior_endpoints')

out=dict(source_hashes={'analysis.json':sha(analysis_path),'descriptive_results.json':sha(S/'descriptive_results.json'),'evidence_manifest.json':sha(S/'evidence/evidence_manifest.json')},
    arms=A['arms'],contrasts={a+'-'+b:A['primary' if p else 'secondary'][a+'-'+b] for a,b,_,p in CONTRASTS},blocks=D['blocks'],
    behavior={a:D['arms'][a]['post'] for a in ARMS},positive_counts={a+'-'+b:int((matrix[i]>0).sum()) for i,(a,b,_,_) in enumerate(CONTRASTS)},
    independent_biological_recordings=1,digital_seed_blocks=8,statistical_tests_reused_without_changes=True)
(R/'chart_data.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
audit=dict(passed=True,source_identity_verified=True,all_40_seed_arm_records_checked=True,all_10_contrast_vectors_checked=True,
    primary_post_p_corrected=True,other_p_uncorrected=True,matplotlib_version=matplotlib.__version__,outputs={n:sha(R/n) for n in generated},chart_data_sha256=sha(R/'chart_data.json'))
(R/'chart_validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,images=4,svg=4,positive_counts=out['positive_counts'])))
