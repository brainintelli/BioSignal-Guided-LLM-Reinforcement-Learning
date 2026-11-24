"""Paired analysis of the fixed sixth game tick; consume checked offline evidence only."""
import csv,itertools,os,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.append(str(ROOT/'.tools/mea_baseline_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import audit_r2_beta_clock as audit
O=audit.OUT
FONT=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
if FONT.exists():font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb','font.size':10})
AR=audit.ARMS; BS=[0,.02,.05,.10]; M=audit.MET; V=list(audit.VERBS)
LABEL=dict(baseline='刺激前自发',sham='同轮零输出',ordered='有序',shuffled='打乱')
MN=dict(cells='探索新格',r1='原生 R1',health='生命',hunger='饱食',thirst='水分')
VN=['移动','导航','采集','进食','饮水','休息','扎营','等待']; CM={'22021':'#19786b','22554':'#b35b29'}
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def plot_save(fig,path):
 tmp=path.with_name(path.stem+'.tmp.png');fig.savefig(tmp,dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig);tmp.replace(path)
def fmt(v,m,signed=False):
 return format(v,('+' if signed else '')+('.6f' if m=='r1' else '.2f'))
def ci(v,m):return f"[{fmt(v['ci95_low'],m)}, {fmt(v['ci95_high'],m)}]"
def summary():
 cov=readcsv(O/'coverage.csv'); worlds=readcsv(O/'fixed_tick_worlds.csv'); acts=readcsv(O/'actions_by_world.csv')
 seedmap={ch:audit.study(ch) for ch in AR}; cube={}; actioncube={}; firstcube={}; empirical={}; charges={}
 for c in cov:
  key=(c['chip'],float(c['beta']),c['arm']);cube[key]=np.full((16,48,len(M)),np.nan);actioncube[key]=np.full((16,48,len(V)),np.nan);firstcube[key]=np.full((16,48,len(V)),np.nan);empirical[key]=np.full((16,48,len(V)),np.nan)
 indices={ch:({s:i for i,s in enumerate(ss)},{w:i for i,w in enumerate(ww)}) for ch,(ss,ww) in seedmap.items()}
 def ix(r):
  ch=r['chip'];p,q=indices[ch];return (ch,float(r['beta']),r['arm']),p[int(r['train_seed'])],q[int(r['world_seed'])]
 for r in worlds:
  k,i,j=ix(r);assert np.isnan(cube[k][i,j]).all() and int(r['tick'])==6;cube[k][i,j]=[float(r[x]) for x in M]
 for r in acts:
  k,i,j=ix(r);v=V.index(r['verb']);n=int(r['decisions']);nr=int(r['probability_decisions']); assert n>0 and nr in (0,n) and np.isnan(empirical[k][i,j,v])
  empirical[k][i,j,v]=int(r['selected'])/n
  if nr:actioncube[k][i,j,v]=float(r['probability_sum'])/nr
  if r.get('first_probability','')!='':firstcube[k][i,j,v]=float(r['first_probability'])
 eligible=set(k for k,m in cube.items() if np.isfinite(m).all())
 for c in cov:
  k=c['chip'],float(c['beta']),c['arm'];assert int(c['complete_worlds'])==int(np.isfinite(cube[k][:,:,0]).sum())
  if k in eligible:
   assert np.allclose(empirical[k].sum(axis=2),1)
   if np.isfinite(actioncube[k]).all():assert np.allclose(actioncube[k].sum(axis=2),1,atol=1e-5)
 rr=readcsv(O/'training_reward_audit.csv')
 for k,g in itertools.groupby(sorted(rr,key=lambda r:(r['chip'],r['arm'],float(r['beta']))),key=lambda r:(r['chip'],r['arm'],float(r['beta']))):
  group=list(g);ch,arm,b=k
  charges[(ch,b,arm)]=dict(chip=ch,beta=b,arm=arm,n_seeds=len(group),steps=sum(int(r['decisions']) for r in group),nonzero=sum(int(r['nonzero']) for r in group),ratio_abs=sum(float(r['sum_abs_contribution']) for r in group)/sum(float(r['sum_abs_R1']) for r in group),max_step_ratio=max(float(r['max_ratio']) for r in group))
 # Resample the same seed and world indices on both sides, allowing for both trained policies and reused game maps.
 N=20000; rng=np.random.default_rng(2026112006)
 idx=rng.multinomial(16,np.repeat(1/16,16),size=N); wid=rng.multinomial(48,np.repeat(1/48,48),size=N);wts=(idx[:,:,None]*wid[:,None,:]).reshape(N,768).astype('float64')/768
 means=[];effects=[];ac=[];ae=[];cohorts=[]; sims={}; asc={}
 for ch,b,arm in sorted(eligible):
  k=ch,b,arm;g=cube[k]
  for m,ar in [(m,g[:,:,l]) for l,m in enumerate(M)]:means.append(dict(chip=ch,beta=b,arm=arm,metric=m,mean=ar.mean(),n_seeds=16,n_worlds_per_seed=48))
  for kind,mat in [('实际选择频率',empirical[k]),('策略概率',actioncube[k])]:
   if not np.isfinite(mat).all():continue
   for l,v in enumerate(V):ac.append(dict(chip=ch,beta=b,arm=arm,kind=kind,verb=v,mean=mat[:,:,l].mean(),n_seeds=16,n_worlds_per_seed=48))
  sims[k]=wts@g.reshape(768,len(M))
  if np.isfinite(actioncube[k]).all():asc[k]=wts@actioncube[k].reshape(768,len(V))
 def pair(ch,b,name,detail,terms):
  if not all(k in eligible for _,k in terms):return
  d=sum(f*cube[k] for f,k in terms);bs=sum(f*sims[k] for f,k in terms)
  for l,m in enumerate(M):
   ar=d[:,:,l];per=ar.mean(1);lo,hi=np.quantile(bs[:,l],[.025,.975]);blo,bhi=np.quantile(bs[:,l],[.025/3,1-.025/3])
   effects.append(dict(chip=ch,beta=b,contrast=name,detail=detail,metric=m,delta=ar.mean(),ci95_low=lo,ci95_high=hi,ci_three_low=blo,ci_three_high=bhi,positive_seeds=int((per>1e-10).sum()),negative_seeds=int((per< -1e-10).sum()),n_seeds=16,n_worlds_per_seed=48))
   if name in ('arm','vs_r1_anchor','interaction'):
    for label,sub in [('原8个种子',per[:8]),('新增8个种子',per[8:])]:
     cohorts.append(dict(chip=ch,beta=b,contrast=name,detail=detail,metric=m,registered_cohort=label,mean_paired_difference=sub.mean(),positive_seeds=int((sub>1e-10).sum()),negative_seeds=int((sub< -1e-10).sum()),n_seeds=8,n_worlds_per_seed=48))
  if all(k in asc for _,k in terms):
   d=sum(f*actioncube[k] for f,k in terms);bs=sum(f*asc[k] for f,k in terms)
   for l,v in enumerate(V):
    lo,hi=np.quantile(bs[:,l],[.025,.975]);ae.append(dict(chip=ch,beta=b,contrast=name,detail=detail,verb=v,delta_pp=100*d[:,:,l].mean(),ci95_low_pp=100*lo,ci95_high_pp=100*hi))
 for ch,(a,z) in AR.items():
  for b in BS:
   pair(ch,b,'arm','%s − %s'%(LABEL[a],LABEL[z]),[(1,(ch,b,a)),(-1,(ch,b,z))])
   if b:
    for arm in (a,z):pair(ch,b,'vs_beta0',arm,[(1,(ch,b,arm)),(-1,(ch,0,arm))])
    pair(ch,b,'interaction','组间差相对 β=0 的变化',[(1,(ch,b,a)),(-1,(ch,b,z)),(-1,(ch,0,a)),(1,(ch,0,z))])
 firsteffects=[]
 for ch,(a,z) in AR.items():
  for arm,b in itertools.product((a,z),BS[1:]):
   k,k0=(ch,b,arm),(ch,0,arm)
   if k not in eligible or not all(np.isfinite(firstcube[v]).all() for v in (k,k0)):continue
   contrast=100*(firstcube[k]-firstcube[k0]);vv=wts@contrast.reshape(768,len(V));bounds=np.quantile(vv,[.025,.975],axis=0)
   for j,verb in enumerate(V):firsteffects.append(dict(chip=ch,beta=b,arm=arm,verb=verb,delta_pp=float(contrast[:,:,j].mean()),ci95_low_pp=bounds[0,j],ci95_high_pp=bounds[1,j],n_seeds=16,n_worlds_per_seed=48))
 audit.csvout(O/'first_action_effects.csv',firsteffects)
 # Historical masked/native R1-only anchors are not new beta cells or additional biological samples.
 refpath=O/'r1_reference_worlds.csv'
 if refpath.is_file():
  rrset={'22021':'masked','22554':'pure'}
  reflist=readcsv(refpath)
  for ch,arm in rrset.items():cube[(ch,0.,arm)]=np.full((16,48,len(M)),np.nan)
  for r in reflist:
   k,i,j=ix(r);assert k==(r['chip'],0.,rrset[r['chip']]) and int(r['tick'])==6 and np.isnan(cube[k][i,j]).all();cube[k][i,j]=[float(r[x]) for x in M]
  for ch,ra in rrset.items():
   k=ch,0.,ra;assert np.isfinite(cube[k]).all();sims[k]=wts@cube[k].reshape(768,len(M));eligible.add(k)
   for b,arm in itertools.product(BS,AR[ch]):pair(ch,b,'vs_r1_anchor',arm,[(1,(ch,b,arm)),(-1,k)])
  timing=[];bygroup=defaultdict(list)
  for r in [*reflist,*worlds]:
   if float(r['beta'])==0:bygroup[r['chip'],r['arm']].append(r)
  for ch,ra in rrset.items():
   ctrl=sorted(bygroup[ch,ra],key=lambda r:(int(r['train_seed']),int(r['world_seed'])))
   for arm in AR[ch]:
    group=sorted(bygroup[ch,arm],key=lambda r:(int(r['train_seed']),int(r['world_seed'])))
    assert [(r['train_seed'],r['world_seed']) for r in group]==[(r['train_seed'],r['world_seed']) for r in ctrl]
    for metric in ('r1','cells'):
     for horizon,field in [('6次决策','decision6_'+metric),('6个游戏刻',metric)]:
      a=np.array([float(r[field]) for r in group]);z=np.array([float(r[field]) for r in ctrl])
      t=np.mean([float(r['end_ticks']) for r in group]) if horizon=='6次决策' else 6
      ct=np.mean([float(r['end_ticks']) for r in ctrl]) if horizon=='6次决策' else 6
      timing.append(dict(chip=ch,beta=0,arm=arm,reference=ra,horizon=horizon,metric=metric,arm_mean=a.mean(),reference_mean=z.mean(),delta=(a-z).mean(),relative_percent=100*(a-z).mean()/z.mean(),arm_elapsed_ticks=t,reference_elapsed_ticks=ct))
  audit.csvout(O/'clock_sensitivity.csv',timing)
 for dest,data in [('fixed_tick_means.csv',means),('paired_effects.csv',effects),('action_probabilities.csv',ac),('action_effects.csv',ae),('reward_scale.csv',list(charges.values()))]:audit.csvout(O/dest,data)
 audit.csvout(O/'registered_seed_cohorts.csv',cohorts)
 return cov,means,effects,ac,ae,charges,len(worlds)

def figures(cov,effects,ac):
 fig,axes=plt.subplots(2,5,figsize=(16.2,6.6));fig.subplots_adjust(left=.058,right=.985,top=.84,bottom=.14,hspace=.54,wspace=.40)
 for row,(ch,(a,z)) in enumerate(AR.items()):
  for col,m in enumerate(M):
   ax=axes[row,col];ax.axhline(0,color='#84928d',lw=.8);ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
   for i,b in enumerate(BS):
    v=next((x for x in effects if x['chip']==ch and x['beta']==b and x['contrast']=='arm' and x['metric']==m),None)
    if v:
     y=v['delta'];ax.vlines(i,v['ci95_low'],v['ci95_high'],color=CM[ch],lw=1.7);ax.plot(i,y,'o',color=CM[ch],ms=5.5)
    else:ax.text(i,.04,'待齐',transform=ax.get_xaxis_transform(),color='#7b827e',ha='center',fontsize=8)
   if m=='health':ax.set_ylim(-.1,.1)
   if col==0:ax.set_ylabel(ch+'  '+LABEL[a]+' − '+LABEL[z],fontsize=9)
   ax.set_title(MN[m],fontweight='bold');ax.set_xticks(range(4),['0','.02','.05','.10']);ax.set_xlim(-.35,3.35);ax.tick_params(labelsize=8)
 fig.suptitle('同一游戏时长：不同 R2 权重下，两组相差多少？',fontsize=18,x=.5,y=.97,fontweight='bold')
 fig.text(.5,.895,'第 6 个游戏刻 · 点为配对均值差，线为训练种子与游戏世界双向重采样的 95% 区间',ha='center',color='#40534d')
 fig.text(.5,.04,'横轴：β   |   正值表示前一组更高；各指标单位不同。只绘出 16 个训练种子 × 48 个世界均完整的比较。',ha='center',fontsize=9)
 plot_save(fig,O/'fixed_tick_effects.png')
 # Same-signal beta comparisons answer a different question from arm differences.
 fig,axes=plt.subplots(2,5,figsize=(16.4,6.6));fig.subplots_adjust(left=.06,right=.99,top=.81,bottom=.15,hspace=.48,wspace=.43)
 from matplotlib.lines import Line2D
 palette=['#117568','#ca793b']
 for row,(ch,arms) in enumerate(AR.items()):
  for col,m in enumerate(M):
   ax=axes[row,col];ax.axhline(0,color='#687873',lw=.8);ax.grid(axis='y',alpha=.17)
   for j,arm in enumerate(arms):
    for i,b in enumerate(BS[1:]):
     v=next((v for v in effects if v['chip']==ch and v['metric']==m and v['beta']==b and v['contrast']=='vs_beta0' and v['detail']==arm),None)
     if not v:continue
     x=i+(j-.5)*.16;ax.vlines(x,v['ci95_low'],v['ci95_high'],color=palette[j],lw=1.5);ax.plot(x,v['delta'],'o' if j==0 else 's',color=palette[j],ms=4.6)
   if m=='health':ax.set_ylim(-.1,.1)
   if col==0:ax.set_ylabel(ch)
   ax.set_xlim(-.4,2.4);ax.set_xticks(range(3),['.02','.05','.10']);ax.set_title(MN[m]);ax.tick_params(labelsize=8)
 fig.suptitle('在同一信号组内：开小权重 R2 相对关 R2 改变了什么？',fontsize=18,y=.97,fontweight='bold')
 fig.legend([Line2D([0],[0],color=palette[0],marker='o',lw=1),Line2D([0],[0],color=palette[1],marker='s',lw=1)],['22021 自发 / 22554 有序','22021 零输出 / 22554 打乱'],loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
 fig.text(.5,.035,'横轴：β；基准为相同信号组的 β=0，第 6 个游戏刻。误差线为配对 95% 区间；空位表示该组尚未齐全。',ha='center',fontsize=9)
 plot_save(fig,O/'r2_increment_effects.png')
 refs=[e for e in effects if e['contrast']=='vs_r1_anchor' and e['beta']==0]
 if refs:
  fig,axes=plt.subplots(1,3,figsize=(12.6,4.1));fig.subplots_adjust(left=.18,right=.985,top=.76,bottom=.22,wspace=.23)
  ordering=[(ch,arm) for ch,arms in AR.items() for arm in arms]
  for j,(ax,metric) in enumerate(zip(axes,['cells','r1','thirst'])):
   ax.axvline(0,color='#768681',lw=1);ax.grid(axis='x',alpha=.16);ax.set_title(MN[metric],fontweight='bold',fontsize=11)
   for i,(ch,arm) in enumerate(ordering):
    v=next(x for x in refs if x['chip']==ch and x['detail']==arm and x['metric']==metric);y=3-i
    ax.hlines(y,v['ci95_low'],v['ci95_high'],color=CM[ch],lw=2);ax.plot(v['delta'],y,'o',color=CM[ch],ms=6)
   ax.set_ylim(-.6,3.6);ax.set_yticks(range(4),[ch+'  '+LABEL[arm] for ch,arm in ordering[::-1]] if j==0 else []);ax.tick_params(labelsize=9)
  fig.suptitle('先关掉 R2：仅接入 MEA 后，游戏表现怎样变化？',y=.98,fontsize=16,fontweight='bold')
  fig.text(.5,.08,'第 6 个游戏刻；点为相对本芯片基线的差，线为配对 95% 区间。22021 对比屏蔽组，22554 对比原生纯 R1。',ha='center',fontsize=9)
  plot_save(fig,O/'mea_vs_r1_baseline.png')
 fig,axes=plt.subplots(2,2,figsize=(12.5,7.3));fig.subplots_adjust(left=.09,right=.98,bottom=.1,top=.82,hspace=.65,wspace=.23)
 lookup={(v['chip'],v['arm'],v['beta'],v['kind'],v['verb']):v['mean'] for v in ac}
 panel=[] 
 for ch,arms in AR.items():
  for arm in arms:
   kind='策略概率' if (ch,arm,0.,'策略概率',V[0]) in lookup else '实际选择频率'
   mat=np.full((3,8),np.nan)
   for k,b in enumerate(BS[1:]):
    if (ch,arm,b,kind,V[0]) in lookup:
     for l,v in enumerate(V):mat[k,l]=(lookup[ch,arm,b,kind,v]-lookup[ch,arm,0.,kind,v])*100
   panel.append((ch,arm,kind,mat))
 vmax=max(.05,max(np.nanmax(abs(m)) for *_,m in panel if np.any(np.isfinite(m)))); cm=plt.get_cmap('BrBG').copy();cm.set_bad('#e8e9e6')
 for ax,(ch,arm,kind,mat) in zip(axes.ravel(),panel):
  ax.imshow(mat,cmap=cm,vmin=-vmax,vmax=vmax,aspect='auto')
  for (i,j),x in np.ndenumerate(mat):
   ax.text(j,i,f'{x:+.2f}' if np.isfinite(x) else '—',ha='center',va='center',fontsize=8,color='white' if np.isfinite(x) and abs(x)>.6*vmax else '#253630')
  ax.set_xticks(range(8),VN,fontsize=9);ax.set_yticks(range(3),['.02','.05','.10']);ax.set_ylabel('β');ax.set_title(f'{ch}  {LABEL[arm]}｜{kind}',fontsize=11)
 fig.suptitle('加入 R2 后，动作选择变化了多少？',fontsize=18,y=.97,fontweight='bold');fig.text(.5,.895,'相对同组 β=0，单位为百分点；以第 6 刻前已发起的决策计算，每个世界等权。',ha='center',fontsize=10)
 fig.text(.5,.03,'— 表示尚无完整比较；实际选择频率与策略给出的概率分列标注。',ha='center',fontsize=9)
 plot_save(fig,O/'action_probability_changes.png')

def report(cov,means,ef,ac,ae,charges,n):
 done=sum(int(c['complete_seeds'])==16 for c in cov); full=done==16
 primary=[e for e in ef if e['contrast']=='arm' and e['metric']=='cells']; interactions=[e for e in ef if e['contrast']=='interaction' and e['metric']=='cells']; found=[e for e in primary if e['chip']=='22554' and e['beta']>0 and e['ci_three_low']>0]; improved=[e for e in interactions if e['chip']=='22554' and e['ci_three_low']>0]
 key=lambda ch,b,c,m,detail=None:next((e for e in ef if e['chip']==ch and e['beta']==b and e['contrast']==c and e['metric']==m and (detail is None or e['detail']==detail)),None)
 tnow=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
 s=['# 小权重 R2 的离线复验：22021 与 22554','',f'报告更新时间：{tnow}（北京时间）。覆盖：{done}/16 个条件完整；已核验 {n:,} 局（包括已完成的单个种子）。下表、区间与图只纳入每组全部 16 个训练种子和 48 个留存世界。每个种子 48 个训练决策、4 次更新。主终点取同一个游戏时钟的第 6 刻；“6 次决策”的实际时长另存于逐局表。','',
 '## 结果','',
 ('22554 的有序信号在以下非零权重相对打乱组出现探索优势（扫描三个权重校正）：'+', '.join(str(e['beta']) for e in found)) if found else '已完成的 22554 对照中，没有一个非零权重显示有序信号比打乱信号稳定多探新格（三个权重校正后）。',
 ('相对 β=0，有序与打乱间的探索优势在以下权重进一步扩大：'+', '.join(str(e['beta']) for e in improved)) if improved else '与 β=0 相比，也未见加 R2 能稳定放大有序信号的探索优势。',('这是全量结果。' if full else '当前为阶段结果，未跑齐的权重不作结论。'),'',
 '| 芯片与比较（前减后） | β | 探索格差〔95% 区间〕 | 探索差为正的种子 | R1 差 | 生命差 | 饱食差 | 水分差 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v in primary:
  ch,b=v['chip'],v['beta'];s.append('| '+ ' | '.join([ch+' '+v['detail'],f'{b:.2f}',fmt(v['delta'],'cells',True)+' '+ci(v,'cells'),str(v['positive_seeds'])+'/16']+[fmt(key(ch,b,'arm',m)['delta'],m,True) for m in M[1:]])+' |')
 s+=['','**R2 自身的增量。** 以下先在同一信号组中减去 β=0，再比较这两个增量。它回答的是“改变权重是否更有利于有序信号”，与仅看某一组的平均探索不同。','',
 '| 芯片 | β | 前一组探索变化 | 后一组探索变化 | 两组增量之差〔95% 区间〕 |','|---|---:|---:|---:|---:|']
 for v in interactions:
  ch,b=v['chip'],v['beta'];a,z=AR[ch];s.append(f"| {ch} | {b:.2f} | {fmt(key(ch,b,'vs_beta0','cells',a)['delta'],'cells',True)} | {fmt(key(ch,b,'vs_beta0','cells',z)['delta'],'cells',True)} | {fmt(v['delta'],'cells',True)} {ci(v,'cells')} |")
 anchors=[e for e in ef if e['contrast']=='vs_r1_anchor' and e['beta']==0 and e['metric']=='cells']
 if anchors:
  s+=['','**只接入 MEA 时，是否超过各自的 R1 基线？** 同一时钟第 6 刻，使用已有的独立基线轨迹（每芯片 768 局）。22021 用信号屏蔽组；22554 用没有 MEA 输入的原生纯 R1 模型。下表固定 β=0，因而不含 R2；[基线对照图](mea_vs_r1_baseline.png)。其他权重相对该基线的差异见配对明细。','',
  '| 芯片 | 信号组减去相应基线 | 探索新格差〔95% 区间〕 | 原生 R1 差〔95% 区间〕 |','|---|---|---:|---:|']
  for v in anchors:
   u=key(v['chip'],0,'vs_r1_anchor','r1',v['detail']);s.append(f"| {v['chip']} | {LABEL[v['detail']]} | {fmt(v['delta'],'cells',True)} {ci(v,'cells')} | {fmt(u['delta'],'r1',True)} {ci(u,'r1')} |")
  s+=['','22021 的自发和零输出都比屏蔽组多探，属于同一历史样本、共享基线上的线索；不能据此归因于刺激、加 R2 或特定信号内容，也不构成跨样本复现。']
  rs=[e for e in ef if e['chip']=='22554' and e['contrast']=='vs_r1_anchor' and e['metric']=='thirst']
  if rs:s.append(f"22554 各完整信号组相对纯 R1 的水分差为 {min(e['delta'] for e in rs):+.2f} 至 {max(e['delta'] for e in rs):+.2f} 点。"+('所列条件的单项配对 95% 区间均在零以下。' if all(e['ci95_high']<0 for e in rs) else '请按权重查看区间。')+'纯模型与两种信号组共享初始 Actor，但原生 Critic 有 108 维输入，接入 MEA 后为 112 维。该比较衡量整体接入的表现；有序与打乱使用同一结构，才直接检验信号配对的额外作用。')
  cc=readcsv(O/'registered_seed_cohorts.csv');s+=['','按原有 8 个和后来新增的 8 个训练种子拆开复看（均为同一批游戏世界，不是新类器官）：']
  for arm in AR['22021']:
   z=[r for r in cc if r['chip']=='22021' and r['metric']=='cells' and float(r['beta'])==0 and r['contrast']=='vs_r1_anchor' and r['detail']==arm]
   if len(z)==2:s.append(f"- 22021 {LABEL[arm]}相对屏蔽组，原 8 个种子平均 {float(z[0]['mean_paired_difference']):+.2f} 格，新增 8 个种子 {float(z[1]['mean_paired_difference']):+.2f} 格。")
  s+=['[两批种子的完整对照](registered_seed_cohorts.csv)。']
  times=readcsv(O/'clock_sensitivity.csv')
  def tm(arm,horizon):return next(r for r in times if r['chip']=='22021' and r['arm']==arm and r['metric']=='r1' and r['horizon']==horizon)
  x,y=tm('baseline','6次决策'),tm('baseline','6个游戏刻');s+=['',f"**为什么等游戏时长会改变 R1 结论？** 在完全相同的 16 × 48 对 22021 轨迹上，执行 6 次决策时，自发组平均已走过 {float(x['arm_elapsed_ticks']):.2f} 刻，屏蔽组只有 {float(x['reference_elapsed_ticks']):.2f} 刻。此时 R1 差为 {float(x['delta']):+.6f}（{float(x['relative_percent']):+.1f}%）；截在同一个第 6 刻，差缩至 {float(y['delta']):+.6f}（{float(y['relative_percent']):+.1f}%），95% 区间跨零。[两种计时口径的逐组对照](clock_sensitivity.csv)。"]
 s+=['','**生存与动作。**']
 for ch,(a,z) in AR.items():
  es=[e for e in ef if e['chip']==ch and e['contrast']=='arm' and e['beta']>0 and e['metric'] in ['hunger','thirst']];decreases=[e for e in es if e['ci95_high']<0];increases=[e for e in es if e['ci95_low']>0]
  if es:s.append(f"- {ch}，{LABEL[a]}减{LABEL[z]}：饱食差范围 {min(e['delta'] for e in es if e['metric']=='hunger'):+.2f} 至 {max(e['delta'] for e in es if e['metric']=='hunger'):+.2f}，水分差范围 {min(e['delta'] for e in es if e['metric']=='thirst'):+.2f} 至 {max(e['delta'] for e in es if e['metric']=='thirst'):+.2f}。"+('单项 95% 区间完整位于零一侧：'+'；'.join(f"β={e['beta']:.2f} {MN[e['metric']]} {e['delta']:+.2f}" for e in decreases+increases)+'（探索性）。' if increases or decreases else '这些差异的单项 95% 区间均含零。'))
  aset=[e for e in ae if e['chip']==ch and e['contrast']=='vs_beta0']
  if aset:
   d=max(aset,key=lambda e:abs(e['delta_pp']));s.append(f"- {ch} 已完成条件中最大的动作概率点估计变化：{LABEL[d['detail']]}、β={d['beta']:.2f}，{VN[V.index(d['verb'])]} {d['delta_pp']:+.2f} 个百分点（相对本组 β=0；这是描述性最大值）。")
 if key('22021',.02,'interaction','thirst') and key('22021',.1,'interaction','thirst'):
  for b in (.02,.1):
   v=key('22021',b,'interaction','thirst');s.append(f"- 将原有水分差扣除：22021，β={b:.2f} 相对 β=0 的组间增量为 {v['delta']:+.3f} 点，95% 区间 {ci(v,'thirst')}。")
 if all(v['mean']==100 for v in means if v['metric']=='health'):s.append('- 所有完整组的第 6 刻生命均为 100；这个短窗口无法检验避死行为，也不能据此证明长期生存无代价。')
 delta_action={(e['beta'],e['verb']):e for e in ae if e['chip']=='22554' and e['contrast']=='arm'}
 if delta_action:
  s+=['','22554 有序减打乱的策略概率差（百分点，括号为配对 95% 区间）。这是模型在有机会或没有机会执行这些动作的所有已访问状态上的平均概率。','','| β | 移动 | 进食 | 饮水 |','|---:|---:|---:|---:|']
  for b in BS:
   if (b,'move') not in delta_action:continue
   cells=[]
   for verb in ['move','eat','drink']:
    e=delta_action[b,verb];cells.append(f"{e['delta_pp']:+.3f} [{e['ci95_low_pp']:.3f}, {e['ci95_high_pp']:.3f}]")
   s.append(f'| {b:.2f} | '+' | '.join(cells)+' |')
 draw=O/'audit_cache/22021-action-draw-check.json'
 if draw.exists():
  dc=audit.read(draw)
  if dc['matched_logged_verbs']==dc['decisions']==36864 and dc['frozen_models']==128:s.append('- 22021 的概率取自 128 个冻结模型的逐状态重载。用原种子从这些概率重新抽样，36,864 次动作类别全部与原日志一致。22554 则直接读取每一步的八类概率，并与被选动作保存的对数概率核对。')
 firstfile=O/'first_action_effects.csv'
 if firstfile.exists():
  f=readcsv(firstfile);s+=['','整段轨迹的动作概率也受模型走到了哪些状态影响。为分开这一点，另在每个世界的**第一个决策**核对相同游戏状态和相同 MEA 输入，并在同组内比较非零 β 与 β=0。']
  for ch in AR:
   z=[e for e in f if e['chip']==ch]
   if z:
    v=max(z,key=lambda e:abs(float(e['delta_pp'])));s.append(f"- {ch} 在相同起点的最大概率差点估计：{LABEL[v['arm']]}、β={float(v['beta']):.2f}，{VN[V.index(v['verb'])]} {float(v['delta_pp']):+.2f} 个百分点。完整估计见[同起点动作对照](first_action_effects.csv)，不据最大值认定效果。")
 tc=O/'technical_check/technical_check.json'
 if tc.exists():
  rr=audit.read(tc);v=next(z for z in rr['metrics'] if z['metric']=='cells')
  s+=['','**同设置重复的波动有多大？** 另行核验 22554 β=0.05 有序组在另一台服务器已完成的 '+str(len(rr['seeds']))+' 个种子。初始权重、注册配置和 48 个世界逐一相同。'+f"同一设置重跑的平均探索差为 {v['mean_host23_minus_host16']:+.2f} 格（95% 区间 [{v['ci95_low']:.2f}, {v['ci95_high']:.2f}]）；逐种子平均差的绝对值再取平均为 {v['mean_absolute_seed_difference']:.2f} 格。两机运行环境与训练随机性无法在这项复核中拆开；这些重复不进入主结果。[技术重复明细](technical_check/replicate_effects.csv)"]
 s+=['','## 奖励到底有多大','',
 '总奖励为 R1 + β × sign × 0.1 × min(|R1|, 0.009775 × D)，D 是动作历时对应的折扣刻数，γ=0.99985。β=0.02、0.05、0.10 的单步贡献最多分别是 |R1| 的 **0.2%、0.5%、1.0%**。R1 为零时该步贡献也为零。下表的实测占比是训练全程 Σ|βR2| / Σ|R1|。','',
 '| 芯片 | 信号组 | β | 非零步数 / 训练步数 | 实测占比 | 最大单步占比 |','|---|---|---:|---:|---:|---:|']
 for (ch,b,arm),r in sorted(charges.items()):
  if r['n_seeds']==16:s.append(f"| {ch} | {LABEL[arm]} | {b:.2f} | {r['nonzero']} / {r['steps']} | {100*r['ratio_abs']:.3f}% | {100*r['max_step_ratio']:.3f}% |")
 s+=['','22021：state2 来自刺激前自发或同轮零输出 MEA；两组都没有当轮刺激。沿用该轮注册方案，每第 3 个合格决策由游戏动作结果给正负号，其他决策为零。这一行检验模拟游戏辅助奖励，不能算作刺激奖惩证据。','',
 '22554：正负号来自存档刺激事件的时相（刺激前正、刺激后负，终止无贡献）。有序组与打乱组共享这一时序；打乱仅置换训练块内、同一时相的实测特征，留存评估的输入次序相同。9 个物理试次中 6 个用作训练、3 个留作评估；48 个游戏世界反复使用已登记的测试窗，不等于另有 48 次物理实验。22021 的 MEA 测试窗同样按既定序列回放，当下的游戏动作不会改变历史记录。两枚芯片分开分析。','',
 '每个条件在同一芯片内配对训练种子、世界和起始场景，评估时冻结模型。探索是相对起始状态新增的已揭示格数；R1 只计原生游戏回报（包括跨越第 6 刻动作的实际前缀）。生命、饱食和水分取游戏内部同刻日志，未做插值。动作统计包括第 6 刻以前发起的决策，每个世界先独立求均值，再在世界和种子间等权。','',
 '95% 区间按训练种子和世界两层交叉配对重采样 20,000 次；同一芯片的各比较共用抽样。另在配对表提供针对 3 个非零 β 的 Bonferroni 区间（每项 98.33%），指标与方向同时披露。每芯片只代表一个历史样本，多种子、多世界不是生物学重复。本轮不包含新刺激、实时反馈或独立新样本。','',
 '图：[有序与对照的差异](fixed_tick_effects.png) · [每组对比 β=0](r2_increment_effects.png) · [动作概率变化](action_probability_changes.png)；明细：[各组绝对值](fixed_tick_means.csv) · [全部配对区间](paired_effects.csv) · [策略概率与选择频率](action_probabilities.csv) · [动作概率差及区间](action_effects.csv) · [训练奖励占比](reward_scale.csv) · [逐世界等时刻数据](fixed_tick_worlds.csv) · [既有 R1 基线](r1_reference_worlds.csv) · [完成度](coverage.csv)。','']
 file=O/'report.md';tmp=O/'report.tmp';tmp.write_text('\n'.join(s),encoding='utf8');tmp.replace(file)
if __name__=='__main__':
 cov,means,ef,act,ae,charges,n=summary();figures(cov,ef,act);report(cov,means,ef,act,ae,charges,n);print('report and figures written',n,'checked games')
