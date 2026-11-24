import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'.tools'/'mea_baseline_deps')]+[str(ROOT/'standalone'/name) for name in ('mea_live_pair','mea_state2_r2','mea_binary_r2','mea_stimulation_v2','mea_online')]
from livepair.figures import draw_sites,_plot_style,_save
from matplotlib.figure import Figure
import numpy as np

def read(name):
    return json.loads((HERE/name).read_text(encoding='utf8'))

def positions():
    plan=read('site-plan.json')
    selection=read('site-candidates.json')
    selection.update({key:plan[key] for key in ('suggested','distances_um','region_count','quadrant_midpoint_um')})
    b=read('binding.json')
    b['electrodes']={site:row['electrode'] for site,row in plan['suggested'].items()}
    b['organoid_id']='编号未提供 · 扫描 000016'
    draw_sites(HERE,b,selection)

def traces():
    records=read('event-evidence.json')
    plan=read('site-plan.json')
    with _plot_style():
        fig=Figure(figsize=(9,6.4),layout='constrained',dpi=155)
        axes=fig.subplots(3,1,sharex=True,sharey=True)
        fig.suptitle('芯片 22554 · 逐点刺激的 MEA 原始回读\n事件后最初数毫秒出现刺激伪迹，平台段已达采集量程',fontsize=14)
        for label,color,ax in zip('HWF',('#a23d2d','#196896','#277d52'),axes):
            row=records.get(label,{})
            ax.set_title(f"{label}  ·  电极 {plan['suggested'][label]['electrode']}",loc='left',fontsize=11)
            if 'relative_mv' in row:
                y=np.asarray(row['relative_mv'])
                x=(np.arange(len(y))-60)/20
                ax.plot(x,y,lw=1.3,color=color)
                ax.axvline(0,c='#434c51',ls=':',lw=1)
                ax.set_ylabel('相对基线 (mV)')
                ax.text(.99,.86,f"事件帧 {row['events'][0]['frame']:,}",transform=ax.transAxes,ha='right',fontsize=9)
            else:
                ax.text(.5,.5,'无已核实的单脉冲事件波形',transform=ax.transAxes,ha='center')
            ax.spines[['top','right']].set_visible(False)
        axes[-1].set_xlabel('相对硬件事件帧的时间 (ms)')
        _save(fig,HERE/'HWF-raw-response.png')

def unavailable_matrix():
    state=read('trial-status.json')
    if state.get('analysis'):return
    with _plot_style():
        fig=Figure(figsize=(7.8,4.1),layout='constrained',dpi=155)
        ax,info=fig.subplots(1,2,gridspec_kw={'width_ratios':[1.2,1]})
        fig.suptitle('芯片 22554 · 响应混淆矩阵尚不可估计',fontsize=15)
        ax.imshow(np.full((3,4,3),.93),interpolation='nearest')
        for i in range(3):
            for j in range(4):ax.text(j,i,'—',ha='center',va='center',color='#646e74')
        ax.set(xticks=range(4),xticklabels=['H','W','F','未判定'],yticks=range(3),yticklabels=list('HWF'),
               xlabel='预测位点',ylabel='实际刺激位点')
        ax.spines[:].set_visible(False)
        info.axis('off')
        rows=state.get('trials',[])
        info.text(.03,.93,'已取得的正式判别试次',transform=info.transAxes,weight='bold')
        for i,site in enumerate('HWF'):
            info.text(.03,.81-i*.10,f"{site}     {sum(row['site']==site for row in rows)} / 3",transform=info.transAxes)
        qc=read('connected-baseline-qc.json')
        info.text(.03,.44,f"接线基线 · {qc['roi']} 个独立通道",transform=info.transAxes,weight='bold')
        fractions='、'.join(f"{r['fraction']:.1%}" for r in qc['checks'])
        info.text(.03,.21,f"20、40、55 秒的稳定通道占比\n{fractions}\n各窗口要求 ≥ 95%",transform=info.transAxes,fontsize=9)
        info.text(.03,.04,'灰格表示缺少可验证的预测。',transform=info.transAxes,fontsize=9)
        _save(fig,HERE/'HWF-confusion-status.png')

if __name__=='__main__':
    if (HERE/'site-plan.json').exists():positions()
    if (HERE/'event-evidence.json').exists():traces()
    if (HERE/'trial-status.json').exists():unavailable_matrix()
