"""Workstation-side figures for one bound sample, using its closed preflight records."""
from collections import Counter
from itertools import combinations
import math

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager
import numpy as np

from .protocol import digest
from .qualification import SITES, VERSION, analysis, require_pass


def _plot_style():
    family=font_manager.findfont(font_manager.FontProperties(
        family=['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']))
    font=font_manager.FontProperties(fname=family).get_name()
    return mpl.rc_context({'font.family':font,'font.size':10,'axes.unicode_minus':False})


def _save(fig,path):
    FigureCanvasAgg(fig).print_png(str(path))
    return path.resolve()


def _identity(b):
    return f"芯片 {b['chip_id']}   ·   类器官 {b['organoid_id']}   ·   会话 {b['session_id']}"


def draw_sites(out,b,selection):
    sites=selection['suggested']
    if (selection['chip']!=b['chip_id'] or set(sites)!=set(SITES)
            or {s:r['electrode'] for s,r in sites.items()}!=b['electrodes']):
        raise ValueError('Selected sites differ from bound electrodes')
    xy={s:(r['x_um'],r['y_um']) for s,r in sites.items()}
    distances={a+'-'+c:math.dist(xy[a],xy[c]) for a,c in combinations(SITES,2)}
    if any(not math.isfinite(v) or v<=0 or not math.isclose(v,selection['distances_um'][k])
           for k,v in distances.items()):
        raise ValueError('Stored and measured-coordinate distances differ')
    with _plot_style():
        fig=Figure(figsize=(9,5),dpi=150,layout='constrained')
        ax,info=fig.subplots(1,2,gridspec_kw={'width_ratios':[1.6,1]})
        fig.suptitle('H、W、F 刺激电极分布\n'+_identity(b),fontsize=14)
        candidates=selection['candidates']
        ax.scatter([r['x_um'] for r in candidates],[r['y_um'] for r in candidates],
                   s=13,c='#bbc7ce',label='基线合格电极')
        mx,my=selection['quadrant_midpoint_um']
        ax.axvline(mx,c='#8e9ba4',ls=':',lw=.8)
        ax.axhline(my,c='#8e9ba4',ls=':',lw=.8)
        ax.plot(*zip(*(xy[s] for s in 'HWFH')),c='#566871',lw=1)
        for s,color,offset in zip(SITES,('#aa4934','#176b96','#227754'),((5,7),(5,-13),(5,7))):
            ax.scatter(*xy[s],s=64,c=color,edgecolor='white',zorder=3)
            ax.annotate(s,xy[s],xytext=offset,textcoords='offset points',color=color,weight='bold')
        ax.set(aspect='equal',xlabel='阵列横坐标（µm）',ylabel='阵列纵坐标（µm）')
        ax.margins(.12);ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='best',frameon=False,fontsize=8)
        info.axis('off')
        info.text(.02,.94,'位点及坐标（µm）',weight='bold',transform=info.transAxes)
        for i,s in enumerate(SITES):
            x,y=xy[s]
            info.text(.02,.83-i*.105,f"{s}   电极 {sites[s]['electrode']}    ({x:g}, {y:g})",
                      transform=info.transAxes)
        info.text(.02,.46,'两点间距（µm）',weight='bold',transform=info.transAxes)
        for i,(pair,v) in enumerate(distances.items()):
            info.text(.02,.35-i*.095,f'{pair}       {v:,.1f}',transform=info.transAxes)
        return _save(fig,out/'HWF-sites.png')


def draw_responses(out,b,report):
    bid=digest(b)
    if report.get('schema')!=VERSION or report.get('binding')!=bid or type(report.get('passed')) is not bool:
        raise ValueError('Response report differs from current binding')
    rows=report['trials']
    stats=report.get('analysis')
    if report['passed']:
        require_pass(report,bid)
    elif stats is not None and stats!=analysis(bid,rows):
        raise ValueError('Reported confusion matrices differ from the recorded trials')
    with _plot_style():
        fig=Figure(figsize=(9,4.5),dpi=150,layout='constrained')
        status='通过' if report['passed'] else ('未通过' if stats else '未完成')
        fig.suptitle(f'三位点响应预检 · {status}\n'+_identity(b),fontsize=14)
        if stats is None:
            ax=fig.subplots();ax.axis('off')
            counts=Counter(r['site'] for r in rows)
            ax.text(.5,.62,f'已记录 {len(rows)}/9 个试次，未生成可验证的混淆矩阵',
                    ha='center',transform=ax.transAxes,fontsize=14)
            ax.text(.5,.43,'     '.join(f'{s}  {counts[s]}/3' for s in SITES),
                    ha='center',transform=ax.transAxes)
        else:
            for ax,key,label in zip(fig.subplots(1,2),('post','pre'),('刺激后 12–13 秒','刺激前 3–2 秒 · 背景对照')):
                data=stats[key]
                counts=np.asarray(data['confusion'])
                ax.imshow(counts,cmap='Blues',vmin=0,vmax=3,interpolation='nearest')
                for (i,j),v in np.ndenumerate(counts):
                    ax.text(j,i,str(v),ha='center',va='center',fontsize=14,
                            color='white' if v>=2 else '#203344')
                ax.set(xticks=range(4),xticklabels=[*SITES,'未判定'],yticks=range(3),yticklabels=list(SITES),
                       ylabel='实际刺激位点',xlabel='预测位点（格内为试次数）',
                       title=f"{label}\n正确 {data['correct']}/9    置换 p = {data['p_exact']:.3f}")
                ax.spines[:].set_visible(False)
        return _save(fig,out/'HWF-responses.png')
