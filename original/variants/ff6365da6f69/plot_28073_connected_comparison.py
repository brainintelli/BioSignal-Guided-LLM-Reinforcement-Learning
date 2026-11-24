import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt,font_manager
from PIL import Image
out=Path(sys.argv[1]);r=json.loads((out/'test-outcome.json').read_text(encoding='utf-8'))
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family":font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),"axes.unicode_minus":False})
c=r['connected_baseline'];n=c['roi_count']; t=c['window_offsets_s']
fig,ax=plt.subplots(figsize=(9.2,5.1),dpi=150)
fig.subplots_adjust(left=.14,right=.91,top=.73,bottom=.21)
fig.text(.08,.91,"28073  |  刺激线路接入前后的通道稳定性",fontsize=17,weight="bold",color="#1d3f4a")
fig.text(.08,.82,f"两段均为无刺激记录。用相同的 {n} 个通道相对于各自第 3 秒的基线复算。",fontsize=11,color="#51626a")
y=np.arange(3)
for vals,delta,col,name in [(r['same_channel_comparison_before_routing']['stable_counts_by_window'],-.18,'#6d8995','线路接入前'),(c['stable_counts_by_window'],.18,'#ad6244','线路接入后')]:
    p=np.asarray(vals)/n*100
    ax.barh(y+delta,p,height=.32,color=col,label=name)
    for i,v in enumerate(vals):ax.text(p[i]-1.1,i+delta,f"{100*v/n:.1f}%  ({v}/{n})",color="white",weight="bold",ha="right",va="center",fontsize=10)
ax.axvline(95,ls="--",lw=1.4,color="#173b49");ax.text(95,-.50,"预检线 95%",ha="center",va="bottom",fontsize=10,color="#173b49")
ax.set(xlim=(0,101),ylim=(2.6,-.66),yticks=y,yticklabels=[f"{v} 秒" for v in t],xlabel="达到稳定性条件的通道（%）")
ax.spines[["top","left","right"]].set_visible(False);ax.tick_params(axis="y",length=0)
fig.legend(loc="lower left",bbox_to_anchor=(.10,.045),ncol=2,frameon=False);fig.savefig(out/'baseline-comparison.png')
with Image.open(out/'baseline-comparison.png') as f:f.verify()
