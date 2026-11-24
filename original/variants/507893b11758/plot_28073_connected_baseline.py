import sys,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from PIL import Image
out=Path(sys.argv[1])
a=json.loads((out/"connected-baseline-qc-detail.json").read_text(encoding="utf8"))
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family":font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),"axes.unicode_minus":False})
wins=a["windows"];n=a["roi_n"];xmin=0
fig,ax=plt.subplots(figsize=(8.4,4.4),dpi=150)
fig.subplots_adjust(left=.16,right=.9,top=.74,bottom=.24)
fig.text(.08,.885,"28073  |  刺激线路接入后的基线质控",fontsize=17,weight="bold",color="#193b46")
fig.text(.08,.80,f"与第 3 秒的参考窗口比较，每个窗口至少需 {a['required_count']} / {n} 个独立通道稳定。",fontsize=11,color="#4e636b")
for i,r in enumerate(wins):
    pct=r["fraction"]*100;ax.barh(i,pct,height=.52,color="#368479" if r["passed"] else "#aa6042")
    ax.text(pct-1.8,i,f"{pct:.1f}%   ({r['passed_channels']}/{n})",fontsize=11,weight="bold",va="center",ha="right",color="white")
ax.axvline(95,color="#263f4a",ls="--",lw=1.4)
ax.text(95,-.53,"95%",ha="center",va="bottom",fontsize=10,color="#263f4a")
ax.set(yticks=range(3),yticklabels=[str(r["offset_s"])+" 秒" for r in wins],xlim=(0,102),ylim=(2.55,-.65),xlabel="达到稳定性条件的独立通道（%）")
ax.spines[["top","left","right"]].set_visible(False);ax.tick_params(axis="y",length=0)
fig.text(.08,.055,"稳定性条件包括噪声幅度、基线中位数漂移和 ADC 饱和检查。",fontsize=10,color="#4e636b")
fig.savefig(out/"connected-baseline-qc.png")
with Image.open(out/"connected-baseline-qc.png") as f:f.verify()
