import json, itertools, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
from PIL import Image
d = Path(sys.argv[1])
def load(name):
    return json.loads((d/name).read_text(encoding="utf-8-sig"))
final_route = len(sys.argv)>2 and sys.argv[2]=="final"
prefix="routeB-" if final_route else ""
a=load(prefix+("audit.json" if final_route else "baseline-audit.json"))
s=load(prefix+"site-candidates.json")
route=load(prefix+"route-evidence.json")
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family":font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(), "axes.unicode_minus":False, "font.size":11})
colors = {"H":"#1B718A","W":"#B2661C","F":"#82448A"}
m=a["mapping_activity"]; sites=s["suggested"]
if final_route:
    assert route["three_distinct"] and s["suggested"]==a["selected_qc"]["suggested"]
    assert all(route["readback"][k]["electrode"]==sites[k]["electrode"] for k in "HWF")
assert a["chip_id"]==s["chip"]=="28073" and len(m)==a["mapped_channels"]
for u,v in itertools.combinations("HWF",2):
    r,q=sites[u],sites[v]
    assert math.isclose(math.hypot(r["x_um"]-q["x_um"],r["y_um"]-q["y_um"]),s["distances_um"][u+"-"+v])
fig,(ax,info)=plt.subplots(1,2,figsize=(12,5.9),dpi=155,gridspec_kw={"width_ratios":[1.6,1]})
fig.subplots_adjust(left=.074,right=.98,top=.82,bottom=.17,wspace=.17)
fig.text(.075,.945,"28073  |  三个候选刺激位点",weight="bold",fontsize=20,color="#173444")
fig.text(.075,.885,"芯片与类器官 28073     ·     "+("最终记录路由" if final_route else "原始记录路由")+"     ·     2026-11-18 静默基线",fontsize=11,color="#465c66")
for subset,lab,c,size in [(m,"映射电极","#cdd5da",6),([r for r in m if any(r["counts_20s"])],"前 60 秒有尖峰","#73919b",13)]:
    ax.scatter([r["x_um"] for r in subset],[r["y_um"] for r in subset],s=size,c=c,lw=0,label=lab)
if final_route:
    ax.scatter([r["x_um"] for r in s["candidates"]],[r["y_um"] for r in s["candidates"]],s=40,facecolor="none",edgecolor="#3d7869",linewidth=.85,label="通过选点质控")
mx,my=s["quadrant_midpoint_um"];ax.axvline(mx,color="#8a9aa4",ls=":",lw=1);ax.axhline(my,color="#8a9aa4",ls=":",lw=1)
coords={k:(r["x_um"],r["y_um"]) for k,r in sites.items()}
ax.plot(*zip(*(coords[k] for k in "HWFH")),c="#506671",lw=1.3)
for k in "HWF":
    ax.scatter(*coords[k],s=95,c=colors[k],edgecolor="white",linewidth=1.1,zorder=5)
    ax.annotate(k,coords[k],xytext={"H":(-17,2),"W":(5,7),"F":(6,6)}[k],textcoords="offset points",fontsize=14,color=colors[k],weight="bold")
ax.invert_yaxis();ax.set_aspect("equal",adjustable="box");ax.set_xlabel("横坐标（µm）");ax.set_ylabel("纵坐标（µm）")
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="lower left",bbox_to_anchor=(0,-.235),ncol=3,frameon=False,fontsize=9)
info.axis("off")
def tab(x,y,msg,**kw): info.text(x,y,msg,transform=info.transAxes,**kw)
tab(0,.93,"位点 / 电极",weight="bold");tab(.45,.93,"坐标（µm）",weight="bold")
for i,k in enumerate("HWF"):
    rr=sites[k];y=.82-.105*i
    tab(0,y,k+"   "+str(rr["electrode"]),weight="bold",color=colors[k])
    tab(.45,y,f'{rr["x_um"]:g}, {rr["y_um"]:g}')
tab(0,.44,"两点间距",weight="bold")
for i,(k,dist) in enumerate(s["distances_um"].items()):
    tab(0,.345-.09*i,k);tab(.45,.345-.09*i,f"{dist:,.1f} µm")
fig.text(.075,.045,f"虚线按本次映射范围划分象限。所选三点覆盖 {s['region_count']} 个象限。位置取自 H5 的物理映射。",fontsize=10,color="#415461")
p=d/("HWF-selected-sites.png" if final_route else "HWF-baseline-sites.png");fig.savefig(p);plt.close(fig)
if final_route:
    fig,ax=plt.subplots(figsize=(8,4.4),dpi=145)
    fig.subplots_adjust(top=.73,bottom=.26,left=.18,right=.90)
    fig.text(.08,.88,"28073  |  刺激响应混淆矩阵待测",fontsize=18,weight="bold",color="#173444")
    fig.text(.08,.79,"实测刺激试次 0 / 9",fontsize=12,color="#415461")
    for i in range(3):
        for j in range(4):
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor="#f0f4f5",edgecolor="white",linewidth=2))
            ax.text(j,i,"—",ha="center",va="center",color="#87969b")
    ax.set(xlim=(-.5,3.5),ylim=(2.5,-.5),yticks=range(3),yticklabels=list("HWF"),xticks=range(4),xticklabels=["H","W","F","未判定"],ylabel="实际刺激位点",xlabel="预测位点")
    ax.spines[:].set_visible(False);ax.tick_params(length=0,pad=6)
    fig.text(.08,.07,"横线表示尚无试次可用于估计。H、W、F 各完成 0 / 3 次。",fontsize=10,color="#415461")
    q=d/"HWF-pending-matrix.png";fig.savefig(q);plt.close(fig)
    for x in [p,q]:
        with Image.open(x) as im: im.verify()
    print("verified",p.name,q.name)
    sys.exit(0)
fig,ax=plt.subplots(figsize=(8.5,4.7),dpi=145);fig.subplots_adjust(top=.73,bottom=.23,left=.16,right=.96)
fig.text(.085,.91,"28073  |  逐点响应预检进度",fontsize=18,weight="bold",color="#173444")
fig.text(.085,.825,"已记录 0 / 9 个刺激试次。响应混淆矩阵尚无可估计的单元格。",fontsize=11)
for i,k in enumerate("HWF"):
    ax.text(-.48,i,k,ha="right",va="center",weight="bold",color=colors[k],size=14)
    for j in range(3):
        ax.scatter(j,i,s=850,facecolor="#f0f4f5",edgecolor="#c2cfd4",marker="s")
        ax.text(j,i,"待采",fontsize=9,color="#647580",ha="center",va="center")
ax.set(xlim=(-.65,2.5),ylim=(2.5,-.6),xticks=range(3),xticklabels=["第 1 轮","第 2 轮","第 3 轮"],yticks=[])
ax.tick_params(axis="x",length=0,pad=8);ax.spines[:].set_visible(False)
fig.text(.085,.065,"每轮计划 H、W、F 各一次。当前文件只含无刺激基线。",fontsize=10,color="#415461")
q=d/"HWF-trial-status.png";fig.savefig(q);plt.close(fig)
for x in [p,q]:
    with Image.open(x) as im: im.verify()
print("verified",p.name,q.name)
