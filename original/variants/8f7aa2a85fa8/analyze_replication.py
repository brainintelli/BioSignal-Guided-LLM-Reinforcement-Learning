"""Reapply the registered analysis and compare the two samples without pooling them."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OLD=ROOT/'output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
sys.path.insert(0,str(ROOT/'standalone/brain_device_session'))
from report_recorded_pulses import Figure,_plot_style,_save,np
comparison_spec=importlib.util.spec_from_file_location('five_arm_registered_comparison',OUT/'compare.py')
compare=importlib.util.module_from_spec(comparison_spec)
comparison_spec.loader.exec_module(compare)

def read(p):return json.loads(p.read_bytes())
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def clock():
    path=ROOT/'output/22021_复验诊断/replay_clock.py'
    spec=importlib.util.spec_from_file_location('original_tick6_analysis',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.OUT,mod.STUDY=OUT,OUT
    mod.main()

LABEL={'masked':'屏蔽 MEA','baseline':'首次刺激前自发','sham':'当轮零输出','regular':'固定规律刺激后','irregular':'多点随机刺激后'}
CONTRASTS=[('baseline','masked'),('sham','masked'),('regular','sham'),('irregular','sham')]
PAIR_NAMES=['刺激前自发 − 屏蔽','当轮零输出 − 屏蔽','规律刺激 − 零输出','随机刺激 − 零输出']

def extra(spec):
    controls=dict(same_actor=0,same_critic=0,same_episode_actions=0,actor_max_abs_difference=0.0)
    means={}
    hosts=collections.Counter()
    for arm in LABEL:
        root=OUT/'pre-stim-baseline' if arm=='baseline' else OUT
        rows=[]
        for seed in spec['training_seeds']:
            model=root/'models'/str(seed)/arm
            host=read(model/'transfer.json')
            hosts[host['host']]+=1
            games=[json.loads(line) for line in (model/'holdout.jsonl').read_text(encoding='utf8').splitlines()]
            assert [g['world'] for g in games]==spec['holdout_seeds']
            if arm=='masked':
                previous=OLD/'models'/str(seed)/arm
                prev=read(previous/'update.json')['after'];now=read(model/'update.json')['after']
                assert read(previous/'update.json')['initial']==read(model/'update.json')['initial']
                for key in ['actor','critic']:controls['same_'+key]+=int(prev[key]==now[key])
                def tensors(directory):
                    data=(directory/'checkpoint/adapter/adapter_model.safetensors').read_bytes()
                    size=int.from_bytes(data[:8],'little')
                    header=json.loads(data[8:8+size]);payload=memoryview(data)[8+size:]
                    result={}
                    for key,item in header.items():
                        if key=='__metadata__':continue
                        assert item['dtype']=='F32'
                        result[key]=np.frombuffer(payload[slice(*item['data_offsets'])],dtype='<f4').reshape(item['shape'])
                    return result
                prior=tensors(previous); current=tensors(model)
                assert current.keys()==prior.keys()
                max_delta=max(float(np.abs(t-prior[key]).max()) for key,t in current.items())
                controls['actor_max_abs_difference']=max(max_delta,controls['actor_max_abs_difference'])
                prev_games=[json.loads(line) for line in (previous/'holdout.jsonl').read_text(encoding='utf8').splitlines()]
                for g,pg in zip(games,prev_games,strict=True):
                    assert g['world']==pg['world']
                    controls['same_episode_actions']+=int([r['action'] for r in g['rows']]==[r['action'] for r in pg['rows']])
            for g in games:
                seq=g['rows'];end=seq[-1]['after']
                living=0.;last=0
                for r in seq:
                    elapsed=r['after']['tick']-last
                    assert elapsed>0 and not r['after']['dead']
                    living+=sum(.01*.99985**j for j in range(elapsed))
                    last=r['after']['tick']
                rows.append([last,sum(q['r1'] for q in seq),living,end['health'],end['thirst'],end['hunger'],
                             sum(q['action']['verb']=='drink' for q in seq),sum(q['action']['verb']=='eat' for q in seq)])
        means[arm]=dict(zip(['ticks','r1','living','health','water','food','drink','eat'],np.mean(rows,axis=0).tolist()))
    return dict(masked_reproducibility=controls,models_by_host=dict(hosts),secondary=means)

def plot(old,new,previous_tick,tick):
    with _plot_style():
        fig=Figure(figsize=(11.5,5.8),layout='constrained',dpi=155)
        fig.suptitle('芯片 22021 · 两枚类器官在同一批留存地图上的配对结果',fontsize=15)
        for ax,(metric,title,unit) in zip(fig.subplots(1,2),[('r1_sum','游戏原生回报 R1','R1 差值'),('new_cells','新增探索格数','格数差')]):
            ax.axvline(0,color='#68747e',ls='--',lw=1)
            for j,(a,b) in enumerate(CONTRASTS):
                pair=a+'-'+b
                for z,(obj,tag,color) in enumerate([(old,'11/18 六次决策','#768699'),(new,'11/19 六次决策','#087e79'),(previous_tick,'11/18 第六刻','#a28bb0'),(tick,'11/19 第六刻','#c07336')]):
                    if z>=2: v=obj['results'][a]['r1' if metric=='r1_sum' else 'cells'];m=v['mean']
                    else:v=obj['contrasts'][pair]['metrics'][metric];m=v['difference']
                    y=3-j+(1.5-z)*.17
                    ax.hlines(y,*v['interval'],lw=2.2,color=color)
                    ax.plot(m,y,'o',ms=4.8,color=color,label=tag if j==0 else None)
            ax.set(title=title,xlabel=unit,yticks=[3,2,1,0],yticklabels=PAIR_NAMES,ylim=(-.55,3.55))
            ax.spines[['left','top','right']].set_visible(False)
            ax.grid(axis='x',alpha=.18)
        fig.legend(*fig.axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
        _save(fig,OUT/'两枚样本复验.png')

def fm(v):return f"{v:+.6f}"
def fc(v):return f"{v:+.2f}"
def interval(m,fn,field='difference'):return f"{fn(m[field])} [{fn(m['interval'][0])}, {fn(m['interval'][1])}]"

def report():
    spec=read(OUT/'design.json');old=read(OLD/'comparison.json');new=read(OUT/'comparison.json');tick=read(OUT/'clock6.json')
    assert (new['audited_models'],new['audited_games'],new['audited_actions'])==(40,1920,11520)
    assert (tick['archived_games'],tick['matched_action_ends'])==(1920,11520)
    meta=extra(spec);write('replication-checks.json',meta)
    previous_tick=read(ROOT/'output/22021_复验诊断/clock6.json')
    plot(old,new,previous_tick,tick)
    positive=[a for a in ['baseline','sham'] if new['contrasts'][a+'-masked']['metrics']['r1_sum']['interval'][0]>0]
    joint=[p for p,v in new['contrasts'].items() if v['stable']]
    headline=('、'.join(LABEL[a] for a in positive)+'的六决策 R1 单项增益在第二枚样本上复现。' if positive else '刺激前自发和当轮零输出的六决策 R1 均未在第二枚样本上达到原校正区间下界大于零的复现条件。')
    overall=('达到探索与 R1 联合提升原判据的比较为 '+ '、'.join(PAIR_NAMES[[a+'-'+b for a,b in CONTRASTS].index(j)] for j in joint)+'。') if joint else '四项比较均未达到探索与 R1 同时稳定改善的原判据。'
    s=['# 22021 第二枚样本五组离线训练复验','',headline+overall,'',
       '使用 11 月 19 日登记的新类器官 anon-22021-20261119T042416Z。两枚样本共用芯片编号、训练种子、训练游戏和留存地图。每枚样本内分别估计配对差。','',
       '![两枚样本的配对结果](两枚样本复验.png)','',
       '## 原定六次决策终点','',
       '| 对照差 | 第一枚 R1 差 | 第二枚 R1 差及校正区间 | 第二枚探索差及校正区间 | R1 上升种子 | 两项同升种子 |',
       '|---|---:|---:|---:|---:|---:|']
    for n,(a,b) in enumerate(CONTRASTS):
        v=new['contrasts'][a+'-'+b];r=v['metrics']['r1_sum'];cells=v['metrics']['new_cells']
        s.append(f"| {PAIR_NAMES[n]} | {fm(old['contrasts'][a+'-'+b]['metrics']['r1_sum']['difference'])} | {interval(r,fm)} | {interval(cells,fc)} | {sum(x>1e-10 for x in r['by_seed'])}/8 | {v['jointly_positive']}/8 |")
    s+=['','原方案有三项主比较（零输出对屏蔽、规律对零输出、随机对零输出），每项两个终点，共六项，单项双侧 99.17% 区间。刺激前自发对屏蔽沿用单独的 97.5% 区间。训练种子与地图分别配对重采样 30,000 次。单项 R1 复现要求该项在两枚样本上方向为正且各自校正下界大于零。原联合判据还要求两项区间下界均高于零且至少 6/8 个种子两项同时上升。','',
        '## 各组的游戏表现','',
        '| 组别 | R1 | 新格数 | 平均游戏刻 | 期末生命 | 期末水分 | 期末饱食 | 每局饮水 | 每局进食 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in LABEL:
        m=meta['secondary'][a];v=new['arm_means'][a]
        s.append('| '+LABEL[a]+f" | {v['r1_sum']:.6f} | {v['new_cells']:.2f} | "+' | '.join(f"{m[k]:.2f}" for k in ('ticks','health','water','food','drink','eat'))+' |')
    s+=['','六次动作覆盖的游戏刻数取决于动作种类。按逐步日志分解第二枚样本的 R1 差，存活计时按实际经过刻数和单动作折扣 0.99985 重算。','',
        '| 相对屏蔽组 | 平均经过刻数差 | 总 R1 差 | 存活计时项差 | 其余状态项差 |','|---|---:|---:|---:|---:|']
    masked=meta['secondary']['masked']
    for a in ('baseline','sham'):
        g=meta['secondary'][a];gain=g['r1']-masked['r1'];timer=g['living']-masked['living']
        s.append(f"| {LABEL[a]} | {g['ticks']-masked['ticks']:+.3f} | {fm(gain)} | {fm(timer)} | {fm(gain-timer)} |")
    s+=['','## 相同游戏时间的核对','',
        '沿用上一轮的第六个游戏刻。对全部 1,920 局原动作核对 11,520 个动作终点，在引擎中读取跨越截止刻的状态。第六刻是补充分析时点。','',
        '| 比较 | R1 差及区间 | 新格数差及区间 | 探索上升种子 | 生命差 | 水分差及区间 | 饱食差及区间 |','|---|---:|---:|---:|---:|---:|---:|']
    for n,(a,b) in enumerate(CONTRASTS):
        v=tick['results'][a]
        s.append(f"| {PAIR_NAMES[n]} | {interval(v['r1'],fm,'mean')} | {interval(v['cells'],fc,'mean')} | {v['cells']['positive_seeds']}/8 | {fc(v['health']['mean'])} | {interval(v['water'],fc,'mean')} | {interval(v['food'],fc,'mean')} |")
    s+=['','原六决策轨迹均以角色仍存活结束，无法据此判断长期生存率。训练和评估都回放归档信号，游戏动作与该次实际生物放电之间没有实时因果对应。两枚类器官的结果可以逐样本比较，区间表示每枚样本内由游戏地图与模型种子产生的不确定性。','',
        '## 训练与数据核验','',
        '按原方案从同一 Qwen3.5-0.8B 暖启动检查点训练 5 组 × 8 个种子。每模型执行 4 批 × 12 个训练决策、8 次 Actor 优化步和 64 次 Critic 优化步，每组 384 个训练决策。R1 系数 1，模拟 R2 逐步记录，实际奖励贡献全部为零；各模型重载后冻结评估。游戏起始生命 100、水分 95、饱食 95。48 张地图与训练分离，沿用第一枚的 48 张评估地图，以配对观察更换样本的影响。','',
        '新样本的 324 个登记非刺激通道用于计算四项特征，第一枚为 322 个。独立的 64.04 秒静默记录提供归一化与刺激前自发组的分段记录。前四个物理轮次用于训练，后两个物理轮次整轮留存。四种实测条件各含 48 个训练窗、4 个价值估计窗、6 个留存窗。每个种子都核对 Actor、Critic 的实际输入、训练步数、文件校验值、游戏回报、记录的模拟 R2 和留存动作轨迹。','']
    ctrl=meta['masked_reproducibility']
    s.append(f"第二枚的屏蔽组也从头按同一预算重训。与第一枚的屏蔽组相比，{ctrl['same_actor']}/8 个 Actor、{ctrl['same_critic']}/8 个 Critic 的训练后张量散列一致。逐元素核对八对 Actor 的 LoRA 权重，最大绝对差为 {ctrl['actor_max_abs_difference']:.3g}。{ctrl['same_episode_actions']}/384 对地图与种子上的整局动作完全相同。40 个检查点位置及模型实际分配见传输回执。")
    s+=['','[第二枚样本配对统计](comparison.json) · [逐地图配对数据](paired-worlds.csv) · [第六刻的游戏回放](clock6.json) · [核验汇总](replication-checks.json) · [主比较登记](design.json) · [刺激前补充比较登记](pre-stim-baseline/design.json)']
    (OUT/'五组训练复验结果.md').write_text('\n'.join(s)+'\n',encoding='utf8')
    print(headline,overall,meta['models_by_host'])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('step',choices=['compare','clock','report'])
    {'compare':compare.main,'clock':clock,'report':report}[parser.parse_args().step]()
