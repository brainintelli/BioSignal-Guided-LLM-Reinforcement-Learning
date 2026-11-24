"""Prepare the preregistered 28080 three-arm offline exploration study."""
import json, random, hashlib
from bisect import bisect_left, bisect_right
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'standalone/mea_state2_r2'))
from mea_state2_r2.features import digest, extract, transform

RECORD=ROOT/'output/mea_predictability_28080_20261119T065953Z/cpp-crossover-v1'
OUT=RECORD/'offline-low-health-three-arms'
TRAIN_SEEDS=[28080101+i for i in range(8)]
HOLDOUT_WORLDS=[28080901+i for i in range(8)]
LOW_HEALTH={'health':30.0,'hunger':82.0,'thirst':78.0}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=read(RECORD/'ppo-data.json')
    design=read(RECORD/'design.json')
    norm=data['normalization']; roi=list(design['roi']); include=set(roi)
    stream=[]
    with (RECORD/'cpp-spikes.jsonl').open(encoding='utf-8') as f:
        for line in f:
            ms,ch,amp,_=json.loads(line)
            if int(ch) in include:
                stream.append((int(ms),int(ch),float(amp)))
    stream.sort(key=lambda x:(x[0],x[1],x[2]))
    times=[r[0] for r in stream]
    trials=sorted(data['trials'],key=lambda x:int(x['index']))
    def make_window(tr,offset,phase):
        start=int(tr['start_ms'])+int(offset)*1000
        lo=bisect_left(times,start); hi=bisect_left(times,start+1000)
        events=stream[lo:hi]
        features=list(extract(events,start,start+1000,roi,1000))
        return dict(start_ms=start,end_ms=start+1000,index=int(tr['index'])*100+int(offset),
                    trial=int(tr['index']),block=int(tr['block']),condition=str(tr['condition']),
                    phase=phase,features=features,normalized=list(transform(features,norm)),
                    events_sha=digest(events),count=len(events))
    train=[]
    holdout=[]
    for tr in trials:
        if int(tr['block'])<=4:
            for off in range(4): train.append(make_window(tr,off,'after'))
        else:
            for off in range(10): holdout.append(make_window(tr,off,'after'))
    assert len(train)==48 and len(holdout)==60
    context=dict(schema='mea.28080.offline_context.v1',profile='cpp_28080_low_health_60tick',chip='28080',
                 source_sha256=data['stream_sha256'],replay_digest=digest(data),roi_sha=data['roi_sha'],
                 normalization=norm,train_windows=train,holdout_windows=holdout,
                 split='blocks 1-4 train; blocks 5-6 held out',window_policy='train 4 contiguous 1s windows per trial; holdout 10 contiguous 1s windows per trial')
    context_sha=digest(context)
    rng=random.Random(2808011901)
    train_order={}
    for seed in TRAIN_SEEDS:
        order=[]
        for b in range(4):
            trial_offsets=[list(range(t*4,t*4+4)) for t in range(b*3,(b+1)*3)]
            rr=random.Random(2808011902+seed+b)
            rr.shuffle(trial_offsets)
            order.extend([i for group in trial_offsets for i in group])
        train_order[str(seed)]=order
        assert sorted(order)==list(range(b*12+12))
    eval_order={}
    for world in HOLDOUT_WORLDS:
        order=list(range(60)); rr=random.Random(2808011903+world); rr.shuffle(order)
        while any(i==v for i,v in enumerate(order)): rr.shuffle(order)
        eval_order[str(world)]=order
    episodes={str(seed):[28082000+i*10+j for j in range(4)] for i,seed in enumerate(TRAIN_SEEDS)}
    study=dict(schema='mea.28080.offline_exploration.v1',study='offline_exploration_28080_low_health',chip='28080',
        arms=['ordered','masked','shuffled'],alpha=1.0,r2_mode='logged_only',r2_beta=0.0,
        stimulation=False,realtime_training=False,starting_scene='low_health_v1',starting_reserves=LOW_HEALTH,
        batch=12,updates=4,train_decisions=48,holdout_windows=60,horizon=60,
        training_seeds=TRAIN_SEEDS,holdout_worlds=HOLDOUT_WORLDS,episodes=episodes,
        train_order_shuffled=train_order,eval_order_shuffled=eval_order,
        context_sha=context_sha,replay_digest=digest(data),source_sha=data['stream_sha256'],roi_sha=data['roi_sha'],
        primary='new_cells',secondary=['r1_sum','survival_at_60','survival_duration','first_success_drink','first_success_eat','final_health','final_thirst','final_hunger'],
        criterion={'comparison_pairs':['ordered-minus-masked','ordered-minus-shuffled'],
                   'unit':'training-seed mean across the same 8 holdout worlds',
                   'bootstrap_draws':40000,'interval':'two-sided Bonferroni-adjusted 97.5% paired bootstrap percentile interval',
                   'stable':'both primary intervals lower bound > 0 and at least 6 of 8 seed-level differences > 0'},
        locked_at='2026-11-19T00:00:00+08:00',
        provenance='28080 six-round C++ MEA recording; no stimulation or real-time training in this replay')
    write(OUT/'context.json',context); write(OUT/'design.json',study); write(OUT/'ppo-data.json',data)
    (OUT/'LOCKED_CRITERIA.md').write_text('''# 芯片 28080 离线三组对照锁定判据\n\n- 主终点：低生命开局（health=30, hunger=82, thirst=78），60 刻新增探索格数 `new_cells`。\n- 组别：`ordered`（真实 MEA 顺序）、`masked`（MEA 特征置零）、`shuffled`（同一批 MEA 窗口打乱）。\n- 三组从同一初始模型、同一优化器、同一训练预算、同一 8 个留存地图开始。\n- 前四轮（12 个试次）只用于训练，后两轮（6 个试次）只用于留存输入；留存窗口不进入训练。\n- R2 仅记录，`alpha=1`，训练奖励中的 R2 贡献固定为 0；不发送刺激，不开启实时训练。\n- 统计单位是每个训练种子在 8 个相同留存地图上的均值；两项主比较使用 40,000 次配对 bootstrap，并按 Bonferroni 使用双侧 97.5% 区间。\n- 只有 `ordered` 同时优于 `masked` 与 `shuffled`、两项区间下界都大于 0，且 8 个种子至少 6 个差值为正，才记为“预注册稳定提升”。\n- 次要指标：原生 R1、60 刻存活、存活刻数、首次成功补水/进食时间、终点生命/水分/饱食。\n''',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'context_sha':context_sha,'train_windows':len(train),'holdout_windows':len(holdout),'seeds':TRAIN_SEEDS,'worlds':HOLDOUT_WORLDS},ensure_ascii=False))
if __name__=='__main__': main()

