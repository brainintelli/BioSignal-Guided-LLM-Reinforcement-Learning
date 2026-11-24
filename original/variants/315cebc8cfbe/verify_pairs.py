from pathlib import Path
import csv, json, statistics as st, sys, shutil, math, collections, importlib.util
from tempfile import TemporaryDirectory
root=Path.cwd();w=Path(__file__).parent.resolve();out=root/'output';run=out/'mea_six_arm_r1_off_20261120'
base=out/'mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
conf=out/'mea_predictability_22021_20261119T042416Z/offline-confirmatory-results'
arms=['masked','baseline','sham']
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
raw=read(run/'metrics_by_world.csv'); records={(int(r['train_seed']),int(r['world_seed']),r['arm']):r for r in raw if r['chip']=='22021'}
seeds=sorted({s for s,k,a in records}); worlds=sorted({k for s,k,a in records});assert len(records)==3*16*48 and len(worlds)==48 and len(seeds)==16
routes=collections.defaultdict(list);actions={a:collections.Counter() for a in arms}
for s in seeds:
 for a in arms:
  prefix=(run/('baseline' if a=='baseline' else 'main')) if s>=102600001 else (base/('pre-stim-baseline' if a=='baseline' else ''))
  src=prefix/'models'/str(s)/a/'holdout.jsonl'
  eps=[json.loads(l) for l in src.read_text(encoding='utf8').splitlines()];assert sorted(e['world'] for e in eps)==worlds
  for e in eps:
   k=e['world'];r=e['rows'];assert len(r)==6
   route=tuple(json.dumps(step['action'],sort_keys=True) for step in r)
   routes[k,route].append((s,a,r))
   actions[a].update(x['action']['verb'] for x in r)
print('Checking',len(routes),'unique world/action sequences for',len(records),'episodes',flush=True)
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('clockscan',out/'22021_复验诊断/replay_clock.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def measures():
 from agent.m9_contract import Reward
 from agent.m9_env import GodotEnv
 verified={}
 with TemporaryDirectory(prefix='native_pairs_',dir=w) as temp:
  proj=Path(temp)
  mod.observer(proj)
  with GodotEnv(str(root/'.tools/godot/Godot_v4.7.2-stable_win64_console.exe'),str(proj),map_mode='infinite_freeze',navigation_steps=4) as env:
   for i,((world,route),copies) in enumerate(routes.items()):
    start=env.reset(world)
    init=start['observation'];assert start['starting_scene']=='supplied_95' and init['world_tick']==0
    tracker=Reward(init,'bottleneck',.99985);rsum=0
    for j,action in enumerate(route):
     step=env.step(json.loads(action));tr=step['transition'];obs=step['observation']
     reward=tracker.score(tr,obs)['reward'];rsum+=reward
     for s,a,r in copies:
      assert tr['after']==r[j]['after'] and math.isclose(reward,r[j]['r1'],abs_tol=1e-11),(s,world,a,j)
    new=obs['visited_count']-init['visited_count']
    for s,a,r in copies:
     orig=records[s,world,a]
     assert new==int(orig['exploration']) and math.isclose(rsum,float(orig['r1']),abs_tol=1e-11) and not obs.get('dead',False)
     verified[s,world,a]=(new,init['visited_count'],obs['visited_count'],tr['after']['tick'])
    if i%200==0:print('replayed',i+1,'/',len(routes),flush=True)
 return verified
v=measures();assert len(v)==2304
paired=[{'训练种子':s,'地图种子':i,'决策次数':6,'开局已揭示格数':v[s,i,'masked'][1],
 '屏蔽组新增格':v[s,i,'masked'][0],'自发组新增格':v[s,i,'baseline'][0],'零输出组新增格':v[s,i,'sham'][0],
 '自发减屏蔽_格':v[s,i,'baseline'][0]-v[s,i,'masked'][0],'零输出减屏蔽_格':v[s,i,'sham'][0]-v[s,i,'masked'][0],
 '屏蔽组结束刻':v[s,i,'masked'][3],'自发组结束刻':v[s,i,'baseline'][3],'零输出组结束刻':v[s,i,'sham'][3]} for s in seeds for i in worlds]
assert all(len({v[s,k,a][1] for a in arms})==1 for s in seeds for k in worlds)
write(w/'paired.csv',paired)
byseed=[[s]+[sum(v[s,i,a][0] for i in worlds) for a in arms] for s in seeds]
af=[x for x in read(run/'effects.csv') if x['chip']=='22021' and x['metric']=='exploration']
tot=[sum(x[c] for x in byseed) for c in (1,2,3)]
for j,arm in enumerate(arms[1:],2):
 r=next(x for x in af if x['arm_a']==arm and x['arm_b']=='masked')
 assert math.isclose((tot[j-1]-tot[0])/768,float(r['mean_difference'])) and sum(row[j]>row[1] for row in byseed)==int(r['positive_seeds'])
bc=read(conf/'episodes.csv');br=read(conf/'verification/native-replay.csv')
key=lambda r:(r['scene'],r['seed'],r['world'],r['arm'])
src={key(r):r for r in br}
assert len(bc)==len(src)==len(br)==len(set(map(key,bc)))==18432
for r in bc:
 rr=src[key(r)]
 for f in ['cells','survived','decisions','end_tick','drink_wait','eat_wait']: assert int(r[f])==int(rr[f])
 assert abs(float(r['r1'])-float(rr['r1_recorded']))<1e-10
assert sum(abs(float(row['r1_recorded'])-float(row['r1_native']))>1e-10 for row in br)==64
b={}
for scene in ('low_health','thirst','hunger'):
 b[scene]={}
 for arm in ('ordered','masked','shuffled'):
  g=[r for r in bc if r['scene']==scene and r['arm']==arm]
  assert len(g)==2048
  b[scene][arm]=[sum(int(r[col]) for r in g) for col in ('cells','survived')]
meta=dict(seeds=byseed,tot=tot,examples=paired[:5],actions=actions,confirm=b,routes=len(routes),n=len(v),actioncount=sum(sum(x.values()) for x in actions.values()))
(w/'numbers.json').write_text(json.dumps(meta,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:meta[k] for k in ['tot','actions','confirm','examples','routes']},ensure_ascii=False),flush=True)
