"""Fixed-clock offline evidence for the 22021/22554 R2 sweep; no device connection."""
import argparse,collections,csv,gzip,hashlib,importlib.util,json,math,os,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.m9_env import GodotEnv
from agent.m9_contract import Reward,potential
from agent.mea_contract import observation_key
from agent.m8s3_contract import VERBS
OUT=ROOT/'output/mea_r2_beta_sweep_20261120'; REM=OUT/'remote'; PAST=ROOT/'output/mea_six_arm_r1_off_20261120'; EARLY=ROOT/'output/mea_six_arm_r1_off_20261119/22554'; FIRST=ROOT/'output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
TAGS={0:'0',2:'002',5:'005',10:'010'}; ARMS={'22021':['baseline','sham'],'22554':['ordered','shuffled']}; GM=.99985; MET=['cells','r1','health','hunger','thirst']; CACHE=OUT/'audit_cache'; OBSLINK=ROOT/'output/22021_复验诊断/replay_clock.py';spec=importlib.util.spec_from_file_location('prior_clock_observer',OBSLINK);clock=importlib.util.module_from_spec(spec);spec.loader.exec_module(clock)
def read(p):return json.loads(p.read_bytes())
def lines(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def canon(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def agg(v):return float(np.mean(v))
def near(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=2e-9)
def chk(test,msg):
 if not test:raise ValueError(str(msg))
def writejson(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)
def csvout(p,rows):
 if not rows:return
 q=p.with_suffix('.tmp');p.parent.mkdir(exist_ok=True,parents=True)
 with q.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 q.replace(p)
def gzload(p):
 if not p.exists():return {}
 with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def gzsave(p,x):
 tmp=p.with_suffix('.tmp');p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
 tmp.replace(p)
def rewardcheck(r1,sign,dt,con,total,b):
 expected=(sign or 0)*b/100*.1*min(abs(r1),.00977500000000008*math.expm1(dt*math.log(GM))/math.expm1(math.log(GM)))
 chk(dt>=1 and near(expected,con) and near(total,r1+con),('reward mismatch',r1,sign,b,dt,expected,con,total));return abs(con),abs(r1), int(abs(con)>0), abs(con/r1) if r1 else 0

def study(chip):
 if chip=='22021':
  d=read(REM/chip/'beta_002/design.json');chk(d['training_seeds']==d['seed_cohorts']['prior_8']+d['seed_cohorts']['added_8'] and len(d['seed_cohorts']['prior_8'])==len(d['seed_cohorts']['added_8'])==8,'registered cohorts');chk(not {world for seed in d['training_seeds'] for world in d['episodes'][str(seed)]}.intersection(d['holdout_seeds']) and not set(d['physical_training_blocks']).intersection(d['physical_holdout_blocks']),'22021 train/holdout overlap');return d['training_seeds'],d['holdout_seeds']
 b=read(OUT/'configs/22554/budget-beta_002.json');chk(b['training_seeds']==list(range(2255401,2255417)),'registered paired cohorts');return b['training_seeds'],b['game_eval_seeds']
def loc(chip,b,arm,s):
 if b:return REM/chip/('beta_'+TAGS[b])/('models/'+str(s)+'/'+arm if chip=='22021' else arm+'/training/'+str(s))
 if chip=='22021':
  new=s in read(PAST/'main/design.json')['training_seeds'];base=(PAST/('baseline' if arm=='baseline' else 'main')) if new else (FIRST/'pre-stim-baseline' if arm=='baseline' else FIRST);return base/'models'/str(s)/arm
 root=EARLY if s<=2255408 else PAST/'22554_extra';return root/arm/'training'/str(s)
def load_cells(armset=ARMS,betas=None,locator=loc):
 allgames=[];seen=[];train=[]; prows=[];identities={};first_inputs={}
 for chip,arms in armset.items():
  seeds,worlds=study(chip);chk(len(seeds)==len(set(seeds))==16 and len(worlds)==len(set(worlds))==48,'duplicate seed/world')
  for b in (TAGS if betas is None else betas):
   for arm in arms:
    complete=0
    for seed in seeds:
     d=locator(chip,b,arm,seed);res=d/'result.json';tr=d/('holdout.jsonl' if chip=='22021' else 'eval-trace.json')
     if not (res.exists() and tr.exists()):continue
     r=read(res);ev=r['evaluation' if chip=='22021' else 'eval'];chk([e['seed'] for e in ev]==worlds,('heldout split',d));reg=read(d/'registration.json'); complete+=1
     if chip=='22021':
      chk(reg['training_seed']==seed and reg['arm']==arm,('registration',d)); byworld={g['world']:g['rows'] for g in lines(tr)};chk(list(byworld)==worlds,'22021 trace world mismatch')
      if b:
       audit=read(d/'replay-audit.json');chk(audit['seed']==seed and audit['arm']==arm and audit['worlds']==48 and reg['reward_mix']['beta']==b/100,'22021 replay/weight'); rr=read(d/'reward-audit.json');chk(len(rr)==48,'training size');check=[rewardcheck(*v,b) for v in rr];chk(all(sig==0 if (i+1)%3 else sig in (-1,1) for i,(_,sig,*_) in enumerate(rr)),'reward schedule')
      else:check=[]
     else:
      chk(reg['config']['seed']==seed and reg['eval_seeds']==worlds and r['training_seed']==seed and r['arm']==arm and r['checkpoint']['reload_verified'] and r['update']['world_size']==8 and r['training_steps']==48,('22554 registration',d));byworld={int(k):v for k,v in read(tr).items()};chk(list(byworld)==worlds,'22554 trace world mismatch');check=[]
      if b:
       chk(near(reg['config']['r2_beta'],b/100),'22554 beta'); rt=read(d/'training-trace.json');chk(len(rt)==48,'training size'); pairs=reg['train_pairs'];tests=reg['test_pairs'];chk(all(i==j for i,j,ph in tests),'evaluation donor changed');chk(all((i==j)==(arm=='ordered') for i,j,ph in pairs),'training donors not as registered')
       chk(reg['physical_files']==9 and {x[0] for x in pairs}=={x[1] for x in pairs}==set(range(6)) and {x[0] for x in tests}=={6,7,8} and not set(reg['budget']['training_seeds']).intersection(worlds),'22554 physical/game holdout split')
       for i,step in enumerate(rt):
        src,don,ph=pairs[i%len(pairs)];sg=(1 if ph=='pre' else -1) if not step['terminal'] else None
        chk(step['r2']==sg and step['recorded_phase']==ph and step['r2_trial']==(src if b else None) and step['next_state2_trial']==don,'stimulus phase/assignment');check.append(rewardcheck(step['r1'],sg,step['duration'],step['r2_contribution'],step['reward'],b))
     if chip=='22021':sig=(reg['initial'],reg['config'],reg['source_sha'],reg['windows_digest'])
     else:
      cfg={k:v for k,v in reg['config'].items() if k!='r2_beta'}
      sig=(r['initial'],cfg,reg['train_pairs'],reg['test_pairs'],reg['replay_digest'],[reg['budget'][k] for k in ('train_steps','rollout_batch','eval_horizon','game_eval_seeds')])
     chk(identities.setdefault((chip,arm,seed),sig)==sig,('initial policy, input, or budget changed',chip,arm,seed,b))
     if check:
      trn=np.array(check); train.append(dict(chip=chip,arm=arm,beta=b/100,train_seed=seed,decisions=len(check),nonzero=int(trn[:,2].sum()),sum_abs_contribution=trn[:,0].sum(),sum_abs_R1=trn[:,1].sum(),max_ratio=trn[:,3].max()))
     for e in ev:
      w=e['seed'];rows=byworld[w];chk(len(rows)==e['decisions']==6 and e.get('terminated',e.get('death'))==0,'terminal/horizon');actions=[z['action'] for z in rows]; end=tuple(e[k] for k in (['new_cells','r1_sum','final_health','final_hunger','final_thirst'] if chip=='22021' else ['explored','r1','health','hunger','thirst']));duration=e['elapsed_ticks' if chip=='22021' else 'ticks'];chk(duration>=6,'short clock');idx=digest([chip,w,actions]); signs=[]
      for i,z in enumerate(rows):
       if chip=='22554':
        k=VERBS.index(z['action']['verb']); pp=z['verb_probabilities'];chk(k==z['verb'] and len(pp)==len(VERBS) and all(p>=0 and math.isfinite(p) for p in pp) and abs(sum(pp)-1)<2e-5 and abs(math.exp(z['verb_logp'])-pp[k])<1e-5,'bad probabilities or chosen-action log probability')
        if b:
         chk(z['r2']==(1 if z['recorded_phase']=='pre' else -1),'eval sign');rewardcheck(z['r1'],z['r2'],z['duration'],z['r2_contribution'],z['reward'],b)
        if arm=='pure':chk(reg['physical_files']==0 and z['actor_native_messages'] and z['critic_input_size']==108 and z['r2_contribution']==0 and near(z['r1'],z['reward']),'unexpected pure input/reward')
       else:
        chk(z['action']['verb'] in VERBS,'unknown verb')
        if arm=='masked':chk(all(x==0 for x in z['consumed_features']),'unmasked reference')
      z=rows[0];first_input=(z['context'],z['wetware_events_sha'],z['consumed_features']) if chip=='22021' else (z['obs_hash'],z['context_donor'],z.get('state2'))
      chk(first_inputs.setdefault((chip,arm,seed,w),first_input)==first_input,'matched starting policy input changed across beta')
      allgames.append((chip,b,arm,seed,w,idx,rows,end,duration));
     if chip=='22021':prows.append(dict(seed=seed,beta=b/100,arm=arm,checkpoint_digest=r['checkpoint_digest'],source_sha=reg['source_sha'],windows=reg.get('windows_digest'),initial_actor=reg['initial']['actor']))
    seen.append(dict(chip=chip,arm=arm,beta=b/100,complete_seeds=complete,expected_seeds=16,complete_worlds=complete*48));
 return allgames,seen,train,prows

def make_observer(game,chip):
 clock.observer(game)
 if chip=='22554':
  base=(ROOT/'scripts/m9_environment_session.gd').read_text(encoding='utf8'); needle='func handle(request: Dictionary) -> Dictionary:\n';chk(base.count(needle)==1,'observer schema');(game/'scripts/mea_supplied_session.gd').write_text(base.replace(needle,needle+'\tif request.get("op", "") == "landmark":\n\t\treturn {"ok": true, "id": request.id, "landmark": world.archive_landmark}\n'),encoding='utf8')

def runreplay(games,native_name='22021-native-clock.json.gz'):
 CACHE.mkdir(exist_ok=True,parents=True); bind=digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/m9_world.gd', ROOT/'scripts/m9_environment_session.gd',ROOT/'scripts/survival_world.gd', ROOT/'agent/m9_contract.py', OBSLINK,ROOT/'.tools/godot/Godot_v4.7.2-stable_win64.exe']}); obs_cache=gzload(CACHE/'22021-game-observations.json.gz');cache=gzload(CACHE/('tick6-v2-'+bind[:12]+'.json.gz')); alluniq={g[5]:g for g in games}; n0=len(cache)
 native=gzload(CACHE/native_name)
 if any(g[0]=='22021' for g in games):
  chk(all(k in native for k,g in alluniq.items() if g[0]=='22021'),'Missing original Linux context replay; no report produced')
 for chip in ARMS:
  if chip=='22021':continue
  todo=[g for k,g in alluniq.items() if g[0]==chip and k not in cache]; print('physical tick replay',chip,len(todo),'new sequences',flush=True)
  if not todo:continue
  with tempfile.TemporaryDirectory(prefix='r2-observer-') as tmp:
   gd=Path(tmp);make_observer(gd,chip)
   with GodotEnv(str(ROOT/'.tools/godot/Godot_v4.7.2-stable_win64_console.exe'),str(gd),map_mode='infinite_freeze',navigation_steps=4) as env:
    for count,(c,b,arm,s,w,idx,rows,end,duration) in enumerate(todo):
     ob=env.reset(w)['observation'];chk(ob['world_tick']==0 and [ob['player'][a] for a in ('health','hunger','thirst')]==([100,95,95] if c=='22021' else [100,82,78]),('invalid initial game',chip,ob['player']));tracker=Reward(ob,'bottleneck',GM);start=ob['visited_count'];cum=0;cut=None; checks=[]
     for i,z in enumerate(rows):
      bef=tracker.previous;prev=ob['world_tick']; k=observation_key(ob) if c=='22021' else digest(ob)
      if c=='22021':obs_cache.setdefault(k,ob)
      nxt=env.step(z['action']);ob=nxt['observation'];t=nxt['transition'];val=tracker.score(t,ob)['reward'];sub={a:t['after'][a] for a in ('visited_count','health','hunger','thirst')};checks.append((k,digest(t['after']) if c=='22021' else sub,t['elapsed_ticks'],val));
      if cut is None and ob['world_tick']>=6:
       frame=next(v for v in t['trace'] if v['tick']==6);mid=ob if ob['world_tick']==6 else env.call('landmark')['landmark'];chk(mid['world_tick']==6 and mid['visited_count']==frame['visited_count'] and not frame['dead'],'invalid cut');dt=6-prev; rr=cum+sum(.01*GM**v for v in range(dt))+GM**dt*potential(mid,'bottleneck')-bef;cut=[mid['visited_count']-start,rr,frame['health'],frame['hunger'],frame['thirst']]; taken=i+1
      cum+=val
     chk(cut is not None,'missing game tick');cache[idx]=dict(cut=cut,clock_actions=taken,checks=checks,final=[ob['visited_count']-start,cum,t['after']['health'],t['after']['hunger'],t['after']['thirst']],ticks=ob['world_tick'])
     if (count+1)%100==0: print(chip,count+1,'/',len(todo),flush=True);gzsave(CACHE/('tick6-v2-'+bind[:12]+'.json.gz'),cache);gzsave(CACHE/'22021-game-observations.json.gz',obs_cache)
  gzsave(CACHE/('tick6-v2-'+bind[:12]+'.json.gz'),cache);gzsave(CACHE/'22021-game-observations.json.gz',obs_cache)
 cache.update(native)
 print('unique fully replayed',len(alluniq),'Linux 22021',len(native),'new 22554',len(cache)-n0,flush=True);return cache

def audit(games,cache,write_index=True):
 rowsout=[];counts=collections.defaultdict(lambda:[0,0.,0,None,None]); requests=collections.defaultdict(list); premade=gzload(CACHE/'22021-distributions.json.gz')
 for chip,b,arm,s,w,idx,rows,end,dt in games:
  v=cache[idx];chk(all(near(x,y) for x,y in zip(end,[*v['final'][:2],*[rows[-1]['after'][q] for q in ('health','hunger','thirst')]])) and v['ticks']==dt,('endpoint',chip,s,w,idx,v['final'],end));start=0
  for j,(z,(oh,after,dur,r1)) in enumerate(zip(rows,v['checks'])):
   key=z['context'] if chip=='22021' else z['obs_hash'];chk(oh==key and near(z['r1'],r1),('replay state or R1',chip,w));chk((digest(z['after'])==after if chip=='22021' else all(near(val,z['after'][k]) for k,val in after.items()) and z['duration']==dur),'per step endpoint mismatch')
   if chip=='22021':requests[(b,arm,s)].append([w,j,oh,z['wetware_window'],z['wetware_events_sha'],z['consumed_features']]);p=premade.get(f'{b}:{arm}:{s}',{}).get(f'{j}:{oh}')
   else:p=z['verb_probabilities']
   if p is not None:chk(len(p)==8 and abs(sum(p)-1)<2e-5 and min(p)>=0,'reloaded probabilities')
   if start<6:
    for k,a in enumerate(VERBS):
     acc=counts[(chip,b,arm,s,w,a)];acc[0]+=int(z['action']['verb']==a);acc[1]+=p[k] if p is not None else 0;acc[2]+=int(p is not None)
     if j==0:acc[3]=int(z['action']['verb']==a);acc[4]=p[k] if p is not None else None
   start+=dur
  rowsout.append(dict(chip=chip,beta=b/100,arm=arm,train_seed=s,world_seed=w,tick=6,**dict(zip(MET,v['cut'])),actions_started=v['clock_actions'],end_ticks=dt,**{'decision6_'+k:v_ for k,v_ in zip(MET,end)}))
 actions=[]; norm={(r['chip'],round(r['beta']*100),r['arm'],r['train_seed'],r['world_seed']):r['actions_started'] for r in rowsout}
 for (chip,b,arm,s,w,verb),(num,p,n,first,firstp) in counts.items():
  den=norm[(chip,b,arm,s,w)];chk(n in (0,den) and first is not None,'partial model output');actions.append(dict(chip=chip,beta=b/100,arm=arm,train_seed=s,world_seed=w,verb=verb,decisions=den,selected=num,probability_sum=p if n else '',probability_decisions=n,first_selected=first,first_probability=firstp if firstp is not None else ''))
 if write_index:gzsave(CACHE/'22021-evaluation-index.json.gz',{f'{b}:{a}:{s}':lst for (b,a,s),lst in requests.items()})
 return rowsout,actions

if __name__=='__main__':
 g,status,rewards,models=load_cells(); print([(x['chip'],x['beta'],x['arm'],x['complete_seeds']) for x in status],flush=True);data,actions=audit(g,runreplay(g));csvout(OUT/'fixed_tick_worlds.csv',data);csvout(OUT/'actions_by_world.csv',actions);csvout(OUT/'training_reward_audit.csv',rewards);writejson(CACHE/'22021-identities.json',models);csvout(OUT/'coverage.csv',status);print('checked',len(data),'seed x worlds; ',len(rewards),'training-reward audits',flush=True)

