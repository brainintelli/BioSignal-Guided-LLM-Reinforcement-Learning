"""Compare only existing, identically registered, cross-host repeat runs. No seeds enter the primary sweep."""
import csv,shutil,numpy as np
import audit_r2_beta_clock as a
if __name__=='__main__':
 root=a.OUT/'technical_check'
 def other(ch,b,arm,seed):return root/ch/'beta_005'/arm/'training'/str(seed)
 oldcache=a.CACHE;a.CACHE=root/'cache';a.CACHE.mkdir(exist_ok=True)
 for f in oldcache.glob('tick6-v2*.gz'):
  if not (a.CACHE/f.name).exists():shutil.copyfile(f,a.CACHE/f.name)
 games,cov,_,_=a.load_cells({'22554':['ordered']},[5],other);assert 0<len(games)<=16*48
 ids=sorted(set(g[3] for g in games));same_actor=same_critic=0
 for seed in ids:
  x=other('22554',5,'ordered',seed);y=a.loc('22554',5,'ordered',seed)
  assert a.read(x/'registration.json')==a.read(y/'registration.json')
  xr,yr=[a.read(p/'result.json') for p in (x,y)];assert xr['initial']==yr['initial'] and xr['training_steps']==yr['training_steps']==48
  same_actor+=xr['after']['actor']==yr['after']['actor'];same_critic+=xr['after']['critic']==yr['after']['critic']
 res,act=a.audit(games,a.runreplay(games),False)
 with (a.OUT/'fixed_tick_worlds.csv').open(encoding='utf-8-sig') as f: control={(int(r['train_seed']),int(r['world_seed'])):r for r in csv.DictReader(f) if r['chip']=='22554' and r['arm']=='ordered' and r['beta']=='0.05'}
 worlds=a.study('22554')[1];d=np.empty((len(ids),len(worlds),len(a.MET)));pairs=set()
 for r in res:
  idx=r['train_seed'],r['world_seed'];assert idx not in pairs;pairs.add(idx);c=control[idx]
  d[ids.index(idx[0]),worlds.index(idx[1])]=[r[m]-float(c[m]) for m in a.MET]
 assert len(res)==len(pairs)==len(ids)*48
 rng=np.random.default_rng(2026112023);ss=rng.multinomial(len(ids),np.repeat(1/len(ids),len(ids)),20000);ww=rng.multinomial(48,np.repeat(1/48,48),20000)
 bs=((ss[:,:,None]*ww[:,None,:]).reshape(20000,-1)/(len(ids)*48)) @ d.reshape(-1,len(a.MET))
 scores=[]
 for j,met in enumerate(a.MET):
  v=d[:,:,j];lo,hi=np.quantile(bs[:,j],[.025,.975]);scores.append(dict(metric=met,mean_host23_minus_host16=v.mean(),ci95_low=lo,ci95_high=hi,mean_absolute_seed_difference=np.abs(v.mean(1)).mean(),mean_absolute_world_difference=np.abs(v).mean(),positive_seeds=int((v.mean(1)>1e-10).sum()),negative_seeds=int((v.mean(1)<-1e-10).sum())))
 a.csvout(root/'replicate_effects.csv',scores)
 a.writejson(root/'technical_check.json',dict(seeds=ids,worlds=worlds,matched_world_episodes=len(pairs),same_training_registrations=len(ids),identical_initial_tensors=len(ids),identical_trained_actor=same_actor,identical_trained_critic=same_critic,chip='22554',arm='ordered',beta=.05,metrics=scores))
 print('matched registrations',len(ids),'matched final actors',same_actor,'cells',scores[0])
