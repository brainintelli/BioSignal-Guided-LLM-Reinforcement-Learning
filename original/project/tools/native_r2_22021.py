"""Read-only clock and policy replay on the original 22021 training runtime."""
import collections, concurrent.futures, gzip, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace

HOME=Path.home()
E=HOME/'r2-evidence-20261120'
STUDY=HOME/'mea-six-beta-22021-beta002'
sys.path.insert(0,str(STUDY))
from resources import PYTHON, GODOT, PROJECT, environment
def get(p): return json.loads(Path(p).read_bytes())
def gz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def dump(p,x):
    tmp=p.with_suffix('.part.gz')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=3) as f: json.dump(x,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    tmp.replace(p)
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def native_clock(reference=False):
    from agent.m9_env import GodotEnv
    from agent.m9_contract import Reward,potential
    from agent.mea_contract import observation_key
    prefix='22021-reference' if reference else '22021-native'
    req=gz(E/(prefix+'-index.json.gz')); result={}; obsmap={};gamma=.99985
    with tempfile.TemporaryDirectory(prefix='r2-landmark-') as tmp:
        project=Path(tmp)
        for name in ('scripts','configs','godot_tests'):shutil.copytree(PROJECT/name, project/name)
        shutil.copy2(PROJECT/'project.godot',project/'project.godot')
        for file,needle,addition in [('scripts/mea_supplied_session.gd','func handle(request: Dictionary) -> Dictionary:\n','\tif request.get("op", "") == "landmark":\n\t\treturn {"ok": true, "id": request.id, "landmark": world.archive_landmark}\n'),('scripts/m9_world.gd','var navigation_steps: int = 1','\nvar archive_landmark: Dictionary = {}'),('scripts/m9_world.gd','\t\tsuper._advance(1)\n\t\t_sync_deadlines()','\n\t\tif tick == 6:\n\t\t\tarchive_landmark = observation().duplicate(true)')]:
            path=project/file;src=path.read_text();assert src.count(needle)==1
            path.write_text(src.replace(needle,needle+addition))
        with GodotEnv(str(GODOT),str(project),map_mode='infinite_freeze',navigation_steps=4) as env:
            for key,(world,rows) in req['unique'].items():
                start=env.reset(world);assert start['starting_scene']=='supplied_95';ob=start['observation']; visits=ob['visited_count']; rr=Reward(ob,'bottleneck',gamma);cum=0.; checks=[];cut=None
                for j,row in enumerate(rows):
                    ctx=observation_key(ob);assert ctx==row['context'],('model input',world,j,ctx,row['context'])
                    if ctx not in obsmap:obsmap[ctx]=ob
                    prev=rr.previous; beg=ob['world_tick'];st=env.step(row['action']);ob=st['observation'];tr=st['transition'];v=rr.score(tr,ob)['reward']
                    assert tr['after']==row['after'] and math.isclose(v,row['r1'],abs_tol=1e-11)
                    checks.append([ctx,digest(tr['after']),tr['elapsed_ticks'],v])
                    if cut is None and ob['world_tick']>=6:
                        frame=next(z for z in tr['trace'] if z['tick']==6); mid=ob if ob['world_tick']==6 else env.call('landmark')['landmark']; dt=6-beg
                        assert mid['world_tick']==6 and mid['visited_count']==frame['visited_count'] and not frame['dead']
                        cut=[mid['visited_count']-visits,cum+sum(.01*gamma**i for i in range(dt))+gamma**dt*potential(mid,'bottleneck')-prev,frame['health'],frame['hunger'],frame['thirst']];n=j+1
                    cum+=v
                assert cut is not None
                result[key]=dict(cut=cut,clock_actions=n,checks=checks,final=[ob['visited_count']-visits,cum,*[tr['after'][m] for m in ('health','hunger','thirst')]],ticks=ob['world_tick'])
    dump(E/(prefix+'-clock.json.gz'),result)
    if not reference:dump(E/(prefix+'-obs.json.gz'),obsmap)
    print('native clock checked',len(result),'trajectories,',len(obsmap),'distinct model inputs',flush=True)

def infer(lane,lanes):
    import torch
    from agent.m9_policy import IDENTITY_KEYS
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    from train import FixedContextPolicy,CONFIG,ReplayRewardMix
    req=gz(E/'22021-native-index.json.gz');obs=gz(E/'22021-native-obs.json.gz'); out=E/'probability-parts';out.mkdir(exist_ok=True)
    cache={}; policies={}; ctxmap={a:get(STUDY/(a+'.json')) for a in ['baseline','sham']}; seen=0
    for name in sorted(req['models'])[lane::lanes]:
        dest=out/(name.replace(':','_')+'.json.gz');item=req['models'][name];b,arm,s=name.split(':');s=int(s);b=int(b)
        if dest.exists():
            prev=gz(dest)
            if prev['checkpoint']==item['checkpoint'] and len(prev['p'])==len(item['frames']):continue
            raise ValueError(('existing mismatched inference',name))
        root=HOME/('mea-six-beta-22021-beta'+{2:'002',5:'005',10:'010'}[b])/'models'/str(s)/arm if b else (HOME/'mea-six-next-22021'/('baseline' if arm=='baseline' else 'main')/'models'/str(s)/arm if 102600001<=s<=102600008 else E/'original22021'/str(s)/arm)
        cp=root/'checkpoint'; man=get(cp/'manifest.json'); assert digest(man)==item['checkpoint']==get(root/'result.json')['checkpoint_digest']
        reg=get(root/'registration.json'); context=ctxmap[arm]; assert reg['source_sha']==context['source_sha256'] and reg['windows_digest']==digest(context['series'])
        if arm not in policies:
            torch.manual_seed(6100000);cfg=SimpleNamespace(**(CONFIG|dict(minibatch=12,critic_minibatch=12)),**get(STUDY/'model_config.json'),rng_seed=6100000)
            policies[arm]=FixedContextPolicy(cfg,arm,context,None)
        policy=policies[arm];policy.reward_mix=ReplayRewardMix(b/100) if b else None
        meta=get(cp/'policy.json');current=policy.metadata()
        assert all(meta[k]==current[k] for k in IDENTITY_KEYS)
        for f in ('policy.json','critic.pt','adapter/adapter_model.safetensors'):
            assert hashlib.sha256((cp/f).read_bytes()).hexdigest()==man['files'][f]
        set_peft_model_state_dict(policy.model,load_file(str(cp/'adapter/adapter_model.safetensors')))
        policy.critic.load_state_dict(torch.load(cp/'critic.pt',map_location=policy.device_name,weights_only=True),strict=True)
        policy.execution_cache.revisions.clear();policy.clear_prefix_cache();assert policy.digest()==man['actor_sha256']
        p={}
        with torch.inference_mode():
            for key,expected in item['frames'].items():
                j,ctx=key.split(':'); ck=(arm,key)
                if ck not in cache:
                    prep,snap=policy.replay_input(obs[ctx],context['series']['holdout'][int(j)])
                    assert snap['context_id']==ctx and [snap['window_index'],snap['events_sha256'],prep['wetware_features']]==expected
                    cache[ck]=prep
                p[key]=policy.verb_distribution([cache[ck]]).probs[0].tolist()
        assert all(len(row)==8 and min(row)>=0 and abs(sum(row)-1)<2.e-5 for row in p.values())
        dump(dest,{'checkpoint':item['checkpoint'],'p':p});seen+=1;print('inferred',name,len(p),flush=True)
    print('worker completed',lane,seen,flush=True)

def driver():
    subprocess.run([str(PYTHON),'-u',__file__,'clock'],env=environment(),check=True)
    space=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    cards=sorted([(int(line.split(',')[1]),int(line.split(',')[0])) for line in space.splitlines()],reverse=True)
    cards=[i for mb,i in cards if mb>=12500][:4]
    if not cards:raise RuntimeError('No GPU with adequate free memory; clocks saved.')
    def start(pair):
        lane,card=pair;env=environment();env['CUDA_VISIBLE_DEVICES']=str(card)
        with (E/('infer-lane-%d.log'%lane)).open('a') as f:
            subprocess.run([str(PYTHON),'-u',__file__,'infer',str(lane),str(len(cards))],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards)) as ex:list(ex.map(start,enumerate(cards)))
    req=gz(E/'22021-native-index.json.gz');out={}
    for name,item in req['models'].items():
        row=gz(E/'probability-parts'/(name.replace(':','_')+'.json.gz'));assert row['checkpoint']==item['checkpoint'] and row['p'].keys()==item['frames'].keys()
        out[name]=row['p']
    dump(E/'22021-distributions.json.gz',out);print('verified all',len(out),'frozen models',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('clock','reference'):native_clock(sys.argv[1]=='reference')
    elif sys.argv[1]=='infer':infer(int(sys.argv[2]),int(sys.argv[3]))
    else:driver()
