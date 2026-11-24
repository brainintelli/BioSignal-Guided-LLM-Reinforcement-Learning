"""Resume the original offline 22554 sweep without changing its training/evaluation code."""
from pathlib import Path
import fcntl, hashlib, json, os, subprocess, sys, time

def get(p): return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()

def verified(old, reg, seed, arm):
    if not (old/'result.json').is_file(): return False
    res=get(old/'result.json'); c=res['checkpoint']; b=reg['budget']; m=old/'checkpoint/manifest.json'; manifest=get(m)
    if not (get(old/'registration.json')==json.loads(json.dumps(reg,allow_nan=False)) and res['training_seed']==seed and res['arm']==arm and res['training_steps']==b['train_steps'] and res['update_count']==b['train_steps']//b['rollout_batch'] and res['update']['world_size']==8 and all(c[k] for k in ('reload_verified','optimizer_roundtrip_verified','roundtrip_verified')) and sha(m)==c['manifest_sha256'] and manifest['registration_sha256']==c['registration_sha256'] and [x['seed'] for x in res['eval']]==b['game_eval_seeds'] and all(sha(m.parent/k)==v for k,v in manifest['files'].items()) and all((old/k).is_file() for k in ('training-trace.json','eval-trace.json','updates.json'))): raise ValueError(f'Completed seed does not match frozen registration: {old}')
    return True

def main(root):
    # The original source file is unchanged. The shim only places/reuses output directories.
    root=root.resolve(); os.chdir(root); lock=(root/'continuation_v1.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=root/'code/train_long_recorded.py'; frozen=code.read_text(); expected='13cd22917ddf6896'
    if not sha(code).startswith(expected): raise ValueError('Upstream trainer changed')
    seedstart='        out.mkdir()\n        cfg ='; register="        write(out/'registration.json', reg)"; assert frozen.count(seedstart)==frozen.count(register)==1
    insertion='''        prev = Path(os.environ['MEA_PREVIOUS_ARM'])/str(seed)
        if verified(prev, reg, seed, args.arm):
            out.symlink_to(prev.resolve(), target_is_directory=True)
            results.append(read(prev/'result.json'))
            print('verified and reused',seed,flush=True)
            continue
        out.mkdir()
'''
    worker=root/'continuation_v2_train.py'; worker_code='import os\nfrom resume_r2_sweep_22554 import verified\n'+frozen.replace(seedstart,'        cfg =').replace(register,insertion+register)
    if worker.exists(): assert worker.read_text()==worker_code
    else: worker.write_text(worker_code)
    env=os.environ.copy(); env.update(get(Path.home()/'mea-state2-r2-20261118/run02/environment.json')); env['PYTHONPATH']=str(root/'code')+os.pathsep+env.get('PYTHONPATH','')
    py=Path.home()/'mea-r2-22554/runtime_20261116_01/assets/python/bin/python'
    def launch(tag,arm,out,previous):
        budget=get(root/f'budget-{tag}.json'); old_done=out/'training/result.json'
        if old_done.is_file() and (out/'exit.json').is_file() and get(out/'exit.json')['exit_code']==0 and [v['training_seed'] for v in get(old_done)['seeds']]==budget['training_seeds']: return
        if out.exists(): raise FileExistsError(f'Inspect interrupted output before re-launch: {out}')
        out.mkdir(parents=True); env['MEA_PREVIOUS_ARM']=str(previous); cmd=[str(py),'-u',str(worker),'--arm',arm,'--out',str(out/'training'),'--data',str(root/'replay.json'),'--prime',str(root/'preceding-quiet.json'),'--settings',str(root/'settings.json'),'--budget',str(root/f'budget-{tag}.json')]; print(time.time(),'launch',tag,arm,flush=True)
        with (out/'process.log').open('x') as log: ret=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        (out/'exit.json').write_text(json.dumps(dict(exit_code=ret,finished=time.time(),trainer_sha256=sha(code)))); print('exit',tag,arm,ret,flush=True)
        if ret: raise RuntimeError((tag,arm,ret))
        a=get(old_done); assert a['budget']==budget and [v['training_seed'] for v in a['seeds']]==budget['training_seeds']
    launch('beta_005','shuffled',root/'resume005_v2/shuffled',root/'beta_005/shuffled/training')
    for arm in ('ordered','shuffled'): launch('beta_010',arm,root/'beta_010'/arm,root/'no_previous_completed_seeds')

if __name__=='__main__': main(Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd())

