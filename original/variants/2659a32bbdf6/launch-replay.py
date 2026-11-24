import json,os,subprocess,sys,pathlib
root=pathlib.Path(__file__).resolve().parent
run=pathlib.Path.home()/'mea-state2-r2-20261118/run02'
assets=pathlib.Path.home()/'mea-r2-22554/runtime_20261116_01/assets'
python=assets/'python/bin/python'
env=dict(os.environ);env.update(json.loads((run/'environment.json').read_text()))
pkg=pathlib.Path.home()/'mea-predictability-check-369c6af196e0'
env['PYTHONPATH']=':'.join(map(str,[pkg,pkg/'mea_binary_r2',pkg/'mea_state2_r2']))+':'+env['PYTHONPATH']
if '--check' in sys.argv:
 sys.exit(subprocess.call([python,root/'crossover_replay_ppo.py','--help'],env=env))
if not (root/'ppo-data.json').exists():raise FileNotFoundError('Current crossover data')
arms=['r1','ordered','shuffled','masked']
deviceinfo=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free=[int(s.split(',')[0]) for s in deviceinfo.splitlines() if int(s.split(',')[1])>20000]
if len(free)<1:raise RuntimeError('No free training GPU')
attempt=sys.argv[1] if len(sys.argv)>1 else 'run01'
out=root/attempt;out.mkdir(exist_ok=False)
for offset in range(0,4,len(free)):
 jobs={}
 for i,arm in enumerate(arms[offset:offset+len(free)]):
  log=(out/(arm+'.log')).open('x')
  cmd=[python,'-u',root/'crossover_replay_ppo.py','--data',root/'ppo-data.json','--arm',arm,'--out',out/arm,'--model-config',run/'model_config.json','--godot',assets/'godot/Godot_v4.7.2-stable_linux.x86_64','--project',run/'project']
  jobs[arm]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=str(free[i])));log.close()
 rc={arm:j.wait() for arm,j in jobs.items()};print(rc,flush=True)
 if any(rc.values()):sys.exit(1)
