import json,os,pathlib,subprocess,sys,time,traceback
root=pathlib.Path(__file__).resolve().parent
arm=sys.argv[1]; out=root/arm
out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ);env.update(json.loads((pathlib.Path.home()/'mea-state2-r2-20261118/run02/environment.json').read_text()))
env['PYTHONPATH']=str(root/'code')+os.pathsep+env.get('PYTHONPATH','')
py=str(pathlib.Path.home()/'mea-r2-22554/runtime_20261116_01/assets/python/bin/python')
cmd=[py,'-u',str(root/'code/train_long_recorded.py'),'--arm',arm,'--out',str(out/'training'),'--data',str(root/'replay.json'),'--prime',str(root/'preceding-quiet.json'),'--settings',str(root/'settings.json'),'--budget',str(root/'budget.json')]
try:
 p=subprocess.run(cmd,env=env)
 state={'exit_code':p.returncode,'finished':time.time()}
except Exception:
 state={'exit_code':1,'error':traceback.format_exc(),'finished':time.time()}
(out/'exit.json').write_text(json.dumps(state))
sys.exit(state['exit_code'])
