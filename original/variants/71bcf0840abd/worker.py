"""One isolated GPU fit and an independent CPU replay of the archived actions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback

root=Path(__file__).resolve().parent
arm,seed,gpu=sys.argv[1:]
job=root/'jobs'/(arm+'-'+seed)
folder=root/'pre-stim-baseline' if arm=='baseline' else root
source=folder/'models'/seed/arm
assets=Path.home()/'mea-r2-22554/runtime_20261116_01/assets'
py=str(assets/'python/bin/python')
game=root/'game'
godot=assets/'godot/Godot_v4.7.2-stable_linux.x86_64'
args=['--root',str(folder),'--arm',arm,'--seed',seed,'--project',str(game),'--godot',str(godot)]
env=dict(os.environ)
env.update(json.loads((Path.home()/'mea-state2-r2-20261118/run02/environment.json').read_bytes()))
pkg=root/'pkg'
env['PYTHONPATH']=os.pathsep.join(map(str,[pkg,pkg/'mea_binary_r2',pkg/'mea_state2_r2',root/'code',env['PYTHONPATH']]))
env['CUDA_VISIBLE_DEVICES']=gpu
def atom(name,obj):
    p=job/name
    tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)
try:
    if not (source/'result.json').exists():
        subprocess.run([py,'-u',str(folder/'train.py'),*args,'--model-config',str(folder/'model_config.json')],env=env,check=True)
    replay_args=['--executable' if a=='--godot' else a for a in args]
    subprocess.run([py,'-u',str(folder/'verify_game.py'),*replay_args],env=env,check=True)
    path=job/'result.tar'
    with tarfile.open(path.with_suffix('.partial'),'w') as tar:
        for file in sorted(source.rglob('*')):
            if file.is_file():tar.add(file,arcname=file.relative_to(source).as_posix(),recursive=False)
    path.with_suffix('.partial').replace(path)
    with path.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    atom('done.json',dict(sha256=h.hexdigest(),bytes=path.stat().st_size,arm=arm,seed=int(seed),gpu=int(gpu)))
except Exception:
    atom('error.json',dict(message=traceback.format_exc()))
    raise
