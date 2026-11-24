"""Linux entry points for the immutable deployed package. No live action by default."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
OLD=Path.home()/'mea-state2-r2-20261118/run02'
ASSETS=Path.home()/'mea-r2-22554/runtime_20261116_01/assets'


def main(mode):
    if mode not in ('device','gpu','check'):raise ValueError('Expected check, device or gpu')
    paths=[ROOT, ROOT/'mea_online', ROOT/'mea_binary_r2', ROOT/'mea_state2_r2', ROOT/'mea_stimulation_v2']
    env=dict(os.environ)
    if mode!='device':
        env.update(json.loads((OLD/'environment.json').read_text()))
    env['PYTHONPATH']=os.pathsep.join([str(p) for p in paths]+[env.get('PYTHONPATH','')])
    if mode=='device':
        os.execve(sys.executable,[sys.executable,'-u','-m','livepair.device_rpc'],env)
    python=str(ASSETS/'python/bin/python')
    if mode=='gpu':
        os.execve(python,[python,'-u','-m','livepair.worker'],env)
    os.execve(python,[python,'-u','-m','livepair.server_check'],env)


if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'check')
