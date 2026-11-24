"""Explicit offline preparation/registration/run entry. Never controls an MEA."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from sr_data import Data,build,canonical,read,sha,write,require

ROOT=Path(__file__).resolve().parent
PROJECT_SOURCE=Path('/home/researcher/mea-r2-22554/state_reward_20261116_01/project')
RUNTIME=Path('/home/researcher/mea-r2-22554/runtime_20261116_01/assets')


def bootstrap_runtime():
    project=ROOT/'project'
    if not project.exists():
        require(PROJECT_SOURCE.is_dir(),'validated project missing: '+str(PROJECT_SOURCE))
        project.symlink_to(PROJECT_SOURCE,target_is_directory=True)
    require(project.resolve()==PROJECT_SOURCE.resolve(),'unexpected project dependency; use verified previous project')
    paths=[str(ROOT),str(RUNTIME/'overlay'),str(RUNTIME/'site'),str(project)]
    sys.path[:0]=paths
    os.environ['PYTHONPATH']=os.pathsep.join(paths)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['OMP_NUM_THREADS']='4'


def validate_dataset():
    pin=read(ROOT/'dataset_pin.json')['sha256']
    data=Data(ROOT/'dataset.json',pin,ROOT/'input'/'manifest.json')
    return dict(dataset_sha256=pin,manifest_sha256=data.data['manifest_sha256'],windows=len(data.data['windows']),
                excluded_windows=len(data.data['excluded_windows']),r1_weight=1.,r2_enabled=False,aux_enabled=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('build',help='CPU-only; derive pinned dataset from input/manifest.json')
    sub.add_parser('validate',help='CPU-only; reconstruct features and verify pins')
    for name in ('register','run','analyze'):
        q=sub.add_parser(name);q.add_argument('--root',type=Path,required=True)
    a=p.parse_args(argv)
    if a.action=='build':
        result=build(ROOT/'input'/'manifest.json',ROOT/'dataset.json')
        write(ROOT/'dataset_pin.json',dict(sha256=result['dataset_sha256']))
        print(canonical(result));return
    result=validate_dataset()
    if a.action=='validate':print(canonical(result));return
    require(a.root.resolve().is_relative_to(ROOT),'experiment output must stay within this new directory')
    bootstrap_runtime()
    if a.action=='register':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','register with CUDA_VISIBLE_DEVICES=0')
        from sr_run import register
        register(a.root.resolve(),False)
    elif a.action=='run':
        from sr_driver import main as driver
        driver(a.root.resolve())
        subprocess.run([sys.executable,str(ROOT/'sr_analyze.py'),'--root',str(a.root.resolve())],check=True,env=os.environ.copy())
    else:
        from sr_analyze import main as analyze
        analyze(a.root.resolve())


if __name__=='__main__':main()
