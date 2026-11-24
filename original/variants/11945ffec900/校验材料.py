"""用 Python 3 标准库核对压缩包，可再逐个验证确认性实验的参数文件。"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

def digest(f):
    h=hashlib.sha256()
    for block in iter(lambda:f.read(4*1024*1024),b''):
        h.update(block)
    return h.hexdigest()

def check_components(root):
    pairs={}
    for line in (root/'确认性实验_原始外部清单.sha256').read_text(encoding='utf-8').splitlines():
        h,k=line.split(maxsplit=1)
        pairs[k]=h
    if len(pairs)!=576:
        raise ValueError('预期有 576 项外部文件指纹')
    with ZipFile(root/'确认性实验_checkpoint.zip') as z:
        for k,h in pairs.items():
            with z.open(k) as f:
                if digest(f)!=h:raise ValueError('参数指纹不符 '+k)
        indexes=[s for s in z.namelist() if s.endswith('/checkpoint/manifest.json')]
        if len(indexes)!=96:raise ValueError('内部参数清单数不等于 96')
        for idx in indexes:
            data=json.loads(z.read(idx))
            prefix=idx.rsplit('/',1)[0]
            if len(data['files'])!=6 or any(pairs[prefix+'/'+k]!=h for k,h in data['files'].items()):
                raise ValueError('内外参数清单不符 '+idx)
    print('通过：96 个模型、576 个参数组件、96 份内部清单')

def check_training(root):
    decisions=0
    batches=0
    with (root/'确认性实验_96个模型索引.csv').open(encoding='utf-8-sig',newline='') as f:
        models=list(csv.DictReader(f))
    if len(models)!=96:raise ValueError('确认性实验模型数不符')
    with ZipFile(root/'确认性实验_训练日志与程序.zip') as z:
        for m in models:
            reg=json.loads(z.read(m['日志包内登记']))
            up=json.loads(z.read(m['日志包内更新']))
            group=reg['arm']
            if (str(reg['training_seed']),group)!=(m['训练种子'],m['输入组']):raise ValueError('训练登记与索引不一致')
            if up['actor_steps']!=[8] or up['critic_steps']!=[64]:raise ValueError('更新步数不符')
            gamma=reg['config']['gamma_tick']
            lam=reg['config']['gae_lambda']
            folder=m['日志包内登记'].rsplit('/',1)[0]
            for i,rnd in enumerate(up['rounds'],1):
                name=f'{folder}/train-{i}.jsonl'
                with z.open(name) as f:rows=[json.loads(line) for line in f]
                if len(rows)!=12:raise ValueError('批决策数有异 '+name)
                nval=rnd['bootstrap'];carry=0.0
                for j in reversed(range(len(rows))):
                    r=rows[j];last=(j==len(rows)-1) or r['terminal']
                    vnext=0.0 if r['terminal'] else nval
                    disc=gamma**r['duration']
                    reward=r['r1_score']['reward']
                    delta=reward+disc*vnext-r['value']
                    carry=delta if last else delta+disc*lam**r['duration']*carry
                    expected=[(r['contribution'],0),(r['reward'],reward),(r['next_value'],vnext),
                              (r['advantage'],carry),(r['return'],carry+r['value'])]
                    if r['boundary']!=last or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-10) for a,b in expected):
                        raise ValueError(f'轨迹数值有异 {name} 第 {j+1} 行')
                    nval=r['value']
                decisions+=len(rows);batches+=1
    if (batches,decisions)!=(384,4608):raise ValueError('训练日志覆盖数有异')
    print('通过：384 份训练轨迹、4,608 次训练决策的回报、时长折扣、优势及价值目标')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--checkpoints-only',action='store_true',help='只核对确认性实验的模型组件和内部清单')
    group.add_argument('--training-only',action='store_true',help='只核对逐批训练记录、回报和优势')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    if not args.checkpoints_only and not args.training_only:
        with (root/'交付文件校验.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                p=root/row['文件']
                if p.stat().st_size!=int(row['字节']):raise ValueError('文件长度不符 '+row['文件'])
                with p.open('rb') as stream:
                    if digest(stream)!=row['SHA256']:raise ValueError('文件指纹不符 '+row['文件'])
                print('通过',row['文件'],flush=True)
    if not args.training_only:check_components(root)
    if not args.checkpoints_only:check_training(root)
if __name__=='__main__':
    try:main()
    except (KeyError,ValueError,OSError) as error:
        print('校验中断',str(error),file=sys.stderr)
        sys.exit(1)
