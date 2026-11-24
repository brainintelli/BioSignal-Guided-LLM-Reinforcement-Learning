"""Eight-GPU worker. Stdin/stdout RPC carries windows and receipts, never device commands."""
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

from mea_binary_r2.encoding import Encoder
from mea_online.m9_runtime import DEFAULTS, EXPECTED_ACTOR, M9Runtime, canonical, digest
from .protocol import Buffer, PROTOCOL, STEPS, metrics, reward, snapshot, transaction, scheduled_intent
from . import predictability


def publish(path, row):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(canonical(row))


class Worker:
    def __init__(self):
        self.rt = self.phase = self.pending = None
        self.updated = False

    def initialize(self, config, out):
        if self.rt is not None:
            raise ValueError('Already initialized')
        self.config, self.out = config, Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        if config['experiment'] != PROTOCOL:
            raise ValueError('Wrong study')
        publish(self.out/'registration.json',config)
        self.rt = M9Runtime(config,self.out)
        return dict(initial=self.rt.state(), devices=config['devices'])

    def start(self, phase, windows, episode=0):
        if self.pending or phase not in ('train','eval') or (phase=='eval') != self.updated:
            raise ValueError('Invalid phase')
        if episode:
            if phase != 'train' or self.phase != phase or not self.rows[-1]['terminal'] or episode != self.episode+1:
                raise ValueError('Invalid episode reset')
        elif self.phase == phase:
            raise ValueError('Duplicate phase start')
        else:
            self.rows = []
        self.phase, self.episode = phase, episode
        seed = self.config['seed'] + (100000 if phase == 'eval' else 0) + episode
        self.obs = self.rt.reset(seed)
        self.buffer = Buffer(self.config['arm'],seed,windows,self.config['binding_sha256'])
        self.encoder = Encoder(3)
        return True

    def prepared(self, w, key):
        from agent.m9_contract import encode
        ctx = digest(encode(self.obs, DEFAULTS['state_variant']))
        return self.rt.prepare(self.obs, snapshot(w,self.config,ctx,key))

    def act(self):
        import torch
        if self.pending or len(self.rows)>=STEPS or (self.rows and self.rows[-1]['terminal']
                and self.rows[-1]['episode']==self.episode):
            raise ValueError('Decision outside active episode')
        index = len(self.rows)
        key = f'{self.phase}-{self.episode}-{index}'
        prepared = self.prepared(self.buffer.take(),key)
        torch.manual_seed(self.config['seed'] + (200000 if self.phase=='eval' else 10000) + index)
        sampled = self.rt.act(prepared)
        step = self.rt.step(sampled['action'])
        self.obs = step['observation']
        req = None
        if not step['terminal']:
            args = (self.config['binding_sha256'], self.config['arm'], self.phase, self.episode, index, digest(self.obs))
            if self.config.get('feedback_encoding') == predictability.MODE:
                req = predictability.request(*args, step['transition'])
            else:
                p = self.obs['player']
                intent = scheduled_intent(self.encoder,dict(H=p['health'],W=p['thirst'],F=p['hunger']),False,index)
                req = transaction(*args, intent)
        self.pending = dict(**sampled,prepared=prepared,terminal=step['terminal'],boundary=step['terminal'],
                            duration=step['duration'],r1=step['reward'],transition=step['transition'],
                            episode=self.episode,request=req)
        return dict(request=req,terminal=step['terminal'])

    def accept(self, receipt):
        row = self.pending
        if row is None:
            raise ValueError('No pending action')
        score = reward(row['r1'],row['duration'],row['terminal'],row['request'],receipt)
        if receipt is not None:
            self.buffer.push(receipt['window'])
        row.update(score=score,reward=score['total'],receipt=receipt)
        with (self.out/(self.phase+'.jsonl')).open('a',encoding='utf-8') as f:
            f.write(canonical(row)+'\n')
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        self.pending = None
        return True

    def finish(self):
        if self.pending or not self.rows:
            raise ValueError('Unresolved transition')
        if self.phase == 'eval':
            result=metrics(self.rows)
            if self.rt.state()!=self.evaluation_weights:
                raise RuntimeError('Weights changed during held-out evaluation')
            publish(self.out/'eval.json',result)
            return result
        if self.updated or len(self.rows) != STEPS:
            raise ValueError('Complete fixed training budget required')
        boot=0. if self.rows[-1]['terminal'] else self.rt.value(self.prepared(self.buffer.peek(),'train-bootstrap'))
        result=self.rt.update(self.rows,boot,1)
        if not result['actor_changed'] or not result['critic_changed']:
            raise RuntimeError('Optimizer update missing')
        ck=self.rt.checkpoint(self.out/'checkpoint',digest(self.config),1)
        publish(self.out/'update.json',dict(result=result,checkpoint=ck))
        self.updated=True
        self.evaluation_weights=self.rt.state()
        return dict(checkpoint=ck['manifest_sha256'],devices=self.config['devices'])


def main():
    # Preserve a dedicated wire FD even if Godot, libraries or replica children print.
    wire=os.fdopen(os.dup(1),'w',encoding='utf-8',buffering=1)
    os.dup2(sys.stderr.fileno(),1)
    w=Worker()
    try:
        for line in sys.stdin:
            try:
                cmd=json.loads(line)
                if cmd['op']=='close':
                    break
                if cmd['op'] not in ('initialize','start','act','accept','finish'):
                    raise ValueError('Unknown operation')
                result=getattr(w,cmd['op'])(**cmd.get('args',{}))
                print(canonical(dict(ok=True,value=result)),file=wire,flush=True)
            except Exception:
                print(canonical(dict(ok=False,error=traceback.format_exc())),file=wire,flush=True)
                break
    finally:
        if w.rt is not None:w.rt.close()


if __name__=='__main__':
    main()
