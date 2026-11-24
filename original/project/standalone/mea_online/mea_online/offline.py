"""Deterministic synthetic source and tiny numeric PPO control-flow verifier.

No network, MaxLab, GPU or Qwen. These results never qualify biological input.
"""
import math
from datetime import datetime,timezone
from pathlib import Path
import random
import tempfile
from .common import clone,digest,read,write
from .config import Budget,template
from .trainer import Trainer,finish_rows

class Clock:
    def __init__(self):self.wall=1000.;self.mono=0.
    def time(self):return self.wall
    def monotonic(self):return self.mono
    def advance(self,n):self.wall+=n;self.mono+=n

def fixture_binding():
    return dict(session_id='synthetic-online',chip_id='synthetic-chip',organoid_id='synthetic-organoid',
      started_utc='1970-01-01T00:16:40+00:00',channel_map=[[0,10],[1,11]],sample_rate_hz=20000,window_frames=20000,well_id=0,amplitude_scale_uv=1.)

def fixture_calibration(binding):
    c=dict(schema='mea.online.calibration.v1',binding_sha256=digest(binding),source='simulation',
      coverage_basis='filtered_transport_frames',centers=[1.,.5,.5,2.],scales=[1.,.5,.5,1.],
      calibration_window_ids=[digest(['synthetic-calibration',i]) for i in range(60)],
      validation_window_ids=[digest(['synthetic-validation',i]) for i in range(30)],status='engineering_validated')
    return c|dict(normalization_id=digest(c))

class SyntheticSource:
    def __init__(self,binding,clock,fault_at=None,fault='disconnect',expire_at=None):
        self.binding=binding;self.clock=clock;self.fault_at=fault_at;self.fault=fault;self.expire_at=expire_at;self.calls=0;self.closed=False
    def capture(self,request_id):
        self.calls+=1;n=self.calls;self.clock.advance(2.)
        start=(n-1)*40000+20000
        w=dict(schema='mea.online.window.v1',source='simulation',request_id=request_id,binding_sha256=digest(self.binding),session_id=self.binding['session_id'],
          start_frame=start,end_frame=start+20000,frame_count=20000,complete=True,dropped=0,corrupted=0,
          stream_opened_after_request=True,warmup_frames=20000,warmup_elapsed_s=1.,capture_elapsed_s=1.,
          events=[[start+i*100,start+i*100-8,i%2,-float(i+2)] for i in range(1+n%3)])
        if n==self.expire_at:self.clock.advance(7200)
        if n==self.fault_at:
            if self.fault=='disconnect':raise ConnectionError('Synthetic disconnect')
            if self.fault=='duplicate':
                delta=40000;w['start_frame']-=delta;w['end_frame']-=delta
                w['events']=[[e[0]-delta,e[1]-delta,*e[2:]] for e in w['events']]
            elif self.fault=='stale':self.clock.advance(9)
            elif self.fault=='wrong_source':w['source']='live'
            elif self.fault=='wrong_binding':w['binding_sha256']='0'*64
            elif self.fault=='missing_frame':w['frame_count']-=1
            elif self.fault=='nan':w['capture_elapsed_s']=float('nan')
        w['window_sha256']=digest(w)
        if n==self.fault_at and self.fault=='mutated_hash':w['events'][0][3]=999
        return w
    def close(self):self.closed=True

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def probability(weights,x):return 1/(1+math.exp(-max(-30,min(30,dot(weights,x)))))

class TinyRuntime:
    """Small logistic actor/linear critic; validates orchestration, not M9."""
    def __init__(self,terminal_every=None,reward_violation=False,clock=None,inference_delay=0.,state_delay=0.,budget=None):
        self.actor=[.01]*6;self.critic=[.02]*6;self.steps=0;self.reset_count=0;self.closed=False
        self.terminal_every=terminal_every;self.reward_violation=reward_violation;self.rng=random.Random(42);self.last_rows=[]
        self.clock=clock;self.inference_delay=inference_delay;self.state_delay=state_delay;self.audit_pending=False;self.budget=budget;self.actions_remaining=[]
    def state(self):
        if self.audit_pending and self.clock:self.clock.advance(self.state_delay);self.audit_pending=False
        return dict(actor=digest(self.actor),critic=digest(self.critic),base=digest('frozen-tiny-base'),reference=digest('frozen-tiny-reference'))
    def reset(self,seed):self.reset_count+=1;self.pos=0;self.episode_step=0;return dict(position=0,seed=seed)
    def prepare(self,observation,snapshot):
        if snapshot['snapshot_id']!=digest({k:v for k,v in snapshot.items() if k!='snapshot_id'}):raise ValueError('Snapshot changed')
        return dict(x=[1.,observation['position']/10.]+snapshot['normalized_values'],snapshot_id=snapshot['snapshot_id'],messages=[dict(role='user',content=str(snapshot['normalized_values']))])
    def value(self,prepared):return dot(self.critic,prepared['x'])
    def act(self,prepared):
        if self.clock:self.clock.advance(self.inference_delay)
        self.audit_pending=True
        p=probability(self.actor,prepared['x']);a=int(self.rng.random()<p);logp=math.log(p if a else 1-p)
        return dict(action=a,verb=a,target=0,value=self.value(prepared),logp=logp,verb_logp=logp,reference_verb=[.5,.5],reference_target=[1.])
    def step(self,action):
        if self.budget:self.actions_remaining.append(self.budget.remaining())
        self.episode_step+=1;self.pos+=action
        r=1. if action else -.2;terminal=self.terminal_every is not None and self.episode_step==self.terminal_every
        return dict(observation=dict(position=self.pos,seed=42),transition=dict(action=action,position=self.pos),reward=r,
          duration=1+self.episode_step%2,terminal=terminal,total_reward=r+1 if self.reward_violation else r,r2_enabled=self.reward_violation)
    def update(self,rows,bootstrap,update_id):
        # Independent reverse TD recursion checks the controller's GAE.
        carry=0.
        for i in range(len(rows)-1,-1,-1):
            r=rows[i];nxt=0. if r['terminal'] else (rows[i+1]['value'] if i+1<len(rows) else bootstrap)
            d=.99985**r['duration'];carry=r['reward']+d*nxt-r['value']+(d*.999**r['duration']*carry if i+1<len(rows) and not r['terminal'] else 0.)
            if abs(carry-r['advantage'])>1e-10:raise ValueError('GAE mismatch')
        self.last_rows=clone(rows)
        for r in rows:
            x=r['prepared']['x'];p=probability(self.actor,x);chosen=p if r['action'] else 1-p
            ratio=math.exp(math.log(chosen)-r['logp']);adv=r['advantage']
            active=not ((adv>0 and ratio>1.2) or (adv<0 and ratio<.8))
            factor=adv*ratio*(r['action']-p) if active else 0.
            err=r['return']-dot(self.critic,x)
            self.actor=[w+.002*max(-10,min(10,factor*z)) for w,z in zip(self.actor,x)]
            self.critic=[w+.002*max(-10,min(10,err*z)) for w,z in zip(self.critic,x)]
        self.steps+=1
        return dict(backend='offline_tiny_numeric_not_m9',gae_verified=True,optimizer_steps=self.steps,finite=all(math.isfinite(v) for v in self.actor+self.critic))
    def checkpoint(self,path,registration_sha,update_id):
        path=Path(path);path.mkdir();saved=dict(actor=self.actor,critic=self.critic,steps=self.steps,registration_sha=registration_sha,update=update_id)
        write(path/'weights.json',saved);loaded=read(path/'weights.json')
        if loaded!=saved:raise ValueError('Tiny checkpoint reload mismatch')
        self.actor=loaded['actor'];self.critic=loaded['critic'];self.steps=loaded['steps']
        return dict(reload_verified=True,backend='offline_tiny_numeric_not_m9',path=str(path),sha256=digest(loaded))
    def close(self):self.closed=True

def exercise(*,output=None,terminal_every=None,fault_at=None,fault='disconnect',expire_at=None,runtime_reward_violation=False,inference_delay=0.,state_delay=0.,initial_remaining=7200.):
    temporary=tempfile.TemporaryDirectory() if output is None else None
    out=Path(temporary.name)/'run' if temporary else Path(output)
    clock=Clock();binding=fixture_binding();start=1000.-(7200.-initial_remaining);binding['started_utc']=datetime.fromtimestamp(start,timezone.utc).isoformat()
    c=template();c.update(binding=binding,calibration=fixture_calibration(binding),batch_size=4,batches=2,update_reserve_seconds=30)
    budget=Budget(start,clock);src=SyntheticSource(binding,clock,fault_at,fault,expire_at);runtime=TinyRuntime(terminal_every,runtime_reward_violation,clock,inference_delay,state_delay,budget)
    result=Trainer(c,src,runtime,out,budget,live=False,context_fn=digest,clock=clock).run()
    result.update(backend='offline_tiny_numeric_not_m9',source_closed=src.closed,runtime_closed=runtime.closed,reset_count=runtime.reset_count,
      gae_verified=bool(result['updates']),actions_remaining=runtime.actions_remaining,reload_verified=bool(result['updates']) and all(x['reload_verified'] for x in result['checkpoint_receipts']))
    if temporary:temporary.cleanup()
    return result
