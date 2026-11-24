"""One biological observation timeline; immutable sampled inputs, R1 only."""
import copy
from pathlib import Path
import time
import uuid
from .common import append,clone,digest,integer,number,write
from .config import validate_config,validate_binding,validate_calibration
from .adapter import validate_window

def evidence(prepared):return clone({k:v for k,v in prepared.items() if k!='semantic_trees'})

def features(window,binding):
    events=window['events'];n=len(events);channels=len(binding['channel_map'])
    seconds=(window['end_frame']-window['start_frame'])/binding['sample_rate_hz']
    return [float(n),n/(seconds*channels),len({e[2] for e in events})/channels,
            sum(abs(e[3])*binding['amplitude_scale_uv'] for e in events)/n if n else 0.]

def game_context(observation):
    from agent.m9_contract import encode
    return digest(encode(observation,'reserves'))

def snapshot(window,binding,calibration,context_id):
    vals=features(window,binding)
    normalized=[max(-5.,min(5.,(v-c)/s)) for v,c,s in zip(vals,calibration['centers'],calibration['scales'])]
    s=dict(schema='mea.online.snapshot.v1',binding_sha256=digest(binding),request_id=window['request_id'],
      window_sha256=window['window_sha256'],source=window['source'],context_id=context_id,
      start_frame=window['start_frame'],end_frame=window['end_frame'],feature_values=vals,normalized_values=normalized,
      normalization_id=calibration['normalization_id'],coverage_basis='filtered_transport_frames')
    return s|{'snapshot_id':digest(s)}

def finish_rows(rows,bootstrap,gamma=.99985,lam=.999):
    """Variable-duration GAE; terminal and batch boundaries are distinct."""
    if not rows:raise ValueError('Empty rollout')
    number(bootstrap,'bootstrap')
    carry=0.
    for i in range(len(rows)-1,-1,-1):
        r=rows[i];integer(r['duration'],'duration',1)
        if type(r['terminal']) is not bool:raise ValueError('Boolean terminal required')
        number(r['reward'],'reward');number(r['value'],'value')
        boundary=r['terminal'] or i==len(rows)-1
        nxt=0. if r['terminal'] else (rows[i+1]['value'] if i+1<len(rows) else bootstrap)
        discount=gamma**r['duration'];delta=r['reward']+discount*nxt-r['value']
        carry=delta+(0. if boundary else discount*lam**r['duration']*carry)
        r.update(boundary=boundary,next_value=nxt,advantage=carry,**{'return':carry+r['value']})

class Trainer:
    def __init__(self,config,source,runtime,output,budget,live=True,context_fn=game_context,clock=time):
        self.config=validate_config(config,live=live);self.binding=validate_binding(config['binding'])
        self.calibration=validate_calibration(config['calibration'],self.binding,live=live)
        self.source=source;self.runtime=runtime;self.out=Path(output);self.budget=budget
        self.live=live;self.context_fn=context_fn;self.clock=clock
        self.previous_end=None;self.seen=set();self.run_id=uuid.uuid4().hex;self.received_mono=None
        self.result=dict(status='running',updates=0,decisions=0,windows=0,r2_enabled=False,stimulation=False,
          realtime_training=live,coverage_basis='filtered_transport_frames',errors=[],cleanup_errors=[],
          decision_window_ids=[],bootstrap_window_ids=[],bootstrap_values=[],terminal_indices=[],checkpoint_receipts=[])

    def acquire(self,observation,kind):
        self.budget.require(self.config['source']['timeout']+1)
        request_id=self.run_id+'-'+str(self.result['windows']+1)
        before=self.clock.monotonic();raw=self.source.capture(request_id);self.received_mono=self.clock.monotonic();elapsed=self.received_mono-before
        if elapsed<0 or elapsed>self.config['source']['timeout']:raise TimeoutError('Window response stale or timed out')
        self.budget.require()
        w=validate_window(raw,self.binding,request_id,self.previous_end,live=self.live)
        if w['window_sha256'] in self.seen:raise ValueError('Already consumed window')
        self.seen.add(w['window_sha256']);self.previous_end=w['end_frame']
        s=snapshot(w,self.binding,self.calibration,self.context_fn(observation))
        append(self.out/'windows.jsonl',w);append(self.out/'snapshots.jsonl',dict(kind=kind,received_monotonic=self.clock.monotonic(),snapshot=s))
        self.result['windows']+=1;self.result[kind+'_window_ids'].append(w['window_sha256'])
        return s

    def require_fresh(self):
        age=self.clock.monotonic()-self.received_mono
        if not 0<=age<=self.config['max_observation_age_seconds']:raise TimeoutError('Observation expired during inference')

    def run(self):
        self.out.mkdir(parents=True,exist_ok=False)
        registration=dict(config=self.config,source='live' if self.live else 'simulation',r2_enabled=False,
          stimulation=False,run_id=self.run_id,hard_deadline_wall=self.budget.wall_end)
        write(self.out/'registration.json',registration);reg=digest(registration)
        initial=None;final=None
        try:
            self.budget.require();initial=self.runtime.state();final=initial;write(self.out/'initial_state.json',initial)
            self.budget.require(1);seed=self.config['seed'];obs=self.runtime.reset(seed)
            for batch in range(1,self.config['batches']+1):
                self.budget.require(self.config['update_reserve_seconds'])
                rows=[];version=self.runtime.state();write(self.out/f'batch_{batch:03d}_start.json',dict(policy_version=version))
                for _ in range(self.config['batch_size']):
                    s=self.acquire(obs,'decision');pp=self.runtime.prepare(obs,s)
                    pp_hash=digest(evidence(pp));self.budget.require(1);self.require_fresh()
                    sampled=self.runtime.act(pp);self.budget.require(1)
                    if self.runtime.state()!=version:raise ValueError('Policy changed while collecting rollout')
                    self.require_fresh()
                    self.budget.require(1)
                    outcome=self.runtime.step(sampled['action'])
                    reward=number(outcome['reward'],'R1');integer(outcome['duration'],'duration',1)
                    if outcome.get('r2_enabled',False) is not False or outcome.get('total_reward',reward)!=reward:raise ValueError('Runtime violated R1-only reward')
                    if type(outcome['terminal']) is not bool:raise ValueError('Native terminal required')
                    if digest(evidence(pp))!=pp_hash:raise ValueError('Sampled input mutated')
                    row={k:copy.deepcopy(sampled[k]) for k in ('action','verb','target','value','logp','verb_logp','reference_verb','reference_target')}
                    for k in ('value','logp','verb_logp'):number(row[k],k)
                    row.update(index=self.result['decisions'],prepared=copy.deepcopy(pp),observation=clone(obs),snapshot=s,
                      transition=clone(outcome['transition']),next_observation=clone(outcome['observation']),
                      policy_version=clone(version),reward=reward,duration=outcome['duration'],terminal=outcome['terminal'],boundary=outcome['terminal'],
                      reward_record=dict(r1=reward,total_reward=reward,r2_enabled=False,alpha=1.))
                    rows.append(row);append(self.out/'decisions.jsonl',row|{'prepared':evidence(pp)})
                    self.result['decisions']+=1;obs=outcome['observation']
                    if row['terminal']:
                        self.result['terminal_indices'].append(row['index']);seed+=1;self.budget.require(1);obs=self.runtime.reset(seed)
                if rows[-1]['terminal']:boot=0.;boot_info=dict(terminal=True,value=0.,snapshot=None,prepared=None)
                else:
                    bs=self.acquire(obs,'bootstrap');bp=self.runtime.prepare(obs,bs);self.budget.require()
                    boot=number(self.runtime.value(bp),'bootstrap value');self.require_fresh()
                    boot_info=dict(terminal=False,value=boot,snapshot=bs,prepared=evidence(bp))
                self.result['bootstrap_values'].append(boot)
                finish_rows(rows,boot)
                write(self.out/f'batch_{batch:03d}_bootstrap.json',boot_info)
                write(self.out/f'batch_{batch:03d}_rollout.json',[r|{'prepared':evidence(r['prepared'])} for r in rows])
                rows_hash=digest([r|{'prepared':evidence(r['prepared'])} for r in rows])
                self.budget.require(self.config['update_reserve_seconds'])
                if self.runtime.state()!=version:raise ValueError('Policy changed before update')
                self.budget.require(self.config['update_reserve_seconds'])
                metrics=self.runtime.update(rows,boot,batch);after=self.runtime.state();final=after
                # A partial/failed update can alter weights: retain receipt but never collect more.
                write(self.out/f'batch_{batch:03d}_update.json',dict(metrics=metrics,before=version,after=after))
                if digest([r|{'prepared':evidence(r['prepared'])} for r in rows])!=rows_hash:raise ValueError('Update mutated sampled inputs')
                if any(after[k]!=initial[k] for k in ('base','reference')):raise ValueError('Frozen model or reference changed')
                if any(after[k]==version[k] for k in ('actor','critic')):raise ValueError('No actor/critic parameter change')
                self.budget.require()
                receipt=self.runtime.checkpoint(self.out/f'checkpoint_{batch:03d}',reg,batch)
                if receipt.get('reload_verified') is not True:raise ValueError('Checkpoint reload not verified')
                if self.runtime.state()!=after:raise ValueError('Checkpoint changed optimizer or model state')
                self.result['checkpoint_receipts'].append(receipt);self.result['updates']+=1
                write(self.out/f'batch_{batch:03d}_complete.json',dict(after=after,checkpoint=receipt))
            self.result['status']='completed'
        except BaseException as error:
            self.result['status']='failed';self.result['errors'].append(type(error).__name__+': '+str(error))
        finally:
            for label,obj in [('source',self.source),('runtime',self.runtime)]:
                try:obj.close()
                except BaseException as e:self.result['cleanup_errors'].append(label+': '+repr(e))
            if self.result['cleanup_errors']:self.result['status']='failed'
            self.result['finished_wall']=self.clock.time();self.result['hard_deadline_wall']=self.budget.wall_end
            if initial is not None and final is not None:
                self.result.update(actor_changed=final['actor']!=initial['actor'],critic_changed=final['critic']!=initial['critic'])
            write(self.out/'result.json',self.result)
        return clone(self.result)
