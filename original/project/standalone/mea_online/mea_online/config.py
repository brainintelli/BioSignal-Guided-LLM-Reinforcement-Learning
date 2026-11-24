"""Unbound templates and original biological session budgets."""
from datetime import datetime, timezone, timedelta
import re
import time
from pathlib import Path
from .common import clone,digest,integer,number,write

BINDING_KEYS={'session_id','chip_id','organoid_id','started_utc','channel_map','sample_rate_hz','window_frames','well_id','amplitude_scale_uv'}

def template():
    return dict(schema='mea.online.run.v1',gpu_host='192.0.2.16',devices=list(range(8)),
      binding=None,calibration=None,source=dict(host='127.0.0.1',port=15561,timeout=8.),
      model_paths=dict(model=None,initial_checkpoint=None,parent_checkpoint=None,warm_checkpoint=None),project=None,godot=None,
      seed=740001,batch_size=16,batches=2,max_session_seconds=7200,finish_reserve_seconds=120,update_reserve_seconds=300,max_observation_age_seconds=10,
      r2_enabled=False,stimulation=False,reward='r1_only',registry=None)

def started_wall(binding):
    try:
        t=datetime.fromisoformat(binding['started_utc'].replace('Z','+00:00'))
        if t.tzinfo is None or t.utcoffset()!=timedelta(0):raise ValueError('UTC timezone required')
        return t.timestamp()
    except (TypeError,KeyError,AttributeError,OverflowError) as e:raise ValueError('Invalid original experiment start') from e

def validate_binding(binding):
    if not isinstance(binding,dict) or set(binding)!=BINDING_KEYS:raise ValueError('Complete explicit binding required')
    for k in ('session_id','chip_id','organoid_id'):
        if not isinstance(binding[k],str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',binding[k]):raise ValueError('Invalid binding '+k)
        if binding[k].lower() in ('unknown','null','none','todo','placeholder'):raise ValueError('Unbound '+k)
    started_wall(binding)
    for key,expected in [('sample_rate_hz',20000),('window_frames',20000),('well_id',0)]:
        if type(binding[key]) is not int or binding[key]!=expected:raise ValueError('Binding '+key)
    if number(binding['amplitude_scale_uv'],'amplitude_scale_uv')<=0:raise ValueError('Measured amplitude scale required')
    pairs=binding['channel_map']
    if not isinstance(pairs,list) or not 1<=len(pairs)<=1024:raise ValueError('Nonempty fixed channel map required')
    for pair in pairs:
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError('Mapping pair')
        if integer(pair[0],'channel')>=1024 or integer(pair[1],'electrode')>=26400:raise ValueError('Mapping range')
    if pairs!=sorted(pairs) or len({x[0] for x in pairs})!=len(pairs) or len({x[1] for x in pairs})!=len(pairs):raise ValueError('One-to-one sorted mapping')
    return clone(binding)

def validate_calibration(calibration,binding,live=True):
    keys={'schema','binding_sha256','source','coverage_basis','centers','scales','calibration_window_ids','validation_window_ids','normalization_id','status'}
    if not isinstance(calibration,dict) or set(calibration)!=keys:raise ValueError('Bound calibration required')
    if calibration['schema']!='mea.online.calibration.v1' or calibration['binding_sha256']!=digest(binding):raise ValueError('Calibration binding')
    if calibration['source']!=('live' if live else 'simulation') or calibration['coverage_basis']!='filtered_transport_frames' or calibration['status']!='engineering_validated':raise ValueError('Calibration scope')
    for key in ('centers','scales'):
        if not isinstance(calibration[key],list) or len(calibration[key])!=4:raise ValueError('Four normalization values required')
        for x in calibration[key]:number(x,key)
    if any(x<=0 for x in calibration['scales']):raise ValueError('Positive normalization scales required')
    ids=calibration['calibration_window_ids']+calibration['validation_window_ids']
    if len(calibration['calibration_window_ids'])<60 or len(calibration['validation_window_ids'])<30 or len(set(ids))!=len(ids):raise ValueError('Independent 60+30 calibration windows required')
    if any(not isinstance(x,str) or not re.fullmatch('[0-9a-f]{64}',x) for x in ids):raise ValueError('Calibration evidence hashes required')
    if calibration['normalization_id']!=digest({k:v for k,v in calibration.items() if k!='normalization_id'}):raise ValueError('Calibration digest')
    return clone(calibration)

def validate_config(value,live=False,allow_pending_calibration=False):
    expected=template()
    if not isinstance(value,dict) or set(value)!=set(expected):raise ValueError('Exact configuration fields required')
    c=clone(value)
    if c['schema']!=expected['schema'] or c['gpu_host']!='192.0.2.16':raise ValueError('Target/schema mismatch')
    if c['r2_enabled'] is not False or c['stimulation'] is not False or c['reward']!='r1_only':raise ValueError('Only R1/no-stimulation mode is implemented')
    for k,v in [('max_session_seconds',7200),('finish_reserve_seconds',120)]:
        if type(c[k]) is not int or c[k]!=v:raise ValueError('Fixed session deadline '+k)
    if c['devices']!=list(range(8)) or any(type(x) is not int for x in c['devices']):raise ValueError('Eight configured GPU devices required')
    for k,lo,hi in [('batch_size',1,128),('batches',1,100),('seed',0,2147483647),('update_reserve_seconds',30,600)]:
        if not lo<=integer(c[k],k,lo)<=hi:raise ValueError(k+' bound')
    if not 1<=number(c['max_observation_age_seconds'],'max observation age')<=30:raise ValueError('Observation age bound')
    source=c['source']
    if not isinstance(source,dict) or set(source)!={'host','port','timeout'}:raise ValueError('Source fields')
    if source['host'] not in ('127.0.0.1','localhost','::1'):raise ValueError('Use an authenticated SSH tunnel on loopback')
    if integer(source['port'],'source port',1)>65535 or not 1<=number(source['timeout'],'source timeout')<=8:raise ValueError('Source bounds')
    if not isinstance(c['model_paths'],dict) or set(c['model_paths'])!=set(expected['model_paths']):raise ValueError('Model path fields')
    if c['binding'] is not None:c['binding']=validate_binding(c['binding'])
    if live and c['binding'] is None:raise ValueError('New chip/organoid/start binding required')
    if c['calibration'] is not None:
        if c['binding'] is None:raise ValueError('Calibration without binding')
        c['calibration']=validate_calibration(c['calibration'],c['binding'],live=live)
    if live:
        if c['calibration'] is None and not allow_pending_calibration:raise ValueError('New session calibration required')
        for k in ('project','godot','registry'):
            if not isinstance(c[k],str) or not c[k]:raise ValueError('Explicit '+k+' required')
        if any(not isinstance(x,str) or not x for x in c['model_paths'].values()):raise ValueError('Explicit model paths required')
    return c

class Budget:
    def __init__(self,start,clock=time):
        self.clock=clock;self.wall_end=number(start,'start')+7200
        self.m0=clock.monotonic();self.w0=clock.time();self.initial=self.wall_end-self.w0
        if self.w0<start:raise ValueError('Experiment start is in the future')
        self.last_m=self.m0;self.last_remaining=self.initial
    def remaining(self):
        now=self.clock.monotonic()
        if now<self.last_m:raise TimeoutError('Monotonic clock regressed')
        remaining=min(self.wall_end-self.clock.time(),self.last_remaining-(now-self.last_m))
        self.last_m=now;self.last_remaining=remaining
        return remaining
    def require(self,seconds=0):
        if self.remaining()<120+number(seconds,'reserve'):raise TimeoutError('Original two-hour deadline/finish reserve')

def claim_session(registry,binding):
    registry=Path(registry);registry.mkdir(parents=True,exist_ok=True)
    token=digest({'organoid_id':binding['organoid_id'],'started_utc':datetime.fromtimestamp(started_wall(binding),timezone.utc).isoformat()})
    path=registry/(token+'.json');write(path,dict(binding=binding,status='claimed_no_automatic_restart'))
    return path
