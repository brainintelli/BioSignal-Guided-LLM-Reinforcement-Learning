"""Strict future identity binding. No chip, organoid or clock defaults for live use."""
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re

class ConfigurationError(ValueError): pass

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def route_mapping(path):
 text=Path(path).read_text(encoding='utf-8').split()
 rows=[(int(c),int(e)) for c,e in re.findall(r'(\d+)\((\d+)\)',text[0] if text else '')]
 if len(rows)<51 or len({c for c,e in rows})!=len(rows) or len({e for c,e in rows})!=len(rows):
  raise ConfigurationError('Route must have >=51 unique physical channels/electrodes')
 if any(not 0<=c<1024 or not 0<=e<26400 for c,e in rows):raise ConfigurationError('Route channel/electrode out of range')
 return sorted(rows)

@dataclass(frozen=True)
class Config:
 mode: str='offline'
 session_id: str|None=None
 chip_id: str|None=None
 organoid_id: str|None=None
 started_utc: str|None=None
 device_hostname: str|None=None
 device_model: str='MaxOne'
 electrode_material: str='Pt-Black'
 route_file: str|None=None
 route_sha256: str|None=None
 stim_electrode: int|None=None
 threshold_value: float|None=None
 threshold_h5_dataset: str|None=None
 threshold_units: str='standard_deviations'
 seed: int=20261117
 prior_true: int=0
 prior_sham: int=0
 r2_enabled: bool=False
 training_enabled: bool=False
 live_authorized: bool=False
 authorization_sha256: str|None=None

 @classmethod
 def template(cls):return cls()
 @classmethod
 def from_dict(cls,d):
  if set(d)-{f.name for f in fields(cls)}:raise ConfigurationError('Unknown configuration fields')
  c=cls(**d)
  if c.r2_enabled is not False or c.training_enabled is not False:raise ConfigurationError('R2 and training must be false')
  if type(c.live_authorized) is not bool:raise ConfigurationError('Authorization must be a boolean')
  if c.mode not in ('offline','simulate','live'):raise ConfigurationError('Unknown mode')
  if type(c.seed) is not int or not 0<=c.seed<2**63:raise ConfigurationError('Invalid seed')
  if c.device_model!='MaxOne' or c.electrode_material!='Pt-Black':raise ConfigurationError('Protocol is for ordinary MaxOne Pt-Black only')
  if c.threshold_units!='standard_deviations':raise ConfigurationError('Unknown detector units')
  if any(type(x) is not int or x!=0 for x in (c.prior_true,c.prior_sham)):raise ConfigurationError('Full v2 requires a new unstimulated biological session')
  return c
 def to_dict(self):return asdict(self)
 def binding_sha256(self):
  d=self.to_dict();d.pop('live_authorized');d.pop('authorization_sha256')
  return digest({'protocol':'maxone-v2-300mV-200us-11true-11sham-R2off',**d})
 def started_wall(self):
  try:
   x=datetime.fromisoformat(self.started_utc.replace('Z','+00:00'))
   if x.tzinfo is None:raise ValueError()
   return x.timestamp()
  except (AttributeError,ValueError,TypeError):raise ConfigurationError('An explicit timezone-aware biological start is required') from None
 def validate_binding(self):
  Config.from_dict(self.to_dict())
  for label in ('session_id','organoid_id','chip_id','device_hostname'):
   x=getattr(self,label)
   if not isinstance(x,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',x):raise ConfigurationError('Missing/invalid '+label)
  self.started_wall()
  if not self.route_file or not self.route_sha256:raise ConfigurationError('Route binding required')
  if hashlib.sha256(Path(self.route_file).read_bytes()).hexdigest()!=self.route_sha256:raise ConfigurationError('Route hash mismatch')
  rows=route_mapping(self.route_file)
  if type(self.stim_electrode) is not int or self.stim_electrode not in dict((e,c) for c,e in rows):raise ConfigurationError('Stimulation electrode is not in bound route')
  if not isinstance(self.threshold_value,(int,float)) or isinstance(self.threshold_value,bool) or not math.isfinite(self.threshold_value) or self.threshold_value<=0:
   raise ConfigurationError('Confirmed spike detector value required')
  p=self.threshold_h5_dataset
  if not isinstance(p,str) or not p.startswith('settings/') or '..' in p or not re.fullmatch(r'[A-Za-z0-9_/.-]+',p):
   raise ConfigurationError('Explicit detector H5 readback path under settings/ required')
  return rows
 def validate_live(self):
  self.validate_binding()
  if self.mode!='live' or not self.live_authorized or self.authorization_sha256!=self.binding_sha256():
   raise ConfigurationError('Live execution requires separate authorization matching the exact binding')
  if self.organoid_id.startswith('SIM-'):raise ConfigurationError('Simulation identity cannot authorize live execution')

