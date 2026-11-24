"""Fixed paired schedule, stage gates and cumulative biological-session budget."""
from pathlib import Path
import json,os,random,math

SESSION_ID='22554-20261117T032900Z'


def sequence_commands(schedule,lsb):
 if not math.isfinite(lsb) or lsb<=0:raise ValueError('Invalid DAC step')
 bits=int(300/lsb)
 if not 1<=bits<=511:raise ValueError('Invalid nominal 300mV DAC encoding')
 commands=['system_delay_samples 100']
 for i,t in enumerate(schedule):
  if t['arm'] not in ('true','sham') or t['amplitude_mv']!=(300 if t['arm']=='true' else 0):
   raise ValueError('Only fixed 300mV and zero sham allowed')
  b=bits if t['arm']=='true' else 0
  commands.extend([f"system_trigger_event 0 1 {t['event_id']} arm {t['arm']} trial {i+1} pair {t['pair']}",
   f'mea_set_dac 0 {512-b} 512','system_delay_samples 4',
   f'mea_set_dac 0 {512+b} 512','system_delay_samples 4','mea_set_dac 0 512 512'])
  if i<len(schedule)-1:
   left=900000-8
   while left:
    n=min(left,65535);commands.append(f'system_delay_samples {n}');left-=n
 return commands


def make_control_schedule():
 orders=['true','true','true','sham','sham','sham']
 random.Random(225542026).shuffle(orders)
 result=[]
 for pair,first in enumerate(orders,1):
  for arm in (first,'sham' if first=='true' else 'true'):
   result.append({'pair':pair,'arm':arm,'event_id':9001+len(result),
                  'amplitude_mv':300 if arm=='true' else 0})
 return result


def validate_gate(q,stage):
 if q.get('status')!='pass' or q.get('session_id')!=SESSION_ID or q.get('stage')!=stage:
  raise ValueError('Quality gate not passed for this session/stage')
 roi=q.get('roi_channels',[])
 if len(set(roi))<50 or any(type(c) is not int or not 0<=c<1024 for c in roi):
  raise ValueError('Insufficient valid fixed recording ROI')


class Budget:
 def __init__(self,root,*,deadline_wall,wall_now,mono_now,prior_true):
  if not all(math.isfinite(v) for v in (deadline_wall,wall_now,mono_now)) or deadline_wall<=wall_now:
   raise RuntimeError('Invalid or expired biological deadline')
  self.root=Path(root);self.root.mkdir(exist_ok=True,parents=True)
  self.deadline_wall=deadline_wall;self.deadline_mono=mono_now+deadline_wall-wall_now
  self.prior_true=prior_true
 def remaining(self,wall,mono):return min(self.deadline_wall-wall,self.deadline_mono-mono)
 def totals(self):
  true,sham=self.prior_true,0
  for p in self.root.glob('claim_*.json'):
   r=json.loads(p.read_text());true+=r['true'];sham+=r['sham']
  return true,sham
 def reserve(self,stage,true,sham,duration,wall,mono):
  if any(type(n) is not int or n<0 for n in (true,sham)):raise ValueError('Invalid dose')
  if duration<0 or not math.isfinite(duration):raise ValueError('Invalid sequence duration')
  # Full post-recording and cleanup remain available even after a lost API reply.
  # The transport can spend 5 s connecting plus 20 s awaiting a reply.
  if self.remaining(wall,mono)<duration+30+600+120:raise RuntimeError('Insufficient recovery/closeout time')
  t,s=self.totals()
  if t+true>25 or s+sham>12:raise RuntimeError('Cumulative session dose exceeded')
  record={'session_id':SESSION_ID,'stage':stage,'true':true,'sham':sham,
          'attempt_wall':wall,'deadline_wall':self.deadline_wall,
          'state':'reserved_before_send_no_retry'}
  with (self.root/('claim_'+stage+'.json')).open('x') as f:
   json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
