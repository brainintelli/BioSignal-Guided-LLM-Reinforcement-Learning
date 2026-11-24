"""Protocol mechanics independent of MaxLab and network transports."""
import math
import random
import time

class BudgetError(RuntimeError):pass
class RealClock:
 def wall(self):return time.time()
 def mono(self):return time.monotonic()
 def sleep(self,s):time.sleep(s)

def schedule(seed):
 rng=random.Random(seed);orders=['true']*5+['sham']*5;rng.shuffle(orders)
 return [{'pair':i+1,'interval_s':rng.randint(45,75),'events':[
  {'id':9001+2*i,'arm':first,'pair':i+1},
  {'id':9002+2*i,'arm':'sham' if first=='true' else 'true','pair':i+1}]} for i,first in enumerate(orders)]

def pulse_codes(arm,lsb):
 if arm not in ('true','sham') or not math.isfinite(lsb) or lsb<=0:raise ValueError('Invalid arm/DAC step')
 bits=int(300/lsb)
 if not 1<=bits<=511:raise ValueError('300mV cannot be represented')
 bits=bits if arm=='true' else 0
 return [('dac',512-bits),('delay',4),('dac',512+bits),('delay',4),('dac',512)]

class Budget:
 def __init__(self,started_wall,clock):
  self.clock=clock;now=clock.wall();mono=clock.mono()
  if not all(math.isfinite(x) for x in (started_wall,now,mono)) or started_wall>now or now-started_wall>=7200:
   raise BudgetError('Biological start is future/expired/invalid')
  self.wall_end=started_wall+7200;self.mono_end=mono+self.wall_end-now;self.last_mono=mono
 def remaining(self):
  wall=self.clock.wall();mono=self.clock.mono()
  if not all(math.isfinite(x) for x in (wall,mono)) or mono<self.last_mono:raise BudgetError('Invalid clock/monotonic rollback')
  self.last_mono=mono
  return min(self.wall_end-wall,self.mono_end-mono)
 def require(self,seconds):
  if not math.isfinite(seconds) or seconds<0 or self.remaining()<seconds:raise BudgetError('Insufficient deadline reserve')
 def admit_pair(self):self.require(1035)
 def before_output(self,remaining_work):
  self.require(max(900,remaining_work))

