"""Bounded 15-pulse batches, scheduled at 1 Hz. No automatic retry or catch-up."""
import math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mea_next_recording/vendor'))
from brain_stimulation import maxlab as mx

def available_batches(remaining,*,post_seconds=1800):
    if post_seconds not in (600,1800):raise ValueError('Explicit supported post duration required')
    reserve=post_seconds+207
    if not math.isfinite(remaining) or remaining<reserve+1:raise RuntimeError('No time for a batch and full post/cleanup')
    return min(600,int(remaining-reserve))

def network_gate(now,start,index):
    if not 0<=now-(start+index)<=.05:raise RuntimeError('Late before actual network send; do not retry')

def batch_commands(lsb,units,*,zero,switch_units=True):
    if (type(zero) is not bool or type(switch_units) is not bool or not 2<=len(units)<=32 or len(set(units))!=len(units)
        or any(type(u) is not int or not 0<=u<32 for u in units)):
        raise ValueError('Need 2..32 unique units')
    if not math.isfinite(lsb) or lsb<=0 or not 1<=int(600/lsb)<=511:raise ValueError('Invalid DAC LSB')
    bits=0 if zero else int(600/lsb);eid=900000 if zero else 910000
    commands=[mx.DAC(0,512).set()]
    if switch_units:
        for u in units:commands.append(mx.StimulationUnit(u).power_up(True).connect(True).set_voltage_mode().dac_source(0).set())
    commands.append(mx.DelaySamples(100).set())
    for b in range(5):
        for p in range(3):
            i=b*3+p
            commands.extend([mx.Event(0,1,eid+i,f'mode {"zero" if zero else "true"} pulse_in_batch {i}').set(),
                mx.DAC(0,512-bits).set(),mx.DelaySamples(4).set(),
                mx.DAC(0,512+bits).set(),mx.DelaySamples(4).set(),mx.DAC(0,512).set()])
            if p<2:commands.append(mx.DelaySamples(192).set())
            elif b<4:commands.append(mx.DelaySamples(3592).set())
    if switch_units:
        for u in range(32):commands.append(mx.StimulationUnit(u).power_up(False).connect(False).set())
    commands.extend([mx.DAC(0,512).set(),mx.Event(0,1,eid+15,'batch_completed 1').set()])
    return dict(commands=commands,duration_s=.8254,pulse_count=15,event_base=eid,switch_units=switch_units,
                nominal_phase_mv=bits*lsb,units=list(units),zero=zero)

def drive_batches(clock,reserve,send,health,*,batch_count=600,start_mono=None):
    if type(batch_count) is not int or not 1<=batch_count<=600:raise ValueError('Invalid batch count')
    start=clock.monotonic() if start_mono is None else start_mono;attempted=acked=0;timings=[];result={}
    for i in range(batch_count):
        target=start+i
        if clock.monotonic()<target:clock.sleep(target-clock.monotonic())
        late=clock.monotonic()-target
        if late>.05:
            result=dict(status='stopped_late',late_s=late);break
        try:health();reserve(i)
        except BaseException as exc:
            result=dict(status='stopped_before_send',error=repr(exc));break
        late=clock.monotonic()-target
        if late>.05:
            result=dict(status='stopped_late',late_s=late,reserved_but_unsent_batch=i);break
        attempted+=1;stamp=clock.monotonic();timings.append(dict(batch=i,target=target,requested=stamp))
        try:send(i);acked+=1
        except BaseException as exc:
            result=dict(status='output_unknown',error=repr(exc));break
        # Wait past this bounded batch's tail before considering another batch.
        end=clock.monotonic()+.84
        if clock.monotonic()<end:clock.sleep(end-clock.monotonic())
    else:
        if clock.monotonic()<start+batch_count:clock.sleep(start+batch_count-clock.monotonic())
        result=dict(status='completed')
    result.update(attempted_batches=attempted,acknowledged_batches=acked,timings=timings,
                  max_reserved_pulses_per_site=attempted*15,start_mono=start,end_mono=clock.monotonic())
    return result
