"""Finite multipoint waveform compiler. This module never opens a socket."""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mea_next_recording/vendor'))
from brain_stimulation import maxlab as mx

PULSES=9000
PERIOD_TICKS=4000
DURATION_TICKS=12000100

def required_seconds():
    return 600.005+1800+120+20

def compile_sequence(lsb,*,zero):
    if type(zero) is not bool:raise ValueError('Explicit zero/true mode required')
    if not math.isfinite(lsb) or lsb<=0 or not 1<=int(600/lsb)<=511:
        raise ValueError('Unrepresentable DAC amplitude')
    bits=int(600/lsb);codes=[512-bits,512+bits,512]
    start_id=100000 if zero else 200000
    mode='zero_qualification' if zero else 'multi10_true'
    commands=[mx.DelaySamples(100).set()];onsets=[]
    for b in range(3000):
        for p in range(3):
            i=b*3+p;onsets.append(100+b*4000+p*200)
            commands.extend([
                mx.Event(0,1,start_id+i,f'mode {mode} pulse {i}').set(),
                mx.DAC(0,512 if zero else codes[0]).set(),
                mx.DelaySamples(4).set(),
                mx.DAC(0,512 if zero else codes[1]).set(),
                mx.DelaySamples(4).set(),
                mx.DAC(0,512).set(),
                mx.DelaySamples(192 if p<2 else 3592).set(),
            ])
    for u in range(32):commands.append(mx.StimulationUnit(u).power_up(False).connect(False).set())
    commands.append(mx.DAC(0,512).set())
    commands.append(mx.Event(0,1,start_id+PULSES,f'mode {mode} completed 1').set())
    return dict(mode=mode,pulse_count=PULSES,phase_ticks=4,codes=codes,
                nominal_phase_mv=bits*lsb,onset_ticks=onsets,duration_ticks=DURATION_TICKS,
                terminal_event_id=start_id+PULSES,
                terminal_event_command=mx.Event(0,1,start_id+PULSES,f'mode {mode} completed 1').set(),
                commands=commands,training_admitted=False)

def check_pulse_events(events,*,zero):
    start_id=100000 if zero else 200000
    if len(events)!=PULSES:return ['pulse_count']
    if [(e['type'],e['id']) for e in events]!=[(1,start_id+i) for i in range(PULSES)]:
        return ['pulse_ids']
    expected=[b*4000+p*200 for b in range(3000) for p in range(3)]
    if [e['frame']-events[0]['frame'] for e in events]!=expected:return ['pulse_frame_intervals']
    return []

def validate_routes(routes,actual_links):
    if not 2<=len(routes)<=32:raise ValueError('Need 2..32 simultaneous points')
    for key,bound in [('unit',32),('channel',1024),('electrode',26400)]:
        values=[r[key] for r in routes]
        if any(type(v) is not int or not 0<=v<bound for v in values) or len(set(values))!=len(values):
            raise ValueError('Invalid or duplicate '+key)
    if actual_links!={r['unit']:r['channel'] for r in routes}:raise ValueError('Reverse route mismatch')
