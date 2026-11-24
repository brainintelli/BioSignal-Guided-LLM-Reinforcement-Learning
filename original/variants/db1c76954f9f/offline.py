"""Run deterministic engineering scenarios. No LLM, network, SDK or wetware."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .contract import AttemptBudget, TransitionGate
from .encoding import Encoder, compile_sequence


class SimulationClock:
    def __init__(self):
        self.seconds=0.
    def time(self):return 1000.+self.seconds
    def monotonic(self):return 10.+self.seconds
    def advance(self,seconds):self.seconds+=seconds


def simulated_evidence(identity,intent,anchor,event_id,window_id):
    """Synthetic producer, explicitly inadmissible as live hardware evidence."""
    return dict(source='simulation',identity=deepcopy(identity),event_id=event_id,
        window_id=window_id,status='confirmed',sample_rate_hz=20000,
        anchor_frame=anchor,pulse_frames=[anchor+6000*i for i in range(intent.pulses)],
        max_abs_output_mv=298.7 if intent.pulses else 0.,output_verified=True,
        route_verified=True,window_start_frame=anchor+240000,
        window_end_frame=anchor+260000,coverage_start_frame=anchor-60000,
        coverage_end_frame=anchor+260000,complete=True,dropped_frames=0,qc_passed=True)


def _write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _jsonl(path,rows):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def _arguments(identity,intent,evidence,terminal=False):
    return dict(r1=-1. if terminal else .01,duration=1,gamma=.99985,beta=.002,
                intent=intent,evidence=evidence,expected_identity=identity,terminal=terminal)


def _rejections(binding):
    rows=[]
    for name in ('unknown_output','recovery_failure','missing_coverage','wrong_context','duplicate_window'):
        current=binding|dict(episode_id='fault-'+name,transition_id='1',context_id='post-action-1')
        gate=TransitionGate(binding)
        intent=Encoder().decide({'H':100,'F':100,'W':80})
        receipt=simulated_evidence(current,intent,100000,'event-1','window-1')
        if name=='unknown_output':receipt['status']='unknown'
        elif name=='recovery_failure':receipt['qc_passed']=False
        elif name=='missing_coverage':receipt['coverage_end_frame']-=1
        elif name=='wrong_context':receipt['identity']['context_id']='another-action'
        else:
            gate.accept(**_arguments(current,intent,receipt))
            current=current|dict(transition_id='2',context_id='post-action-2')
            receipt=simulated_evidence(current,intent,500000,'event-2','window-1')
        result=None;error=None
        try:
            result=gate.accept(**_arguments(current,intent,receipt))
            rejected=not result['valid'] and result['total_reward'] is None
        except ValueError as exc:
            rejected=True;error=str(exc)
        if not rejected or not gate.stopped:raise RuntimeError('Fault admitted: '+name)
        rows.append(dict(case=name,source='simulation',rejected=rejected,session_stopped=gate.stopped,
                         receipt=receipt,reward=result,error=error))
    return rows


def run_offline(output):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    registration=dict(schema='mea.binary_r2.offline.v1',source='simulation',
        hardware_commands_sent=False,model_training_performed=False,r2_enabled=True,
        r1_source='scripted_engineering_fixture_not_game_performance',
        beta=.002,gamma_tick=.99985,lsb_mv=2.9,phase_us=200,pulse_onset_interval_ms=300,
        minimum_cycle_seconds=20,hard_deadline_seconds=7200,nonzero_cutoff_seconds=5700)
    _write(out/'registration.json',registration)
    binding=dict(session_id='offline-engineering-session',chip_id='SIMULATED-CHIP',
                 organoid_id='SIMULATED-ORGANOID',route_id='SIMULATED-ROUTE')
    encoder=Encoder();gate=TransitionGate(binding);clock=SimulationClock()
    budget=AttemptBudget(clock.time(),clock)
    cases=[('healthy',100,100,100,False),('thirsty',100,100,80,False),
           ('water_hysteresis',100,100,87,False),('water_recovered',100,100,90,False),
           ('hungry',100,40,100,False),('health_low',40,100,100,False),
           ('restored',100,100,100,False),('terminal',0,0,0,True)]
    rows=[]
    for index,(name,health,food,water,terminal) in enumerate(cases):
        reserves=dict(H=health,F=food,W=water)
        current=binding|dict(episode_id='episode-1',transition_id=str(index),context_id='post-action-'+str(index))
        intent=encoder.decide(reserves,terminal)
        sequence=compile_sequence(intent,2.9)
        receipt=None
        if not terminal:
            budget.admit(intent)
            receipt=simulated_evidence(current,intent,100000+400000*index,'event-'+str(index),'window-'+str(index))
            clock.advance(13);budget.complete(True)
            if budget.stopped:raise RuntimeError('Budget ended before response could be accepted')
        reward=gate.accept(**_arguments(current,intent,receipt,terminal))
        if not reward['valid']:raise RuntimeError('Nominal scenario was rejected: '+name)
        rows.append(dict(case=name,source='simulation',reserves_after_action=reserves,
            intent=asdict(intent),sequence_description=sequence,receipt=receipt,reward=reward))
        if not terminal:clock.advance(7)
    rejected=_rejections(binding)
    _jsonl(out/'transitions.jsonl',rows)
    _jsonl(out/'rejections.jsonl',rejected)
    result=dict(passed=True,source='simulation',hardware_commands_sent=False,
        model_training_performed=False,biological_effect_evaluated=False,r2_enabled=True,
        positive_rows=sum(r['reward']['r2_raw']==1 for r in rows),
        negative_rows=sum(r['reward']['r2_raw']==-1 for r in rows),
        terminal_rows=sum(not r['reward']['eligible'] for r in rows),
        rejected_cases=len(rejected),simulated_nonzero_events=budget.events,simulated_pulses=budget.pulses,
        simulated_elapsed_seconds=clock.seconds,
        limitations=['simulation_only','no_live_receipt_verifier','no_persistent_hardware_ledger',
                     'no_state2_signal_extraction_in_this_package','no_LLM_or_PPO_update_in_this_runner'])
    _write(out/'result.json',result)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    _write(out/'manifest.json',manifest)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    _write(out/'source_manifest.json',source)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    arguments=parser.parse_args()
    print(json.dumps(run_offline(arguments.output),ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
