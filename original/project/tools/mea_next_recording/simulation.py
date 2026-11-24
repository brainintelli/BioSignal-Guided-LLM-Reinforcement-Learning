"""Virtual time + SDK wire simulation; deliberately produces no biological H5."""
from pathlib import Path
import socket
from unittest.mock import patch
from device import MaxLabBackend
from protocol import atomic_json
from session import Session


class VirtualClock:
    def __init__(self):self.t=1800000000.;self.m=0.
    def time(self):return self.t
    def monotonic(self):return self.m
    def sleep(self,seconds):self.t+=seconds;self.m+=seconds


class SimulatedDevice(MaxLabBackend):
    def __init__(self,binding,output,clock,scenario):
        self.scenario=scenario;self.wire=[];self.connected=False
        super().__init__(binding,output,clock=clock,request=self.reply)
    def reply(self,message,timeout):
        self.wire.append(dict(mono=self.clock.monotonic(),command=message))
        if message=='system_query_dac_lsb_mv':return '2.981901'
        if message.startswith('sequence_new '):return message.split()[1]
        if 'mea_array_connect_electrode_to_stimulation ' in message:self.connected=True;return 'Ok'
        if 'mea_array_query_amplifier_at_stimulation ' in message:return '2' if self.connected and message.split()[-1]=='7' else ''
        if 'mea_array_query_stimulation_at_amplifier ' in message:return '7' if message.split()[-1]=='2' else '-1'
        if 'mea_array_get_config' in message:return self.route
        if 'mea_array_query_stimulation_at_electrode' in message:return '7'
        if 'mea_array_query_amplifier_at_' in message:return '2'
        if message.startswith('sequence_send full_pulse_') and self.scenario=='lost-reply':
            raise TimeoutError('SIMULATED acknowledgement loss')
        return 'Ok'
    def preflight(self):pass
    def identity(self):self.candidate={'channel':2,'electrode':42};return self.candidate
    def qualify(self):return dict(status='failed' if self.scenario=='baseline-failure' else 'passed',synthetic=True)
    def progress(self):pass
    def verify_recording_ready(self):pass
    def audit(self):return dict(status='passed',synthetic=True,training_admitted=False)


def simulate(output,scenario):
    from datetime import datetime,timezone
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    route=root/'SYNTHETIC.cfg';route.write_text('0(40)1(41)2(42)3(43)')
    clock=VirtualClock()
    b=dict(protocol='photo_full_v1',chip_id='SYNTHETIC',
           session_started_at=datetime.fromtimestamp(clock.time(),timezone.utc).isoformat(),
           acquisition_host='SIMULATED',route_cfg=str(route),alpha=1,r2_mode='record_only',
           realtime_training=False,offline_training=False)
    d=SimulatedDevice(b,root/'run',clock,scenario)
    with patch.object(socket,'create_connection',side_effect=AssertionError('Simulation forbids sockets')):
        with d.sdk_context():result=Session(b,d,clock,root/'run',root/'simulation_journal').run()
    sends=sum(x['command'].startswith('sequence_send full_pulse_') for x in d.wire)
    summary=dict(status=result['status'],synthetic=True,hardware_connected=False,
                 nonzero_sequence_attempts=sends,virtual_elapsed_seconds=clock.monotonic(),
                 scenario=scenario,training_admitted=False,alpha=1,r2_mode='record_only',
                 limitation='SDK serialization and orchestration only; no biological response or real H5 simulated.')
    atomic_json(root/'simulation.json',summary);atomic_json(root/'wire_commands.json',d.wire)
    return summary
