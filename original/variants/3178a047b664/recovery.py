"""Narrow, explicitly invoked recovery for the observed pre-download amp360 error.

Not an automatic retry. Original launch and stimulus claims remain untouched.
All other failures, any download, send, unclear cleanup or changed binding reject.
"""
import hashlib,json
from pathlib import Path
from protocol import atomic_json,session_key
from session import Session


def verify_preoutput_resume(launch,binding,stimulus_claim):
    entry=json.loads(Path(launch).read_text());old=Path(entry['output'])
    if entry['binding']!=binding or Path(stimulus_claim).exists():raise RuntimeError('Changed binding or output claim exists')
    if Path(binding['route_cfg']).read_bytes()!=(old/'route_input.cfg').read_bytes():raise RuntimeError('Frozen route changed')
    progress=json.loads((old/'progress.json').read_text())
    names=[x['name'] for x in progress['stages']]
    if (progress.get('status')!='failed' or progress.get('cleanup_errors')!=[] or
        progress.get('stimulus_attempted') is not False or progress.get('stimulus_acknowledged') is not False or
        'configure_current_route' not in names or any(x in names for x in ('baseline_600s','baseline_qc','single_stimulus_attempt','post_1800s'))):
        raise RuntimeError('Not the verified pre-output configuration failure')
    identity=json.loads((old/'identity_qc.json').read_text())
    if identity['status']!='passed' or identity['chip_id']!=binding['chip_id'] or identity['candidate']!=progress['candidate']:
        raise RuntimeError('Identity/candidate proof missing')
    data=(old/'api_audit.jsonl').read_bytes();logs=[json.loads(x) for x in data.splitlines()]
    commands=[x['command'] for x in logs]
    if any('sequence_send ' in c or 'sequence_append ' in c or 'mea_array_download' in c or
           (c.startswith('mea_set_dac ') and c!='mea_set_dac 0 512 512') for c in commands):
        raise RuntimeError('Prior download/output/prepared sequence prevents recovery')
    errors=[(i,x) for i,x in enumerate(logs) if 'error' in x]
    if len(errors)!=1 or not errors[0][1]['command'].endswith('\nmea_array_disconnect_amplifier_from_stimulation 360'):
        raise RuntimeError('Different or ambiguous failure')
    error_entry=errors[0][1]
    first_line=error_entry['command'].splitlines()[0]
    expected_error=repr(RuntimeError('MaxLab rejected '+first_line+': Error'))
    if error_entry['error']!=expected_error:raise RuntimeError('Failure was not an explicit MaxLab rejection')
    tail=logs[errors[0][0]+1:]
    ack=[x['command'] for x in tail if str(x.get('ack','')).lower()=='ok']
    for unit in range(32):
        if not any(c.startswith(f'mea_set_stimulation_unit {unit} 0 0 ') for c in ack):
            raise RuntimeError('Output cleanup acknowledgement missing')
    if 'mea_set_dac 0 512 512' not in ack:raise RuntimeError('DAC zero acknowledgement missing')
    first=next(x for x in progress['stages'] if x['name']=='preflight')
    return dict(original_output=str(old),audit_sha256=hashlib.sha256(data).hexdigest(),
                candidate=identity['candidate'],original_clock={'wall':first['wall'],'mono':first['mono']},
                restriction='Same session; one explicit recovery before any hardware download; stimulus budget unchanged')


class VerifiedPreoutputResume(Session):
    def claim_launch(self,start):
        key=session_key(self.binding,start)
        proof=verify_preoutput_resume(self.journal/(key+'.launch.json'),self.binding,self.journal/(key+'.stimulus.json'))
        clock=proof['original_clock']
        if self.clock.monotonic()<clock['mono']:raise RuntimeError('Monotonic clock reset')
        self.deadline.mono_end=min(self.deadline.mono_end,clock['mono']+start+7200-clock['wall'])
        self.deadline.require(60+10+20+600+60+1940.8254)
        atomic_json(self.journal/(key+'.verified_preoutput_recovery.json'),dict(**proof,new_output=str(self.output)),exclusive=True)
        self.report['verified_preoutput_recovery']=proof
        self.device.frozen_candidate=proof['candidate']
