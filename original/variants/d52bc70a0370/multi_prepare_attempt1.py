"""Owned software-object preparation only: no device download or sequence send."""
import json,sys,time,uuid
from pathlib import Path
from multi_protocol import compile_sequence,validate_routes

def software_allowed(message,array_token,sequence_tokens):
    if message=='system_query_dac_lsb_mv':return True
    if message in ('mea_array_new '+array_token,'mea_array_delete '+array_token):return True
    prefix='mea_array_command '+array_token+'\n'
    if message.startswith(prefix):
        command=message[len(prefix):].split(' ',1)[0]
        return command in {'mea_array_set_config','mea_array_get_config',
            'mea_array_query_amplifier_at_stimulation','mea_array_query_stimulation_at_electrode',
            'mea_array_query_amplifier_at_electrode','mea_array_connect_electrode_to_stimulation',
            'mea_array_disconnect_amplifier_from_stimulation'}
    for token in sequence_tokens:
        if message in ('sequence_new '+token,'sequence_delete '+token):return True
        if message.startswith('sequence_append '+token+'\n'):
            payload=message.split('\n',1)[1]
            return '\n' not in payload and payload.split(' ',1)[0] in {
                'system_delay_samples','system_trigger_event','mea_set_dac','mea_set_stimulation_unit'}
    return False

def prepare(base):
    base=Path(base);sys.path.insert(0,str(base/'corrected/mea_next_recording'))
    from device import MaxLabBackend
    from protocol import atomic_json
    from quality import parse_mapping
    from brain_stimulation import maxlab as mx
    b=json.loads((base/'binding.json').read_text())
    candidates=json.loads((base/'multipoint_candidates.json').read_text())
    if b['chip_id']!='28080' or candidates['chip_id']!=b['chip_id']:raise ValueError('Unbound candidate identity')
    out=base/'multi_software_prepare';out.mkdir(exist_ok=False)
    d=MaxLabBackend(b,out);network=d.request
    array_token='multi_route_'+uuid.uuid4().hex
    zero_token='multi_zero_'+uuid.uuid4().hex
    tokens={zero_token}
    def restricted(message,timeout):
        if not software_allowed(message,array_token,tokens):raise RuntimeError('No-execution allowlist rejected '+message[:160])
        return network(message,timeout)
    d.request=restricted;started=time.monotonic();success=False
    try:
        with d.sdk_context():
            d.array_token=array_token;d.array=mx.Array(array_token,persistent=True)
            d.array.load_config_data(d.route)
            for amplifier in d.active_links().values():d.array.disconnect_amplifier_from_stimulation(amplifier)
            if d.active_links():raise RuntimeError('Inherited software links remain')
            routes=[];rejected=[];links={}
            for candidate in candidates['candidates']:
                el=candidate['electrode'];ch=candidate['channel']
                possible=d.array.query_stimulation_at_electrode(el).strip()
                if not possible.isdigit() or not 0<=int(possible)<32 or int(possible) in links:
                    rejected.append(dict(electrode=el,reason='no_unused_nominal_unit'));continue
                unit=int(possible)
                d.array.connect_electrode_to_stimulation(el)
                expected=dict(links);expected[unit]=ch
                if d.active_links()!=expected:raise RuntimeError('Connection changed existing or unexpected software links')
                if int(d.array.query_amplifier_at_electrode(el))!=ch:raise RuntimeError('Candidate mapping changed')
                links=expected;routes.append(dict(candidate,unit=unit))
                if len(routes)==32:break
            validate_routes(routes,links)
            cfg=d._raw('mea_array_command '+array_token+'\nmea_array_get_config')
            if parse_mapping(cfg)!=d.mapping:raise RuntimeError('Record mapping changed')
            lsb=float(d._raw('system_query_dac_lsb_mv'));preview=compile_sequence(lsb,zero=True)
            d.sequence_tokens.append(zero_token)
            if d._raw('sequence_new '+zero_token).strip()!=zero_token:raise RuntimeError('Unexpected sequence token')
            for i,command in enumerate(preview['commands']):
                if time.monotonic()-started>120:raise TimeoutError('Software preparation exceeded 120 seconds')
                ack=d._raw('sequence_append '+zero_token+'\n'+command)
                if ack.strip().lower()!='ok':raise RuntimeError('Sequence append acknowledgement unknown')
                if i%1000==0:atomic_json(out/'progress.json',dict(stage='zero_sequence_upload',uploaded=i,total=len(preview['commands'])))
            result=dict(status='software_prepared_hardware_not_qualified',routes=routes,rejected=rejected,
                        array_token=array_token,zero_token=zero_token,lsb_mv=lsb,
                        zero_commands=len(preview['commands']),duration_s=preview['duration_ticks']/20000,
                        hardware_downloads=0,sequence_sends=0,true_stimulation_sends=0,
                        started_epoch=time.time()-(time.monotonic()-started),seconds=time.monotonic()-started,
                        training_admitted=False)
            atomic_json(out/'preparation.json',result);success=True
            print(json.dumps({k:v for k,v in result.items() if k not in ('routes','rejected')},indent=2),flush=True)
    finally:
        if not success:
            with d.sdk_context():errors=d.cleanup()
            atomic_json(out/'failure_cleanup.json',dict(errors=errors,hardware_downloads=0,sequence_sends=0))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Explicit remote evidence base required')
    prepare(sys.argv[1])
