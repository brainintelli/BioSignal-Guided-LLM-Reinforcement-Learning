"""Inspect an owned software array; never download it or enable output."""
import json,sys,uuid
from pathlib import Path
BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
sys.path.insert(0,str(BASE/'mea_next_recording/vendor'))
from brain_stimulation import maxlab as mx
from brain_stimulation.transport import bind_sdk,Endpoint
token='diagnostic_'+uuid.uuid4().hex
report={'no_hardware_download':True,'no_output_commands':True,'queries':[]}
with bind_sdk(mx,Endpoint(timeout=3)):
    try:
        a=mx.Array(token,persistent=True);a.load_config(str(BASE/'route_frozen.cfg'))
        for kind,ids in [('amplifier_at_stimulation',range(32)),('stimulation_at_amplifier',[0,359,360,361,640,1023])]:
            for i in ids:
                try:reply=mx.send_raw('mea_array_command '+token+'\nmea_array_query_'+kind+' '+str(i))
                except Exception as exc:reply={'error':repr(exc)}
                report['queries'].append({'kind':kind,'index':i,'reply':reply})
        report['software_connect_reply']=a.connect_electrode_to_stimulation(8858)
        report['chosen_forward']=a.query_stimulation_at_electrode(8858)
        report['after_software_connect']={str(i):a.query_amplifier_at_stimulation(i) for i in range(32)}
    finally:mx.send_raw('mea_array_delete '+token)
(BASE/'route_diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
