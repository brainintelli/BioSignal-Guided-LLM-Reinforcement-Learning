"""Only upload and delete prepared sequences; prohibit every execution command."""
import sys,json
from pathlib import Path
BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
sys.path.insert(0,str(BASE/'corrected/mea_next_recording'))
from device import MaxLabBackend
out=BASE/'sequence_upload_only';out.mkdir(exist_ok=False)
b=json.loads((BASE/'binding.json').read_text());d=MaxLabBackend(b,out)
network=d.request
def upload_only(message,timeout):
    if not (message in ('system_query_dac_lsb_mv','system_clear_events') or
            message.startswith(('sequence_new ','sequence_append ','sequence_delete '))):
        raise RuntimeError('No-execution allowlist rejected '+message)
    return network(message,timeout)
d.request=upload_only
with d.sdk_context():
    try:d.prepare()
    finally:errors=d.cleanup()
if errors:raise RuntimeError(repr(errors))
report=dict(status='passed',sequence_sends=0,hardware_downloads=0,nonzero_direct_dac_commands=0,
            prepared_sequence_duration=d.preview['duration_s'],nominal_phase_mv=d.preview['nominal_phase_mv'])
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
