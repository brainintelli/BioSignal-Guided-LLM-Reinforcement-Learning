"""Offline bundle/source verification and native CLI preview; no SDK execution."""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SOURCE = Path('./github_program/BrainDesktop')
BUNDLE = HERE/'package/BrainStimulation'
TEMPLATE = Path('standalone/stimulation/profiles/maxone_first_probe.template.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source_manifest = json.loads((BUNDLE/'SOURCE-MANIFEST.json').read_text(encoding='utf-8'))
for rel, expected in source_manifest['source_sha256'].items():
    assert sha(SOURCE/rel) == expected, rel
archive = HERE/'package/BrainStimulation-ubuntu22.04-python.tar.gz'
assert sha(archive) == (HERE/'package/SHA256SUMS').read_text(encoding='ascii').split()[0]
native_template = BUNDLE/'profiles/maxone_first_probe.template.json'
assert native_template.read_bytes() == (SOURCE/TEMPLATE).read_bytes()
data = json.loads(native_template.read_text(encoding='utf-8'))
assert data['electrodes_selection'] == {'recording': [], 'stimulation': []}
assert data['amplitude_range'] == [300, 300, 1]
assert data['pulses'] == 1 and data['pulse_trains'] == 5 and data['ipti'] == 10
assert data['phase'] == 200 and data['record_only_spikes'] is False
assert data['pre_stim_period'] == data['post_stim_period'] == 10
assert data['control'] is False and data['randomize'] is False
rejected = subprocess.run([sys.executable, str(BUNDLE/'run_stimulation.py'),
                           str(native_template), '--dry-run'], capture_output=True)
assert rejected.returncode == 1

with tempfile.TemporaryDirectory() as tmp:
    # Required solely to exercise the native parser. No executable test profile is shipped.
    data['electrodes_selection'] = {'recording': [3580, 4887], 'stimulation': [3580]}
    test_profile = Path(tmp)/'mock_only.json'
    test_profile.write_text(json.dumps(data), encoding='utf-8')
    command = [sys.executable, str(BUNDLE/'run_stimulation.py'), str(test_profile), '--dry-run']
    proc = subprocess.run(command, capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode('utf-8', errors='replace')
    preview = json.loads(proc.stdout)
assert preview['amplitude_schedule_mv'] == [300]*5
assert preview['stimulation_seconds'] == 40.0054
assert preview['recording_seconds'] == 1240.0054
assert preview['actual_phase_us'] == 200
assert preview['actual_ipti_seconds'] == 10
preview.pop('recording_electrodes')
preview.pop('stimulation_electrodes')
preview.update(status='offline_timing_preview_only', electrode_route_verified=False,
               device_commands_sent=False, chip_id=None, organoid_id=None)
(HERE/'parameters_preview.json').write_text(json.dumps(preview, indent=2)+'\n', encoding='utf-8')

lsb = Decimal('2.981901')
bits = int(Fraction(300)/Fraction(lsb))
assert bits == 100 and bits*lsb == Decimal('298.190100')
registration = PROJECT/'configs/mea_maxone_stimulation_20261117.json'
reg = json.loads(registration.read_text(encoding='utf-8'))
assert reg['target_phase_amplitude_mv'] == 300 and reg['stimulation_sent'] is False
assert reg['current_session']['chip_id'] is None
assert reg['session_limits']['max_biological_seconds'] == 7200
assert (65535*3+3387+8)*50 == 10_000_000
files = [SOURCE/TEMPLATE, SOURCE/'standalone/stimulation/brain_stimulation/runner.py',
         SOURCE/'standalone/stimulation/README.md', SOURCE/'tests/unit/test_standalone_stimulation.py',
         registration, PROJECT/'docs/MEA_MAXONE_STIMULATION_PROFILE_20261117.md', archive]
result = {'status': 'passed', 'source_files_in_export_verified': len(source_manifest['source_sha256']),
          'archive_sha256': sha(archive), 'template_empty_electrodes_rejected': True,
          'native_cli_timing_preview_passed': True,
          'target_mv': 300, 'historical_lsb_mv': str(lsb), 'nominal_phase_mv': str(bits*lsb),
          'current_device_lsb_requeried': False, 'physical_output_verified': False,
          'stimulation_sent': False, 'remote_deployment_performed': False,
          'files': [{'path': str(p), 'sha256': sha(p)} for p in files]}
(HERE/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status','source_files_in_export_verified','native_cli_timing_preview_passed','nominal_phase_mv','stimulation_sent')}))
