"""Independent arithmetic and source-integrity audit, no preparation imports."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = json.loads((ROOT/'preparation_report_final.json').read_text(encoding='utf-8'))
source = Path(report['source_path'])
reference = json.loads(source.read_text(encoding='utf-8'))
assert sha(source) == report['reference_sha256']
assert sha(ROOT/'prepare.py') == report['tool_sha256']
assert sha(ROOT/'session_template.json') == report['session_template_sha256']
lookup = {row['electrode_id']: row for row in reference['limits']
          if row['chip_id'] == '22554' and row['scope'] == 'per_electrode'}
assert set(lookup) == {r['electrode_id'] for r in report['electrodes']} == {9764, 1388, 9985}
for row in report['electrodes']:
    original = lookup[row['electrode_id']]
    cap_uv = Fraction(original['p90_uv']) * Fraction(6, 5)
    cap_mv = cap_uv / 1000
    assert Fraction(row['cap_uv']) == cap_uv == Fraction(original['cap_phase_uv'])
    assert Fraction(row['cap_mv']) == cap_mv == Fraction(original['cap_phase_mv'])
    assert cap_mv < Fraction(row['dac_lsb_mv'])
    assert row['offset_bits'] == 0 and row['candidate_phase_mv'] is None
    assert row['nonzero_encodable'] is False
assert report['execution_enabled'] is False and report['stimulation_sent'] is False
assert report['hardware_connected'] is False and report['r2_enabled'] is False
assert report['hardware_deadline_enforced'] is False and report['vendor_configuration'] is None
assert report['limits']['planned_seconds'] == (10+15+15+40+10+2)*60 == 5520
assert report['limits']['reserve_seconds'] == 1680
assert report['limits']['planned_seconds'] + report['limits']['reserve_seconds'] == 7200
assert report['limits']['latest_work_end_seconds'] + 600 + 120 == 7200
assert report['timing']['status'] == 'not_started'
assert all(report['current_session'][k] is None for k in ('device_host', 'chip_id', 'organoid_id', 'started_at'))
sources = [source, PROJECT/'output/mea_22554_state_reward_20261116/sr_policy.py',
           PROJECT/'agent/mea_reward.py',
           Path('./github_program/BrainDesktop/standalone/stimulation/brain_stimulation/runner.py'),
           Path('./github_program/BrainDesktop/standalone/stimulation/example.json')]
result = {'status': 'passed', 'historical_electrode_count': 3, 'all_nonzero_outputs_blocked': True,
          'experiment_started': False, 'runtime_stop_verified': False,
          'source_files_read_only': [{'path': str(p), 'sha256': sha(p)} for p in sources],
          'artifacts': [{'path': str(p), 'sha256': sha(p)} for p in sorted(ROOT.glob('*'))
                        if p.is_file() and p.name not in {'delivery_validation.json'}]}
(ROOT/'delivery_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'historical_electrode_count', 'all_nonzero_outputs_blocked', 'experiment_started', 'runtime_stop_verified')}))
