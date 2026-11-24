"""Audit this session's finalized recordings; never contacts acquisition hardware."""
import argparse
from decimal import Decimal, ROUND_FLOOR
import datetime
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--software-root', type=Path, default=Path(__file__).resolve().parents[2])
args = p.parse_args()
sys.path.insert(0, str(args.software_root))
from agent.mea_recording import load_h5, sha256
from agent.mea_timing import audit_progress, audit_filtered_progress
from tools.mea_timing_audit import rows

run = args.run.resolve()
def read(name): return json.loads((run / name).read_text(encoding='utf-8'))
def write(name, data):
    with (run / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')

acq = read('acquisition.json')
strict = read('audit.json')
assert acq['success'] and acq['stimulation'] is False and acq['alpha'] == 1
assert acq['r2_reward_enabled'] is False and acq['legacy_restored']
start = datetime.datetime.fromisoformat('2026-11-17T03:28:40+00:00')
deadline = datetime.datetime.fromisoformat('2026-11-17T05:28:40+00:00')
assert start <= datetime.datetime.fromisoformat(acq['recording_started_utc'])
assert datetime.datetime.fromisoformat(acq['finished_utc']) < deadline
assert strict['metadata']['chip_id'] == '22021'
strict_before = sha256(run / 'audit.json')
write('chip_confirmation.json', dict(schema='mea.user_chip_confirmation.v1', source='explicit_user_reply',
      statement='芯片编号22021 ip 192.0.2.101 刺激电极编号你来根据历史数据选择',
      original_expected_chip_id='22021', chip_id='22021'))
for action in ('admit', 'verify'):
    with (run / ('subset_' + action + '.log')).open('x') as log:
        result = subprocess.run([sys.executable, '-s', '-m', 'tools.mea_h5_admit', action, str(run),
                                 '--chip-id', '22021'], cwd=str(args.software_root),
                                stdout=log, stderr=subprocess.STDOUT, timeout=240)
    assert result.returncode == 0, 'Subset ' + action + ' failed; inspect log'
assert sha256(run / 'audit.json') == strict_before, 'Strict audit must remain unchanged'
admission = read('admission.json')
meta, frames, spikes = load_h5(run / 'baseline.raw.h5')
event_rows = rows(run / 'events.jsonl')
timing = audit_progress(rows(run / 'events.jsonl.progress.jsonl'), frames, meta['sample_rate_hz'], event_rows)
filtered = audit_filtered_progress(event_rows, frames, meta['sample_rate_hz'], admission['offline_integrity_passed'])
timing.update(filtered_stream=filtered, mapping_basis='finalized_h5_metadata',
              hardware_mapping_independently_verified=False,
              recorded_event_agreement_passed=admission['offline_integrity_passed'],
              stimulation_clock_aligned=False, live_spike_window_complete=False)
timing['subset_diagnostic_passed'] = all(timing[k] for k in (
    'raw_progress_passed', 'raw_freshness_passed', 'receive_clock_fit_passed', 'wall_clock_stable')) and filtered['observed_transport_gate_passed']
timing['inputs'] = {n: sha256(run / n) for n in ('admission.json', 'events.jsonl', 'events.jsonl.progress.jsonl')}
write('subset_timing_audit.json', timing)
assert timing['subset_diagnostic_passed'], 'Timing check failed'

lo, hi = admission['audit_interval']
middle = (lo + hi) // 2
mapping = admission['analysis_mapping']
scale = Decimal(str(meta['lsb_volts'])) * 1000000
preflight = json.loads((run.parent / 'preflight.json').read_text(encoding='utf-8'))
assert preflight['host'] == '192.0.2.101' and preflight['expected_chip_id'] == '22021'
lsbs = [Decimal(str(v)) for v in preflight['lsb_mv']]
assert len(lsbs) == 3 and all(v.is_finite() and v > 0 and v == lsbs[0] for v in lsbs)
lsb = lsbs[0]
candidates = []
for electrode in (7587, 2047, 4267):
    channels = [c for c, e in mapping if e == electrode]
    selected = [(f, abs(Decimal(str(a))) * scale) for f, c, a in spikes if lo <= f < hi and c in channels]
    values = sorted(a for f, a in selected)
    p90 = cap = bits = None
    if values:
        index = Decimal('0.9') * (len(values) - 1)
        low = int(index)
        p90 = values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)
        cap = Decimal('1.2') * p90 / 1000
        bits = min(511, int((cap / lsb).to_integral_value(rounding=ROUND_FLOOR)))
    first = sum(f < middle for f, a in selected)
    candidates.append(dict(electrode_id=electrode, h5_channels=channels, events=len(values),
        first_half_events=first, second_half_events=len(values)-first,
        p90_uv=float(p90) if p90 is not None else None,
        cap_mv_exact=str(cap) if cap is not None else None,
        cap_mv=float(cap) if cap is not None else None,
        nominal_dac_steps_best_case=bits,
        enough_for_candidate_review=len(values) >= 100 and 0 < first < len(values),
        executable_stimulation=False))
preferred = next((c['electrode_id'] for c in candidates if c['enough_for_candidate_review']), None)
result = dict(schema='mea.live_session.offline_candidate_review.v1', chip_id='22021',
    recording_seconds=admission['raw_duration_seconds'], analyzed_seconds=admission['audited_seconds'],
    strict_audit_passed=strict['offline_integrity_passed'], subset_admission_passed=True,
    analysis_channels=admission['mapped_channels'], matched_events=admission['matched_events'],
    preferred_historical_candidate=preferred, candidate_selection='historical_order_then_current_presence_and_event_coverage',
    candidates=candidates, dac_lsb_mv=str(lsb), dac_query_source='../preflight.json',
    dac_query_sha256=sha256(run.parent / 'preflight.json'),
    loaded_output_calibration_available=False, stimulation_sent=False,
    alpha=1, r2_reward_enabled=False, realtime_training=False,
    limits=['H5 metadata provides mapping; independent hardware mapping is not established.',
            'No spike-time maximum-lateness guarantee or absolute stimulation clock alignment.',
            'No biological efficacy claim; physical stimulation remains disabled.'])
write('candidate_review.json', result)
print(json.dumps(result, indent=2, ensure_ascii=False))
