"""Reproducible local-only validation. Output run directories must not exist."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.mea_game_loop import simulate, replay, default_config
from tools.mea_p0 import read_json, write_json

OUT = Path(__file__).resolve().parent
with patch('socket.socket.connect', side_effect=AssertionError('Device/network access forbidden')):
    report = simulate(OUT / 'run01', steps=37)
    replay_report = replay(OUT / 'run01')
    write_json(OUT / 'replay.json', replay_report)
    config = default_config()
    config['output_path'].update(path_id='historical_lsb_rejection_fixture', dac_lsb_mv=2.981901,
                                 uncertainty_mv=0, calibration_id='historical_nominal_not_loaded_calibration')
    try:
        simulate(OUT / 'historical_lsb_rejected', config=config, steps=1)
    except ValueError as exc:
        assert 'nonzero DAC level' in str(exc), str(exc)
    else:
        raise AssertionError('Sub-LSB request was not rejected')

rows = [json.loads(s) for s in (OUT / 'run01/rollout.jsonl').read_text(encoding='utf-8').splitlines()]
intervals = []
profile = default_config()['stimulation_parameters']
for index, row in enumerate(rows):
    response = row['response']
    plan = response['plan']
    assert plan['parameters'] == profile
    assert plan['phase_us'] == 200
    assert plan['peak_upper_mv'] <= plan['cap_mv']
    assert plan['dac_codes'] == [512-plan['dac_offset'], 512+plan['dac_offset'], 512]
    assert row['reward']['total_reward'] == row['r1_score']['reward']
    if index:
        previous = rows[index-1]['response']['plan']
        interval = (plan['stimulus_frame'] - previous['stimulus_frame']) / 20
        assert 10000 <= interval <= 14000
        assert interval == response['timing']['onset_interval_ms']
        assert response['timing']['post_pulse_delay_samples'] * 50 + 400 == interval * 1000
        intervals.append(interval)
    else:
        assert response['timing'] == dict(onset_interval_ms=None, post_pulse_delay_samples=None)
failure = read_json(OUT / 'historical_lsb_rejected/failure.json')
assert failure['pulse_attempts'] == 0 and failure['committed_decisions'] == 0
assert not (OUT / 'historical_lsb_rejected/manifest.json').exists()
assert report['simulated_nonzero_attempts'] == 25 and report['simulated_sham_attempts'] == 12
assert report['device_commands_sent'] == report['optimizer_steps'] == 0
result = dict(passed=True, decisions=len(rows), simulated_nonzero_attempts=25, simulated_sham_attempts=12,
              intervals_ms=intervals, min_onset_interval_ms=min(intervals), max_onset_interval_ms=max(intervals),
              parameters=profile, replay_passed=replay_report['passed'], historical_lsb_rejected=True,
              device_commands_sent=0, optimizer_steps=0, alpha=1, r2_reward_enabled=False,
              actual_stimulation_enabled=False, biological_effect_verified=False)
write_json(OUT / 'validation.json', result)
print(json.dumps({k: v for k, v in result.items() if k != 'intervals_ms'}, indent=2))
