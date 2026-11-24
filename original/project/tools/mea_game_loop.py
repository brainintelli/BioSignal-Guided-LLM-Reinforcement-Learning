"""Native game + synthetic stimulation/response, never hardware or training.

python -m tools.mea_game_loop simulate --output output/mea_game_loop/run01
python -m tools.mea_game_loop replay --input output/mea_game_loop/run01
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import uuid

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_closed_loop import ClosedLoop
from agent.mea_protocol import canonical, identifier, integer
from agent.mea_reward import record_reward
from agent.mea_simulator import Simulator, State2Probe
from agent.mea_snapshot import FEATURE_NAMES
from agent.mea_stimulation import Baseline, OutputPath, StimulationParameters
from tools.mea_p0 import DEFAULT_GODOT, ROOT, _same, _source_identity, read_json, sha, write_json

CONFIG_PATH = ROOT / 'configs/mea_game_loop_sim.json'
FILES = ('config.json', 'baseline.json', 'rollout.jsonl', 'report.json')
FIXED = dict(schema='mea.game_loop.sim.v2', source='simulation', actual_stimulation_enabled=False,
             acquisition_enabled=False, realtime_training=False, r2_reward_enabled=False,
             alpha=1, action_source='state2_probe_not_llm', map_mode='infinite_freeze', navigation_steps=4,
             state_variant='reserves', reward_variant='bottleneck', gamma_tick=.99985)


def default_config(): return read_json(CONFIG_PATH)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(FIXED) | {
            'session_seconds', 'recovery_reserve_seconds', 'max_pulses', 'max_shams', 'output_path', 'stimulation_parameters'}:
        raise ValueError('Unexpected simulation configuration fields')
    for key, value in FIXED.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise ValueError('Simulation-only setting required: ' + key)
    if not isinstance(config['output_path'], dict) or set(config['output_path']) != {
            'path_id', 'dac_lsb_mv', 'uncertainty_mv', 'calibration_id', 'simulation_only'}:
        raise ValueError('Explicit virtual output path required')
    OutputPath(**config['output_path'])
    parameters = config['stimulation_parameters']
    if not isinstance(parameters, dict) or set(parameters) != set(StimulationParameters.__dataclass_fields__):
        raise ValueError('Explicit stimulation parameter profile required')
    StimulationParameters(**parameters)
    integer(config['session_seconds'], 'session_seconds', 1)
    integer(config['recovery_reserve_seconds'], 'recovery_reserve_seconds', 900)
    integer(config['max_pulses'], 'max_pulses'); integer(config['max_shams'], 'max_shams')
    if (config['session_seconds'] > 7200 or config['recovery_reserve_seconds'] >= config['session_seconds']
            or config['max_pulses'] > 25 or config['max_shams'] > 12):
        raise ValueError('Session limits exceeded')
    record_reward(0, alpha=config['alpha'])
    return deepcopy(config)


def synthetic_baseline(config, seed, session_id=None):
    session_id = 'simulation-' + uuid.uuid4().hex if session_id is None else identifier(session_id, 'session_id')
    if not session_id.startswith('simulation-'): raise ValueError('Synthetic session identity required')
    return Baseline(session_id=session_id,
                    chip_id='SYNTHETIC-NOT-A-REAL-CHIP', well_id=0, sample_rate_hz=20000,
                    channels=((0, 100), (1, 101), (2, 102), (3, 103)), electrode_id=100,
                    start_frame=0, end_frame=18_000_000,
                    events=tuple((i * 1000, 0, -50.) for i in range(1, 101)))


def source_identity(project):
    return _source_identity(project) | {'code/tools/mea_game_loop.py': sha(Path(__file__))}


def behavior_metrics(rows):
    positions = set()
    damage = zero_thirst = low_water = low_water_drinks = ticks = 0
    for row in rows:
        obs = row['observation']
        positions.add((obs['player']['x'], obs['player']['y']))
        low = obs['player']['thirst'] <= 20
        low_water += int(low)
        low_water_drinks += int(low and row['response']['action']['verb'] == 'drink')
        for tick in row['transition']['trace']:
            positions.add((tick['player']['x'], tick['player']['y']))
            damage += tick['damage']; zero_thirst += int(tick['thirst'] <= 0); ticks += 1
    return dict(unique_walked_positions=len(positions), total_damage=damage, zero_thirst_ticks=zero_thirst,
                low_water_decision_opportunities=low_water, low_water_drink_decisions=low_water_drinks,
                low_water_drink_fraction=low_water_drinks / low_water if low_water else None,
                observed_game_ticks=ticks, meaning='software_probe_diagnostics_not_behavioral_efficacy')


def simulate(output, *, config=None, godot=DEFAULT_GODOT, project=ROOT, steps=12, seed=5100001):
    return _simulate(output, config=config, godot=godot, project=project, steps=steps, seed=seed)


def _simulate(output, *, config=None, godot=DEFAULT_GODOT, project=ROOT, steps=12, seed=5100001,
              replay_session_id=None):
    config = validate_config(default_config() if config is None else config)
    integer(steps, 'steps', 1); integer(seed, 'seed')
    shams = steps // 3
    if steps > 37 or steps - shams > config['max_pulses'] or shams > config['max_shams']:
        raise ValueError('Requested decisions exceed stimulus/sham budgets')
    project, godot, output = Path(project).resolve(), Path(godot).resolve(), Path(output)
    if not godot.is_file() or not (project / 'project.godot').is_file():
        raise ValueError('Native Godot executable and project required')
    baseline = synthetic_baseline(config, seed, replay_session_id)
    factory = ClosedLoop if replay_session_id is None else ClosedLoop.for_replay
    loop = factory(baseline, OutputPath(**config['output_path']), max_seconds=config['session_seconds'],
                   reserve_seconds=config['recovery_reserve_seconds'],
                   max_pulses=config['max_pulses'], max_shams=config['max_shams'],
                   parameters=StimulationParameters(**config['stimulation_parameters']))
    sources, engine_sha = source_identity(project), sha(godot)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        write_json(output / 'config.json', config)
        write_json(output / 'baseline.json', baseline.to_dict())
        backend, actor = Simulator(baseline), State2Probe()
        with (output / 'rollout.jsonl').open('x', encoding='utf-8', newline='\n') as log:
            with GodotEnv(str(godot), str(project), map_mode=config['map_mode'], navigation_steps=config['navigation_steps']) as env:
                obs = env.reset(seed)['observation']
                reward = Reward(obs, config['reward_variant'], config['gamma_tick'])
                for index in range(steps):
                    condition = 'sham' if index % 3 == 2 else 'stimulus'
                    response = loop.transact(obs, backend, actor, condition)
                    result = env.step(response['action'])
                    score = reward.score(result['transition'], result['observation'])
                    metrics = dict(zip(FEATURE_NAMES, response['snapshot']['feature_values']))
                    row = dict(index=index, observation=obs, response=response, transition=result['transition'],
                               next_observation=result['observation'], terminated=result['terminated'], r1_score=score,
                               reward=record_reward(score['reward'], alpha=config['alpha'], neural_metrics=metrics))
                    log.write(canonical(row) + '\n'); log.flush()
                    rows.append(row)
                    obs = result['observation']
                    if result['terminated']: break
        _same(source_identity(project), sources, 'sources changed during simulation')
        report = dict(status='complete', simulation_only=True, source='simulation',
                      action_source=actor.action_source, actual_stimulation_enabled=False, acquisition_enabled=False,
                      realtime_training=False, model_training_performed=False, model_inference_performed=False,
                      device_commands_sent=0, optimizer_steps=0, r2_reward_enabled=False, alpha=1,
                      biological_effect_verified=False, session_id=baseline.session_id,
                      decisions=len(rows), requested_decisions=steps, seed=seed,
                      elapsed_ticks=sum(r['r1_score']['elapsed_ticks'] for r in rows),
                      terminated=rows[-1]['terminated'], truncated=not rows[-1]['terminated'],
                      simulated_nonzero_attempts=loop.pulse_attempts, simulated_sham_attempts=loop.sham_attempts,
                      virtual_elapsed_seconds=(loop.reference_frame-baseline.start_frame)/baseline.sample_rate_hz,
                      peak_cap_mv=baseline.cap_mv, virtual_path=config['output_path'],
                      critic_input_dim=112, behavior=behavior_metrics(rows))
        write_json(output / 'report.json', report)
        write_json(output / 'manifest.json', dict(schema='mea.game_loop.run.v1',
                   created_utc=datetime.now(timezone.utc).isoformat(), seed=seed, requested_decisions=steps,
                   session_id=baseline.session_id,
                   sources=sources, engine_sha256=engine_sha, files={name: sha(output / name) for name in FILES}))
        return report
    except (Exception, KeyboardInterrupt) as exc:
        # repr retains malformed/non-finite input evidence without normalizing
        # it into a valid JSON packet. Never write a success manifest on error.
        (output / 'failure_audit.txt').write_text(repr(loop.audit) + '\n', encoding='utf-8')
        write_json(output / 'failure.json', dict(status='invalid', error_type=type(exc).__name__, error=str(exc),
                   committed_decisions=len(rows), pulse_attempts=loop.pulse_attempts, sham_attempts=loop.sham_attempts,
                   loop_state=loop.state, retry_performed=False, device_commands_sent=0, optimizer_steps=0))
        raise


def replay(input_dir, *, godot=DEFAULT_GODOT, project=ROOT):
    root, project, godot = Path(input_dir), Path(project).resolve(), Path(godot).resolve()
    if (root / 'failure.json').exists() or not (root / 'manifest.json').is_file():
        raise ValueError('Incomplete or failed simulation')
    manifest = read_json(root / 'manifest.json')
    if manifest.get('schema') != 'mea.game_loop.run.v1' or set(manifest.get('files', {})) != set(FILES):
        raise ValueError('Unknown completion manifest')
    for name in FILES:
        if sha(root / name) != manifest['files'][name]: raise ValueError('File hash mismatch: ' + name)
    _same(source_identity(project), manifest['sources'], 'source identity')
    _same(sha(godot), manifest['engine_sha256'], 'engine identity')
    config = validate_config(read_json(root / 'config.json'))
    # Regenerate the entire deterministic fixture, actor choice and native
    # transition. A rehashed arbitrary legal action must not pass a replay.
    with tempfile.TemporaryDirectory(prefix='mea-game-replay-') as tmp:
        expected = Path(tmp) / 'expected'
        report = _simulate(expected, config=config, godot=godot, project=project,
                           steps=manifest['requested_decisions'], seed=manifest['seed'],
                           replay_session_id=manifest['session_id'])
        for name in FILES:
            if (expected / name).read_bytes() != (root / name).read_bytes():
                raise ValueError('Replay content mismatch: ' + name)
    return dict(passed=True, simulation_only=True, verified_decisions=report['decisions'],
                verified_ticks=report['elapsed_ticks'], device_commands_sent=0, optimizer_steps=0,
                alpha=1, r2_reward_enabled=False, model_inference_performed=False,
                run_manifest_sha256=sha(root / 'manifest.json'),
                checks=['hashes', 'source_and_engine', 'baseline_p90', 'plans_and_quantization',
                        'receipts_raw_events_and_windows', 'actor_projection_and_probe_actions',
                        'native_transitions', 'exact_r1_only_rewards', 'behavior_metrics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sim = commands.add_parser('simulate')
    sim.add_argument('--output', type=Path, required=True)
    sim.add_argument('--config', type=Path, default=CONFIG_PATH)
    sim.add_argument('--steps', type=int, default=12)
    sim.add_argument('--seed', type=int, default=5100001)
    rep = commands.add_parser('replay')
    rep.add_argument('--input', type=Path, required=True)
    rep.add_argument('--report', type=Path)
    for sub in (sim, rep):
        sub.add_argument('--godot', type=Path, default=DEFAULT_GODOT)
        sub.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'simulate':
        result = simulate(args.output, config=read_json(args.config), godot=args.godot, project=args.project,
                          steps=args.steps, seed=args.seed)
    else:
        result = replay(args.input, godot=args.godot, project=args.project)
        if args.report: write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__': main()
