"""Offline P0 acceptance runner. Never imports a model or a device driver.

python -m tools.mea_p0 simulate --output output/mea_p0/run01
python -m tools.mea_p0 replay --input output/mea_p0/run01
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import VERSION, encode, observation_key
from agent.mea_protocol import ResearchStream, canonical, digest, integer, strict_json
from agent.mea_reward import record_reward
from agent.mea_snapshot import FEATURE_NAMES, Normalization, WetwareSnapshot, build_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GODOT = ROOT / '.tools/godot/Godot_v4.7.2-stable_win64_console.exe'
CONFIG_PATH = ROOT / 'configs/mea_p0.json'
DATA_FILES = ('stream.jsonl', 'rollout.jsonl', 'report.json')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(value) + '\n')


def read_json(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def _config(alpha):
    record_reward(0., alpha=alpha)
    config = read_json(CONFIG_PATH)
    record_reward(0., alpha=config['alpha'])
    if config['source'] != 'simulation' or config['action_source'] != 'scripted_p0':
        raise ValueError('This runner supports only offline simulation')
    Normalization(**config['normalization'])
    integer(config['max_decisions'], 'max_decisions', 1)
    integer(config['blank_frames'], 'blank_frames')
    integer(config['window_frames'], 'window_frames', 1)
    return config


def _source_files(project):
    paths = [CONFIG_PATH, Path(__file__)]
    paths += list((ROOT / 'agent').glob('*.py'))
    paths += list((project / 'scripts').rglob('*.gd'))
    paths += list((project / 'godot_tests').glob('*environment.gd'))
    paths += list((project / 'config').rglob('*.json'))
    paths += list((project / 'configs').rglob('*.json'))
    paths += [project / 'project.godot']
    return sorted(set(p.resolve() for p in paths if p.is_file()))


def _source_identity(project):
    # Record paths relative to their declared roots; replay does not open paths
    # supplied by the manifest. Recompute the complete current source set.
    return {('code/' + p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT)
             else 'project/' + p.relative_to(project).as_posix()): sha(p) for p in _source_files(project)}


def _same(actual, expected, name):
    if canonical(actual) != canonical(expected):
        raise ValueError(f'Replay mismatch: {name}')


class SyntheticProducer:
    """Seeded engineering fixture. Pattern labels have no hardware meaning."""
    def __init__(self, handle, config, seed):
        self.handle, self.config = handle, config
        self.stream = ResearchStream(max_line_bytes=config['max_line_bytes'], max_spikes=config['max_spikes'])
        self.normalization = Normalization(**config['normalization'])
        self.rng = random.Random(seed)
        self.seq, self.index = 0, 0
        self.session_id = 'sim-' + digest(dict(seed=seed, config=config))[:24]
        self.emit(dict(type='session', protocol='mea.spikes.v1', source='simulation', sample_rate_hz=config['sample_rate_hz'],
                       well_id=0, start_frame=0, channel_map_id='synthetic-four-channel-v1',
                       channels=[[i, 100 + i] for i in range(4)]))

    def emit(self, row):
        packet = dict(row, session_id=self.session_id, seq=self.seq)
        raw = (canonical(packet) + '\n').encode('utf-8')
        self.handle.write(raw)
        # Deliberately fragment transport boundaries, including inside numbers.
        for start in range(0, len(raw), 37):
            self.stream.feed(raw[start:start + 37])
        self.seq += 1

    def capture(self, observation):
        config = self.config
        context = observation_key(observation, config['state_variant'])
        pattern = int(context[:8], 16) % 4
        frame = self.index * (config['blank_frames'] + config['window_frames'] + 50)
        event_id = f'event-{self.index:04d}'
        self.emit(dict(type='stimulus', event_id=event_id, frame=frame, context_id=context, pattern_id=f'synthetic-{pattern}'))
        start, end = frame + config['blank_frames'], frame + config['blank_frames'] + config['window_frames']
        # Include regular valid quiet windows, and an artifact within blanking.
        count = 0 if self.index % 7 == 0 else 3 + pattern
        frames = sorted(self.rng.sample(range(start, end), count))
        events = ([[frame, 0, -999.]] if config['blank_frames'] else [])
        events += [[f, i % 4, -float(self.rng.randrange(10, 61))] for i, f in enumerate(frames)]
        self.emit(dict(type='spikes', events=events))
        self.emit(dict(type='watermark', end_frame=end, complete=True, dropped=0))
        self.index += 1
        return build_snapshot(self.stream, event_id, blank_frames=config['blank_frames'], window_frames=config['window_frames'],
                              normalization=self.normalization)


def _action(prepared, index):
    # Cycle available verb classes for transport coverage, not survival quality.
    branches = [branch for branch in prepared['branches'] if branch]
    branch = branches[index % len(branches)]
    return dict(branch[index % len(branch)])


def _reward(score, snapshot, config):
    return record_reward(score['reward'], alpha=config['alpha'],
                         neural_metrics=dict(zip(FEATURE_NAMES, snapshot.feature_values)))


def simulate(output, *, godot=DEFAULT_GODOT, project=ROOT, steps=32, seed=4100001, alpha=1.):
    config = _config(alpha)
    integer(steps, 'steps', 1); integer(seed, 'seed')
    if steps > config['max_decisions']:
        raise ValueError('P0 bounded run exceeds max_decisions')
    project, godot, output = Path(project).resolve(), Path(godot).resolve(), Path(output)
    if not godot.is_file() or not (project / 'project.godot').is_file():
        raise ValueError('Native Godot executable and project required')
    # Before the first action, record the exact code and engine identity.
    sources, engine_sha = _source_identity(project), sha(godot)
    output.mkdir(parents=True, exist_ok=False)
    decisions, total_ticks, terminated = 0, 0, False
    try:
        with (output / 'stream.jsonl').open('xb') as raw, (output / 'rollout.jsonl').open('x', encoding='utf-8', newline='\n') as log:
            producer = SyntheticProducer(raw, config, seed)
            with GodotEnv(str(godot), str(project), map_mode=config['map_mode'], navigation_steps=config['navigation_steps']) as env:
                observation = env.reset(seed)['observation']
                tracker = Reward(observation, config['reward_variant'], config['gamma_tick'])
                snapshot = producer.capture(observation)
                for index in range(steps):
                    prepared = encode(observation, snapshot, config['state_variant'])
                    action = _action(prepared, index)
                    result = env.step(action)
                    next_observation, transition = result['observation'], result['transition']
                    terminated = result['terminated']
                    score = tracker.score(transition, next_observation)
                    next_snapshot = None if terminated else producer.capture(next_observation)
                    row = dict(index=index, observation=observation, snapshot=snapshot.to_dict(), prepared=prepared,
                               action_source='scripted_p0', action=action, transition=transition, next_observation=next_observation,
                               next_snapshot=next_snapshot.to_dict() if next_snapshot else None,
                               terminated=terminated, r1_score=score, reward=_reward(score, snapshot, config))
                    log.write(canonical(row) + '\n')
                    log.flush(); raw.flush()
                    decisions += 1; total_ticks += score['elapsed_ticks']
                    if terminated:
                        break
                    observation, snapshot = next_observation, next_snapshot
            producer.stream.finish()
        _same(_source_identity(project), sources, 'source changed during run')
        report = dict(status='complete', simulation_only=True, action_source='scripted_p0', model_training_performed=False,
                      device_commands_sent=0, alpha=1., r2_enabled=False, contract=VERSION,
                      decisions=decisions, requested_decisions=steps, seed=seed, elapsed_ticks=total_ticks,
                      terminated=terminated, truncated=not terminated, snapshots=producer.index,
                      actor_input_verified='formatted_text_only', critic_input_dim=112,
                      biological_effect_verified=False)
        write_json(output / 'report.json', report)
        # Manifest is the completion marker, written last; failed/partial runs
        # have no valid marker and are retained for diagnosis, never retried.
        write_json(output / 'manifest.json', dict(schema='mea.p0.run.v1', created_utc=datetime.now(timezone.utc).isoformat(),
                                                config=config, seed=seed, requested_decisions=steps,
                                                sources=sources, engine_sha256=engine_sha,
                                                files={name: sha(output / name) for name in DATA_FILES}))
        return report
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='invalid', error_type=type(exc).__name__, error=str(exc),
                                               committed_decisions=decisions, retry_performed=False))
        raise


def replay(input_dir, *, godot=DEFAULT_GODOT, project=ROOT):
    root, project, godot = Path(input_dir), Path(project).resolve(), Path(godot).resolve()
    if (root / 'failure.json').exists() or not (root / 'manifest.json').is_file():
        raise ValueError('Incomplete or invalid run; no replay acceptance')
    manifest = read_json(root / 'manifest.json')
    if manifest.get('schema') != 'mea.p0.run.v1' or set(manifest.get('files', {})) != set(DATA_FILES):
        raise ValueError('Unknown manifest schema or files')
    for name in DATA_FILES:
        if not (root / name).is_file() or sha(root / name) != manifest['files'][name]:
            raise ValueError(f'Evidence hash mismatch: {name}')
    config = _config(manifest['config']['alpha'])
    _same(config, manifest['config'], 'configuration identity')
    _same(_source_identity(project), manifest['sources'], 'source identity')
    _same(sha(godot), manifest['engine_sha256'], 'engine identity')
    stream = ResearchStream(max_line_bytes=config['max_line_bytes'], max_spikes=config['max_spikes'])
    with (root / 'stream.jsonl').open('rb') as handle:
        while chunk := handle.read(8191):
            stream.feed(chunk)
    stream.finish()
    if stream.session is None or stream.session.source != 'simulation':
        raise ValueError('Synthetic evidence source required')
    normalization = Normalization(**config['normalization'])
    used = []

    def restore(data):
        snapshot = WetwareSnapshot.from_dict(data)
        rebuilt = build_snapshot(stream, snapshot.event_id, blank_frames=config['blank_frames'], window_frames=config['window_frames'],
                                 normalization=normalization)
        _same(snapshot.to_dict(), rebuilt.to_dict(), 'snapshot from persisted raw spikes')
        return rebuilt

    count, total_ticks, previous_next, terminated = 0, 0, None, False
    with GodotEnv(str(godot), str(project), map_mode=config['map_mode'], navigation_steps=config['navigation_steps']) as env:
        observation = env.reset(manifest['seed'])['observation']
        tracker = Reward(observation, config['reward_variant'], config['gamma_tick'])
        with (root / 'rollout.jsonl').open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n') or count >= config['max_decisions'] or terminated:
                    raise ValueError('Truncated, oversized or post-terminal rollout')
                row = strict_json(line)
                _same(row['index'], count, 'decision ordering')
                _same(row['observation'], observation, 'native observation')
                snapshot = restore(row['snapshot'])
                if snapshot.event_id in used:
                    raise ValueError('Snapshot consumed more than once')
                if previous_next is not None:
                    _same(snapshot.to_dict(), previous_next, 'next snapshot reused exactly once')
                used.append(snapshot.event_id)
                prepared = encode(observation, snapshot, config['state_variant'])
                _same(prepared, row['prepared'], 'actor/critic input')
                _same(row['action_source'], 'scripted_p0', 'action source')
                _same(row['action'], _action(prepared, count), 'scripted legal action')
                result = env.step(row['action'])
                for name in ('transition', 'terminated'):
                    _same(result[name], row[name], f'native {name}')
                observation, terminated = result['observation'], result['terminated']
                _same(observation, row['next_observation'], 'native next observation')
                score = tracker.score(result['transition'], observation)
                _same(score, row['r1_score'], 'R1 calculation')
                _same(_reward(score, snapshot, config), row['reward'], 'alpha=1 reward and telemetry')
                previous_next = row['next_snapshot']
                if terminated:
                    _same(previous_next, None, 'terminal snapshot')
                else:
                    following = restore(previous_next)
                    encode(observation, following, config['state_variant'])
                    if following.event_id in used:
                        raise ValueError('Next snapshot reused old event')
                count += 1; total_ticks += score['elapsed_ticks']
    report = read_json(root / 'report.json')
    if count < 1 or (not terminated and count != manifest['requested_decisions']) or count > manifest['requested_decisions']:
        raise ValueError('Incomplete rollout decision count')
    expected_snapshots = count + int(not terminated)
    if len(stream.stimuli) != expected_snapshots:
        raise ValueError('Extra or missing stimulus evidence')
    for key, value in dict(status='complete', decisions=count, elapsed_ticks=total_ticks, snapshots=expected_snapshots,
                           requested_decisions=manifest['requested_decisions'], seed=manifest['seed'],
                           terminated=terminated, truncated=not terminated, alpha=1., simulation_only=True,
                           r2_enabled=False, action_source='scripted_p0', model_training_performed=False,
                           device_commands_sent=0, biological_effect_verified=False, contract=VERSION,
                           actor_input_verified='formatted_text_only', critic_input_dim=112).items():
        _same(report.get(key), value, f'report {key}')
    return dict(passed=True, verified_decisions=count, verified_snapshots=expected_snapshots,
                verified_ticks=total_ticks, alpha=1., simulation_only=True, model_training_performed=False,
                checks=['file_hashes', 'source_and_engine_identity', 'stream_integrity', 'snapshots_from_raw_spikes',
                        'actor_critic_input', 'next_snapshot_reuse', 'native_transitions', 'r1_exact_reward'],
                run_manifest_sha256=sha(root / 'manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sim = commands.add_parser('simulate')
    sim.add_argument('--output', type=Path, required=True)
    sim.add_argument('--steps', type=int, default=32)
    sim.add_argument('--seed', type=int, default=4100001)
    sim.add_argument('--alpha', type=float, default=1.)
    rep = commands.add_parser('replay')
    rep.add_argument('--input', type=Path, required=True)
    rep.add_argument('--report', type=Path, help='Optional new JSON file; never overwrite run evidence')
    for sub in (sim, rep):
        sub.add_argument('--godot', type=Path, default=DEFAULT_GODOT)
        sub.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'simulate':
        result = simulate(args.output, godot=args.godot, project=args.project, steps=args.steps, seed=args.seed, alpha=args.alpha)
    else:
        result = replay(args.input, godot=args.godot, project=args.project)
        if args.report:
            write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
