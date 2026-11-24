"""Bind immutable prior evidence; reuse completed runs, recover evaluation only."""
import json
from pathlib import Path
import shutil

from agent.m8s3_training import sha, write
from agent.m9_metrics import summarize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect_resume(config):
    from tools.m9_campaign import verify_registration, task_spec
    previous_path = Path(config['resume_from']).resolve()
    previous = read(previous_path)
    verify_registration(previous)
    old = previous['config']
    if set(config) != set(old) | {'resume_from'}:
        raise ValueError('Unexpected resume configuration fields')
    allowed = {'project', 'output', 'job', 'hard_wall_seconds', 'common'}
    for key in set(old) - allowed:
        if config[key] != old[key]:
            raise ValueError('Resume changes study definition: ' + key)
    common = dict(config['common']);common['max_tokens'] = 1536
    if common != old['common'] or config['common']['max_tokens'] != 4096:
        raise ValueError('Only the lossless context capacity may change')
    if any(Path(config[k]).resolve() == Path(old[k]).resolve() for k in ('project', 'output', 'job')):
        raise ValueError('Resume requires separate source, output, and job paths')
    root = Path(old['output'])
    if list(root.rglob('*claim*.json')) or list(root.rglob('test_*.json')) or (root/'selection.json').exists():
        raise ValueError('Cannot resume an exposed final-test or selected campaign')
    finished_path = Path(old['job'])/'finished.json'
    finished = read(finished_path)
    if finished['status'] != 'failed' or not 0 < config['hard_wall_seconds'] <= old['hard_wall_seconds']-finished['seconds']+1:
        raise ValueError('Resume must retain the remaining active wall budget')
    inputs = {}
    def bind(key, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected immutable regular evidence file: '+str(path))
        inputs['resume:'+key] = dict(path=str(path.resolve()), sha256=sha(path))
    bind('registration', previous_path);bind('finished', finished_path)
    for name in previous['source_hashes']:
        bind('source:'+name, Path(old['project'])/name)
    runs = {};updates = decisions = 0
    for rep in old['phases']['explore']['replicates']:
        for arm in old['arms']:
            spec = task_spec(previous, 'explore', arm, rep)
            path = Path(spec['output'])
            if not path.exists():
                continue
            if read(path/'config.json') != spec:
                raise ValueError('Prior task config differs: '+path.name)
            metrics = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines() if line.strip()]
            if (len(metrics) != spec['updates'] or
                [m.get('update') for m in metrics] != list(range(1, spec['updates']+1)) or
                any(m.get('decisions') != m.get('update')*spec['envs']*spec['rollout_steps'] for m in metrics)):
                raise ValueError('Cannot recover incomplete optimization: '+path.name)
            checkpoint = path/f"checkpoint_{spec['updates']:04d}"
            manifest = read(checkpoint/'manifest.json')
            if manifest['update'] != spec['updates'] or manifest['registration_sha256'] != previous['registration_sha256']:
                raise ValueError('Checkpoint identity differs')
            for name, expected in manifest['files'].items():
                if sha(checkpoint/name) != expected:
                    raise ValueError('Checkpoint file changed: '+name)
            if not (checkpoint/'reload_probe.json').is_file():
                raise ValueError('Missing original probability probe')
            complete = (path/'training_complete.json').exists()
            if complete != (path/'evaluation.json').exists():
                raise ValueError('Incomplete result pair')
            for kind in ('initial', 'evaluation') if complete else ('initial',):
                report = read(path/(kind+'.json'))
                expected_actor = previous['warm_actor_sha256'] if kind=='initial' else manifest['actor_sha256']
                if (report['registration_sha256'] != previous['registration_sha256'] or
                    report['actor_sha256'] != expected_actor or
                    sorted(e['seed'] for e in report['episodes']) != spec['evaluation_seeds'] or
                    summarize(report['episodes'], spec['horizon']) != report['summary'] or
                    sha(path/(kind+'.jsonl.gz')) != report['trace_sha256']):
                    raise ValueError('Invalid prior assessment: '+path.name+'/'+kind)
            if complete:
                reload = read(checkpoint/'independent_reload.json')
                result = read(path/'training_complete.json')
                if (not reload.get('passed') or reload.get('logp_error', 1)>1e-6 or
                    reload.get('actor_sha256') != manifest['actor_sha256'] or
                    reload.get('registration_sha256') != previous['registration_sha256'] or
                    result.get('updates') != spec['updates'] or result.get('decisions') != metrics[-1]['decisions'] or
                    result.get('final_actor') != manifest['actor_sha256']):
                    raise ValueError('Incomplete reload/training proof')
            elif not any(s['stage']=='explore_'+path.name and s['exit_code']!=0 for s in finished['stages']):
                raise ValueError('Evaluation recovery must match the recorded failure')
            files = {}
            for entry in sorted(path.rglob('*')):
                if entry.is_file():
                    rel = entry.relative_to(path).as_posix()
                    bind(path.name+'/'+rel, entry);files[rel] = sha(entry)
            runs[path.name] = dict(kind='complete' if complete else 'evaluate_saved', source=str(path),
                files=files, checkpoint=str(checkpoint), actor_sha256=manifest['actor_sha256'],
                original_registration_sha256=previous['registration_sha256'])
            updates += spec['updates'];decisions += metrics[-1]['decisions']
    if not runs:
        raise ValueError('No reusable work')
    return dict(previous_registration=str(previous_path), previous_registration_sha256=previous['registration_sha256'],
        previous_active_seconds=finished['seconds'], runs=runs, reused_updates=updates,
        reused_decisions=decisions, semantics='Immutable results or complete-checkpoint evaluation; never resume partial optimizer/world state'), inputs


def verify_import(spec):
    source = Path(spec['resume_source'])
    for name, expected in spec['resume_files'].items():
        if sha(source/name) != expected:
            raise ValueError('Resume evidence changed: '+name)
    return source


def import_completed(spec):
    if spec['resume_kind'] != 'complete':
        raise ValueError('Only complete results may be imported')
    source = verify_import(spec)
    out = Path(spec['output']);out.mkdir(parents=True, exist_ok=False)
    for name in ('initial.json', 'evaluation.json', 'training_complete.json'):
        shutil.copy2(source/name, out/name)
    write(out/'config.json', spec)
    write(out/'reuse.json', dict(source=str(source), original_registration_sha256=spec['resume_registration_sha256'],
        registration_sha256=spec['registration_sha256'], kind='complete', original_reports_preserved=True))


def copy_evaluation_recovery(spec):
    if spec['resume_kind'] != 'evaluate_saved':
        raise ValueError('Only a completed checkpoint may recover assessment')
    source = verify_import(spec)
    out = Path(spec['output']);out.mkdir(parents=True, exist_ok=False)
    checkpoint = f"checkpoint_{spec['updates']:04d}"
    shutil.copytree(source/checkpoint, out/checkpoint)
    for name in spec['resume_files']:
        # Never overwrite or reinterpret the incomplete previous evaluation.
        if name in ('initial.json', 'initial.jsonl.gz', 'metrics.jsonl', 'episodes.jsonl') or name.startswith('rollout_'):
            shutil.copy2(source/name, out/name)
    write(out/'config.json', spec)
    write(out/'reuse.json', dict(source=str(source), original_registration_sha256=spec['resume_registration_sha256'],
        registration_sha256=spec['registration_sha256'], kind='evaluate_saved',
        optimization_repeated=False, evaluation_restarted_from_registered_seeds=True))
