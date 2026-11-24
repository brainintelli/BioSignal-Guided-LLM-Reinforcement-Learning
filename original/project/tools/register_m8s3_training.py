"""Freeze a derived open4 training run without changing prior registrations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.m8s3_training import sha, write, verify_files, validate_seed_split, training_slots
from agent.m8s3_diag_contract import registration_digest, verify_registration_digest


def registered_sources(root, inherited, formal=False):
    names = {'agent/m8s3_training.py', 'agent/m8s3_run.py', 'tools/register_m8s3_training.py',
             'tools/m8s3_job.py', 'tools/audit_m8s3_training.py', 'tests/test_m8s3_training.py',
             'agent/m8s3_supply.py', 'tests/test_m8s3_supply.py'}
    if formal:
        from agent.m8x_identity import CORE_FILES
        names.update(CORE_FILES)
    sources = dict(inherited)
    for name in sorted(names):
        sources[name] = sha(root / name)
    return sources


def warm_checkpoint_inputs(checkpoint):
    root = Path(checkpoint).resolve()
    manifest = root / 'manifest.json'
    files = json.loads(manifest.read_text(encoding='utf-8'))['files']
    result = {'warm_start_manifest': dict(path=str(manifest), sha256=sha(manifest))}
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Warm checkpoint file changed: ' + name)
        result['warm_start:' + name] = dict(path=str(path), sha256=digest)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    c = json.loads(Path(a.config).read_text())
    root = Path(c['project'])
    parent = json.loads(Path(c['parent_registration']).read_text())
    verify_registration_digest(parent)
    verify_files(root, parent)
    validate_seed_split(list(range(c['train_seed_start'], c['train_seed_start']+c['train_seed_count'])),
                        c['validation_seeds'], c['test_seeds'], c['pilot_eval_seeds'])
    if c['mode'] not in ('pilot', 'formal') or len(c['validation_seeds']) != 16 or len(c['test_seeds']) != 100:
        raise ValueError('Invalid assessment matrix')
    if c['validation_every'] % c['save_every'] or c['updates'] % c['validation_every']:
        raise ValueError('Every assessment candidate must have a checkpoint')
    if c['training_horizon'] is not None or c['resume'] is not None:
        raise ValueError('Natural episodes and fresh training required')
    training_slots(1, c['envs'], c.get('environment_pool_size', c['envs']))
    from agent.m8s3_supply import reward_config
    reward_config(c)
    if type(c.get('critic_readiness',False)) is not bool:raise ValueError('Critic extension must be explicit')
    if c.get('optimization_anchor_checkpoint'):
        if c['optimization_anchor_checkpoint']!=c.get('warm_start_checkpoint'):
            raise ValueError('This experiment anchors only to its frozen warm candidate')
        meta=json.loads((Path(c['optimization_anchor_checkpoint'])/'policy.json').read_text())
        if meta['actor_sha256']!=c['optimization_anchor_actor_sha256']:
            raise ValueError('Optimization anchor digest differs')
    if c.get('stop_after_consecutive_perfect', 0):
        if (c['mode'] != 'formal' or c['stop_after_consecutive_perfect'] < 2
                or not c['validation_every'] <= c['minimum_updates'] <= c['updates']):
            raise ValueError('Invalid preregistered adaptive stopping rule')
    sources = registered_sources(root, parent['source_hashes'], formal=c['mode'] == 'formal')
    inputs = dict(parent['input_files'])
    if c.get('warm_start_checkpoint'):
        if c.get('warm_start_critic') is not True or c.get('reset_critic') is not False:
            raise ValueError('Warm start explicitly loads the compatible 96-feature critic, with fresh optimizers')
        inputs.update(warm_checkpoint_inputs(c['warm_start_checkpoint']))
    if c['mode'] == 'formal':
        critic = Path(c['parent_checkpoint']) / 'critic.pt'
        inputs['legacy_parent_critic'] = dict(path=str(critic), sha256=sha(critic))
    reg = dict(config=c, source_hashes=sources, input_files=inputs,
        parent_registration_sha256=parent['registration_sha256'],
        created_utc=datetime.now(timezone.utc).isoformat(),
        selection='all registered checkpoints; rank successes, CVaR20, RMST, earliest; require 15/16, no early deaths, same-rule parent NI',
        test='one frozen 100-world matrix; >=90 alive at 1800, <=5 deaths before360; paired RMST lowerCI>-90, success lowerCI>-.05, CVaR>=90% parent',
        complete_requires='numeric gate + independent reload/replay/RNG audit + real HTTP and visible game acceptance',
        stopping='finite registered maximum or explicitly configured consecutive perfect screens after minimum updates; failure never means milestone complete',
        resume='no automatic resume; preserve optimizer and source evidence only')
    reg['registration_sha256'] = registration_digest(reg)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'registration.json', reg)
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(root / name, name)
    print(json.dumps(dict(registration_sha256=reg['registration_sha256'], sources=len(sources))))


if __name__ == '__main__':
    main()
