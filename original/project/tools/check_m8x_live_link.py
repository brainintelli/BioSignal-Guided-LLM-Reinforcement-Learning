"""Two-phase post-release HTTP/restart probe using fixed training states.

Writes isolated probe journeys into the real service log. Does not start/stop
services, step Godot, or claim the second fixed observation follows its new action.
Run on the service host, prepare then operator restart (same log) then resume.
"""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.m8x_live_sampler_probe import preflight, training_pairs, write_new
from tools.check_m6_live_link import request
from tools.m8x_remote_service import identity, validate_health
from agent.m8x_identity import sha
from agent.m8x_server import validate_log_identity


def compare_decision(expected, actual):
    if any(actual.get(k) != value for k,value in expected.items()):
        raise ValueError('HTTP probabilities/indices/action/context/RNG differ from CUDA fixture')
    if 'source' in actual and (actual['source'] != 'model' or actual.get('raw_valid') is not True):
        raise ValueError('HTTP did not return a valid model decision')


def require_restart(before, after):
    if (before['pid'], before['start_ticks']) == (after['pid'], after['start_ticks']):
        raise ValueError('No actual service process restart observed')


def service_identity(path, verification_sha, checkpoint):
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    current = identity(saved['pid'])
    if any(saved.get(k) != v for k,v in current.items()):
        raise ValueError('Service record no longer owns the live process')
    if (saved.get('verification_sha256') != verification_sha or saved.get('checkpoint') != str(checkpoint.resolve())
            or 'agent.m8x_server' not in current['command']):
        raise ValueError('Owned service differs from accepted release')
    command = current['command']
    log = Path(command[command.index('--log')+1]).resolve()
    return current, log


def audit_log(path, contexts, expected_rows, actor, verification_sha):
    wanted = {c['session_id'] for c in contexts}
    found = {c['session_id']:[] for c in contexts}
    if not path.exists():
        if any(expected_rows): raise ValueError('Missing acknowledged service log')
        return {sid:0 for sid in wanted}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'): raise ValueError('Incomplete service log')
            record = json.loads(line)
            sid = record['agent_context']['session_id']
            if sid not in wanted: continue
            validate_log_identity(record, actor, verification_sha)
            d = record['decision']
            if hashlib.sha256(base64.b64decode(record['generator_state_b64'], validate=True)).hexdigest() != d['rng_after_sha256']:
                raise ValueError('Persisted RNG bytes disagree with decision')
            found[sid].append(record)
    for context, rows in zip(contexts, expected_rows):
        records = found[context['session_id']]
        if len(records) != len(rows): raise ValueError('Duplicate/missing durable probe record')
        for record, expected in zip(records, rows):
            if record['observation'] != expected['observation'] or record['decision'] != expected['decision']:
                raise ValueError('Durable record differs from acknowledged HTTP request')
    return {sid:len(rows) for sid,rows in found.items()}


def exercise(origin, observation, context, expected):
    status, decision = request(origin, '/decide', observation, context)
    if status != 200 or decision.get('source') != 'model' or decision.get('raw_valid') is not True:
        raise ValueError('Model request failed: '+str(decision))
    compare_decision(expected, decision)
    status, retry = request(origin, '/decide', observation, context)
    if status != 200 or retry != decision: raise ValueError('Retry was not exactly cached')
    changed = copy.deepcopy(observation)
    changed['inventory']['berries'] += 1
    status, error = request(origin, '/decide', changed, context)
    if status != 400 or error.get('detail') != 'Changed observation at an already decided tick':
        raise ValueError('Changed observation not rejected')
    status, retry = request(origin, '/decide', observation, context)
    if status != 200 or retry != decision: raise ValueError('Rejected request mutated cached choice')
    return dict(observation=observation, decision=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','resume'])
    parser.add_argument('--verification', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--service-record', required=True)
    parser.add_argument('--receipt', help='prepare output, required for resume')
    parser.add_argument('--project', default=str(ROOT))
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists(): raise FileExistsError(args.output)
    parsed = urllib.parse.urlparse(args.url)
    if not (parsed.scheme=='http' and parsed.hostname=='127.0.0.1' and parsed.port==18080
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and parsed.path in ('','/')):
        raise ValueError('Only owned local M8X service port 18080 is permitted')
    report, checkpoint = preflight(args.verification, args.project)
    digest = sha(args.verification)
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf-8'))
    if not (fixture.get('protocol')=='m8x_live_probe_v1' and fixture.get('passed') is True
            and fixture.get('actual_cuda') is True and fixture.get('verification_sha256')==digest
            and fixture.get('actor_sha256')==report['selected_actor_sha256'] and fixture.get('checkpoint')==str(checkpoint.resolve())):
        raise ValueError('Matching successful CUDA fixture is required')
    expected_sources = ('tools/m8x_live_sampler_probe.py','agent/m8x_server.py','agent/m6_live_session.py')
    if fixture.get('source_sha256') != {p:sha(ROOT/p) for p in expected_sources}:
        raise ValueError('CUDA probe or online implementation changed after fixture creation')
    pairs, _, trace_sha = training_pairs(report)
    if fixture['training_trace_sha256'] != trace_sha or len(fixture['contexts'])!=4 or len(fixture['fixtures'])!=4:
        raise ValueError('Training source identity/shape differs')
    for pair, rows, context in zip(pairs, fixture['fixtures'], fixture['contexts']):
        if len(rows)!=2 or not context['session_id'].startswith('m8x-probe-') or context['world_seed']!=pair[0][1]['seed']:
            raise ValueError('Invalid isolated probe journey')
        for (index, source), row in zip(pair, rows):
            if row['source_index']!=index or row['observation']!=source['observation']:
                raise ValueError('Fixture is not the exact fixed training observation')
    owner, log = service_identity(args.service_record, digest, checkpoint)
    status, health = request(args.url.rstrip('/'), '/health')
    if status != 200: raise ValueError('Service unavailable')
    validate_health(health, report, str(checkpoint.resolve()), digest)
    if health.get('source_sha256')!=report['source_sha256']:
        raise ValueError('Service frozen source differs')
    records = [[] for _ in pairs]
    if args.phase=='resume':
        if not args.receipt: raise ValueError('Prepare receipt is required')
        prior = json.loads(Path(args.receipt).read_text(encoding='utf-8'))
        if (prior.get('phase')!='prepare' or prior.get('passed') is not True or prior.get('fixture_sha256')!=sha(args.fixture)
                or prior.get('verification_sha256')!=digest or prior.get('service_log')!=str(log)):
            raise ValueError('Prepare receipt identity differs')
        require_restart(prior['service_identity'], owner)
        records = prior['records']
        audit_log(log, fixture['contexts'], records, fixture['actor_sha256'], digest)
        for context, rows in zip(fixture['contexts'], records):
            status, cached = request(args.url.rstrip('/'), '/decide', rows[0]['observation'], context)
            if status!=200 or cached!=rows[0]['decision']: raise ValueError('Restart lost original cached response')
    else:
        audit_log(log, fixture['contexts'], records, fixture['actor_sha256'], digest)
    step = 0 if args.phase=='prepare' else 1
    for i,(context, rows) in enumerate(zip(fixture['contexts'], fixture['fixtures'])):
        row = rows[step]
        records[i].append(exercise(args.url.rstrip('/'), row['observation'], context, row['expected']))
    counts = audit_log(log, fixture['contexts'], records, fixture['actor_sha256'], digest)
    final_owner, _ = service_identity(args.service_record, digest, checkpoint)
    if owner!=final_owner: raise ValueError('Service changed during probe')
    write_new(args.output, dict(passed=True, phase=args.phase, fixture_sha256=sha(args.fixture), verification_sha256=digest,
        service_identity=owner, service_log=str(log), health=health, records=records, durable_counts=counts,
        restarted_and_continued=args.phase=='resume', new_world_steps=0, held_out_observations=0,
        probe_log_rows_written=4, scope='Fixed training observations only; HTTP equivalence/retry plus restart only if resume; not survival evidence'))
    print(json.dumps(dict(passed=True, phase=args.phase, restarted_and_continued=args.phase=='resume')))


if __name__ == '__main__': main()
