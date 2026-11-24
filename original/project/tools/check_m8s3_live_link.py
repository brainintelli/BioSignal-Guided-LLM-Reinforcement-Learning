"""Two bounded real HTTP phases, separated by an actual server restart.

Uses fixed training-state CUDA fixtures. These requests do not constitute a new
survival rollout: the second observation followed the original training action.
"""
import argparse
import json
from pathlib import Path
import sys
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.m8s3_online_gate import verify_offline_gate
from agent.m8s3_training import sha
from agent.m8s3_server import validate_record,decision_identity
from tools.check_m6_live_link import request
from tools.check_m8x_live_link import exercise
from tools.m8x_live_sampler_probe import write_new
from tools.m8s3_live_sampler_probe import training_pairs


def validate_health(health,identity):
    if (any(health.get(k)!=v for k,v in identity.items()) or health.get('decision_ready') is not True
            or health.get('validation_authorized') is not True or health.get('numerical_gate_passed') is not True):
        raise ValueError('Live service differs from accepted validation identity')


def audit_records(log,contexts,records,identity):
    wanted={c['session_id'] for c in contexts};found={sid:[] for sid in wanted}
    if Path(log).exists():
        with Path(log).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):raise ValueError('Incomplete durable log')
                row=json.loads(line);sid=row['agent_context']['session_id']
                if sid in wanted:
                    validate_record(row,identity);found[sid].append(row)
    for context,expected in zip(contexts,records):
        actual=found[context['session_id']]
        if len(actual)!=len(expected):raise ValueError('Duplicate/missing durable request')
        for a,b in zip(actual,expected):
            if a['observation']!=b['observation'] or a['decision']!=b['decision']:
                raise ValueError('HTTP response differs from durable commit')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['prepare','resume'])
    for key in ('run','fixture','log','output'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--receipt');ap.add_argument('--url',default='http://127.0.0.1:18081')
    args=ap.parse_args()
    u=urllib.parse.urlparse(args.url)
    if u.scheme!='http' or u.hostname!='127.0.0.1' or u.port!=18081 or u.username or u.password or u.query or u.fragment or u.path not in ('','/'):
        raise ValueError('Only bounded local validation port18081 allowed')
    if Path(args.output).exists():raise FileExistsError(args.output)
    gate=verify_offline_gate(args.run);fixture=json.loads(Path(args.fixture).read_text())
    identity=decision_identity(gate['actor_sha256'],gate['registration_sha256'],gate['config']['verb_exploration_epsilon'])
    if fixture.get('protocol')!='m8s3_live_probe_v1' or fixture.get('passed') is not True or fixture.get('actual_cuda') is not True or fixture['identity']!=identity:
        raise ValueError('Actual-CUDA fixture differs')
    for name,digest in fixture['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Online sampler changed after CUDA probe')
    pairs,trace=training_pairs(gate)
    if fixture['training_trace_sha256']!=sha(trace):raise ValueError('Training fixture changed')
    for pair,rows in zip(pairs,fixture['fixtures']):
        for (_,source),row in zip(pair,rows):
            if row['observation']!=source['observation']:raise ValueError('Not the fixed training observations')
    status,health=request(args.url,'/health')
    if status!=200:raise ValueError('Validation service unavailable')
    validate_health(health,identity)
    if health['source_sha256']!={name:sha(ROOT/'agent'/name) for name in ('m8s3_server.py','m8s3_online_gate.py')}:
        raise ValueError('Live service source differs')
    records=[[] for _ in pairs]
    if args.phase=='resume':
        prior=json.loads(Path(args.receipt).read_text())
        if prior.get('passed') is not True or prior['phase']!='prepare' or prior['fixture_sha256']!=sha(args.fixture) or prior['service_log']!=str(Path(args.log).resolve()):
            raise ValueError('Prepare receipt differs')
        if (prior['health']['process_pid'],prior['health']['process_started_utc'])==(health['process_pid'],health['process_started_utc']):
            raise ValueError('No actual server process restart')
        records=prior['records'];audit_records(args.log,fixture['contexts'],records,identity)
        for context,rows in zip(fixture['contexts'],records):
            status,cached=request(args.url,'/decide',rows[0]['observation'],context)
            if status!=200 or cached!=rows[0]['decision']:raise ValueError('Restart lost exact response')
    else:audit_records(args.log,fixture['contexts'],records,identity)
    step=0 if args.phase=='prepare' else 1
    for i,(context,rows) in enumerate(zip(fixture['contexts'],fixture['fixtures'])):
        row=rows[step];records[i].append(exercise(args.url,row['observation'],context,row['expected']))
    audit_records(args.log,fixture['contexts'],records,identity)
    write_new(args.output,dict(passed=True,phase=args.phase,fixture_sha256=sha(args.fixture),identity=identity,
        health=health,records=records,service_log=str(Path(args.log).resolve()),restarted_and_continued=args.phase=='resume',
        new_world_steps=0,held_out_observations=0,scope='Actual HTTP, exact retries, rejected altered requests, and restart only in resume phase'))
    print(json.dumps(dict(passed=True,phase=args.phase)),flush=True)


if __name__=='__main__':main()
