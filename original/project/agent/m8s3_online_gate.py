"""Rebuild the offline acceptance gate before any new-world online validation.

A positive result authorizes a bounded validation session, not a completed
milestone or a permanent model service. No GPU is imported by this module.
"""
import json
import gzip
from pathlib import Path

from agent.m8s3_contract import VERSION, balance_identity
from agent.m8s3_diag_contract import verify_registration_digest
from agent.m8s3_training import authorize_test, sha, verify_files
from agent.m8x_policy_identity import PARENT_ACTOR_SHA256
from tools.m8b_metrics import compare_test, matrix
from agent.rl_contract import transition_at_horizon


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_report_endpoints(path, report, actor):
    """Bind reported survival to the actual previously audited trace bytes."""
    endpoints, counts = {}, {}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line); seed=row['seed']
            if seed in endpoints or row['actor_sha256']!=actor:
                raise ValueError('Evaluation continued past endpoint or actor changed')
            counts[seed]=counts.get(seed,0)+1
            final=transition_at_horizon(row['transition'],1800)['after']
            if final['dead'] or final['tick']==1800:
                endpoints[seed]=dict(survival_ticks=final['tick'],terminal=final['dead'],
                                    alive_at_horizon=not final['dead'],decisions=counts[seed])
    if len(report['episodes'])!=len(endpoints) or {e['seed'] for e in report['episodes']}!=set(endpoints):
        raise ValueError('Incomplete evaluation endpoint matrix')
    for episode in report['episodes']:
        if any(episode.get(k)!=v for k,v in endpoints[episode['seed']].items()):
            raise ValueError('Report differs from audited endpoint')


def verify_test_reports(config, candidate, parent, original, actor, parent_actor):
    expected={(seed,0) for seed in config['test_seeds']}
    if len(expected)!=100:
        raise ValueError('Registered test world matrix is not100 independent worlds')
    for report in (candidate,parent,original):
        if set(matrix(report['episodes'],100))!=expected:
            raise ValueError('Test world matrix differs from registration')
    for report,reference,digest in ((candidate,False,actor),(parent,True,parent_actor)):
        if (report.get('profile')!='open4' or report.get('world_version')!=6
                or report.get('temperature')!=1 or report.get('reference') is not reference
                or report.get('actor_sha256')!=digest):
            raise ValueError('New-world evaluation policy identity differs')
    if original.get('profile')!='open1' or original.get('actor_sha256')!=parent_actor or original.get('temperature')!=1:
        raise ValueError('Original-rule descriptive parent identity differs')
    result=compare_test(candidate['episodes'],parent['episodes'],original['episodes'])
    if not result['numerical_gate_passed']:
        raise ValueError('Recomputed numerical survival gate did not pass')
    return result


def verify_offline_gate(run):
    run=Path(run).resolve()
    reg=read(run/'registration.json')
    verify_registration_digest(reg)
    config=reg['config']
    if (Path(config['output']).resolve()!=run or config['mode']!='formal'
            or config['profile']!='open4' or config['world_version']!=6
            or config['contract']!=VERSION or config['temperature']!=1
            or config['horizon']!=1800 or config['max_tokens']!=1536):
        raise ValueError('Only the exact registered formal new-world run may validate online')
    verify_files(config['project'],reg)
    checkpoint=authorize_test(run,config)
    manifest=read(checkpoint/'manifest.json')
    for name,digest in manifest['files'].items():
        path=(checkpoint/name).resolve()
        if not path.is_relative_to(checkpoint) or sha(path)!=digest:
            raise ValueError('Selected checkpoint changed: '+name)
    meta=read(checkpoint/'policy.json')
    actor=meta['actor_sha256']
    if (manifest['registration_sha256']!=reg['registration_sha256']
            or manifest['actor_sha256']!=actor
            or meta['registration_sha256']!=reg['registration_sha256']
            or meta['inference_layout']!='semantic_v3_canonical_state_buckets_b4'
            or meta.get('verb_exploration_epsilon',0)!=config.get('verb_exploration_epsilon',0)):
        raise ValueError('Selected actor or categorical policy definition differs')
    if any(meta.get(k)!=v for k,v in balance_identity(config['project']).items()):
        raise ValueError('Selected actor world differs')
    if read(run/'test_claim.json')!={'actor_sha256':actor,'single_use':True}:
        raise ValueError('Single-use test claim differs from selected actor')
    audited={}
    for filename in ('independent_replay.json','independent_test_replay.json'):
        audit=read(run/filename)
        if (audit.get('passed') is not True or audit['registration_sha256']!=reg['registration_sha256']
                or audit['auditor_sha256']!=reg['source_hashes']['tools/audit_m8s3_training.py']
                or audit['transitions']!=sum(x['transitions'] for x in audit['files'].values())
                or audit['max_float_error']>1e-12):
            raise ValueError('Independent replay/RNG auditor identity or coverage differs')
        for name,item in audit['files'].items():
            path=(run/name).resolve()
            if not path.is_relative_to(run) or sha(path)!=item['sha256']:
                raise ValueError('Independently audited trace changed: '+name)
            audited[name]=item
    reports={}
    for stem in ('initial_evaluation','parent_selection','test_candidate','test_parent'):
        report=read(run/(stem+'.json'))
        trace=stem+'.jsonl.gz'
        if (trace not in audited or audited[trace]['sha256']!=report['trace_sha256']
                or audited[trace]['transitions']!=sum(e['decisions'] for e in report['episodes'])):
            raise ValueError('Evaluation report differs from independently replayed trace: '+stem)
        reports[stem]=report
        verify_report_endpoints(run/trace,report,report['actor_sha256'])
    for path in sorted(run.glob('selection_[0-9][0-9][0-9][0-9].json')):
        report=read(path)
        trace=path.with_suffix('.jsonl.gz')
        if trace.name not in audited or audited[trace.name]['sha256']!=report['trace_sha256']:
            raise ValueError('Screen was not independently audited')
        verify_report_endpoints(trace,report,report['actor_sha256'])
    old=read(run/'test_original_v2.json')
    if sha(run/'test_original_v2.jsonl')!=old['trace_sha256']:
        raise ValueError('Original-rule descriptive trace changed')
    result=verify_test_reports(config,reports['test_candidate'],reports['test_parent'],old,actor,PARENT_ACTOR_SHA256)
    if result!=read(run/'numerical_gate.json'):
        raise ValueError('Stored numerical gate differs from independently recomputed result')
    artifacts=['registration.json','training_complete.json','selection.json','test_claim.json',
               'independent_replay.json','independent_test_replay.json','numerical_gate.json',
               'test_candidate.json','test_parent.json','test_original_v2.json']
    return dict(offline_accepted=True,milestone_achieved=False,validation_authorized=True,
                registration_sha256=reg['registration_sha256'],checkpoint=str(checkpoint),
                actor_sha256=actor,config=config,numerical_gate=result,
                evidence_sha256={name:sha(run/name) for name in artifacts},
                checkpoint_manifest_sha256=sha(checkpoint/'manifest.json'))
