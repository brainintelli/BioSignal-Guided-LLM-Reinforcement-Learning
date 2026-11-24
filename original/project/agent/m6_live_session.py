"""Live transport identity, durable retry cache, and decoding-specific release gate.

The context is RNG/logging metadata. It is never passed to the actor encoder.
No automatic session eviction or seed reroll occurs after an inference failure.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import re

STOCHASTIC = 'categorical verb then target; temperature=1; independent episode CUDA generator'
GREEDY = 'argmax verb then conditional target'


def validate_context(context):
    if not isinstance(context,dict) or set(context) != {'session_id','episode','world_seed'}:
        raise ValueError('Three transport identity fields required')
    if not isinstance(context['session_id'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',context['session_id']):
        raise ValueError('Invalid client session identity')
    for key in ('episode','world_seed'):
        if type(context[key]) is not int or not 0 <= context[key] < 2**53:
            raise ValueError('Invalid transport '+key)
    return (context['session_id'],context['episode'])


def live_seed(context):
    validate_context(context)
    text=f"m6_live_v1|{context['session_id']}|{context['episode']}|{context['world_seed']}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big') % (2**63-1)


def observation_sha(observation):
    return hashlib.sha256(json.dumps(observation,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate_release(report, actor_sha, decoding):
    expected = STOCHASTIC if decoding == 'stochastic' else GREEDY if decoding == 'greedy' else None
    if expected is None or report.get('gate_passed') is not True or report.get('candidate_actor_sha256') != actor_sha:
        raise ValueError('Survival verification does not match this checkpoint')
    primary=report.get('primary',{})
    if primary.get('candidate_decoding') != expected or primary.get('gate_passed') is not True:
        raise ValueError('Serving decoding differs from evaluated decoding')
    if decoding == 'stochastic':
        guards=report.get('historical_guards',{})
        if (report.get('protocol') != 'm6_stochastic_v1' or report.get('overall_status') != 'evaluated'
            or report.get('primary_gate_passed') is not True
            or set(guards) != {'legacy_fresh_base','legacy_checkpoint0016'}
            or any(row.get('gate_passed') is not True for row in guards.values())):
            raise ValueError('Both historical guards and primary stochastic gate are required')
        audits=report.get('required_audits',{})
        names={'base_replay_audit.json','candidate_replay_audit.json','rng_base.json','rng_candidate.json'}
        if set(audits)!=names:
            raise ValueError('Four independent replay and RNG audits are required')
        for name,row in audits.items():
            hashes=[value for key,value in report.get('report_sha256',{}).items()
                    if key.replace('\\','/').rsplit('/',1)[-1]==name]
            if (row.get('passed') is not True or type(row.get('replayed_decisions')) is not int
                or row['replayed_decisions']<=0 or hashes!=[row.get('sha256')]
                or any(not isinstance(row.get(k),str) or not re.fullmatch('[a-f0-9]{64}',row[k])
                       for k in ('sha256','trace_sha256'))):
                raise ValueError('Missing or inconsistent audit evidence: '+name)
        for arm in ('base','candidate'):
            replay=audits[arm+'_replay_audit.json'];rng=audits['rng_'+arm+'.json']
            if any(replay[k]!=rng[k] for k in ('trace_sha256','replayed_decisions')):
                raise ValueError('Replay and RNG audit disagree for '+arm)
        historical=report.get('historical_audits',{})
        if set(historical)!={'legacy_base_replay_audit.json','legacy_checkpoint_replay_audit.json'}:
            raise ValueError('Both historical world-replay audits are required')
        for arm,guard in (('legacy_base','legacy_fresh_base'),('legacy_checkpoint','legacy_checkpoint0016')):
            name=arm+'_replay_audit.json';row=historical[name]
            if (row.get('passed') is not True or type(row.get('episodes')) is not int or row['episodes']!=20
                or type(row.get('replayed_decisions')) is not int or row['replayed_decisions']<=0
                or row.get('actor_sha256')!=guards[guard].get('before_actor_sha256')):
                raise ValueError('Historical replay identity/count differs: '+arm)
            for file,field in ((name,'sha256'),(arm+'.json','report_sha256')):
                hashes=[v for k,v in report.get('report_sha256',{}).items() if k.replace('\\','/').rsplit('/',1)[-1]==file]
                if (hashes!=[row.get(field)] or not isinstance(row.get(field),str)
                    or not re.fullmatch('[a-f0-9]{64}',row[field])):
                    raise ValueError('Historical audit/report hash differs: '+file)
    return expected


class SessionRegistry:
    """Called under the server inference lock; snapshot/restore operate on bytes."""
    def __init__(self, generator_factory, commit, max_sessions=4096):
        self.generator_factory=generator_factory
        self.commit=commit
        self.max_sessions=max_sessions
        self.sessions={}
        self.poisoned=False

    def _entry(self,context):
        key=validate_context(context)
        if key not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                raise ValueError('Session capacity reached; refusing to evict an RNG stream')
            self.sessions[key]=dict(context=copy.deepcopy(context),generator=self.generator_factory(live_seed(context)),
                                    tick=-1,request_sha256=None,decision=None)
        entry=self.sessions[key]
        if entry['context'] != context:
            raise ValueError('World seed changed inside the same journey')
        return entry

    def decide(self,context,observation,sample):
        if self.poisoned:
            raise RuntimeError('Durable log failed; inspect and restart before more decisions')
        tick=observation.get('tick')
        if type(tick) is not int or tick<0:
            raise ValueError('Integer observation tick required')
        digest=observation_sha(observation)
        entry=self._entry(context)
        if tick < entry['tick']:
            raise ValueError('Stale observation')
        if tick == entry['tick']:
            if digest != entry['request_sha256']:
                raise ValueError('Changed observation at an already decided tick')
            return copy.deepcopy(entry['decision'])
        generator=entry['generator'];before=generator.snapshot()
        try:
            decision=sample(generator)
            if decision.get('tick')!=tick or decision.get('source')!='model' or decision.get('raw_valid') is not True:
                raise ValueError('Sampler produced an invalid response envelope')
            # Godot's JSON parser represents numbers as doubles. Preserve the
            # exact 63-bit RNG identity as decimal text in live transport/logs.
            decision=dict(decision,agent_context=copy.deepcopy(context),rng_seed=str(live_seed(context)),
                          rng_before_sha256=hashlib.sha256(before).hexdigest(),
                          rng_after_sha256=hashlib.sha256(generator.snapshot()).hexdigest())
            record=dict(agent_context=copy.deepcopy(context),observation=copy.deepcopy(observation),
                        request_sha256=digest,decision=copy.deepcopy(decision),
                        generator_state_b64=base64.b64encode(generator.snapshot()).decode())
            # A response may leave the process only after its exact RNG state and
            # choice are durably logged. A timeout retry then returns this choice.
            try:
                self.commit(record)
            except Exception:
                # A failed fsync may have written all, part, or none of a row.
                # Never guess and draw another action in this process.
                self.poisoned=True
                raise
        except Exception:
            generator.restore(before)
            raise
        entry.update(tick=tick,request_sha256=digest,decision=copy.deepcopy(decision))
        return decision

    def restore_records(self,records):
        for record in records:
            context=record['agent_context'];observation=record['observation'];decision=record['decision']
            if decision['agent_context']!=context or observation_sha(observation)!=record['request_sha256']:
                raise ValueError('Corrupted durable session record')
            state=base64.b64decode(record['generator_state_b64'],validate=True)
            entry=self._entry(context)
            if (type(observation['tick']) is not int or observation['tick']<=entry['tick']
                or decision['tick']!=observation['tick'] or decision['rng_seed']!=str(live_seed(context))
                or decision['rng_before_sha256']!=hashlib.sha256(entry['generator'].snapshot()).hexdigest()
                or decision['rng_after_sha256']!=hashlib.sha256(state).hexdigest()):
                raise ValueError('Broken durable RNG/tick chain; refusing to resample')
            entry['generator'].restore(state)
            entry.update(tick=observation['tick'],request_sha256=record['request_sha256'],decision=copy.deepcopy(decision))
