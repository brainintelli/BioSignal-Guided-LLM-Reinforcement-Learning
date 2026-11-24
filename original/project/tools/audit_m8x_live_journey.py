"""Read-only client/server alignment over an actual Godot replay, after release.

No HTTP, torch, policy load, fresh model choice, or CUDA sampling. Input files
must be immutable snapshots with caller-supplied SHA256. Only natural death
completes an episode; otherwise this verifies a saved prefix, not survival forever.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.m8x_server import validate_log_identity
from agent.m8x_contract import balance_identity, validate_observation_identity
from agent.m6_live_session import live_seed, observation_sha, validate_context
from agent.m8x_env import GodotEnv
from tools.m8x_live_sampler_probe import preflight, write_new
from agent.m8x_identity import sha

CLIENT_FIELDS = ('actor_sha256','decoding','agent_context','rng_seed','rng_before_sha256','rng_after_sha256','raw_output')
VITALS = ('health','hunger','thirst','energy')


def compare(a,b,path='value'):
    if type(a) in (int,float) and type(b) in (int,float):
        if not (math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=0,abs_tol=1e-6)):
            raise ValueError(path+': numerical mismatch')
    elif isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys(): raise ValueError(path+': key mismatch')
        for k in a: compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):raise ValueError(path+': length mismatch')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
    elif type(a)!=type(b) or a!=b:
        raise ValueError(path+': mismatch')


def snapshot(path, expected):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Snapshot SHA differs: '+str(path))
    if raw and not raw.endswith(b'\n'):raise ValueError('Incomplete trailing JSONL record')
    def no_duplicates(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate JSON key: '+k)
            result[k]=v
        return result
    rows=[]
    for line in raw.splitlines():
        if not line.strip():raise ValueError('Blank JSONL record')
        row=json.loads(line,object_pairs_hook=no_duplicates,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        if not isinstance(row,dict):raise ValueError('JSONL record must be object')
        rows.append(row)
    return rows


def matching(rows, context, client=False):
    selected=[]
    for row in rows:
        found=row.get('agent_context',row.get('model_decision',{}).get('agent_context'))
        if not isinstance(found,dict):
            # Manual actions have no journey identity: cannot safely assign or skip.
            raise ValueError('Unattributable record or manual/rule action in snapshot')
        validate_context(found)
        if (found['session_id'],found['episode'])==(context['session_id'],context['episode']):
            if found!=context:raise ValueError('World seed changed inside journey')
            selected.append(row)
    return selected


def runtime_check(row, obs, after, trace):
    status=row['runtime_status']
    if status.get('runtime_day_limit','missing') is not None or status.get('runtime_tick_limit','missing') is not None:
        raise ValueError('Client runtime has a finite or missing limit declaration')
    dead=after['dead']; state=status['state']; conditions=[]
    if dead and trace and trace[-1]['tick']==after['tick'] and trace[-1]['damage']>0:
        for field,label in [('thirst','dehydration'),('hunger','starvation'),('energy','exhaustion')]:
            if trace[-1][field]<=0:conditions.append(label)
        camp=obs['camp'];p=obs['player']
        near=camp['exists'] and math.hypot(p['x']-camp['x'],p['y']-camp['y'])<=2.5
        if obs['phase']=='night' and not near:conditions.append('night_exposure')
        if not near and any(t['distance']<1.6 for t in obs['threats']):conditions.append('nearby_threat')
    labels={'dehydration':'缺水','starvation':'饥饿','exhaustion':'体力耗尽','night_exposure':'营外夜间伤害','nearby_threat':'附近威胁伤害'}
    detail='、'.join(labels[c] for c in conditions)+' · 生命耗尽' if conditions else '生命耗尽（缺少末步伤害记录）' if dead else ''
    compare(status['damage_conditions'],conditions,'damage_conditions')
    compare(status['death_detail'],detail,'death_detail')
    if dead:
        if state!='dead' or status['stop_reason']!='health_depleted' or status['model_error']!='':raise ValueError('Client death flags disagree')
    elif state not in ('running','paused','model_error'):
        raise ValueError('Unexpected live runtime state')
    elif status['stop_reason']!={'running':'','paused':'user_pause','model_error':'model_error'}[state]:
        raise ValueError('Unexpected stop reason')
    elif bool(status['model_error'])!=(state=='model_error'):
        raise ValueError('Model error flag differs')
    compare(row['vitals'],{k:after[k] for k in VITALS},'client vitals')


def audit(clients, servers, context, env, actor, verification_sha):
    validate_context(context)
    if 'probe' in context['session_id'].lower():raise ValueError('Probe journey is not a live survival journey')
    clients=matching(clients,context,True);servers=matching(servers,context)
    if not clients and not servers:raise ValueError('No evidence for requested journey')
    previous=None;last_tick=-1
    for server in servers:
        d=server['decision'];obs=server['observation']
        validate_log_identity(server,actor,verification_sha)
        if d['agent_context']!=context or d['rng_seed']!=str(live_seed(context)) or server['request_sha256']!=observation_sha(obs):
            raise ValueError('Server context/seed/request identity mismatch')
        if obs['profile']!='open3' or obs['world_version']!=5 or obs['tick']<=last_tick:
            raise ValueError('Server world or tick order mismatch/duplicate')
        for key in ('rng_before_sha256','rng_after_sha256'):
            if not re.fullmatch('[a-f0-9]{64}',d[key]):raise ValueError('Malformed RNG hash')
        state=base64.b64decode(server['generator_state_b64'],validate=True)
        if not state or hashlib.sha256(state).hexdigest()!=d['rng_after_sha256']:
            raise ValueError('Saved generator bytes/hash mismatch')
        if previous is not None and previous!=d['rng_before_sha256']:raise ValueError('Broken RNG hash chain')
        previous=d['rng_after_sha256'];last_tick=obs['tick']
    reset=env.reset(context['world_seed'],'open3');obs=reset['observation']
    validate_observation_identity(obs)
    if obs['tick']!=0:raise ValueError('Replay must begin at tick zero')
    after=dict({k:obs['player'][k] for k in VITALS},tick=0,dead=False);trace=[];count=0;events=0
    for client in clients:
        # Runtime events omit profile, but retain the actual world/rules identity.
        validate_observation_identity({**client,'profile':client.get('profile','open3')})
        if 'event' in client:
            if client['event'] not in ('runtime_status','model_error'):raise ValueError('Unknown client event')
            compare(client['tick'],obs['tick'],'event tick')
            runtime_check(client,obs,after,trace)
            if client['event']=='model_error' and (client['source']!='none' or client['runtime_status']['state']!='model_error'):
                raise ValueError('Invalid model_error event flags')
            events+=1;continue
        if after['dead'] or count>=len(servers):raise ValueError('Missing server decision or client action after death')
        if client['source']!='model' or client['profile']!='open3' or client['world_version']!=5:
            raise ValueError('Manual/rule action or changed world')
        server=servers[count];d=server['decision']
        compare(obs,server['observation'],'full server observation')
        compare(client['tick_before'],obs['tick'],'tick_before')
        compare(client['action'],d['action'],'action')
        for key in CLIENT_FIELDS:
            compare(client['model_decision'][key],d[key],'client identity.'+key)
        # Future client logs may retain more response fields: verify every one.
        for key,value in client['model_decision'].items():
            if key not in d:raise ValueError('Unknown client response field')
            if key in ('verb_probabilities','target_probabilities'):
                if len(value)!=len(d[key]) or any(struct.pack('!f',a)!=struct.pack('!f',b) for a,b in zip(value,d[key])):
                    raise ValueError('Client probability FP32 bits differ')
            else:compare(value,d[key],'client response.'+key)
        reply=env.step(client['action']);obs=reply['observation'];after=reply['transition']['after'];trace=reply['transition']['trace']
        validate_observation_identity(obs)
        compare(client['tick_after'],after['tick'],'tick_after')
        compare(client['health'],after['health'],'health')
        compare(client['day'],obs['day'],'day')
        compare(reply['terminated'],after['dead'],'death')
        runtime_check(client,obs,after,trace)
        count+=1
    pending=servers[count:]
    if len(pending)>1 or (pending and after['dead']):raise ValueError('Extra server records beyond one pending decision')
    if pending:compare(pending[0]['observation'],obs,'pending observation at final executed tick')
    return dict(passed=True,agent_context=context,executed_decisions=count,verified_ticks=after['tick'],runtime_events=events,
        complete_episode=after['dead'],status='natural_death' if after['dead'] else 'verified_prefix',pending_decisions=len(pending),
        pending_ticks=[r['observation']['tick'] for r in pending],pending_counts_as_survival=False,
        pending_acknowledgement='server durable commit only; client receipt/execution not established',
        client_identity_fields=list(CLIENT_FIELDS),server_release_encoding_validated=True,
        rng_hash_chain_verified=True,initial_rng_state_reproduced=False,cuda_sampling_reproduced=False,logits_recomputed=False,
        scope='Saved client execution + complete server observations + balanced letter identities and stored RNG hash chain; no new sampling, not unlimited survival',
        **balance_identity())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('client','client-sha256','server','server-sha256','verification','godot','session-id','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--episode',type=int,required=True);parser.add_argument('--world-seed',type=int,required=True)
    parser.add_argument('--server-original',help='Full immutable source if --server is a separately sealed exact byte prefix')
    parser.add_argument('--server-original-sha256')
    parser.add_argument('--project',default=str(ROOT));args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    verification_sha=sha(args.verification)
    report,_=preflight(args.verification,args.project)
    if any(sha(ROOT/path)!=digest for path,digest in report['source_sha256'].items()):
        raise ValueError('Imported local frozen source differs from accepted release')
    clients=snapshot(args.client,args.client_sha256);servers=snapshot(args.server,args.server_sha256)
    context=dict(session_id=args.session_id,episode=args.episode,world_seed=args.world_seed)
    if bool(args.server_original)!=bool(args.server_original_sha256):raise ValueError('Original snapshot path and SHA must be paired')
    original=args.server_original or args.server;original_sha=args.server_original_sha256 or args.server_sha256
    original_rows=snapshot(original,original_sha)
    original_bytes=Path(original).read_bytes();prefix_bytes=Path(args.server).read_bytes()
    if not original_bytes.startswith(prefix_bytes):raise ValueError('Server audit snapshot is not an exact byte prefix of original')
    excluded=original_rows[len(servers):]
    frontier=dict(original_sha256=original_sha,original_bytes=len(original_bytes),prefix_sha256=args.server_sha256,
        prefix_bytes=len(prefix_bytes),excluded_tail_records=len(excluded),
        excluded_selected_context_records=len(matching(excluded,context)))
    with GodotEnv(args.godot,args.project) as env:
        result=audit(clients,servers,context,env,report['selected_actor_sha256'],verification_sha)
    if sha(args.client)!=args.client_sha256 or sha(args.server)!=args.server_sha256 or sha(original)!=original_sha or sha(args.verification)!=verification_sha:
        raise ValueError('Snapshot changed during audit')
    result.update(client_sha256=args.client_sha256,server_sha256=args.server_sha256,verification_sha256=verification_sha,
        actor_sha256=report['selected_actor_sha256'],tool_sha256=sha(__file__),source_sha256=report['source_sha256'],
        snapshot_frontier=frontier,new_model_decisions=0,http_requests=0)
    write_new(args.output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
