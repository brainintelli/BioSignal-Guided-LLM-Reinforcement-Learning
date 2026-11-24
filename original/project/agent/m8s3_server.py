"""Bounded open4 HTTP validation server, gated by complete offline evidence."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

from agent.broker import validate_action
from agent.m6_live_session import SessionRegistry, STOCHASTIC
from agent.m6_stochastic_eval import validate_probabilities
from agent.m8s3_contract import CODES, VERSION, balance_identity, encode, validate_observation_identity
from agent.m8s3_training import sha, verify_files, write
from agent.m8s3_online_gate import verify_offline_gate
from agent.m8x_server import verify_saved_cuda_draws


def decision_identity(actor, registration, epsilon):
    return dict(actor_sha256=actor,registration_sha256=registration,broker_version='m8s3.0',
        observation_format=VERSION,inference_layout='semantic_v3_canonical_state_buckets_b4',
        physical_forward_shape={'batch':4,'widths':[896,1280,1537]},
        decoding=STOCHASTIC,temperature=1,verb_exploration_epsilon=epsilon,**balance_identity())


def semantic_output(row,verb,target):
    return CODES[verb]+':'+row['target_texts'][verb][target]


def validate_observation(obs):
    # The historical broker requires energy; open4 deliberately has no energy.
    if not isinstance(obs,dict) or obs.get('schema_version')!=1:
        raise ValueError('Expected observation schema_version1')
    validate_observation_identity(obs)
    actions=obs.get('allowed_actions')
    if not isinstance(actions,list) or not 1<=len(actions)<=64 or {'verb':'wait'} not in actions:
        raise ValueError('Complete nonempty legal action list required')
    for action in actions:
        validate_action(action,obs)
    encode(obs)  # Positive factual whitelist validates the three finite reserves.
    return obs


def validate_record(record,identity):
    obs,d=record['observation'],record['decision']
    validate_observation(obs)
    validate_observation_identity(obs)
    if any(d.get(k)!=v for k,v in identity.items()):
        raise ValueError('Durable decision actor/world/policy identity differs')
    row=encode(obs)
    v,t=d['verb_index'],d['target_index']
    if (type(v) is not int or not 0<=v<len(CODES) or type(t) is not int
            or not 0<=t<len(row['branches'][v])):
        raise ValueError('Invalid semantic branch index')
    if (d.get('source')!='model' or d.get('raw_valid') is not True
            or d['tick']!=obs['tick'] or d['model_input']!=row['messages']
            or d['action']!=row['branches'][v][t] or d['raw_output']!=semantic_output(row,v,t)):
        raise ValueError('Durable decision differs from observation and semantic action')
    validate_action(d['action'],obs)
    if len(d['verb_probabilities'])!=len(CODES) or len(d['target_probabilities'])!=64:
        raise ValueError('Probability layout differs')
    validate_probabilities(d['verb_probabilities'],row['verb_mask'])
    validate_probabilities(d['target_probabilities'],[i<len(row['branches'][v]) for i in range(64)])
    if d['verb_probabilities'][v]<=0 or d['target_probabilities'][t]<=0:
        raise ValueError('Recorded action has zero probability')


def sample_stochastic(policy,prepared,stream,torch):
    # Same graph-local state roots and two categorical draws as the evaluator.
    with policy._scope(prepared,False):
        vd=policy.verb_distribution(prepared)
        vp=vd.probs[0].tolist()
        validate_probabilities(vp,prepared[0]['verb_mask'])
        verb=int(torch.multinomial(vd.probs[0],1,generator=stream.generator).item())
        td=policy.target_distribution(prepared,[verb])
        tp=td.probs[0].tolist()
        validate_probabilities(tp,[i<len(prepared[0]['branches'][verb]) for i in range(len(tp))])
        target=int(torch.multinomial(td.probs[0],1,generator=stream.generator).item())
    return dict(verb=verb,target=target,action=prepared[0]['branches'][verb][target],
                verb_probabilities=vp,target_probabilities=tp)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--log',required=True)
    ap.add_argument('--port',type=int,default=18081)
    ap.add_argument('--max-seconds',type=int,default=1800)
    args=ap.parse_args()
    if not 1<=args.max_seconds<=7200 or not 1024<=args.port<=65535:
        raise ValueError('A bounded validation duration and unprivileged port are required')
    gate=verify_offline_gate(args.run)
    reg=json.loads((Path(args.run)/'registration.json').read_text())
    # The service has new adapter files, but all inherited policy dependencies
    # must match the frozen trainer, even when installed in a separate project.
    verify_files(Path(__file__).resolve().parents[1],reg)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if active:
        raise RuntimeError('Wait for the owned training worker to release GPU: '+active)
    from tools.benchmark_m8s3 import setup
    setup()
    import torch
    from agent.m8s3_run import make_policy,load_checkpoint
    c=SimpleNamespace(**gate['config'])
    policy=make_policy(c,reference=False)
    load_checkpoint(policy,gate['checkpoint'])
    if policy.digest()!=gate['actor_sha256']:
        raise ValueError('Loaded actor differs from accepted checkpoint')
    identity=decision_identity(policy.digest(),gate['registration_sha256'],c.verb_exploration_epsilon)
    log=Path(args.log)
    log.parent.mkdir(parents=True,exist_ok=True)
    write(log.with_suffix('.offline_gate.json'),gate)
    lock=threading.Lock()

    class CudaStream:
        def __init__(self,seed): self.generator=torch.Generator(device='cuda').manual_seed(seed)
        def snapshot(self): return self.generator.get_state().cpu().numpy().tobytes()
        def restore(self,state): self.generator.set_state(torch.tensor(list(state),dtype=torch.uint8))

    def commit(record):
        with log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),**record),ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    sessions=SessionRegistry(CudaStream,commit)
    if log.exists():
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'): raise ValueError('Incomplete durable log row')
                record=json.loads(line)
                validate_record(record,identity)
                entry=sessions._entry(record['agent_context'])
                verify_saved_cuda_draws(record,entry['generator'],torch)
                sessions.restore_records([record])
    health=dict(identity,status='ok',mode='model',model='meadowmind-m8s3-0.8b',
        process_pid=os.getpid(),process_started_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=gate['checkpoint'],validation_authorized=True,release_authorized=False,
        numerical_gate_passed=True,milestone_achieved=False,release_stage='offline_pass_online_validation',
        scripted_fallback=False,task_goals=False,policy_override=False,
        source_sha256={name:sha(Path(__file__).parent/name) for name in ('m8s3_server.py','m8s3_online_gate.py')},
        max_session_seconds=args.max_seconds,transport_context_required=True)

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data):
            payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path!='/health': return self.reply(404,{'error':'not_found'})
            self.reply(503 if sessions.poisoned else 200,dict(health,decision_ready=not sessions.poisoned))
        def do_POST(self):
            if self.path!='/decide': return self.reply(404,{'error':'not_found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 1<=length<=131072: return self.reply(413,{'error':'body_length'})
                self.connection.settimeout(15)
                obs=json.loads(self.rfile.read(length));validate_observation(obs);validate_observation_identity(obs)
                context=dict(session_id=self.headers.get('X-Agent-Session'),
                    episode=int(self.headers.get('X-Agent-Episode','')),world_seed=int(self.headers.get('X-World-Seed','')))
                with lock:
                    def decide(stream):
                        started=time.perf_counter()
                        with torch.no_grad():
                            prepared=policy.prepare([obs]);chosen=sample_stochastic(policy,prepared,stream,torch)
                        v,t=chosen['verb'],chosen['target']
                        return dict(identity,action=validate_action(chosen['action'],obs),tick=obs['tick'],source='model',raw_valid=True,
                            raw_output=semantic_output(prepared[0],v,t),model=health['model'],
                            reason='新规则模型在线验收',latency_ms=(time.perf_counter()-started)*1000,
                            model_input=prepared[0]['messages'],input_tokens=len(prepared[0]['input_ids']),
                            verb_index=v,target_index=t,verb_probabilities=chosen['verb_probabilities'],target_probabilities=chosen['target_probabilities'])
                    response=sessions.decide(context,obs,decide)
                self.reply(200,response)
            except (ValueError,KeyError,TypeError) as exc:
                self.reply(400,dict(error=type(exc).__name__,detail=str(exc)[:200]))
            except Exception as exc:
                self.reply(503,dict(error=type(exc).__name__,detail=str(exc)[:200]))
        def log_message(self,*_): pass

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    def shutdown(*_): threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    timer=threading.Timer(args.max_seconds,shutdown);timer.daemon=True;timer.start()
    print(json.dumps(dict(event='ready',**health),ensure_ascii=False),flush=True)
    try: server.serve_forever(poll_interval=.25)
    finally:
        timer.cancel();server.server_close()
        print(json.dumps(dict(event='validation_server_exit',actor_sha256=identity['actor_sha256'])),flush=True)


if __name__=='__main__': main()
