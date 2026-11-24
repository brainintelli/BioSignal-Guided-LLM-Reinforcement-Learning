"""Serve the same hierarchical language policy used in M6 training and tests."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import torch

from agent.broker import validate_action,validate_observation
from agent.m6_contract import CODES,LABELS,RULES,VERSION,target_messages
from agent.m6_policy import Policy
from agent.m6_live_session import SessionRegistry,STOCHASTIC,GREEDY,validate_release
from agent.m6_stochastic_eval import validate_probabilities,base_weight_file,BASE_WEIGHT_SHA,sha,runtime_identity


class CudaStream:
    def __init__(self,seed):
        self.generator=torch.Generator(device='cuda').manual_seed(seed)

    def snapshot(self):
        return self.generator.get_state().numpy().tobytes()

    def restore(self,state):
        self.generator.set_state(torch.tensor(list(state),dtype=torch.uint8))


@torch.no_grad()
def sample_stochastic(policy,prepared,stream):
    """Same distributions, T=1 and two CUDA multinomial draws as frozen evaluation."""
    vd=policy.verb_distribution(prepared)
    vp=vd.probs[0].tolist()
    validate_probabilities(vp,prepared[0]['verb_mask'])
    verb=int(torch.multinomial(vd.probs[0],1,generator=stream.generator).item())
    td=policy.target_distribution(prepared,[verb])
    tp=td.probs[0].tolist();n=len(prepared[0]['branches'][verb])
    validate_probabilities(tp,[i<n for i in range(len(tp))])
    target=int(torch.multinomial(td.probs[0],1,generator=stream.generator).item())
    return dict(verbs=[verb],targets=[target],actions=[prepared[0]['branches'][verb][target]],
                verb_probabilities=vp,target_probabilities=tp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--verification");p.add_argument("--port",type=int,default=18080)
    p.add_argument("--decoding",choices=['greedy','stochastic'],default='greedy')
    p.add_argument("--model-id",default="meadowmind-qwen3.5-0.8b-m6")
    p.add_argument("--log",required=True);a=p.parse_args()
    torch.set_num_threads(8)
    policy=Policy(a.model,checkpoint=a.checkpoint)
    digest=policy.digest();passed=False;result={}
    if a.verification:
        result=json.loads(Path(a.verification).read_text())
        validate_release(result,digest,a.decoding)
        for name in result['source_sha256']:
            if sha(Path(__file__).resolve().parents[1]/name)!=result['source_sha256'][name]:
                raise ValueError('Serving source differs from evaluated source: '+name)
        if a.decoding=='stochastic':
            actual_tokenizer={path.name:sha(path) for path in sorted(Path(a.model).iterdir())
                              if path.is_file() and path.suffix in ('.json','.jinja','.txt')}
            if actual_tokenizer!=result['tokenizer_files_sha256'] or runtime_identity(torch)!=result['assessment_runtime']:
                raise ValueError('Serving tokenizer/runtime differs from evaluated model')
            if (sha(base_weight_file(a.model))!=BASE_WEIGHT_SHA or result['base_weight_sha256']!=BASE_WEIGHT_SHA
                or result['assessment_identity']['candidate_actor_sha256']!=digest):
                raise ValueError('Serving base weights or assessment identity differ')
        passed=True
    log=Path(a.log);log.parent.mkdir(parents=True,exist_ok=True);lock=threading.Lock()
    decoding=STOCHASTIC if a.decoding=='stochastic' else GREEDY

    def commit(record):
        with log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),**record),ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())

    sessions=SessionRegistry(CudaStream,commit)
    if a.decoding=='stochastic' and log.exists():
        def records():
            with log.open(encoding='utf-8') as source:
                for line in source:
                    record=json.loads(line)
                    if record['decision']['actor_sha256']!=digest or record['decision']['decoding']!=decoding:
                        raise ValueError('Live log belongs to a different actor or decoding mode')
                    yield record
        sessions.restore_records(records())
    health=dict(status="ok",mode="model",model=a.model_id,broker_version="m6.0",observation_format=VERSION,
        schema_encoding_version="hierarchical_single_token_v2",checkpoint=a.checkpoint,actor_sha256=digest,
        quantization="BF16 + LoRA",policy=policy.metadata(),task_goals=False,scripted_fallback=False,
        release_stage="survival_gate_passed" if passed else "experimental",survival_gate_passed=passed,
        verification_sha256=sha(Path(a.verification)) if a.verification else None,
        protocol_deviations=result.get('protocol_deviations',[]),
        literal_preregistration_conformance=not bool(result.get('protocol_deviations',[])) if passed else None,
        decoding=decoding,temperature=1. if a.decoding=='stochastic' else None,
        transport_context_required=a.decoding=='stochastic',
        retry_policy='durable same-tick cached action; inference failure restores RNG; log failure stops decisions',
        rng_seed_scheme='sha256(m6_live_v1|session_id|episode|world_seed) first 8 bytes mod (2**63-1); context never enters actor input')

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data):
            payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)

        def do_GET(self):
            if self.path!='/health':return self.reply(404,{'error':'not_found'})
            ready=not sessions.poisoned
            self.reply(200 if ready else 503,dict(health,status='ok' if ready else 'error',decision_ready=ready))

        def do_POST(self):
            if self.path!="/decide":return self.reply(404,{"error":"not_found"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 1<=size<=131072:return self.reply(413,{"error":"body_length"})
                self.connection.settimeout(15);obs=json.loads(self.rfile.read(size));validate_observation(obs)
                if obs.get("profile")!="open1":raise ValueError("M6 validated environment is open1")
                with lock:
                    def decide(stream):
                        began=time.perf_counter();prepared=policy.prepare([obs])
                        chosen=sample_stochastic(policy,prepared,stream) if a.decoding=='stochastic' else policy.act(prepared,greedy=True)
                        verb,target=chosen['verbs'][0],chosen['targets'][0]
                        action=validate_action(chosen['actions'][0],obs)
                        result=dict(action=action,tick=obs.get("tick"),source="model",raw_valid=True,
                        raw_output=CODES[verb]+":"+LABELS[target],model=a.model_id,broker_version="m6.0",
                        observation_format=VERSION,schema_encoding_version="hierarchical_single_token_v2",quantization="BF16 + LoRA",
                        latency_ms=round((time.perf_counter()-began)*1000,2),
                        reason="PPO · 生存时间验收通过" if passed else "PPO · 生存试验中",
                        assisted_decoding=True,assistance=dict(legal_action_mask=True,policy_override=False,objective_supplied=False),
                        model_input=prepared[0]["messages"],
                        target_input=target_messages(prepared[0],verb) if len(prepared[0]["branches"][verb])>1 else None,
                        input_tokens=len(prepared[0]["input_ids"]),actor_sha256=digest,
                            prompt_sha256=hashlib.sha256(RULES.encode()).hexdigest(),decoding=decoding,
                            verb_index=verb,target_index=target,
                            verb_probabilities=chosen.get('verb_probabilities'),target_probabilities=chosen.get('target_probabilities'))
                        return result
                    if a.decoding=='stochastic':
                        context=dict(session_id=self.headers.get('X-Agent-Session'),
                                     episode=int(self.headers.get('X-Agent-Episode','')),
                                     world_seed=int(self.headers.get('X-World-Seed','')))
                        result=sessions.decide(context,obs,decide)
                    else:
                        result=decide(None);commit(dict(observation=obs,decision=result))
                self.reply(200,result)
            except (ValueError,KeyError,TypeError) as error:
                self.reply(400,dict(error=type(error).__name__,detail=str(error)[:200]))
            except Exception as error:
                self.reply(503,dict(error=type(error).__name__,detail=str(error)[:200]))

        def log_message(self,*_):pass

    server=ThreadingHTTPServer(("127.0.0.1",a.port),Handler)
    print(json.dumps(dict(event="ready",**health)),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=="__main__":main()
