"""Exploration-world M8X online adapter; offline gate required before start. No training, fallback, world changes or global-RNG sampling.

Transport deliberately reuses SessionRegistry's m6_live_v1 seed recipe. This
namespace is transport compatibility, not the M8X offline assessment seed scheme.
No session identifiers or RNG seeds enter the actor observation.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

from agent.broker import validate_action, validate_observation
from agent.m6_live_session import SessionRegistry, STOCHASTIC
from agent.m6_stochastic_eval import validate_probabilities
from agent.m8x_contract import CODES, TARGET_VOCAB, RULES, target_messages, encode, balance_identity, validate_observation_identity
from agent.rl_contract import LABELS
from agent.m8x_identity import execution_identity, sha
from agent.m8x_release import CONTRACT, LAYOUT, SHAPE, verify_release


def decision_identity(actor, verification_sha):
    return dict(actor_sha256=actor, verification_sha256=verification_sha, broker_version="m8x.0",
        observation_format=CONTRACT, schema_encoding_version="hierarchical_single_token_exploration_v1",
        physical_forward_shape=list(SHAPE), inference_layout=LAYOUT, encoding_salt="m6",
        decoding=STOCHASTIC, temperature=1.0, **balance_identity())


def validate_log_identity(record, actor, verification_sha):
    expected = decision_identity(actor, verification_sha)
    if any(record.get("decision", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Live log belongs to another actor/release/contract/layout/decoding")
    try:
        obs, decision = record["observation"], record["decision"]
        row = encode(obs, salt="m6")
        verb, target = decision["verb_index"], decision["target_index"]
        if (type(verb) is not int or not 0 <= verb < len(CODES) or type(target) is not int
                or not 0 <= target < len(row["branches"][verb])):
            raise ValueError("Invalid recorded exploration branch index")
        if (decision.get("source") != "model" or decision.get("raw_valid") is not True
                or type(decision.get("tick")) is not int
                or decision.get("tick") != obs["tick"]
                or decision.get("action") != row["branches"][verb][target]
                or decision.get("model_input") != row["messages"]
                or decision.get("target_input") != (target_messages(row, verb) if len(row["branches"][verb]) > 1 else None)
                or decision.get("raw_output") != CODES[verb] + ":" + target_code(row, verb, target)):
            raise ValueError("Durable decision does not match original observation/encoding/action")
        if "prompt_sha256" in decision and decision["prompt_sha256"] != hashlib.sha256(RULES.encode()).hexdigest():
            raise ValueError("Recorded rules digest differs")
        validate_action(decision["action"], obs)
        vp, tp = decision["verb_probabilities"], decision["target_probabilities"]
        if len(vp) != len(CODES) or len(tp) != len(LABELS):
            raise ValueError("Recorded probability shape differs")
        validate_probabilities(vp, row["verb_mask"])
        validate_probabilities(tp, [i < len(row["branches"][verb]) for i in range(len(LABELS))])
        if vp[verb] <= 0 or tp[target] <= 0:
            raise ValueError("Recorded choice has zero probability")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete durable exploration record") from exc


def verify_saved_cuda_draws(record, stream, torch):
    """Reproduce saved float32 probabilities on CUDA, not model forward logits.

    Restore the temporary stream afterwards; SessionRegistry restores it once.
    CPU unit doubles verify control flow only and are not CUDA evidence.
    """
    if not str(stream.generator.device).startswith("cuda"):
        raise ValueError("Saved live sampling must be replayed on a CUDA generator")
    before = stream.snapshot()
    decision = record["decision"]
    if hashlib.sha256(before).hexdigest() != decision["rng_before_sha256"]:
        raise ValueError("Saved sampling starts from a different RNG state")
    try:
        choices = []
        for field in ("verb_probabilities", "target_probabilities"):
            probabilities = torch.tensor(decision[field], dtype=torch.float32, device=stream.generator.device)
            choices.append(int(torch.multinomial(probabilities, 1, generator=stream.generator).item()))
        after = stream.snapshot()
        if (choices != [decision["verb_index"], decision["target_index"]]
                or hashlib.sha256(after).hexdigest() != decision["rng_after_sha256"]
                or after != base64.b64decode(record["generator_state_b64"], validate=True)):
            raise ValueError("Saved CUDA choices or post-draw state do not reproduce")
    finally:
        stream.restore(before)


def target_code(row, verb, target):
    return TARGET_VOCAB[row["target_token_indices"][verb][target]]


def sample_stochastic(policy, prepared, stream, torch):
    """Exactly the frozen evaluator's two CUDA draws, including singleton targets."""
    vd = policy.verb_distribution(prepared)
    vp = vd.probs[0].tolist()
    validate_probabilities(vp, prepared[0]["verb_mask"])
    verb = int(torch.multinomial(vd.probs[0], 1, generator=stream.generator).item())
    td = policy.target_distribution(prepared, [verb])
    tp = td.probs[0].tolist()
    size = len(prepared[0]["branches"][verb])
    validate_probabilities(tp, [i < size for i in range(len(tp))])
    target = int(torch.multinomial(td.probs[0], 1, generator=stream.generator).item())
    return dict(verb=verb, target=target, action=prepared[0]["branches"][verb][target],
                verb_probabilities=vp, target_probabilities=tp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model-id", default="meadowmind-qwen3.5-0.8b-m8x")
    args = parser.parse_args()
    import torch
    from agent.m8x_policy import Policy
    torch.set_num_threads(8)
    report = json.loads(Path(args.verification).read_text(encoding="utf-8"))
    config = report["config"]
    if Path(args.model).resolve() != Path(config["model"]).resolve():
        raise ValueError("Model path differs from registration")
    policy = Policy(args.model, checkpoint=args.checkpoint, parent_checkpoint=config["parent_checkpoint"],
                    max_tokens=1536, with_reference=False)
    policy.eval()
    actor = policy.digest()
    metadata = policy.metadata()
    # This recomputes complete result, source, evidence and execution identities.
    report = verify_release(verification=args.verification, project=args.project, checkpoint=args.checkpoint,
        model=args.model, actor_sha256=actor, policy_metadata=metadata, execution_identity=execution_identity(config, torch))
    verification_sha = sha(args.verification)
    identity = decision_identity(actor, verification_sha)
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class CudaStream:
        def __init__(self, seed): self.generator = torch.Generator(device="cuda").manual_seed(seed)
        def snapshot(self): return self.generator.get_state().cpu().numpy().tobytes()
        def restore(self, state): self.generator.set_state(torch.tensor(list(state), dtype=torch.uint8))

    def commit(record):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), **record), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    sessions = SessionRegistry(CudaStream, commit)
    if log.exists():
        with log.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise ValueError("Incomplete durable M8X log row")
                record = json.loads(line)
                validate_observation(record["observation"])
                validate_log_identity(record, actor, verification_sha)
                row = encode(record["observation"], salt="m6")
                if record["decision"].get("input_tokens") != len(policy._tokens(row["messages"])):
                    raise ValueError("Saved tokenizer input length differs")
                entry = sessions._entry(record["agent_context"])
                verify_saved_cuda_draws(record, entry["generator"], torch)
                sessions.restore_records([record])
    health = dict(identity, status="ok", mode="model", model=args.model_id, checkpoint=str(Path(args.checkpoint).resolve()),
        quantization="BF16 + LoRA", policy=metadata, task_goals=False, scripted_fallback=False,
        release_authorized=True, evidence_complete=True, release_phase="test", release_stage="survival_gate_passed",
        survival_gate_passed=True, registration_sha256=report["registration_sha256"],
        source_sha256=report["source_sha256"], environment_profile="open3",
        transport_context_required=True, retry_policy="durable same-tick cached action; failure restores RNG; log failure stops decisions",
        log_restore_validation="exploration letter record + saved-probability CUDA two-draw replay; not model-logit replay",
        rng_seed_scheme="m6_live_v1 transport compatibility: sha256(m6_live_v1|session_id|episode|world_seed), first 8 bytes mod (2**63-1); not offline evaluation seed recipe")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health": return self.reply(404, {"error": "not_found"})
            ready = not sessions.poisoned
            self.reply(200 if ready else 503, dict(health, status="ok" if ready else "error", decision_ready=ready))

        def do_POST(self):
            if self.path != "/decide": return self.reply(404, {"error": "not_found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 131072: return self.reply(413, {"error": "body_length"})
                self.connection.settimeout(15)
                obs = json.loads(self.rfile.read(length))
                validate_observation(obs)
                validate_observation_identity(obs)
                context = dict(session_id=self.headers.get("X-Agent-Session"),
                    episode=int(self.headers.get("X-Agent-Episode", "")), world_seed=int(self.headers.get("X-World-Seed", "")))
                with lock:
                    def decide(stream):
                        began = time.perf_counter()
                        with torch.no_grad():
                            prepared = policy.prepare([obs], salt="m6")
                            chosen = sample_stochastic(policy, prepared, stream, torch)
                        verb, target = chosen["verb"], chosen["target"]
                        action = validate_action(chosen["action"], obs)
                        return dict(identity, action=action, tick=obs["tick"], source="model", raw_valid=True,
                            raw_output=CODES[verb] + ":" + target_code(prepared[0], verb, target),
                            model=args.model_id, quantization="BF16 + LoRA", reason="PPO · M8X 探索规则独立测试已核验",
                            latency_ms=round((time.perf_counter()-began)*1000, 2), assisted_decoding=True,
                            assistance=dict(legal_action_mask=True, policy_override=False, objective_supplied=False),
                            model_input=prepared[0]["messages"], target_input=target_messages(prepared[0], verb) if len(prepared[0]["branches"][verb]) > 1 else None,
                            input_tokens=len(prepared[0]["input_ids"]), prompt_sha256=hashlib.sha256(RULES.encode()).hexdigest(),
                            verb_index=verb, target_index=target, verb_probabilities=chosen["verb_probabilities"], target_probabilities=chosen["target_probabilities"])
                    result = sessions.decide(context, obs, decide)
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, dict(error=type(exc).__name__, detail=str(exc)[:200]))
            except Exception as exc:
                self.reply(503, dict(error=type(exc).__name__, detail=str(exc)[:200]))

        def log_message(self, *_): pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps(dict(event="ready", **health), ensure_ascii=False), flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__": main()
