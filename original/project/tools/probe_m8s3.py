"""Bounded real-actor inference/cache audit. Never creates an optimizer or checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from agent.m8s3_contract import encode, render_scene, STATE_HEADER, FEATURE_NAMES, SurvivalReward
from agent.m8s3_env import GodotEnv
from agent.m8s3_policy import BatchedSemanticPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay")
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--prefix-layout", choices=("system_only", "system_scene"), default="system_only")
    args = parser.parse_args()
    if not 1 <= args.decisions <= 32:
        raise ValueError("Probe budget is 1..32 decisions")
    warnings.filterwarnings("ignore", message="Input tensor shape suggests potential format mismatch:.*")
    torch.set_num_threads(8)
    policy = BatchedSemanticPolicy(args.model, checkpoint=args.checkpoint,
                                  parent_checkpoint=args.parent_checkpoint, reset_critic=True, seed=20261108,
                                  prefix_layout=args.prefix_layout)
    initial_digest = policy.digest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def write(value):
        output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def joint(rows, reference=False):
        with policy._scope(rows, reference):
            v = policy.verb_distribution(rows, reference)
            pieces = []
            for i, row in enumerate(rows):
                parts = []
                for verb, branch in enumerate(row["branches"]):
                    if branch:
                        t = policy.target_distribution([row], [verb], reference)
                        parts.append(v.probs[i, verb] * t.probs[0, :len(branch)])
                pieces.append(torch.cat(parts))
            return pieces

    def error(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    if args.replay:
        original = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        if original["metadata"]["inference_layout"] != "semantic_v3_" + args.prefix_layout:
            raise ValueError("Independent replay must use the recorded prefix layout")
        deviations = []
        with torch.no_grad():
            for record in original["rollout"]:
                row = policy.prepare([record["observation"]])[0]
                verb = next(i for i, branch in enumerate(row["branches"]) if record["action"] in branch)
                target = row["branches"][verb].index(record["action"])
                vd, td = policy.distributions([row], [verb])
                logp = float(vd.logits[0, verb] + td.logits[0, target])
                deviations.append(abs(logp - record["log_probability"]))
        result = dict(independent_process=True, actions_checked=len(deviations), logp_error_max=max(deviations),
                      actor_unchanged=policy.digest() == initial_digest, optimizer_steps=0, checkpoint_writes=0,
                      formal_training_started=False, metadata=policy.metadata())
        write(result)
        print(json.dumps({k: v for k, v in result.items() if k != "metadata"}), flush=True)
        assert result["logp_error_max"] <= 1e-6 and result["actor_unchanged"]
        return

    initial = []
    for seed in range(883810, 883814):
        with GodotEnv(args.godot, args.project) as env:
            initial.append(env.reset(seed)["observation"])
    rows = policy.prepare(initial)
    with torch.no_grad():
        warm = joint(rows)
        again = joint(rows)
        reverse = joint(rows[::-1])[::-1]
        policy.clear_prefix_cache()
        cold = joint(rows)
        singles = [joint([row])[0] for row in rows]
        reference = joint(rows, True)
    grad_rows = joint(rows)
    grad_error = error(again, [x.detach() for x in grad_rows])
    loss = -policy.target_distribution(rows, [1] * len(rows)).logits[:, 0].mean()
    loss.backward()
    gradients = [p.grad for p in policy.model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = float(torch.sqrt(sum(g.float().square().sum() for g in gradients)))
    finite_gradients = bool(gradients) and all(bool(g.isfinite().all()) for g in gradients)
    policy.model.zero_grad(set_to_none=True)
    del loss, grad_rows, gradients
    with torch.no_grad():
        # Tensor-version bump without changing any value tests update invalidation,
        # not optimizer or simulated training. Digest must remain identical.
        parameter = next(p for p in policy.model.parameters() if p.requires_grad)
        before_revision = parameter._version
        parameter.add_(0)
        after_version_bump = joint(rows)
        assert parameter._version > before_revision
    probability = dict(cold_error=error(again, cold), repeat_error=error(warm, again),
                       reorder_error=error(again, reverse), single_error=error(again, singles),
                       grad_mode_error=grad_error, parameter_version_refresh_error=error(again, after_version_bump),
                       normalized_max_error=max(abs(float(x.sum()) - 1) for x in again),
                       reference_normalized_max_error=max(abs(float(x.sum()) - 1) for x in reference),
                       finite_gradients=finite_gradients, gradient_norm=grad_norm)
    assert max(probability[k] for k in ("cold_error", "repeat_error", "reorder_error", "single_error", "grad_mode_error", "parameter_version_refresh_error")) <= 1e-6, probability
    assert finite_gradients and grad_norm > 0

    groups = {}
    for mode in ("stationary", "moving"):
        with GodotEnv(args.godot, args.project) as env:
            observation = env.reset(883810)["observation"]
            observations = [observation]
            for _ in range(3):
                action = {"verb": "wait"} if mode == "stationary" else next(a for a in observation["allowed_actions"] if a["verb"] == "move")
                observation = env.step(action)["observation"]
                observations.append(observation)
        groups[mode] = observations

    def common(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

    text_audit = {}
    timings = {}
    for name, observations in groups.items():
        prepared = policy.prepare(observations)
        before = []
        for row in prepared:
            messages = deepcopy(row["messages"])
            scene = render_scene(row["semantic_state"])
            dynamic = messages[1]["content"][len(scene):]
            messages[1]["content"] = STATE_HEADER + dynamic + "\n" + scene[len(STATE_HEADER):]
            before.append(policy._tokens(messages))
        text_audit[name] = dict(
            input_lengths=[len(r["input_ids"]) for r in prepared],
            fixed_system_prefix_tokens=len(policy.system_ids),
            scene_prefix_lengths=[len(r["scene_prefix_ids"]) for r in prepared],
            reordered_common_prefix_tokens=[common(a["input_ids"], b["input_ids"]) for a, b in zip(prepared, prepared[1:])],
            dynamic_first_common_prefix_tokens=[common(a, b) for a, b in zip(before, before[1:])],
            scene_identical_pairs=sum(a["scene_prefix_ids"] == b["scene_prefix_ids"] for a, b in zip(prepared, prepared[1:])))
        timings[name] = {}
        for layout in ("system_only", "system_scene"):
            policy.prefix_layout = layout
            policy.clear_prefix_cache()
            elapsed = []
            with torch.no_grad():
                for row in prepared:
                    policy.verb_distribution([row])
                counters = dict(policy.cache_stats)
                for _ in range(3):
                    policy.clear_prefix_cache()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for row in prepared:
                        policy.verb_distribution([row])
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - start)
            timings[name][layout] = dict(seconds=elapsed, median_seconds=statistics.median(elapsed),
                                        counters={k: v - counters[k] for k, v in policy.cache_stats.items()})
    policy.prefix_layout = args.prefix_layout
    policy.clear_prefix_cache()
    navigation = []
    with torch.no_grad():
        for seed, observation, row in zip(range(883810, 883814), initial, rows):
            with GodotEnv(args.godot, args.project) as env:
                env.reset(seed)
                t = policy.target_distribution([row], [1])
                index = int(t.probs[0].argmax())
                action = row["branches"][1][index]
                response = env.step(action)
                transition = response["transition"]
                a, b = transition["before"]["player"], transition["after"]["player"]
                assert abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]) == 1 and transition["elapsed_ticks"] == 1
                navigation.append(dict(source="forced_navigation_verb_model_selects_target", state=row["messages"],
                                       semantic_target=row["target_texts"][1][index], action=action, transition=transition,
                                       reward=SurvivalReward(observation).score(transition, response["observation"])))
    rollout = []
    with GodotEnv(args.godot, args.project) as env:
        observation = env.reset(883810)["observation"]
        reward = SurvivalReward(observation)
        torch.manual_seed(20261108)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.decisions):
            row = policy.prepare([observation])[0]
            sampled = policy.act([row])
            action = sampled["actions"][0]
            response = env.step(action)
            rollout.append(dict(step=step, observation=observation, state=row["messages"], action=action,
                                semantic_target=row["target_texts"][sampled["verbs"][0]][sampled["targets"][0]],
                                log_probability=sampled["logp"][0], transition=response["transition"],
                                reward=reward.score(response["transition"], response["observation"])))
            observation = response["observation"]
            if response["terminated"]:
                break
        torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - start
    policy.assert_reference_frozen()
    result = dict(probability=probability, text_audit=text_audit, fixed_sequence_forward_timings=timings,
                  navigation_probe=navigation, rollout=rollout, rollout_seconds=rollout_seconds,
                  rollout_decisions=len(rollout), actor_unchanged=policy.digest() == initial_digest,
                  critic_feature_names=list(FEATURE_NAMES), old_critic_loaded=False, optimizer_steps=0,
                  checkpoint_writes=0, formal_training_started=False, metadata=policy.metadata(), counters=policy.cache_stats)
    write(result)
    assert result["actor_unchanged"]
    print(json.dumps({k: v for k, v in result.items() if k not in ("navigation_probe", "rollout", "metadata", "critic_feature_names")}), flush=True)


if __name__ == "__main__":
    main()
