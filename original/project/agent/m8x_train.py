"""Frozen exploration PPO trial with natural-death-only episode resets.

Imports GPU packages only for actual training. The pure rollout helpers below
can be checked without loading a model or opening any held-out environment.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import zipfile

from agent.m8x_contract import REWARD_CONFIG, SurvivalReward, target_messages, VERSION, balance_identity
from agent.m8x_policy_identity import PARENT_ACTOR_SHA256, TRAINING_DECISIONS, entropy_coefficient
from agent.rl_contract import gae


def validate_training_shape(c):
    expected = dict(envs=4, rollout_steps=512, updates=48, validation_every=16,
                    resume=None, gamma_tick=.9995, gae_lambda=.999,
                    training_horizon=None, horizon=1800, profile="open3",
                    temperature=1., attempt=1)
    for key, value in expected.items():
        actual = getattr(c, key, object())
        if actual != value or (type(value) is int and type(actual) is not int):
            raise ValueError("Unregistered training shape: " + key)
    return c.envs * c.rollout_steps * c.updates


def finish_trajectory(trajectory, bootstrap_value, gamma_tick, gae_lambda):
    """Use true live-tail value; never connect advantages across a natural death."""
    if gamma_tick != REWARD_CONFIG["gamma_tick"] or not trajectory:
        raise ValueError("Reward and GAE discount must agree on a nonempty rollout")
    for i, row in enumerate(trajectory):
        if row["terminal"]:
            row["next_value"] = 0.
            row["boundary"] = True
        elif i + 1 < len(trajectory):
            if row["boundary"]:
                raise ValueError("Only natural death can reset an interior training episode")
            row["next_value"] = trajectory[i+1]["value"]
        else:
            row["next_value"] = float(bootstrap_value)
    trajectory[-1]["boundary"] = True
    advantages, returns = gae(
        [r["reward"] for r in trajectory], [r["value"] for r in trajectory],
        [r["next_value"] for r in trajectory], [r["duration"] for r in trajectory],
        [r["terminal"] for r in trajectory], [r["boundary"] for r in trajectory],
        gamma_tick=gamma_tick, lambda_tick=gae_lambda)
    for row, adv, ret in zip(trajectory, advantages, returns):
        row.update(advantage=adv, **{"return": ret})
        row.pop("next_obs", None)
    return trajectory


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def train(policy, c, output):
    """Collect uninterrupted natural episodes and apply the verified PPO update."""
    import torch
    from agent.m8_train import update
    from agent.m8x_env import GodotEnv
    from agent.m8x_eval import evaluate

    budget = validate_training_shape(c)
    output = Path(output)
    initial = policy.digest()
    if initial != PARENT_ACTOR_SHA256:
        raise ValueError("Training must start from the released M6 parent")
    actor_optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=c.actor_lr, eps=1.e-5, weight_decay=0.)
    critic_optimizer = torch.optim.AdamW(policy.critic.parameters(), lr=c.critic_lr,
                                         eps=1.e-5, weight_decay=0.)
    envs = []
    cursor = 0
    states, trackers, seeds = [], [], []
    episode_actions = [Counter() for _ in range(c.envs)]
    episode_start_updates = [None for _ in range(c.envs)]
    started = time.monotonic()
    total_decisions = total_ticks = 0
    optimizer_steps_total = 0
    best = None
    best_score = -1.

    def reset(i):
        nonlocal cursor
        seed = c.train_seed_start + cursor % c.train_seed_count
        cursor += 1
        state = envs[i].reset(seed, c.profile)
        return state, SurvivalReward(state["observation"]), seed

    def check_deadline():
        if time.monotonic() - started > c.max_train_seconds:
            raise TimeoutError("Registered training time budget exhausted before completion")

    try:
        # Include partial construction/reset in cleanup if an engine fails.
        for _ in range(c.envs):
            envs.append(GodotEnv(c.godot, c.project))
        for i in range(c.envs):
            state, tracker, seed = reset(i)
            states.append(state); trackers.append(tracker); seeds.append(seed)
        for number in range(1, c.updates + 1):
            check_deadline()
            rollout_started = time.monotonic()
            trajectories = [[] for _ in envs]
            episodes = []
            parts = Counter()
            rollout_actor = policy.digest()
            for _ in range(c.rollout_steps):
                check_deadline()
                prepared = policy.prepare([s["observation"] for s in states])
                sampled = policy.act(prepared, with_reference=True)
                for i, env in enumerate(envs):
                    if episode_start_updates[i] is None:
                        # Reset can occur after the last decision of a rollout;
                        # the new episode begins when its first action is drawn.
                        episode_start_updates[i] = number
                    action = sampled["actions"][i]
                    nxt = env.step(action)
                    scored = trackers[i].score(nxt["transition"], nxt["observation"])
                    terminal = nxt["terminated"]
                    if type(terminal) is not bool:
                        raise ValueError("Engine terminal flag must be boolean")
                    row = dict(prepared=prepared[i], verb=sampled["verbs"][i],
                               target=sampled["targets"][i], value=sampled["values"][i],
                               logp=sampled["logp"][i], verb_logp=sampled["verb_logp"][i],
                               reference_verb=sampled["reference_verb"][i],
                               reference_target=sampled["reference_target"][i],
                               reward=scored["reward"], duration=scored["elapsed_ticks"],
                               terminal=terminal, boundary=terminal, next_obs=nxt["observation"])
                    trajectories[i].append(row)
                    total_decisions += 1
                    total_ticks += row["duration"]
                    parts.update(scored["components"])
                    episode_actions[i][action["verb"]] += 1
                    append(output / "transitions.jsonl", dict(
                        update=number, env=i, seed=seeds[i], observation=states[i]["observation"],
                        model_input=prepared[i]["messages"], verb_index=row["verb"],
                        target_index=row["target"], target_input=(
                            target_messages(prepared[i], row["verb"])
                            if len(prepared[i]["branches"][row["verb"]]) > 1 else None),
                        old_log_probability=row["logp"], old_verb_log_probability=row["verb_logp"],
                        action=action, transition=nxt["transition"], next_observation=nxt["observation"], reward=scored,
                        terminal=terminal, truncated=False, source="model", salt=prepared[i]["salt"],
                        contract=VERSION, actor_sha256=rollout_actor,
                        reference_actor_sha256=PARENT_ACTOR_SHA256,
                        input_tokens=len(prepared[i]["input_ids"])))
                    if terminal:
                        episode = dict(
                            update=number, decisions=total_decisions, env=i, seed=seeds[i],
                            survival_ticks=nxt["observation"]["tick"], raw_ticks=nxt["observation"]["tick"],
                            terminal=True, truncated=False, actions=dict(episode_actions[i]),
                            start_update=episode_start_updates[i], end_update=number,
                            crossed_updates=episode_start_updates[i] != number,
                            unique_positions=len(trackers[i].seen), final=nxt["transition"]["after"])
                        append(output / "episodes.jsonl", episode)
                        episodes.append(episode)
                        episode_actions[i] = Counter()
                        states[i], trackers[i], seeds[i] = reset(i)
                        episode_start_updates[i] = None
                    else:
                        states[i] = nxt
            rows = []
            for trajectory in trajectories:
                if trajectory[-1]["terminal"]:
                    bootstrap_value = 0.
                else:
                    with torch.no_grad():
                        bootstrap_value = float(policy.values(policy.prepare([trajectory[-1]["next_obs"]]))[0])
                rows.extend(finish_trajectory(trajectory, bootstrap_value, c.gamma_tick, c.gae_lambda))
            rollout_seconds = time.monotonic() - rollout_started
            update_config = SimpleNamespace(**vars(c))
            update_config.entropy_coef = entropy_coefficient(total_decisions - len(rows))
            metrics = update(policy, actor_optimizer, critic_optimizer, rows, update_config)
            metrics["entropy_coefficient"] = update_config.entropy_coef
            metrics["reference_actor_sha256"] = policy.reference_digest()
            optimizer_steps_total += metrics["optimizer_steps"]
            metrics["completed_update_rollout_decisions"] = len(rows)
            metrics["actor_sample_uses"] = metrics["optimizer_steps"] * c.minibatch
            metrics["actor_unique_coverage_measured"] = False
            check_deadline()
            checkpoint = output / f"checkpoint_{number:04d}"
            policy.save(checkpoint)
            torch.save(dict(actor=actor_optimizer.state_dict(), critic=critic_optimizer.state_dict()),
                       checkpoint / "optimizers.pt")
            report = dict(
                update=number, decisions=total_decisions, ticks=total_ticks,
                rollout_seconds=rollout_seconds,
                update_seconds=time.monotonic()-rollout_started-rollout_seconds,
                mean_reward=sum(r["reward"] for r in rows)/len(rows),
                reward_per_tick=sum(r["reward"] for r in rows)/sum(r["duration"] for r in rows),
                reward_components=dict(parts), completed_episodes=len(episodes),
                mean_completed_survival=(sum(e["survival_ticks"] for e in episodes)/len(episodes)
                                         if episodes else None),
                input_tokens_mean=sum(len(r["prepared"]["input_ids"]) for r in rows)/len(rows),
                input_tokens_max=max(len(r["prepared"]["input_ids"]) for r in rows),
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), **metrics)
            append(output / "training.jsonl", report)
            print(json.dumps(dict(event="update", **report)), flush=True)
            if number % c.validation_every == 0:
                cpu_rng = torch.random.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all()
                python_rng = random.getstate()
                try:
                    assessment = evaluate(policy, c, c.validation_seeds,
                                          output / f"validation_{number:04d}.json")
                finally:
                    torch.random.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state_all(cuda_rng)
                    random.setstate(python_rng)
                score = assessment["mean_survival_ticks"]
                append(output / "validation.jsonl", dict(
                    update=number, decisions=total_decisions, checkpoint=str(checkpoint),
                    mean_survival_ticks=score, alive_at_horizon=assessment["alive_at_horizon"],
                    survival_ticks=[e["survival_ticks"] for e in assessment["episodes"]]))
                if score > best_score:
                    best_score, best = score, str(checkpoint)
            write(output / "progress.json", dict(
                update=number, decisions=total_decisions, ticks=total_ticks,
                checkpoint=str(checkpoint), diagnostic_best_checkpoint=best,
                best_validation_survival=best_score, seconds=time.monotonic()-started))
        check_deadline()
        final = policy.digest()
        summary = dict(
            initial_actor_sha256=initial, final_actor_sha256=final, decisions=total_decisions,
            world_ticks=total_ticks, seconds=time.monotonic()-started,
            diagnostic_best_checkpoint=best, best_validation_survival=best_score,
            source_counts={"model": total_decisions, "fallback": 0, "scripted": 0},
            weight_change_verified=initial != final, optimizer_steps=optimizer_steps_total,
            completed_update_rollout_decisions=total_decisions,
            actor_sample_uses=optimizer_steps_total*c.minibatch,
            actor_unique_coverage_measured=False, **balance_identity(c.project))
        write(output / "training_summary.json", summary)
        if total_decisions != budget or not summary["weight_change_verified"]:
            raise RuntimeError("Incomplete budget or unchanged actor weights")
        return summary
    finally:
        for env in envs:
            env.close()



def main():
    import torch
    from agent.m8x_policy import Policy
    from agent.m8x_eval import evaluate
    from agent.m8x_registration import CORE_FILES, PROTOCOL, create_registration, verify_registration, validate_config
    from agent.m8x_identity import execution_identity, sha, append_ledger_event, read_ledger
    from tools.m8x_metrics import qualify_selection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    c = SimpleNamespace(**config)
    validate_config(c.project, config)
    budget = validate_training_shape(c)
    if c.mode != "train" or budget != TRAINING_DECISIONS:
        raise ValueError("Only the separately registered exploration training trial is supported")
    output = Path(c.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    random.seed(c.seed); torch.manual_seed(c.seed)
    with zipfile.ZipFile(output / "frozen_source.zip", "w", zipfile.ZIP_DEFLATED) as frozen:
        frozen.writestr("run_config.json", json.dumps(config, ensure_ascii=False, indent=2))
        archived = set()
        for name in CORE_FILES:
            frozen.write(Path(c.project) / name, name)
            archived.add(name)
        for folder in ("agent", "scripts", "godot_tests", "tests"):
            for path in sorted((Path(c.project) / folder).glob("*")):
                relative = path.relative_to(c.project).as_posix()
                if path.suffix in (".py", ".gd") and relative not in archived:
                    frozen.write(path, relative)
    registration = create_registration(c.project, config, execution_identity(config, torch),
                                       output / "frozen_source.zip")
    write(output / "registration.json", registration)
    (output / "preregistration.json").write_bytes((output / "registration.json").read_bytes())
    verify_registration(c.project, registration)
    common = dict(attempt=c.attempt, registration_sha256=sha(output / "registration.json"),
                  selected_checkpoint=None, selected_actor_sha256=None, evidence_sha256={})
    train_event = {**common, "phase": "train", "status": "started"}
    selection_event = {**common, "phase": "selection", "status": "started"}
    append_ledger_event(output / "ledger_train.jsonl", train_event)
    append_ledger_event(output / "ledger_selection.jsonl", selection_event)
    try:
        from agent.m6_policy import Policy as LegacyPolicy
        legacy = LegacyPolicy(c.model, checkpoint=c.parent_checkpoint, seed=c.seed, max_tokens=c.max_tokens)
        if legacy.digest() != PARENT_ACTOR_SHA256:
            raise ValueError("Old-world comparison actor must be the released M6 parent")
        legacy_config = SimpleNamespace(**{**vars(c), "profile": "open1"})
        evaluate(legacy, legacy_config, c.validation_seeds, output / "original_v2.json", arm="original_v2")
        del legacy; gc.collect(); torch.cuda.empty_cache()
        parent = Policy(c.model, parent_checkpoint=c.parent_checkpoint, seed=c.seed,
                        max_tokens=c.max_tokens, with_reference=False)
        evaluate(parent, c, c.validation_seeds, output / "balanced_parent.json", arm="balanced_parent")
        del parent; gc.collect(); torch.cuda.empty_cache()
        random.seed(c.seed); torch.manual_seed(c.seed)
        policy = Policy(c.model, parent_checkpoint=c.parent_checkpoint, seed=c.seed,
                        max_tokens=c.max_tokens, gradient_checkpointing=c.gradient_checkpointing)
        write(output / "model.json", policy.metadata())
        summary = train(policy, c, output)
        if summary["decisions"] != budget or summary["seconds"] > c.max_train_seconds:
            raise RuntimeError("Registered training budget was not completed within its deadline")
        policy.assert_reference_frozen()
        verify_registration(c.project, registration)
        validation_updates = tuple(range(c.validation_every, c.updates+1, c.validation_every))
        decisions_per_update = c.envs * c.rollout_steps
        reports = {n*decisions_per_update: json.loads((output/f"validation_{n:04d}.json").read_text(encoding="utf-8"))
                   for n in validation_updates}
        selection = qualify_selection({n: r["episodes"] for n, r in reports.items()},
                                      json.loads((output/"balanced_parent.json").read_text(encoding="utf-8"))["episodes"])
        selected_update = selection["selected_decisions"] // decisions_per_update
        selection.update(selected_checkpoint=str(output/f"checkpoint_{selected_update:04d}"),
                         selected_actor_sha256=reports[selection["selected_decisions"]]["actor_sha256"])
        write(output / "selection.json", selection)
        evidence = {p.name: sha(p) for p in [output/"original_v2.json", output/"balanced_parent.json",
                    *(output/f"validation_{n:04d}.json" for n in validation_updates)]}
        append_ledger_event(output / "ledger_selection.jsonl", {
            **selection_event, "status": "completed", "evidence_sha256": evidence})
        append_ledger_event(output / "ledger_train.jsonl", {
            **train_event, "status": "completed", "evidence_sha256": {
                **evidence, **{p.name: sha(p) for p in [output/"transitions.jsonl", output/"training_summary.json",
                                                      output/"selection.json"]}}})
        write(output / "complete.json", dict(
            status="completed", mode="train", protocol=PROTOCOL, decisions=budget,
            registration_sha256=common["registration_sha256"], selection_sha256=sha(output/"selection.json"),
            eligible_for_test=selection["eligible_for_test"], release_authorized=False,
            finished_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        for name, event in (("ledger_train.jsonl", train_event), ("ledger_selection.jsonl", selection_event)):
            if read_ledger(output / name)[-1]["status"] == "started":
                append_ledger_event(output / name, {**event, "status": "failed",
                                                   "error": type(error).__name__ + ": " + str(error)})
        write(output / "failure.json", dict(error=type(error).__name__ + ": " + str(error)))
        raise


if __name__ == "__main__":
    main()
