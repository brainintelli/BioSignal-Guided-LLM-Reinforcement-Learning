"""CPU checks for rollout boundaries and registered budget; not model results."""
import copy
from contextlib import nullcontext
import importlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ExplorationTrainingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("agent.m8x_train"))
        return importlib.import_module("agent.m8x_train")

    def test_gae_does_not_cross_natural_death_and_preserves_live_tail_bootstrap(self):
        m = self.module()
        gamma, lam = .9995, .999
        rows = [dict(reward=1., value=2., duration=2, terminal=True, boundary=True),
                dict(reward=3., value=4., duration=3, terminal=False, boundary=False),
                dict(reward=5., value=6., duration=1, terminal=False, boundary=False)]
        result = m.finish_trajectory(copy.deepcopy(rows), 8., gamma, lam)
        last_adv = 5. + gamma*8. - 6.
        middle_adv = 3. + gamma**3*6. - 4. + (gamma*lam)**3*last_adv
        self.assertAlmostEqual(result[0]["advantage"], -1.)
        self.assertEqual(result[0]["next_value"], 0.)
        self.assertAlmostEqual(result[1]["advantage"], middle_adv)
        self.assertAlmostEqual(result[2]["advantage"], last_adv)
        self.assertEqual(result[2]["next_value"], 8.)
        self.assertTrue(result[2]["boundary"])
        self.assertFalse(result[2]["terminal"])
        self.assertAlmostEqual(result[1]["return"], middle_adv + 4.)

    def test_final_death_never_bootstraps_and_wrong_gamma_is_rejected(self):
        m = self.module()
        rows = [dict(reward=-.7, value=2., duration=3, terminal=True, boundary=True)]
        result = m.finish_trajectory(copy.deepcopy(rows), 999., .9995, .999)
        self.assertEqual(result[0]["next_value"], 0.)
        self.assertAlmostEqual(result[0]["advantage"], -2.7)
        with self.assertRaises(ValueError):
            m.finish_trajectory(copy.deepcopy(rows), 999., .999, .999)

    def test_training_cannot_use_eval_horizon_or_old_budget(self):
        m = self.module()
        values = dict(envs=4, rollout_steps=512, updates=48, validation_every=16, resume=None,
                      gamma_tick=.9995, gae_lambda=.999, training_horizon=None,
                      horizon=1800, profile="open3", temperature=1., attempt=1)
        self.assertEqual(m.validate_training_shape(SimpleNamespace(**values)), 98304)
        for changed in ({"updates":32}, {"rollout_steps":256}, {"training_horizon":1800},
                        {"profile":"open1"}, {"gamma_tick":.999}, {"resume":"old"}, {"attempt":2}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                m.validate_training_shape(SimpleNamespace(**{**values, **changed}))

    def test_cpu_fixture_checks_real_loop_boundary_accounting_not_model_quality(self):
        """Only orchestration is real here; policy/GPU/environment are explicit fakes."""
        m = self.module()
        from agent.m8x_policy_identity import PARENT_ACTOR_SHA256
        from agent.m8x_contract import balance_identity
        root = Path(__file__).resolve().parents[1]
        rules_sha = balance_identity(root)["balance_sha256"]
        instances = []
        updates = []

        def state(tick, dead=False):
            return dict(tick=tick, health=0. if dead else 100., hunger=50., thirst=50.,
                        energy=50., dead=dead, player={"x":48,"y":48})

        class FakeEnv:
            def __init__(self, *_):
                self.index = len(instances); instances.append(self)
                self.resets = 0; self.closed = False
            def reset(self, seed, profile):
                self.resets += 1; self.tick = 1799 if self.index == 0 else 5399
                self.start = self.tick
                return {"observation": self.obs(), "terminated": False}
            def obs(self, dead=False):
                s = state(self.tick, dead)
                from tests.test_m8x_contract import observation
                return {**observation(self.tick), **s, "player": {**s["player"], **{k:s[k] for k in ("health","hunger","thirst","energy")}},
                        "profile":"open3", "world_version":5, "environment_version":"m8_exploration_v1",
                        "balance_sha256":rules_sha}
            def step(self, action):
                before = state(self.tick); self.tick += 1
                dead = self.index == 0 and self.tick - self.start == 3
                after = state(self.tick, dead)
                return dict(observation=self.obs(dead), terminated=dead,
                            transition=dict(before=before, after=after, trace=[after], elapsed_ticks=1))
            def close(self): self.closed = True

        class FakePolicy:
            def __init__(self):
                self.changed = 0
                self.model = self.critic = SimpleNamespace(parameters=lambda:[SimpleNamespace(requires_grad=True)])
            def digest(self): return PARENT_ACTOR_SHA256 if not self.changed else str(self.changed)*64
            def reference_digest(self): return PARENT_ACTOR_SHA256
            def prepare(self, observations):
                return [dict(messages=[{"role":"system","content":"CPU ORCHESTRATION FIXTURE"}],
                             branches=[[]]*7+[[{"verb":"wait"}]], salt="m6", input_ids=[1]) for _ in observations]
            def act(self, prepared, **_):
                n = len(prepared)
                return dict(actions=[{"verb":"wait"}]*n, verbs=[7]*n, targets=[0]*n,
                            values=[2.]*n, logp=[0.]*n, verb_logp=[0.]*n,
                            reference_verb=[[0.]*7+[1.]]*n, reference_target=[[1.]]*n)
            def values(self, prepared): return [2.]*len(prepared)
            def save(self, directory):
                directory.mkdir()
                (directory/"CPU_FIXTURE_ONLY").write_text("Not a model checkpoint")

        def update(policy, _ao, _co, rows, config):
            updates.append(copy.deepcopy(rows))
            policy.changed += 1
            return dict(optimizer_steps=1, kl_early_stop=False)

        def evaluate(*_args, **_kwargs):
            random.random()  # The real loop must restore even this stream.
            cpu_rng_state["value"] = b"changed-by-fixture-evaluation"
            cuda_rng_state["value"] = [b"changed-by-fixture-evaluation"]
            return dict(mean_survival_ticks=100., alive_at_horizon=0, episodes=[dict(survival_ticks=100)])

        cpu_rng_state = {"value": b"training-cpu-state"}
        cuda_rng_state = {"value": [b"training-cuda-state"]}
        torch = SimpleNamespace(
            optim=SimpleNamespace(AdamW=lambda *a, **k:SimpleNamespace(state_dict=lambda:{"CPU_FIXTURE":True})),
            no_grad=nullcontext, save=lambda obj,path:Path(path).write_text(json.dumps(obj)),
            cuda=SimpleNamespace(max_memory_allocated=lambda:0,
                                 get_rng_state_all=lambda:list(cuda_rng_state["value"]),
                                 set_rng_state_all=lambda value:cuda_rng_state.update(value=list(value))),
            random=SimpleNamespace(get_rng_state=lambda:cpu_rng_state["value"],
                                   set_rng_state=lambda value:cpu_rng_state.update(value=value)))
        c = SimpleNamespace(envs=2, rollout_steps=3, updates=2, validation_every=1,
                            train_seed_start=80000, train_seed_count=2048, profile="open3",
                            godot="unused-fixture", project=str(root), gamma_tick=.9995,
                            gae_lambda=.999, actor_lr=2.5e-6, critic_lr=.0003,
                            minibatch=2, max_train_seconds=30, validation_seeds=[901000])
        modules = {"torch":torch, "agent.m8_train":SimpleNamespace(update=update),
                   "agent.m8x_env":SimpleNamespace(GodotEnv=FakeEnv),
                   "agent.m8x_eval":SimpleNamespace(evaluate=evaluate)}
        before_rng = random.getstate()
        temp_parent = root / ".tools/m8b_cpu_tests"
        temp_parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(temp_parent.resolve().is_relative_to(root.resolve()))
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory, patch.dict("sys.modules", modules), \
                patch.object(m, "validate_training_shape", return_value=12):
            self.assertTrue(Path(directory).resolve().is_relative_to(temp_parent.resolve()))
            summary = m.train(FakePolicy(), c, Path(directory))
            episodes = [json.loads(line) for line in (Path(directory)/"episodes.jsonl").read_text().splitlines()]
            trace = [json.loads(line) for line in (Path(directory)/"transitions.jsonl").read_text().splitlines()]
            self.assertEqual([e["start_update"] for e in episodes], [1,2])
            self.assertTrue(all(not e["crossed_updates"] for e in episodes))
            self.assertEqual(summary["decisions"],12)
            self.assertEqual(summary["actor_sample_uses"],4)
            self.assertEqual(len(trace),12)
            self.assertTrue(all(not r["truncated"] for r in trace))
            self.assertGreater(instances[1].tick,5400)
            self.assertEqual(instances[1].resets,1)
            self.assertTrue(all(env.closed for env in instances))
            self.assertTrue(all(len(rows)==6 for rows in updates))
            self.assertEqual(random.getstate(),before_rng)
            self.assertEqual(cpu_rng_state["value"],b"training-cpu-state")
            self.assertEqual(cuda_rng_state["value"],[b"training-cuda-state"])


if __name__ == "__main__":
    unittest.main()
