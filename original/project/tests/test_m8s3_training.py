"""Acceptance tests for the new-world training lifecycle."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class TrainingContracts(unittest.TestCase):
    def test_environment_cohorts_cover_pool_and_preserve_paused_world_progress(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'training_slots'))
        progress = [0] * 32
        coverage = set()
        for update in range(1, 10):
            active = api.training_slots(update, 4, 32)
            self.assertEqual(len(set(active)), 4)
            if update <= 8:
                self.assertFalse(coverage.intersection(active))
                coverage.update(active)
            for slot in active:
                progress[slot] += 128
        self.assertEqual(coverage, set(range(32)))
        self.assertEqual(sorted(progress), [128] * 28 + [256] * 4)
        self.assertEqual(api.training_slots(9, 4, 32), api.training_slots(1, 4, 32))
        for invalid in (0, 3, 30, 132):
            with self.assertRaises(ValueError):
                api.training_slots(1, 4, invalid)

    def test_warm_start_registration_rejects_mutated_checkpoint(self):
        from tools import register_m8s3_training as registration
        api = self.api()
        self.assertTrue(hasattr(registration, 'warm_checkpoint_inputs'))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output/runtime_tests') as tmp:
            root = Path(tmp)
            (root/'critic.pt').write_bytes(b'compatible96')
            api.write(root/'manifest.json', {'files': {'critic.pt': api.sha(root/'critic.pt')}})
            inputs = registration.warm_checkpoint_inputs(root)
            self.assertEqual(len(inputs), 2)
            api.verify_files(root, {'source_hashes': {}, 'input_files': inputs})
            (root/'critic.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                registration.warm_checkpoint_inputs(root)

    def test_formal_snapshot_includes_legacy_evaluator_identity_dependencies(self):
        from tools import register_m8s3_training as registration
        from agent.m8x_identity import CORE_FILES
        self.assertTrue(hasattr(registration, 'registered_sources'))
        root = Path(__file__).resolve().parents[1]
        inventory = registration.registered_sources(root, {}, formal=True)
        self.assertTrue(set(CORE_FILES).issubset(inventory))
        self.assertIn('docs/M8_RL_DESIGN.md', inventory)
        with tempfile.TemporaryDirectory(dir=root / 'output/runtime_tests') as tmp:
            # A missing historical source dependency must fail before GPU work.
            with self.assertRaises(FileNotFoundError):
                registration.registered_sources(Path(tmp), {}, formal=True)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Runtime supplies RNG replay')
    def test_rng_audit_rejects_broken_chain_even_when_actions_match(self):
        import torch
        from tools import audit_m8s3_training as audit
        self.assertTrue(hasattr(audit, 'verify_generator_draws'))
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().numpy().tobytes().hex()
        oracle = torch.Generator().manual_seed(42)
        torch.multinomial(torch.tensor([1., 0.]), 1, generator=oracle)
        torch.multinomial(torch.tensor([0., 1.]), 1, generator=oracle)
        after = oracle.get_state().numpy().tobytes().hex()
        row = dict(rng_before_hex=before, rng_after_hex=after, verb_probabilities=[1., 0.],
                   target_probabilities=[0., 1.], verb_index=0, target_index=1)
        audit.verify_generator_draws(generator, row, 'cpu')
        with self.assertRaisesRegex(ValueError, 'continuity'):
            audit.verify_generator_draws(generator, row, 'cpu')

    def test_adaptive_stop_requires_two_perfect_complete_screens(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'should_stop_training'))
        rows = [dict(seed=i, replicate=0, survival_ticks=1800, terminal=False,
                     alive_at_horizon=True) for i in range(16)]
        config = dict(stop_after_consecutive_perfect=2, minimum_updates=64)
        self.assertFalse(api.should_stop_training([(32, {'episodes': rows})], rows, config))
        self.assertTrue(api.should_stop_training([(32, {'episodes': rows}), (64, {'episodes': rows})], rows, config))
        failed = [dict(x) for x in rows]
        failed[-1].update(terminal=True, alive_at_horizon=False, survival_ticks=1799)
        self.assertFalse(api.should_stop_training([(32, {'episodes': failed}), (64, {'episodes': rows})], rows, config))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Runtime supplies differentiable distributions')
    def test_exploration_covers_rare_legal_verbs_without_illegal_actions(self):
        import torch
        import agent.m8s3_run as run
        self.assertTrue(hasattr(run, 'exploration_distribution'))
        logits = torch.tensor([[8., -8., -1e9]], requires_grad=True)
        base = torch.distributions.Categorical(logits=logits)
        mask = torch.tensor([[True, True, False]])
        mixed = run.exploration_distribution(base, mask, .1)
        self.assertAlmostEqual(float(mixed.probs.sum()), 1.)
        self.assertGreaterEqual(float(mixed.probs[0, 1]), .05)
        self.assertEqual(float(mixed.probs[0, 2]), 0.)
        mixed.log_prob(torch.tensor([1])).sum().backward()
        self.assertGreater(float(logits.grad[0, 1]), 0.)
        self.assertIs(run.exploration_distribution(base, mask, 0), base)
        with self.assertRaises(ValueError):
            run.exploration_distribution(base, mask, 1.1)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'GPU runtime supplies torch serialization')
    def test_registered_checkpoint_saves_even_when_diagnostic_policy_forbids_save(self):
        import torch
        from agent.m8s3_run import save_checkpoint
        class Adapter:
            def save_pretrained(self, path, **kwargs):
                Path(path).mkdir()
                (Path(path) / 'adapter_model.safetensors').write_bytes(b'test adapter transport')
        class Policy:
            model = Adapter()
            critic = torch.nn.Linear(2, 1)
            def save(self, path):
                raise RuntimeError('Diagnostic save is forbidden')
            def metadata(self):
                return {'inference_layout': 'semantic_v3_canonical_state_buckets_b4',
                        'contract': 'open_world_semantic_survival_v3'}
            def digest(self):
                return 'actor'
        p = Policy()
        optimizer = torch.optim.AdamW(p.critic.parameters())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output/runtime_tests') as tmp:
            out = Path(tmp) / 'checkpoint_0001'
            save_checkpoint(p, out, optimizer, optimizer, 1, 'a' * 64)
            self.assertFalse(out.with_name(out.name + '.partial').exists())
            state = torch.load(out / 'critic.pt', weights_only=True)
            self.assertTrue(torch.equal(state['weight'], p.critic.weight))
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['update'], 1)
            self.assertIn('optimizers.pt', manifest['files'])

    def test_test_worlds_stay_locked_without_independent_audit(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'authorize_test'), 'Held-out test needs an evidence gate')
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output/runtime_tests') as tmp:
            root = Path(tmp)
            (root / 'selection.json').write_text(json.dumps({'eligible_for_test': True}))
            with self.assertRaises((ValueError, FileNotFoundError)):
                api.authorize_test(root, {'updates': 2, 'envs': 4, 'rollout_steps': 1,
                                         'validation_every': 1})

    def test_test_gate_recomputes_winner_and_checks_audited_bytes(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output/runtime_tests') as tmp:
            root = Path(tmp)
            rows = [dict(seed=i, replicate=0, survival_ticks=1800, terminal=False,
                         alive_at_horizon=True) for i in range(16)]
            checkpoint = root / 'checkpoint_0001'
            checkpoint.mkdir()
            api.write(root / 'training_complete.json', {'updates': 1, 'decisions': 4})
            api.write(root / 'parent_selection.json', {'episodes': rows})
            api.write(root / 'selection_0001.json', {'episodes': rows, 'actor_sha256': 'actor'})
            api.write(root / 'selection.json', {'update': 1, 'checkpoint': str(checkpoint), 'actor_sha256': 'actor'})
            api.write(checkpoint / 'independent_reload.json', {'passed': True, 'logp_error': 0, 'actor_sha256': 'actor'})
            files = {}
            for name in ('rollout_0001.jsonl.gz', 'selection_0001.jsonl.gz'):
                (root / name).write_bytes(b'audited fixture bytes')
                files[name] = {'sha256': api.sha(root / name), 'transitions': 4}
            api.write(root / 'independent_replay.json', {'passed': True, 'files': files})
            config = {'updates': 1, 'envs': 4, 'rollout_steps': 1, 'validation_every': 1}
            self.assertEqual(api.authorize_test(root, config), checkpoint)
            (root / 'rollout_0001.jsonl.gz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'trace changed'):
                api.authorize_test(root, config)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8s3_training'),
                             'New-world training contracts must exist')
        from agent import m8s3_training
        return m8s3_training

    def test_dead_at_exact_horizon_is_failure(self):
        api = self.api()
        rows = [dict(seed=i, replicate=0, survival_ticks=1800,
                     terminal=i >= 15, alive_at_horizon=i < 15) for i in range(16)]
        self.assertTrue(api.selection_eligible(rows, rows))
        rows[14].update(terminal=True, alive_at_horizon=False)
        self.assertFalse(api.selection_eligible(rows, rows))

    def test_early_death_rejects_screening(self):
        api = self.api()
        rows = [dict(seed=i, replicate=0, survival_ticks=1800,
                     terminal=False, alive_at_horizon=True) for i in range(16)]
        rows[-1].update(survival_ticks=359, terminal=True, alive_at_horizon=False)
        self.assertFalse(api.selection_eligible(rows, rows))

    def test_no_overlap_between_training_selection_and_test(self):
        api = self.api()
        api.validate_seed_split([1, 2], [3, 4], [5, 6])
        with self.assertRaises(ValueError):
            api.validate_seed_split([1, 2], [2, 4], [5, 6])
        with self.assertRaises(ValueError):
            api.validate_seed_split([1, 1], [3, 4], [5, 6])

    def test_estimate_includes_assessment_and_fixed_overhead(self):
        api = self.api()
        got = api.estimate_seconds(decisions=1000, training_seconds_per_decision=.5,
                                   evaluation_decisions=200, evaluation_seconds_per_decision=.1,
                                   fixed_seconds=30)
        self.assertEqual(got['central_seconds'], 550)
        self.assertEqual(got['planning_seconds'], 825)
        self.assertFalse(got['convergence_guaranteed'])

    def test_test_claim_cannot_be_reused(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output/runtime_tests') as tmp:
            api.claim_test(Path(tmp), 'abc')
            with self.assertRaises(FileExistsError):
                api.claim_test(Path(tmp), 'abc')

    def test_registration_detects_mutated_source(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'output/runtime_tests') as tmp:
            root = Path(tmp)
            (root / 'a.py').write_text('a = 1')
            manifest = {'source_hashes': {'a.py': api.sha(root / 'a.py')}, 'input_files': {}}
            api.verify_files(root, manifest)
            (root / 'a.py').write_text('a = 2')
            with self.assertRaises(ValueError):
                api.verify_files(root, manifest)


if __name__ == '__main__':
    unittest.main()
