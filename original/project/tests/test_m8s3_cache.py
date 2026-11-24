"""Cache lifetime/gradient controls without loading a model checkpoint."""
from collections import OrderedDict
import importlib.util
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'peft'))


@unittest.skipUnless(DEPENDENCIES, 'Torch/transformers/PEFT required')
class SemanticSceneCacheTests(unittest.TestCase):
    def setUp(self):
        import torch
        from agent import m8s3_policy as module
        self.torch, self.module = torch, module
        self.policy = module.Policy.__new__(module.Policy)
        torch.nn.Module.__init__(self.policy)
        p = self.policy
        p.test_weight = torch.nn.Parameter(torch.tensor(.125))
        p.model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(config=None))
        p.execution_cache = SimpleNamespace(revisions={False:0})
        p.prefix_layout, p.scene_cache_size = 'system_scene', 2
        p.system_ids = [10, 11]
        p._persistent_prefixes = {}
        p._persistent_scenes = {False:OrderedDict(), True:OrderedDict()}
        p._scene_revisions = {}
        p.cache_stats = dict(system_hits=0, system_misses=0, state_hits=0, state_misses=0,
                            scene_hits=0, scene_misses=0, scene_persistent_hits=0, scene_reused_tokens=0)
        p.test_calls = []

        def forward(policy, model, sequences, cache):
            self.assertEqual(len(sequences), 1)
            tokens = sequences[0]
            policy.test_calls.append(tuple(tokens))
            cache.tokens += tuple(tokens)
            cache.value = cache.value + policy.test_weight * sum(tokens)
            return cache.value.reshape(1, 1)

        p._forward_batch = MethodType(forward, p)
        self.fresh = patch.object(module, 'fresh_cache', side_effect=lambda cfg: SimpleNamespace(tokens=(), value=torch.tensor(0.)))
        self.branch = patch.object(module, 'branch_cache', side_effect=lambda c: SimpleNamespace(tokens=c.tokens, value=c.value.clone()))
        self.fresh.start(); self.branch.start()
        self.addCleanup(self.fresh.stop); self.addCleanup(self.branch.stop)

    def root(self, scene_tail=(20, 21), dynamic=30):
        p = self.policy
        scene = tuple(p.system_ids) + tuple(scene_tail)
        ids = list(scene) + [dynamic]
        p._kv_scope = {'roots':{tuple(ids):None}, 'scene_declarations':{tuple(ids):scene}}
        return p._root(ids, False), scene

    def test_only_exact_scene_prefix_reuses_and_weight_revision_invalidates(self):
        p = self.policy
        with self.torch.no_grad():
            self.root()
            self.assertEqual(len(p.test_calls), 3)
            cached = p._persistent_scenes[False][(10, 11, 20, 21)]
            self.assertEqual(cached.tokens, (10, 11, 20, 21), 'Dynamic suffix must not mutate cached scene')
            (cache, value), _ = self.root(dynamic=31)
            self.assertEqual(len(p.test_calls), 4)
            self.assertEqual(p.cache_stats['scene_persistent_hits'], 1)
            self.assertAlmostEqual(value.item(), .125 * sum((10, 11, 20, 21, 31)))
            self.root(scene_tail=(20, 22))
            self.assertEqual(len(p.test_calls), 6, 'One changed scene token must force a new scene forward')
            p.execution_cache.revisions[False] = 1
            self.root()
            self.assertEqual(len(p.test_calls), 9, 'Parameter updates invalidate both persistent levels')
            self.assertEqual(len(p._persistent_scenes[False]), 1)

    def test_grad_enabled_does_not_reuse_detached_persistent_caches(self):
        p = self.policy
        with self.torch.no_grad():
            self.root()
        for _ in range(2):
            p.test_weight.grad = None
            (cache, value), _ = self.root()
            value.sum().backward()
            self.assertEqual(p.test_weight.grad.item(), sum((10, 11, 20, 21, 30)))
        self.assertEqual(p.cache_stats['scene_persistent_hits'], 0)
        self.assertEqual(len(p.test_calls), 9, 'Each backward uses its own complete prefix graph')

    def test_scene_lru_is_bounded_and_explicit_clear_removes_all_entries(self):
        p = self.policy
        with self.torch.no_grad():
            for i in range(3):
                self.root(scene_tail=(20, 21+i))
        self.assertEqual(len(p._persistent_scenes[False]), 2)
        self.assertNotIn((10, 11, 20, 21), p._persistent_scenes[False])
        p.clear_prefix_cache()
        self.assertFalse(p._persistent_prefixes)
        self.assertTrue(all(not entries for entries in p._persistent_scenes.values()))
        self.assertFalse(p._scene_revisions)

    def test_old_critic_is_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, 'reset_critic'):
            self.module.Policy('nonexistent-model', checkpoint='not-opened', parent_checkpoint='not-opened')
        p = self.policy
        p.world_identity = self.module.balance_identity()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            p.load_critic_state({}, {'contract':'old', 'critic_features':97})
        metadata = dict(contract=self.module.VERSION, critic_features=self.module.FEATURE_DIM,
                        critic_feature_names=list(self.module.FEATURE_NAMES), **p.world_identity)
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            p.load_critic_state({'0.weight':self.torch.zeros(128, 97)}, metadata)


if __name__ == '__main__':
    unittest.main()
