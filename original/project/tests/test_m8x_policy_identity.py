"""World/reward/runtime identity must survive candidate checkpoint reloads."""
import importlib
import unittest
from copy import deepcopy
from agent.m8x_contract import VERSION, balance_identity, REWARD_CONFIG


class ExplorationIdentityTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('agent.m8x_policy_identity')

    def metadata(self, parent=False):
        m = self.module()
        data = dict(contract='open_world_hierarchical_v2' if parent else VERSION,
                    verb_token_ids=list(range(8)), target_token_ids=list(range(64)),
                    base_config_sha256='base', base_parameters=852985920, max_tokens=1536,
                    inference_layout='fixed_batch4_left_pad_v1')
        if not parent:
            data.update(**balance_identity(), critic_features=97, physical_forward_shape=[4, 1536],
                        reference_actor_sha256=m.PARENT_ACTOR_SHA256, reward_config=deepcopy(REWARD_CONFIG),
                        kernel_identity={'implementation': 'torch_fallback', 'causal_conv1d': None})
        return data

    def check(self, data, parent=False, kernel=None):
        kernel = kernel or {'implementation': 'torch_fallback', 'causal_conv1d': None}
        self.module().validate_checkpoint_metadata(data, list(range(8)), list(range(64)),
                                                  'base', 852985920, 1536, parent, kernel)

    def test_parent_and_exploration_checkpoint_are_not_interchangeable(self):
        self.check(self.metadata(True), True)
        self.check(self.metadata())
        with self.assertRaises(ValueError): self.check(self.metadata(True))
        with self.assertRaises(ValueError): self.check(self.metadata(), True)
        for key, value in [('profile', 'open2'), ('world_version', 4), ('balance_sha256', 'bad'),
                           ('reference_actor_sha256', 'bad'), ('critic_features', 98)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check({**self.metadata(), key: value})

    def test_reward_and_loaded_kernel_must_match_saved_checkpoint(self):
        data = self.metadata()
        data['reward_config']['water_potential']['weight'] = 2.
        with self.assertRaises(ValueError): self.check(data)
        data = self.metadata()
        changed = {'implementation': 'causal_conv1d', 'causal_conv1d': {'version': '1.7.0', 'files_sha256': {'x': 'a'*64}}}
        with self.assertRaises(ValueError): self.check(data, kernel=changed)
        data['kernel_identity'] = changed
        self.check(data, kernel=changed)

    def test_exact_legacy_letter_positions_are_preserved(self):
        m = self.module()
        self.assertEqual(m.branch_token_indices({'branches': [[{}, {}]], 'target_token_indices': [[0, 1]]}, 0, 64), [0, 1])
        for bad in [[1, 0], [0, True], [0, 64]]:
            with self.assertRaises(ValueError):
                m.branch_token_indices({'branches': [[{}, {}]], 'target_token_indices': [bad]}, 0, 64)


if __name__ == '__main__': unittest.main()
