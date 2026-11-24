import copy
import json
from pathlib import Path
import tempfile
import unittest

from agent.m8x_registration import validate_config, CORE_FILES, PROTOCOL


ROOT = Path(__file__).resolve().parents[1]


class BalancedRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/m8x_exploration_attempt1.json').read_text())

    def test_registered_config_and_disjoint_new_identity(self):
        validate_config(ROOT, self.config)
        self.assertEqual(PROTOCOL, 'm8_exploration_reliable_v1')
        self.assertEqual(len(CORE_FILES), len(set(CORE_FILES)))
        self.assertIn('scripts/exploration_world.gd', CORE_FILES)
        self.assertIn('scripts/world_state.gd', CORE_FILES)

    def test_changed_config_rejected(self):
        for key, value in [('profile', 'open1'), ('gamma_tick', .999), ('updates', 24),
                           ('training_horizon', 1800), ('temperature', .7), ('attempt', 2)]:
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(ROOT, changed)

    def test_invalid_matrix_rejected_even_if_both_config_copies_match(self):
        for mutation in ('duplicate_validation', 'overlap', 'short_test', 'wrong_replicate'):
            changed = copy.deepcopy(self.config)
            if mutation == 'duplicate_validation': changed['validation_seeds'][1] = changed['validation_seeds'][0]
            if mutation == 'overlap': changed['test_seeds'][0] = changed['validation_seeds'][0]
            if mutation == 'short_test': changed['test_seeds'].pop()
            if mutation == 'wrong_replicate': changed['replicates'] = [0, 1]
            with tempfile.TemporaryDirectory(prefix='m8b_registration_', dir=ROOT / '.tools') as directory:
                project = Path(directory)
                (project / 'configs').mkdir()
                (project / 'configs/m8x_exploration_attempt1.json').write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_config(project, changed)


if __name__ == '__main__': unittest.main()
