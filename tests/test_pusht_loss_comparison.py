"""Reject confounded loss comparisons before reporting experimental results."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts/pusht'))
spec = importlib.util.spec_from_file_location('compare_losses', root/'scripts/pusht/compare_losses.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class LossComparisonTests(unittest.TestCase):
    def setUp(self):
        self.mae = json.loads((root/'reports/pusht/prefix-history/training/config.json').read_text())
        self.mse = copy.deepcopy(self.mae)
        self.mse['loss_type'] = 'mse'
        self.mse['logging'].update(project='nanowm', name='descriptive-test-name')

    def test_legacy_mae_and_explicit_mse_are_comparable(self):
        comparison.verify_recipe(self.mae, self.mse)

    def test_numerical_changes_are_rejected(self):
        for key, value in [('lr', .001), ('steps', 160000), ('batch_size', 8)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                changed = copy.deepcopy(self.mse)
                changed['training'][key] = value
                comparison.verify_recipe(self.mae, changed)

    def test_same_objective_cannot_be_reported_as_an_ablation(self):
        self.mse['loss_type'] = 'mae'
        with self.assertRaises(AssertionError):
            comparison.verify_recipe(self.mae, self.mse)


if __name__ == '__main__':
    unittest.main()
