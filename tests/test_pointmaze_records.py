"""Undefined angle metrics must not hide invalid measured results."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/pusht/evaluation_records.py'
spec = importlib.util.spec_from_file_location('evaluation_records', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class EpisodeRecordTests(unittest.TestCase):
    def test_pointmaze_without_angle_is_valid_json(self):
        original = {'success':0., 'state_dist':4.3, 'angle_dist':float('nan')}
        record = module.episode_record(original, 'point_maze')
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)),
                         {'success':0., 'state_dist':4.3})
        self.assertIn('angle_dist', original)

    def test_pusht_angle_is_preserved(self):
        original = {'success':1., 'state_dist':.2, 'angle_dist':.1}
        self.assertEqual(module.episode_record(original, 'pusht'), original)

    def test_nonfinite_pointmaze_measurements_still_fail(self):
        for key in ['state_dist','xy_dist','success','planning_seconds']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                record = module.episode_record({key:float('nan')}, 'point_maze')
                json.dumps(record, allow_nan=False)

    def test_nonfinite_pusht_angle_still_fails(self):
        with self.assertRaises(ValueError):
            record = module.episode_record({'angle_dist':float('nan')}, 'pusht')
            json.dumps(record, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
