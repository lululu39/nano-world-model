"""Device logging must work without nvidia-smi or a PyTorch UUID attribute."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import os
import subprocess
import tempfile

path = Path(__file__).resolve().parents[1]/'scripts/pointmaze/record_allocation.py'
spec = importlib.util.spec_from_file_location('pointmaze_allocation', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class AllocationTests(unittest.TestCase):
    def backend(self, **properties):
        return SimpleNamespace(is_available=lambda:True, current_device=lambda:0,
                               get_device_properties=lambda index:SimpleNamespace(**properties))

    def test_device_without_optional_uuid_is_supported(self):
        record = module.device_record(self.backend(name='NVIDIA H200',total_memory=140_000_000_000))
        self.assertEqual(record, {'logical_device':0, 'name':'NVIDIA H200',
                                 'total_memory_bytes':140_000_000_000, 'uuid':None})

    def test_available_uuid_is_preserved(self):
        record = module.device_record(self.backend(name='NVIDIA H200',total_memory=140,uuid='GPU-example'))
        self.assertEqual(record['uuid'],'GPU-example')

    def test_missing_cuda_still_blocks_training(self):
        with self.assertRaisesRegex(RuntimeError,'cannot access CUDA'):
            module.device_record(SimpleNamespace(is_available=lambda:False))

    def test_reviewed_debug_resume_keeps_the_original_process(self):
        worker = path.with_name('worker.sh').read_text()
        prefix = worker.split('trap archive EXIT',1)[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'debug-resume.request').write_text('Test an explicitly reviewed repair.\n')
            (root/'worker.sh').write_text('#!/bin/bash\nprintf "%s %s\\n" "$$" "$POINTMAZE_LIVE_RESUME" > reports/resumed.txt\n')
            script = prefix+'trap archive EXIT\necho $$ > original-pid\nfalse\n'
            env = dict(os.environ);env.pop('POINTMAZE_LIVE_RESUME',None)
            result = subprocess.run(['bash','-c',script],cwd=root,env=env,
                                    capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('POINTMAZE_MANUAL_RESUME',result.stderr)
            pid,live = (root/'reports/resumed.txt').read_text().split()
            self.assertEqual(pid,(root/'original-pid').read_text().strip())
            self.assertEqual(live,'1')
            self.assertEqual(len(list((root/'reports/debug-repairs').glob('*.failure.exit'))),1)
            self.assertFalse((root/'debug-resume.request').exists())


if __name__ == '__main__':
    unittest.main()
