import os
from pathlib import Path
import subprocess
import sys


def test_latent_entry_points_do_not_import_pixel_generation_stack():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc, sys
class BlockPixelStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'diffusers', 'lpips', 'piqa', 'pytorch_fid', 'pytorch_lightning', 'transformers'}:
            raise AssertionError('Pixel/trainer dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockPixelStack())
from latent_world.cli import main
from latent_world.factory import build_model, load_config, _nanowm
from latent_world.contracts import LatentSpec
build_model(load_config('configs/latent/smoke.yaml'))
_nanowm(LatentSpec(4, 2, 2), 2, hidden_dim=32, depth=2, heads=4, frames=3)
'''
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, timeout=60)
