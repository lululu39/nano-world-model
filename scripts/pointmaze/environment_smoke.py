"""Verify legacy PointMaze physics and offscreen rendering on a compute node."""
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from PIL import Image
from planning.envs import create_planning_env


def main():
    env = create_planning_env('point_maze')
    initial = np.array([1., 1., 0., 0.])
    obs, state = env.prepare(42, initial)
    image = obs['visual']
    assert image.shape == (224,224,3) and image.dtype == np.uint8 and image.std() > 1
    np.testing.assert_array_equal(state, initial)
    actions = [np.array([.3,-.2]), np.array([-.1,.4])]
    first = [env.step(a)[3]['state'].copy() for a in actions]
    env.prepare(42, initial)
    second = [env.step(a)[3]['state'].copy() for a in actions]
    np.testing.assert_array_equal(first,second)
    assert np.isfinite(first).all() and not np.array_equal(first[-1], initial)
    Image.fromarray(image).save('reports/environment-render.png')
    report = {'completed':True, 'shape':list(image.shape), 'dtype':str(image.dtype),
              'reset_and_replay_identical':True, 'control_clip':[-1.,1.],
              'physics_frame_skip':env.frame_skip, 'physics_timestep':env.model.opt.timestep,
              'versions':{k:importlib.metadata.version(k) for k in ('gym','mujoco-py','Cython','d4rl')},
              'renderer':'MuJoCo 2.1.0 / OSMesa'}
    Path('reports/environment-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    env.close()
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
