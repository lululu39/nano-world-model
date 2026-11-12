"""PointMaze control chunks, episode windows and held-out planning goals."""
import bisect
import json
from pathlib import Path
import random

import numpy as np
import torch
from .pusht import PushTActions


class PointMazeActions(PushTActions):
    """Five chronological direct controls, without PushT displacement scaling."""
    def pack(self, controls):
        if controls.shape[-2:] != (5, 2):
            raise ValueError('Expected five chronological two-dimensional controls')
        return ((controls-self.mean.to(controls))/self.std.to(controls)).flatten(-2)


class PointMazeWindows:
    def __init__(self, assets, split, frames=4, seed=42):
        assets = Path(assets)
        manifest = json.loads((assets/'split.json').read_text())
        if seed != manifest['split_seed'] or split not in ('train','val'):
            raise ValueError('PointMaze uses the fixed, persisted episode split')
        self.root = assets/'data/point_maze'
        self.episodes = manifest['train' if split == 'train' else 'dev']
        assert not set(self.episodes)&set(manifest['test'])
        self.lengths = torch.load(self.root/'seq_lengths.pth', weights_only=True).long().tolist()
        self.actions = torch.load(self.root/'actions.pth', weights_only=True).float()
        stats = torch.load(assets/'action_stats.pt', weights_only=True)
        self.action_transform = PointMazeActions(stats['mean'],stats['std'])
        self.frames = frames
        self.ends = []; total = 0
        for episode in self.episodes:
            total += max(0,self.lengths[episode]-(frames-1)*5)
            self.ends.append(total)
        if not total:
            raise ValueError('PointMaze split has no complete windows')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self,index):
        from torchvision.transforms.functional import resize
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends,index)
        episode = self.episodes[position]
        start = index-(self.ends[position-1] if position else 0)
        video = np.load(self.root/f'obses/episode_{episode:03d}.npy',mmap_mode='r')
        images = torch.from_numpy(np.array(video[start:start+(self.frames-1)*5+1:5],copy=True))
        images = resize(images.permute(0,3,1,2).float()/255.,[256,256],antialias=True)
        controls = self.actions[episode,start:start+(self.frames-1)*5].reshape(self.frames-1,5,2)
        return {'observations':images.unsqueeze(1),'actions':self.action_transform.pack(controls)}


def sample_goals(env, assets, count, horizon, seed):
    """Replay direct controls from official validation trajectories only."""
    from experiments.planning_experiment import PlanningGoal
    assets = Path(assets); data = assets/'data/point_maze'
    split = json.loads((assets/'split.json').read_text())
    lengths = torch.load(data/'seq_lengths.pth',weights_only=True).tolist()
    states = torch.load(data/'states.pth',weights_only=True).float()
    controls = torch.load(data/'actions.pth',weights_only=True).float()
    eligible = [i for i in split['test'] if lengths[i] >= horizon*5+1]
    assert eligible and not set(eligible)&set(split['official_train'])
    rng = random.Random(seed); goals = []
    for i in range(count):
        episode = rng.choice(eligible)
        offset = rng.randint(0,int(lengths[episode])-horizon*5-1)
        initial = states[episode,offset].numpy().copy()
        init_obs,_ = env.prepare(seed+i,initial)
        init_obs = {k:np.array(v,copy=True) for k,v in init_obs.items()}
        action_sequence = controls[episode,offset:offset+horizon*5].numpy()
        for action in action_sequence:
            goal_obs,_,_,info = env.step(action)
        goals.append(PlanningGoal(init_state=initial,goal_state=np.array(info['state'],copy=True),
            env_info=None,goal_meta={'source':'dset','split':'official_validation','traj_id':episode,
                'offset':offset,'seed':seed+i,'frame_interval':5,'goal_H':horizon,
                'replay_controls':action_sequence.tolist()},
            init_obs=init_obs,goal_obs={k:np.array(v,copy=True) for k,v in goal_obs.items()}))
    return goals


def verify_environment_alignment(assets, output):
    """Check actual direct controls and rendering against held training trajectories."""
    from PIL import Image
    from planning.envs import create_planning_env
    from torchvision.transforms.functional import resize
    dataset = PointMazeWindows(assets,'train')
    states = torch.load(dataset.root/'states.pth',weights_only=True).float()
    transform = dataset.action_transform
    env = create_planning_env('point_maze')
    errors, image_errors, marker_errors = [], [], []
    for episode in dataset.episodes[:3]:
        raw = dataset.actions[episode,:15].reshape(3,5,2)
        restored = transform.unpack(transform.pack(raw))
        torch.testing.assert_close(restored,raw,rtol=1e-6,atol=1e-6)
        obs,_ = env.prepare(42,states[episode,0].numpy())
        recorded = np.load(dataset.root/f'obses/episode_{episode:03d}.npy',mmap_mode='r')[0]
        reference = resize(torch.from_numpy(np.array(recorded,copy=True)).permute(2,0,1).float(),
                           [224,224],antialias=True).permute(1,2,0).numpy()
        rendered = obs['visual'].astype(np.float32)
        image_errors.append(float(np.abs(reference-rendered).mean()/255))
        def center(image):
            mask = (image[:,:,1]>image[:,:,0]+20)&(image[:,:,1]>image[:,:,2]+20)
            y,x = np.nonzero(mask)
            assert len(x)>5, 'Missing green agent marker'
            return np.array([x.mean(),y.mean()])
        marker_errors.append(float(np.linalg.norm(center(reference)-center(rendered))))
        direct = [env.step(a.numpy())[3]['state'].copy() for a in raw.reshape(-1,2)]
        errors.append(float(np.max(np.abs(np.asarray(direct)-states[episode,1:16].numpy()))))
        env.prepare(42,states[episode,0].numpy())
        roundtrip = [env.step(a.numpy())[3]['state'].copy() for a in restored.reshape(-1,2)]
        np.testing.assert_allclose(direct,roundtrip,rtol=1e-4,atol=1e-4)
        if episode == dataset.episodes[0]:
            Image.fromarray(np.concatenate([reference.astype(np.uint8),rendered.astype(np.uint8)],axis=1)).save(Path(output)/'dataset-render-alignment.png')
    env.close()
    result = {'dataset_state_replay_max_abs_errors':errors,'render_mean_absolute_errors_0_1':image_errors,
              'agent_marker_center_errors_pixels':marker_errors,'action_scale':1.0,'relative_actions':False}
    (Path(output)/'environment-alignment.json').write_text(json.dumps(result,indent=2)+'\n')
    assert max(errors)<1e-3, 'Simulator dynamics disagree with the source trajectory'
    assert max(image_errors)<.06 and max(marker_errors)<3., 'Dataset and simulator image coordinates disagree'
    return result
