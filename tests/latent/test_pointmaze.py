import json
import numpy as np
import pytest
import torch
from latent_world.pointmaze import PointMazeActions, PointMazeWindows
from latent_world.pusht import validate_transfer_checkpoint


def test_direct_controls_keep_order_and_do_not_inherit_pusht_scaling():
    raw = torch.arange(30,dtype=torch.float32).reshape(3,5,2)/20-1
    mean,std = torch.tensor([.1,-.2]),torch.tensor([.3,.4])
    adapter = PointMazeActions(mean,std)
    expected = torch.stack([(raw[:,i]-mean)/std for i in range(5)],1).reshape(3,10)
    torch.testing.assert_close(adapter.pack(raw),expected)
    torch.testing.assert_close(adapter.unpack(expected),raw)


def test_pointmaze_checkpoint_cannot_be_loaded_as_pusht():
    with pytest.raises(ValueError,match='different environment'):
        validate_transfer_checkpoint({'config':{'environment':'point_maze'}},{})
    with pytest.raises(ValueError,match='different environment'):
        validate_transfer_checkpoint({'config':{}},{},environment='point_maze')


def test_windows_align_five_controls_and_exclude_planning_episodes(tmp_path):
    root=tmp_path/'data/point_maze';(root/'obses').mkdir(parents=True)
    torch.save(torch.tensor([21,21,21]),root/'seq_lengths.pth')
    actions=torch.arange(3*21*2,dtype=torch.float32).reshape(3,21,2)
    torch.save(actions,root/'actions.pth')
    torch.save({'mean':torch.zeros(2),'std':torch.ones(2)},tmp_path/'action_stats.pt')
    (tmp_path/'split.json').write_text(json.dumps({'split_seed':42,'train':[0],'dev':[1],'test':[2]}))
    for ep in (0,1):
        images=np.broadcast_to(np.arange(21,dtype=np.uint8)[:,None,None,None],(21,4,4,3)).copy()
        np.save(root/f'obses/episode_{ep:03d}.npy',images)
    train=PointMazeWindows(tmp_path,'train')
    assert len(train)==6 and train.episodes==[0]
    window=train[2]
    torch.testing.assert_close(window['actions'],actions[0,2:17].reshape(3,10))
    torch.testing.assert_close(window['observations'][:,0,0,0,0],torch.tensor([2,7,12,17])/255.)
    assert PointMazeWindows(tmp_path,'val').episodes==[1]
    with pytest.raises(IndexError):
        train[6]
