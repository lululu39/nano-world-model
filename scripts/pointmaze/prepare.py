"""Stage pinned PointMaze data and encoder-only assets on a compute node."""
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import torch
from huggingface_hub import snapshot_download

DATA_URL = "https://osf.io/download/vr5gy/?view_only=a56a296ce3b24cceaf408383a175ce28"
MODEL_ID = "knightnemo/nanowm-b2-dino-wm-point-maze-30k"
REVISION = "b84303b7151cbd1836841c3a87bf095e1015b150"


def main():
    root = Path("assets"); root.mkdir(exist_ok=True)
    archive = Path("point_maze.zip")
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL, timeout=180) as source, archive.with_suffix('.partial').open('wb') as out:
            shutil.copyfileobj(source, out, 8*1024*1024)
        archive.with_suffix('.partial').replace(archive)
    data = root/'data/point_maze'; data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = []
        for item in source.infolist():
            p = Path(item.filename)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe dataset member')
            if p.parts[0] != 'point_maze' or item.is_dir():
                continue
            if p.name in ('states.pth','actions.pth','seq_lengths.pth') or (p.parent.name == 'obses' and p.suffix == '.pth'):
                members.append(item)
        for item in members:
            target = root/'data'/item.filename
            # Lossless NPY permits random windows without loading a whole episode each update.
            output = target.with_suffix('.npy') if target.parent.name == 'obses' else target
            if output.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as stream, target.open('wb') as out:
                shutil.copyfileobj(stream, out, 8*1024*1024)
            value = torch.load(target, map_location='cpu', weights_only=False)
            if target.parent.name == 'obses':
                value = value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise ValueError(f'Unexpected source image representation: {value.shape}, {value.dtype}')
                temporary = output.with_suffix('.npy.partial')
                with temporary.open('wb') as stream:
                    np.save(stream, value, allow_pickle=False)
                check = np.load(temporary, mmap_mode='r')
                assert check.shape == value.shape and check.dtype == value.dtype
                assert np.array_equal(check[[0,-1]], value[[0,-1]])
                del check
                temporary.replace(output)
                target.unlink()
            else:
                torch.save(torch.as_tensor(value), target)
    lengths = torch.load(data/'seq_lengths.pth', weights_only=True).long().tolist()
    actions = torch.load(data/'actions.pth', weights_only=True).float()
    states = torch.load(data/'states.pth', weights_only=True).float()
    assert states.shape[:2] == actions.shape[:2] and actions.shape[-1] == 2 and states.shape[-1] == 4
    order = np.random.RandomState(42).permutation(len(lengths)).tolist()
    official_train, test = order[:int(.9*len(order))], order[int(.9*len(order)):]
    inner = np.random.RandomState(43).permutation(official_train).tolist()
    train, dev = inner[:int(.95*len(inner))], inner[int(.95*len(inner)):]
    assert not set(train)&set(dev) and not (set(train)|set(dev))&set(test)
    valid = torch.cat([actions[i,:lengths[i]] for i in official_train])
    assert torch.isfinite(valid).all()
    stats = {'mean':valid.mean(0), 'std':valid.std(0)+1e-6}
    torch.save(stats,root/'action_stats.pt')
    split = {'official_train':official_train,'train':train,'dev':dev,'test':test,
             'split_seed':42,'internal_dev_seed':43}
    (root/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=['config.yaml','model.safetensors'],
                      local_dir=root/'nanowm')
    source_provenance = json.loads(Path('pusht-provenance.json').read_text())
    report = {'dataset_url':DATA_URL,'dataset_bytes':archive.stat().st_size,
              'nanowm':{'model_id':MODEL_ID,'revision':REVISION},'vae':source_provenance['vae'],
              'episodes':len(lengths),'frames':sum(lengths),'image_storage':'lossless_uint8_npy',
              'splits':{k:len(split[k]) for k in ('official_train','train','dev','test')},
              'action_statistics_source':'official 90% training trajectories only',
              'action_scale':1.0,'relative_actions':False,
              'action_stats':{k:v.tolist() for k,v in stats.items()}}
    assert (root/'vae/encoder.safetensors').is_file()
    assert not (root/'vae/diffusion_pytorch_model.safetensors').exists()
    (root/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('reports/assets.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
