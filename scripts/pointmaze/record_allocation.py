"""Record the CUDA device through PyTorch; no optional vendor CLI is required."""
import argparse
import datetime
import json
import os
from pathlib import Path


def device_record(cuda):
    if not cuda.is_available():
        raise RuntimeError('The allocated worker cannot access CUDA')
    index = cuda.current_device()
    properties = cuda.get_device_properties(index)
    uuid = getattr(properties, 'uuid', None)
    return {'logical_device': index, 'name': properties.name,
            'total_memory_bytes': int(properties.total_memory),
            'uuid': str(uuid) if uuid is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    record = device_record(torch.cuda)
    record.update(torch=torch.__version__,
                  recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    with Path(args.output).open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
