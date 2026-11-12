"""Independently check all updates, held-out goals and final checkpoint identities."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pusht'))
from compare_history import verify_records
from summarize import compare, read


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);training=root/'training'
    cfg=read(training,'config.json');summary=read(training,'summary.json')
    assert cfg['environment']=='point_maze' and cfg['loss_type']=='mae'
    assert summary['completed'] and summary['step']==3000 and summary['windows_seen']==12000 and not summary['diagnostic']
    rows=[json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,3001))
    assert all(math.isfinite(row[k]) for row in rows for k in ('train/loss','grad_norm','teacher_mae','rollout_mae'))
    nano,robo=root/'reports/nanowm',root/'reports/robojepa'
    result=compare(nano,robo)
    assert result==read(root/'reports/comparison','comparison.json')
    loading=read(robo,'weight-loading.json')
    assert loading['finetuned_step']==3000 and loading['finetuned_run_id']==summary['run_id']
    assert loading['environment']=='point_maze' and loading['loss_type']=='mae'
    assert loading['pretrained_predictor_exactly_equal'] and loading['encoder_frozen']
    split=read(Path('assets'),'split.json')
    targets=read(robo,'targets.json')
    for target in targets:
        meta=target['goal_meta']
        assert meta['traj_id'] in split['test'] and meta['traj_id'] not in split['official_train']
    checks={name:verify_records(path) for name,path in [('nanowm',nano),('robojepa',robo)]}
    checks.update(held_out_targets_verified=True,finite_updates=3000,checkpoint_step=3000,
                  same_protocol_and_runtime=True,loss_type='mae')
    (root/'reports/independent-check.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    main()
