"""Compare matched 3,000-update MAE/MSE models on the fixed PushT benchmark."""
import argparse
import csv
import json
import math
from pathlib import Path

from compare_history import verify_records
from summarize import read


def verify_recipe(mae, mse):
    # Project/group/display names do not change optimization or data sampling.
    a, b = json.loads(json.dumps(mae)), json.loads(json.dumps(mse))
    assert a.pop('loss_type', 'mae') == 'mae'
    assert b.pop('loss_type') == 'mse'
    a.pop('logging'); b.pop('logging')
    assert a == b, 'The ablation changed more than the prediction loss'
    assert a['training']['steps'] == 3000 and a['training']['batch_size'] == 4
    assert a['seed'] == 42 and a['rollout_context_frames'] == 3


def compare(baseline, current):
    folders = [Path(baseline), Path(current)]
    training = [p/'training' for p in folders]
    evaluations = [p/'reports/robojepa-prefix' for p in folders]
    verify_recipe(*(read(p, 'config.json') for p in training))
    assert read(training[0], 'split.json') == read(training[1], 'split.json')
    provenance = [read(p, 'provenance.json') for p in training]
    for key in ('assets', 'torch', 'gpu', 'robojepa', 'vjepa'):
        assert provenance[0][key] == provenance[1][key], f'Training differs in {key}'
    summaries = [read(p, 'summary.json') for p in training]
    for folder, loss_type, summary, evaluation in zip(training, ('mae', 'mse'), summaries, evaluations):
        assert summary['step'] == 3000 and not summary['diagnostic'] and summary['exit_code'] == 0
        assert summary.get('loss_type', 'mae') == loss_type
        rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 3001))
        assert all(math.isfinite(r[key]) for r in rows
                   for key in ('train/loss', 'grad_norm', f'teacher_{loss_type}', f'rollout_{loss_type}'))
        loading = read(evaluation, 'weight-loading.json')
        assert loading['finetuned_step'] == 3000 and loading['finetuned_run_id'] == summary['run_id']
        assert loading['pretrained_predictor_exactly_equal'] and loading['encoder_frozen']
        assert loading['rollout_context_frames'] == 3 and loading.get('loss_type', 'mae') == loss_type
    for name in ('protocol.json', 'targets.json'):
        assert read(evaluations[0], name) == read(evaluations[1], name), f'Evaluation differs in {name}'
    environments = [read(p, 'environment.json') for p in evaluations]
    for key in ('gpu', 'torch', 'dependencies', 'assets'):
        assert environments[0][key] == environments[1][key], f'Evaluation differs in {key}'
    evaluation_summaries = [read(p, 'summary.json') for p in evaluations]
    rows = []
    for loss_type, folder, summary in zip(('mae', 'mse'), evaluations, evaluation_summaries):
        assert summary['completed'] and not summary['diagnostic_subset'] and summary['model'] == 'robojepa'
        verify_records(folder)
        rows.append({'method':f'robojepa_prefix_{loss_type}', 'success_count':summary['success_count'],
                     'episodes':summary['episodes'], 'success_rate':summary['success_rate'],
                     'goal_distance':summary['goal_distance_mean'],
                     'planning_median_seconds':summary['planning_latency_seconds']['median'],
                     'planning_p95_seconds':summary['planning_latency_seconds']['p95']})
    paired = {'both_success':0, 'mae_only_success':0, 'mse_only_success':0, 'both_fail':0}
    for i in range(64):
        a, b = [read(p, f'episode-{i:04d}.json')['success'] for p in evaluations]
        key = 'both_success' if a and b else 'mae_only_success' if a else 'mse_only_success' if b else 'both_fail'
        paired[key] += 1
    return {'rows':rows, 'paired_successes':paired, 'training':summaries,
            'checks':{'same_recipe_except_loss':True, 'same_split_goals_protocol':True,
                      'finite_updates_per_run':3000, 'evaluated_checkpoint_step':3000,
                      'same_training_gpu_model':True, 'same_evaluation_gpu_model_and_runtime':True},
            'gpu':environments[0]['gpu'], 'goal_distance_definition':evaluation_summaries[0]['goal_distance_definition'],
            'comparison':'One training seed and 64 fixed tasks; no general superiority claim. Same GPU model, not necessarily the same physical device. Training MAE and MSE values are not directly comparable.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--current', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.current)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    with (out/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(result['rows'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
