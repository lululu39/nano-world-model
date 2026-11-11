"""Run one authorized PushT budget after an actual transfer/recovery diagnostic."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def verify_recovery_transfer(history, event_log):
    """Verify checkpoint upload plus final upload, including in-place restarts.

    checkpoint_exit_code restarts the executable in the same sandbox. It need
    not increase NumJobStarts, which counts execution allocations instead.
    The separate diagnostic report must still prove exact restored training.
    """
    assert history.get('JobStatus') == 4 and history.get('ExitCode') == 0
    assert history.get('SuccessCheckpointExitCode') == 85
    files = {name.strip() for name in history.get('TransferCheckpoint', '').split(',')}
    assert {'results.tar', 'checkpoints.tar'} <= files
    started = history.get('NumOutputTransferStarts', 0)
    finished = event_log.count('Finished transferring output files')
    assert started >= 2 and finished >= 2, 'Checkpoint upload and final upload must both complete'
    return {'num_job_starts': history.get('NumJobStarts'),
            'output_transfers_started': started, 'output_transfers_finished': finished,
            'checkpoint_exit_code': 85, 'checkpoint_and_final_upload_verified': True}


class Pipeline:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state_file = self.root / 'state.json'
        self.lock = (self.root / 'controller.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def state(self):
        return json.loads(self.state_file.read_text())

    def update(self, **values):
        state = self.state(); state.update(values)
        temporary = self.state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.state_file)

    def stop_check(self):
        if (self.root / 'STOP').exists():
            raise RuntimeError('STOP marker prevents further scheduling')

    def queue(self, job):
        return json.loads(subprocess.check_output(['condor_q', job, '-json'], text=True).strip() or '[]')

    def prepare(self, phase):
        folder = self.root / phase
        folder.mkdir(exist_ok=True); (folder / 'logs').mkdir(exist_ok=True)
        for name in ['worker.sh', 'source.tar', 'goals.tar', 'robojepa-source.tar', 'vjepa-source.tar']:
            if not (folder / name).exists():
                os.link(self.root / 'inputs' / name, folder / name)
        state = self.state()
        environment = f'PUSHT_PHASE={phase} PUSHT_TRAIN_CONFIG={state["config"]}'
        files = ['source.tar', 'goals.tar', 'robojepa-source.tar', 'vjepa-source.tar']
        credential = ''
        if phase == 'train':
            environment += f' PUSHT_RUN_ID={state["run_id"]}'
            files += ['approval.tar', '/home/yzhong239/.netrc']
            credential = 'encrypt_input_files = /home/yzhong239/.netrc\n'
        elif phase == 'evaluate':
            archive = Path(state['checkpoint']).name
            environment += f' PUSHT_CHECKPOINT_ARCHIVE={archive}'
            files += ['results.tar', 'osdf:///chtc/staging/y/yzhong239/nanowm-research/' + archive]
        base = 'osdf:///chtc/staging/y/yzhong239/nanowm-research/'
        files += [base + name for name in ['pusht-runtime-v1.tar', 'pusht-assets-v1.tar', 'robojepa-22m-v1.tar']]
        gpu = '(GlobalMemoryMb >= 79000) && regexp("H100|H200", DeviceName) && !regexp("MIG", DeviceName)'
        if phase == 'recovery':
            # Diagnostic restarts must retain the same numerical backend.
            gpu = '(GlobalMemoryMb >= 79000) && regexp("H200", DeviceName) && !regexp("MIG", DeviceName)'
        elif phase == 'evaluate':
            gpu = '(GlobalMemoryMb >= 39000) && regexp("A100-SXM4-40GB", DeviceName) && !regexp("MIG", DeviceName)'
        outputs = 'results.tar' if phase == 'evaluate' else 'results.tar,checkpoints.tar'
        text = f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = worker.sh
environment = "{environment}"
request_cpus = 4
request_memory = 32GB
request_disk = 100GB
request_gpus = 1
require_gpus = {gpu}
rank = regexp("H100|H200", GPUs_DeviceName)
requirements = (HasCHTCStaging == true)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
checkpoint_exit_code = 85
transfer_checkpoint_files = {outputs}
transfer_input_files = {','.join(files)}
{credential}transfer_output_files = {outputs}
output = logs/job.$(Cluster).out
error = logs/job.$(Cluster).err
log = logs/job.$(Cluster).log
stream_output = True
stream_error = True
hold = True
on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))
+InteractiveJob = true
+JobBatchName = "{state['batch_name']}-{phase}"
+GPUJobLength = "short"
+MaxRuntime = 129600
queue 1
'''
        (folder / 'job.sub').write_text(text)
        return folder

    def submit(self, phase):
        self.stop_check()
        state = self.state()
        if phase in state['jobs']:
            return state['jobs'][phase]
        folder = self.root / phase
        queued = json.loads(subprocess.check_output(['condor_q', 'yzhong239', '-json', '-attributes',
                                                     'ClusterId,ProcId,JobBatchName'], text=True).strip() or '[]')
        if any(a.get('JobBatchName') == f'{state["batch_name"]}-{phase}' for a in queued):
            raise RuntimeError('Matching job already exists; reconcile state before another submission')
        subprocess.run(['condor_submit', '-dry-run', 'submit-dryrun.ad', 'job.sub'], cwd=folder, check=True)
        job = subprocess.check_output(['condor_submit', '-terse', 'job.sub'], cwd=folder, text=True).split()[0]
        jobs = state['jobs']; jobs[phase] = job
        self.update(jobs=jobs, active_jobs=[job], stage=phase + '_held')
        ad = self.queue(job)[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert ad['RequestGPUs'] == 1 and ad['RequestCpus'] == 4 and ad['SuccessCheckpointExitCode'] == 85
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and 'source.tar' in ad['TransferInput']
        assert state['config'] in ad['Environment']
        if phase == 'train':
            assert '.netrc' in ad['EncryptInputFiles'] and 'approval.tar' in ad['TransferInput']
        fields = ['ClusterId', 'ProcId', 'JobStatus', 'NumJobStarts', 'RequestCpus', 'RequestMemory',
                  'RequestDisk', 'RequestGPUs', 'RequireGPUs', 'TransferInput', 'TransferOutput',
                  'TransferCheckpoint', 'SuccessCheckpointExitCode', 'ContainerImage', 'EncryptInputFiles']
        (folder / 'submission-audit.json').write_text(json.dumps({k:ad.get(k) for k in fields}, indent=2)+'\n')
        subprocess.run(['condor_release', job], check=True)
        self.update(stage=phase + '_queued'); log(f'Released {phase} {job}')
        return job

    def wait(self, phase, job):
        idle_since = None
        state = self.state()
        pinned = bool(state.get('training_gpu')) if phase == 'train' else False
        expanded = bool(state.get('training_fallback'))
        while True:
            self.stop_check()
            queued = self.queue(job)
            if not queued:
                history = json.loads(subprocess.check_output(['condor_history', job, '-limit', '1', '-json'], text=True))
                if not history or history[0].get('ExitCode') != 0 or history[0].get('JobStatus') != 4:
                    raise RuntimeError(f'{phase} did not exit successfully; inspect history')
                ad = history[0]
                fields = ['ClusterId', 'ProcId', 'JobStatus', 'ExitCode', 'NumJobStarts',
                          'NumOutputTransferStarts', 'CheckpointNumber',
                          'SuccessCheckpointExitCode', 'TransferCheckpoint',
                          'LastRemoteHost', 'AssignedGPUs', 'RemoteWallClockTime']
                (self.root / phase / 'history.json').write_text(json.dumps({k:ad.get(k) for k in fields}, indent=2)+'\n')
                return ad
            ad = queued[0]
            if ad['JobStatus'] in (3, 5):
                raise RuntimeError(f'{phase} stopped: {ad.get("HoldReason", ad["JobStatus"])}')
            if ad['JobStatus'] == 1:
                idle_since = idle_since or time.monotonic()
                if phase == 'train' and not pinned and not expanded and time.monotonic()-idle_since >= 3600:
                    subprocess.run(['condor_qedit', job, 'RequireGPUs',
                        '(GlobalMemoryMb >= 39000) && regexp("H100|H200|A100", DeviceName) && !regexp("MIG", DeviceName)'], check=True)
                    expanded = True
                    self.update(training_fallback='Validated A100 enabled after 3600 continuous Idle seconds')
            else:
                idle_since = None
            if phase == 'train' and ad['JobStatus'] == 2 and not pinned:
                try:
                    result = subprocess.run(['condor_ssh_to_job', '-ssh', 'ssh -T', job,
                                             'cat reports/hardware-gate.json'], capture_output=True, text=True, timeout=30)
                except subprocess.TimeoutExpired:
                    result = None
                if result is not None and result.returncode == 0 and result.stdout.strip():
                    gate = json.loads(result.stdout)
                    assert gate['exact_resume_next_update'] and gate['rollout_context_frames'] == 3
                    subprocess.run(['condor_qedit', job, 'RequireGPUs', 'DeviceName == '+json.dumps(gate['gpu'])], check=True)
                    self.update(training_gpu=gate['gpu']); pinned = True
                    log('Pinned all resumed allocations to '+gate['gpu'])
            err = self.root / phase / 'logs' / f'job.{job.split(".")[0]}.err'
            if err.exists() and 'Stopped after failure;' in err.read_text(errors='replace'):
                raise RuntimeError(f'{phase} failed in the retained debug shell; subsequent stages stopped')
            # Only orchestration is polled; per-step training metrics remain in W&B.
            time.sleep(55)

    def collect(self, phase):
        folder = self.root / phase
        output = folder / 'collected'; output.mkdir(exist_ok=True)
        with tarfile.open(folder / 'results.tar') as archive:
            archive.extractall(output, members=[m for m in archive if not (m.issym() or m.islnk())], filter='data')
        return output

    def main(self):
        self.update(controller_pid=os.getpid(), controller_active=True)
        self.prepare('recovery')
        job = self.submit('recovery'); history = self.wait('recovery', job)
        recovery = self.collect('recovery')
        report = json.loads((recovery / 'reports/recovery-check.json').read_text())
        assert report['full_checkpoint_bitwise_equal'] and report['all_update_metrics_equal']
        assert report['final_step'] == 8 and report['diagnostic'] and not report['wandb_run_created']
        event_log = self.root / 'recovery/logs' / f'job.{job.split(".")[0]}.log'
        transfer = verify_recovery_transfer(history, event_log.read_text())
        (self.root / 'recovery/transfer-verification.json').write_text(json.dumps(transfer, indent=2)+'\n')
        assert json.loads((recovery / 'reports/source.json').read_text())['commit'] == self.state()['source']
        self.update(stage='recovery_passed', recovery_result=report, recovery_transfer=transfer,
                    active_jobs=[], error=None)
        train = self.prepare('train')
        with tarfile.open(train / 'approval.tar', 'w') as archive:
            archive.add(recovery / 'reports/recovery.complete', arcname='approval/debug.complete')
            archive.add(recovery / 'reports/recovery-check.json', arcname='approval/recovery-check.json')
            archive.add(recovery / 'reports/source.json', arcname='approval/source.json')
        job = self.submit('train'); self.wait('train', job)
        trained = self.collect('train')
        summary = json.loads((trained / 'training/summary.json').read_text())
        state = self.state()
        assert summary['run_id'] == state['run_id'] and not summary['diagnostic']
        if not summary['completed'] or summary['step'] != state['budget']['steps']:
            self.update(stage='training_budget_exhausted', training_result=summary, active_jobs=[], controller_active=False)
            log('Target incomplete; no budget extension or evaluation submitted')
            return
        assert summary['windows_seen'] == state['budget']['windows'] and summary['exit_code'] == 0
        archive = Path(state['checkpoint_destination'])
        if not state.get('checkpoint'):
            if archive.exists():
                raise FileExistsError('Refusing to overwrite an existing checkpoint archive')
            shutil.move(train / 'checkpoints.tar', archive)
        self.update(stage='training_complete', checkpoint=str(archive), training_result=summary, active_jobs=[])
        evaluation = self.prepare('evaluate')
        if 'evaluate' not in self.state()['jobs']:
            with tarfile.open(evaluation / 'results.tar', 'w') as output:
                output.add(trained / 'training', arcname='training')
        job = self.submit('evaluate'); self.wait('evaluate', job)
        evaluated = self.collect('evaluate')
        result = json.loads((evaluated / 'reports/robojepa-prefix/summary.json').read_text())
        assert result['completed'] and result['episodes'] == 64 and not result['diagnostic_subset']
        self.update(stage='experiments_complete', evaluation_result=result, active_jobs=[], controller_active=False)
        log(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    pipeline = Pipeline(args.root)
    try:
        pipeline.main()
    except Exception as error:
        traceback.print_exc()
        pipeline.update(stage='needs_attention', error=str(error), controller_active=False)
        raise


if __name__ == '__main__':
    main()
