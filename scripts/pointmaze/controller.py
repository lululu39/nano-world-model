"""Complete one authorized PointMaze pilot using a single GPU job."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback


# The launcher copies this dependency beside its immutable controller snapshot.
base_path=Path(__file__).with_name('chtc_controller.py')
if not base_path.exists():
    base_path=Path(__file__).resolve().parents[1]/'pusht/chtc_controller.py'
spec=importlib.util.spec_from_file_location('pusht_controller',base_path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class PointMazePipeline(base.Pipeline):
    def prepare(self,phase):
        assert phase=='train'
        folder=self.root/phase;folder.mkdir(exist_ok=True);(folder/'logs').mkdir(exist_ok=True)
        for name in ('worker.sh','source.tar','robojepa-source.tar','vjepa-source.tar'):
            if not (folder/name).exists():
                os.link(self.root/'inputs'/name,folder/name)
        reports=self.root/'prepare/collected/reports'
        with tarfile.open(folder/'approval.tar','w') as archive:
            for name in ('preparation.json','environment-smoke.json'):
                archive.add(reports/name,arcname='approval/'+name)
        state=self.state();remote='osdf:///chtc/staging/y/yzhong239/nanowm-research/'
        files=['source.tar','robojepa-source.tar','vjepa-source.tar','approval.tar','/home/yzhong239/.netrc']
        files += [remote+name for name in ('pointmaze-assets-v1.tar.gz','pointmaze-runtime-v1.tar.gz','robojepa-22m-v1.tar')]
        text=f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = worker.sh
environment = "POINTMAZE_RUN_ID={state['run_id']} POINTMAZE_TRAIN_CONFIG={state['config']}"
request_cpus = 4
request_memory = 32GB
request_disk = 140GB
request_gpus = 1
require_gpus = (GlobalMemoryMb >= 79000) && DeviceName == "NVIDIA H200" && !regexp("MIG", DeviceName)
requirements = (HasCHTCStaging == true)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
checkpoint_exit_code = 85
transfer_checkpoint_files = results.tar,checkpoints.tar
transfer_input_files = {','.join(files)}
encrypt_input_files = /home/yzhong239/.netrc
transfer_output_files = results.tar,checkpoints.tar
output = logs/job.$(Cluster).out
error = logs/job.$(Cluster).err
log = logs/job.$(Cluster).log
stream_output = True
stream_error = True
hold = True
on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))
+InteractiveJob = true
+JobBatchName = "{state['batch_name']}-train"
+GPUJobLength = "medium"
+MaxRuntime = 28800
queue 1
'''
        (folder/'job.sub').write_text(text)
        return folder

    def wait_preparation(self):
        job=self.state()['jobs']['prepare']
        while True:
            self.stop_check()
            queued=self.queue(job)
            if queued and queued[0]['JobStatus'] not in (4,5):
                time.sleep(55);continue
            if not (self.root/'prepare/results.tar').is_file():
                raise RuntimeError('Preparation ended without returned reports')
            collected=self.collect('prepare');reports=collected/'reports'
            preparation=json.loads((reports/'preparation.json').read_text())
            assert preparation['completed']
            assert json.loads((reports/'environment-smoke.json').read_text())['reset_and_replay_identical']
            status=int((reports/'prepare.exit').read_text())
            if status:
                # The original retained shell still exits with its original failure status.
                # A reviewed repair executed in that same allocation must supply explicit proof.
                repair=json.loads((reports/'manual-repair.json').read_text())
                assert repair['completed'] and repair['same_allocation'] and repair['all_preparation_checks_passed']
                assert repair['initial_failure']=='patchelf missing from PATH'
                self.update(preparation_repair=repair)
            for name in ('pointmaze-assets-v1.tar.gz','pointmaze-runtime-v1.tar.gz'):
                path=Path('/staging/y/yzhong239/nanowm-research')/name
                assert path.is_file() and path.stat().st_size>1_000_000
            if queued and queued[0]['JobStatus']==5:
                fields=('ClusterId','ProcId','JobStatus','ExitCode','HoldReason','NumJobStarts')
                (self.root/'prepare/terminal-ad.json').write_text(json.dumps({k:queued[0].get(k) for k in fields},indent=2)+'\n')
                subprocess.run(['condor_rm',job],check=True)
            self.update(stage='preparation_complete',active_jobs=[],preparation_result=preparation)
            return

    def main(self):
        self.update(controller_pid=os.getpid(),controller_active=True)
        self.wait_preparation()
        self.prepare('train')
        job=self.submit('train');history=self.wait('train',job)
        collected=self.collect('train')
        report=collected/'reports'
        summary=json.loads((collected/'training/summary.json').read_text())
        if (report/'budget-exhausted').exists():
            self.update(stage='training_budget_exhausted',training_result=summary,active_jobs=[],controller_active=False)
            return
        assert (report/'experiment.complete').exists() and (report/'smoke.complete').exists()
        assert summary['completed'] and summary['step']==3000 and summary['run_id']==self.state()['run_id']
        assert json.loads((report/'independent-check.json').read_text())['finite_updates']==3000
        event_log=(self.root/'train/logs'/f'job.{job.split(".")[0]}.log').read_text()
        transfer=base.verify_recovery_transfer(history,event_log)
        comparison=json.loads((report/'comparison/comparison.json').read_text())
        checkpoint=Path(self.state()['checkpoint_destination'])
        if not self.state().get('checkpoint'):
            if checkpoint.exists(): raise FileExistsError('Checkpoint destination already exists')
            shutil.move(self.root/'train/checkpoints.tar',checkpoint)
        self.update(stage='experiments_complete',checkpoint=str(checkpoint),training_result=summary,
                    comparison=comparison,recovery_transfer=transfer,active_jobs=[],controller_active=False)
        base.log(comparison)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args()
    pipeline=PointMazePipeline(args.root)
    try:
        pipeline.main()
    except Exception as error:
        traceback.print_exc();pipeline.update(stage='needs_attention',error=str(error),controller_active=False)
        raise


if __name__=='__main__':
    main()
