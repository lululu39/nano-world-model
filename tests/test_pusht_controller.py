"""Scheduler regressions; stdlib-only so submission checks do not load Torch."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/pusht/chtc_controller.py'
spec = importlib.util.spec_from_file_location('pusht_controller', path)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class RecoveryTransferTests(unittest.TestCase):
    def setUp(self):
        # Observed history from job 8358066.0: exit-85 restarts in-place.
        self.history = {'JobStatus': 4, 'ExitCode': 0, 'NumJobStarts': 1,
                        'SuccessCheckpointExitCode': 85, 'NumOutputTransferStarts': 2,
                        'TransferCheckpoint': 'results.tar,checkpoints.tar'}
        self.events = ('040 Finished transferring output files\n'
                       '040 Finished transferring output files\n'
                       '005 Job terminated.\n')

    def test_in_place_restart_does_not_require_a_second_allocation(self):
        result = controller.verify_recovery_transfer(self.history, self.events)
        self.assertTrue(result['checkpoint_and_final_upload_verified'])
        self.assertEqual(result['num_job_starts'], 1)

    def test_two_allocations_without_checkpoint_upload_are_insufficient(self):
        self.history.update(NumJobStarts=2, NumOutputTransferStarts=1)
        with self.assertRaises(AssertionError):
            controller.verify_recovery_transfer(self.history, self.events)

    def test_started_but_unfinished_upload_is_rejected(self):
        with self.assertRaises(AssertionError):
            controller.verify_recovery_transfer(self.history, 'Finished transferring output files\n')

    def test_missing_checkpoint_payload_is_rejected(self):
        self.history['TransferCheckpoint'] = 'results.tar'
        with self.assertRaises(AssertionError):
            controller.verify_recovery_transfer(self.history, self.events)

    def test_wrong_exit_contract_is_rejected(self):
        self.history['SuccessCheckpointExitCode'] = 0
        with self.assertRaises(AssertionError):
            controller.verify_recovery_transfer(self.history, self.events)

    def test_reviewed_resume_clears_only_the_previous_failure(self):
        old='Stopped after failure; debug shell available for one hour.\n'
        self.assertTrue(controller.has_unrepaired_debug_failure(old))
        repaired=old+'POINTMAZE_MANUAL_RESUME\n'
        self.assertFalse(controller.has_unrepaired_debug_failure(repaired))
        self.assertTrue(controller.has_unrepaired_debug_failure(repaired+old))


if __name__ == '__main__':
    unittest.main()
