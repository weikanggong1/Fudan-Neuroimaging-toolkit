"""执行隔离/失败传播/线程预算回归，模拟仅用于调度测试。"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.hemisphere_parallel import (
    HemisphereGroupError, run_hemisphere_group, validate_hemisphere_workers,
)
from fnit.recon_all.profiling import parallel_intervals
from fnit.recon_all.native_free import main

TARGET = '''
def execute(subject, hemi, device, threads, operation):
    import os, time
    from pathlib import Path
    import torch, numba
    subject = Path(subject)
    if operation == 'failure' and hemi == 'lh':
        raise RuntimeError('intentional worker error')
    if operation == 'failure':
        time.sleep(30)
    if operation == 'undeclared':
        (subject / 'mri/shared-input.mgz').write_bytes(b'changed')
    (subject / 'surf' / (hemi + '.white')).write_bytes(hemi.encode())
    (subject / 'mri/mrisps.white.mgz').write_bytes(hemi.encode())
    (subject / 'scripts/fixed.log').write_text(hemi)
    return dict(pid=os.getpid(), threads=threads, torch_threads=torch.get_num_threads(),
                numba_threads=numba.get_num_threads(), env=os.environ['OMP_NUM_THREADS'])
'''


class HemisphereParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.subject = self.root / 'subject'
        for name in ('mri', 'surf', 'label', 'stats', 'scripts'):
            (self.subject / name).mkdir(parents=True)
        (self.subject / 'mri/shared-input.mgz').write_bytes(b'original')
        (self.root / 'hemi_test_target.py').write_text(TARGET)
        pythonpath = str(self.root) + os.pathsep + os.environ.get('PYTHONPATH', '')
        self.env = patch.dict(os.environ, {'PYTHONPATH': pythonpath})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_group(self, operation='test', workers=2):
        return run_hemisphere_group(self.subject, operation, device='cpu', threads=4,
            workers=workers, callable_path='hemi_test_target:execute')

    def test_real_exec_private_writes_and_right_overwrite(self):
        report = self.run_group()
        self.assertEqual(report['status'], 'complete')
        self.assertEqual((self.subject / 'mri/mrisps.white.mgz').read_bytes(), b'rh')
        self.assertEqual((self.subject / 'mri/shared-input.mgz').read_bytes(), b'original')
        self.assertEqual((self.subject / 'scripts/test.lh.fixed.log').read_text(), 'lh')
        self.assertEqual((self.subject / 'scripts/test.rh.fixed.log').read_text(), 'rh')
        for hemi in ('lh', 'rh'):
            value = report['values'][hemi]
            self.assertNotEqual(value['pid'], os.getpid())
            self.assertEqual(value['torch_threads'], 2)
            self.assertEqual(value['numba_threads'], 2)
            self.assertEqual(value['env'], '2')
            self.assertEqual((self.subject / f'surf/{hemi}.white').read_bytes(), hemi.encode())
        self.assertGreater(report['overlap_seconds'], 0.)
        self.assertFalse(list((self.subject / 'scripts').glob('hemi-*')))

    def test_failure_cancels_sibling_and_does_not_publish(self):
        with self.assertRaises(HemisphereGroupError) as caught:
            self.run_group('failure')
        self.assertEqual(caught.exception.report['status'], 'failed')
        self.assertFalse((self.subject / 'surf/lh.white').exists())
        report = json.loads((self.subject / 'scripts/failure.hemisphere-group.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertIn('intentional worker error', report['workers']['lh']['error'])
        for pid in report['device_process_tree']['worker_pids']:
            self.assertFalse(Path(f'/proc/{pid}').exists())

    def test_undeclared_shared_mutation_is_rejected_before_publish(self):
        with self.assertRaisesRegex(HemisphereGroupError, 'undeclared shared write'):
            self.run_group('undeclared')
        self.assertEqual((self.subject / 'mri/shared-input.mgz').read_bytes(), b'original')
        self.assertFalse((self.subject / 'surf/lh.white').exists())

    def test_serial_exec_uses_four_threads_and_no_overlap(self):
        report = self.run_group(workers=1)
        self.assertEqual(report['overlap_seconds'], 0.)
        self.assertEqual(report['values']['lh']['torch_threads'], 4)
        self.assertEqual(report['values']['rh']['numba_threads'], 4)

    def test_invalid_resource_budget(self):
        for workers, threads in ((2, 1), (3, 4), (True, 4), (2, True), (2, 2.5)):
            with self.assertRaises(ValueError):
                validate_hemisphere_workers(workers, threads)

    def test_intervals_do_not_sum_into_wall(self):
        report = parallel_intervals({'lh': {'started_monotonic': 10., 'finished_monotonic': 15.},
                                     'rh': {'started_monotonic': 12., 'finished_monotonic': 17.}})
        self.assertEqual(report['worker_sum_seconds'], 10.)
        self.assertEqual(report['worker_span_seconds'], 7.)
        self.assertEqual(report['overlap_seconds'], 3.)

    def test_cli_opt_in(self):
        with patch('fnit.recon_all.native_free.run_recon_all_python', return_value={'status':'complete'}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(['t1', 'subject', '--weights-dir', 'weights', '--assets-dir', 'assets',
                      '--hemisphere-workers', '2', '--threads', '4'])
            self.assertEqual(run.call_args.kwargs['hemisphere_workers'], 2)
            self.assertEqual(run.call_args.kwargs['threads'], 4)


if __name__ == '__main__':
    unittest.main()
