"""执行隔离/失败传播/线程预算回归，模拟仅用于调度测试。"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.hemisphere_parallel import (
    HemisphereGroupError, run_hemisphere_group, validate_hemisphere_workers,
    _cancel, _live_group, inherited_allocator_policy, resolve_worker_device,
    release_idle_parent_cuda_cache,
)
from fnit.recon_all.profiling import parallel_intervals, ProcessTreeDeviceSampler
from fnit.recon_all.native_free import main, _finish_cortical_surface

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
                numba_threads=numba.get_num_threads(), env=os.environ['OMP_NUM_THREADS'],
                log=str(subject/'scripts/fixed.log'))
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
            self.assertEqual(value['log'],str(self.subject/f'scripts/test.{hemi}.fixed.log'))
            self.assertTrue(Path(value['log']).is_file())
            self.assertEqual(report['workers'][hemi]['value']['log'],value['log'])
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
        self.assertIn('intentional worker error', (self.subject / 'scripts/failure.lh.worker.log').read_text())
        for pid in report['device_process_tree']['worker_pids']:
            self.assertFalse(Path(f'/proc/{pid}').exists())

    def test_undeclared_shared_mutation_is_rejected_before_publish(self):
        with self.assertRaisesRegex(HemisphereGroupError, 'undeclared shared write'):
            self.run_group('undeclared')
        self.assertEqual((self.subject / 'mri/shared-input.mgz').read_bytes(), b'original')
        self.assertFalse((self.subject / 'surf/lh.white').exists())

    def test_cleanup_failure_cannot_remain_complete(self):
        with patch('fnit.recon_all.hemisphere_parallel.shutil.rmtree', side_effect=OSError('cleanup failed')):
            with self.assertRaises(HemisphereGroupError) as caught:
                self.run_group()
        self.assertEqual(caught.exception.report['status'], 'failed')
        report = json.loads((self.subject / 'scripts/test.hemisphere-group.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['failed_finalization_phase'], 'private_directory_cleanup')

    def test_cleanup_error_preserves_original_worker_failure(self):
        with patch('fnit.recon_all.hemisphere_parallel.shutil.rmtree', side_effect=OSError('cleanup failed')):
            with self.assertRaisesRegex(HemisphereGroupError, 'worker failed') as caught:
                self.run_group('failure')
        self.assertEqual(caught.exception.report['workers']['lh']['status'], 'failed')
        self.assertIn('cleanup failed', ' '.join(caught.exception.__notes__))

    def test_metadata_publication_failure_marks_report_failed(self):
        original = Path.replace
        def replace(path, target):
            if path.name == 'test.hemisphere-group.json.tmp':
                raise OSError('report replace failed')
            return original(path,target)
        with patch.object(Path,'replace',replace):
            with self.assertRaises(HemisphereGroupError) as caught:
                self.run_group()
        self.assertEqual(caught.exception.report['status'], 'failed')
        self.assertEqual(caught.exception.report['failed_finalization_phase'], 'group_report_publication')
        self.assertFalse((self.subject/'scripts/test.hemisphere-group.json.tmp').exists())

    def test_serial_exec_uses_four_threads_and_no_overlap(self):
        report = self.run_group(workers=1)
        self.assertEqual(report['overlap_seconds'], 0.)
        self.assertEqual(report['values']['lh']['torch_threads'], 4)
        self.assertEqual(report['values']['rh']['numba_threads'], 4)

    def test_invalid_resource_budget(self):
        for workers, threads in ((2, 1), (3, 4), (True, 4), (2, True), (2, 2.5), (2.0, 4)):
            with self.assertRaises(ValueError):
                validate_hemisphere_workers(workers, threads)

    def test_intervals_do_not_sum_into_wall(self):
        report = parallel_intervals({'lh': {'started_monotonic': 10., 'finished_monotonic': 15.},
                                     'rh': {'started_monotonic': 12., 'finished_monotonic': 17.}})
        self.assertEqual(report['worker_sum_seconds'], 10.)
        self.assertEqual(report['worker_span_seconds'], 7.)
        self.assertEqual(report['overlap_seconds'], 3.)

    def test_cuda_current_device_is_fixed_without_unneeded_initialization(self):
        with patch('torch.cuda.is_initialized',return_value=True), \
             patch('torch.cuda.current_device',return_value=1) as current:
            self.assertEqual(resolve_worker_device('cuda'),'cuda:1')
            current.assert_called_once()
        with patch('torch.cuda.is_initialized',return_value=False), \
             patch('torch.cuda.current_device') as current:
            self.assertEqual(resolve_worker_device('cuda'),'cuda:0')
            self.assertEqual(resolve_worker_device('cuda:1'),'cuda:1')
            current.assert_not_called()

    def test_allocator_inherits_presence_semantics(self):
        self.assertEqual(inherited_allocator_policy({}),'enabled')
        for value in ('1','0',''):
            self.assertEqual(inherited_allocator_policy({'PYTORCH_NO_CUDA_MEMORY_CACHING':value}),'disabled')
        with patch.dict(os.environ,clear=False):
            os.environ.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
            report=self.run_group()
        self.assertEqual(report['worker_allocator_selection']['selected_policy'],'enabled')
        self.assertEqual(report['workers']['lh']['cuda_allocator']['requested'],'enabled')

    def test_idle_cache_release_never_initializes_cpu_or_fresh_cuda(self):
        with patch('torch.cuda.is_initialized', return_value=False), \
             patch('torch.cuda.synchronize') as synchronize, \
             patch('torch.cuda.empty_cache') as clear:
            for device in ('cpu', 'cuda:0'):
                self.assertEqual(release_idle_parent_cuda_cache(device)['status'], 'not_applicable')
            synchronize.assert_not_called()
            clear.assert_not_called()

    def test_idle_cache_release_records_live_and_reserved_separately(self):
        with patch('torch.cuda.is_initialized', return_value=True), \
             patch('torch.cuda.synchronize') as synchronize, \
             patch('torch.cuda.empty_cache') as clear, \
             patch('torch.cuda.memory_allocated', side_effect=[1024, 1024]), \
             patch('torch.cuda.memory_reserved', side_effect=[4096, 1024]):
            result = release_idle_parent_cuda_cache('cuda:1')
        self.assertEqual(result['before'], {'allocated_bytes': 1024, 'reserved_bytes': 4096})
        self.assertEqual(result['after'], {'allocated_bytes': 1024, 'reserved_bytes': 1024})
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['cuda_context_created'])
        self.assertIn('global initialization', result['cuda_context_measurement'])
        clear.assert_called_once()
        self.assertEqual(str(synchronize.call_args.args[0]), 'cuda:1')

    def test_short_gpu_uuid_is_canonicalized_and_unknown_is_unavailable(self):
        complete='GPU-abc12300-0000-0000-0000-000000000001'
        gpu_list='0, '+complete+'\n1, GPU-def12300-0000-0000-0000-000000000002\n'
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-abc'}), \
             patch('torch.cuda.is_initialized',return_value=False), \
             patch('subprocess.check_output',side_effect=[gpu_list,'155, '+complete+', 10\n']), \
             patch.object(ProcessTreeDeviceSampler,'_tree',return_value={155}):
            sampler=ProcessTreeDeviceSampler(device='cuda:0',parent_pid=155)
            sampler.sample_if_due(force=True)
        self.assertEqual(sampler.report()['target_gpu_uuid'],complete)
        self.assertEqual(sampler.report()['peak_tree_total_bytes'],10*1024*1024)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-'}), \
             patch('torch.cuda.is_initialized',return_value=False), \
             patch('subprocess.check_output',return_value=gpu_list):
            sampler=ProcessTreeDeviceSampler(device='cuda:0',parent_pid=155)
            sampler.sample_if_due(force=True)
        self.assertEqual(sampler.report()['status'],'unavailable')
        self.assertIsNone(sampler.report()['peak_tree_total_bytes'])
        self.assertEqual(len(sampler.report()['failed_samples']),1)

    def test_cancel_ended_leader_reaps_live_own_grandchild_only(self):
        marker=self.root/'grandchild.pid'
        child_code="import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path("+repr(str(marker))+").write_text(str(os.getpid()));time.sleep(60)"
        leader_code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child_code)+"]);time.sleep(.3)"
        survivor=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        leader=subprocess.Popen([sys.executable,'-c',leader_code],start_new_session=True)
        try:
            leader.wait(timeout=10)
            self.assertTrue(marker.exists())
            self.assertTrue(_live_group(leader.pid))
            _cancel([leader])
            deadline=time.monotonic()+3
            while _live_group(leader.pid) and time.monotonic()<deadline:time.sleep(.05)
            self.assertFalse(_live_group(leader.pid))
            self.assertIsNone(survivor.poll())
        finally:
            _cancel([leader,survivor])

    def test_parallel_final_placement_defers_gpu_maps(self):
        with patch('fnit.recon_all.final_white_conda.run_final_white', return_value={'output':'white'}), \
             patch('fnit.recon_all.native_free._run_native_pial', return_value={'output':'pial'}), \
             patch('fnit.recon_all.native_free.shutil.copyfile'), \
             patch('fnit.recon_all.native_free._finish_cortical_metrics') as metrics:
            result = _finish_cortical_surface(self.subject,'lh',Path('/bin/native'),
                    Path('/assets'),device='cuda:0',threads=2,defer_metrics=True)
        self.assertTrue(result['metrics_pending'])
        self.assertFalse(result['placement_pending'])
        metrics.assert_not_called()

    def test_cli_opt_in(self):
        with patch('fnit.recon_all.native_free.run_recon_all_python', return_value={'status':'complete'}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(['t1', 'subject', '--weights-dir', 'weights', '--assets-dir', 'assets',
                      '--hemisphere-workers', '2', '--threads', '4'])
            self.assertEqual(run.call_args.kwargs['hemisphere_workers'], 2)
            self.assertEqual(run.call_args.kwargs['threads'], 4)


if __name__ == '__main__':
    unittest.main()
