"""子进程失败不修改已初始化父CUDA的缓存/精度，不回退到原生。"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from fnit.recon_all.gca_torch_worker import run_isolated_registration, run_worker
from fnit.recon_all.native_free import _run_native_em_register


class GCAIsolationTest(unittest.TestCase):
    def test_exec_failure_preserves_initialized_parent_allocator_and_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(os.environ)
            before_precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            with patch.dict(os.environ, {'PYTORCH_NO_CUDA_MEMORY_CACHING': '1',
                                        'CUDA_VISIBLE_DEVICES': 'GPU-explicit-mapping'}), \
                    patch('torch.cuda.is_initialized', return_value=True), \
                    patch('fnit.recon_all.gca_torch_worker.subprocess.run',
                          side_effect=subprocess.CalledProcessError(17, ['worker'])) as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_isolated_registration(
                        nu_path=root/'nu.mgz', mask_path=root/'brainmask.mgz', atlas_path=root/'fixed.gca',
                        output_path=root/'talairach.lta', report_path=root/'report.json',
                        device='cuda:1', threads=2, candidate_chunk=1024)
                child = launch.call_args.kwargs['env']
                self.assertNotIn('PYTORCH_NO_CUDA_MEMORY_CACHING', child)
                self.assertEqual(os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'], '1')
                self.assertEqual(child['CUDA_VISIBLE_DEVICES'], 'GPU-explicit-mapping')
                self.assertEqual(child['NUMBA_NUM_THREADS'], '2')
                self.assertFalse((root/'talairach.lta').exists())
            self.assertEqual(dict(os.environ), environment)
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), before_precision)

    def test_worker_rejects_inherited_initialized_cuda(self):
        with patch('torch.cuda.is_initialized', return_value=True), \
                patch('fnit.recon_all.gca_torch_worker.register_t1') as registration:
            with self.assertRaisesRegex(ValueError, 'fresh exec'):
                run_worker(nu_path=Path('nu.mgz'), mask_path=Path('brainmask.mgz'), atlas_path=Path('fixed.gca'),
                           output_path=Path('new.lta'), report_path=Path('report.json'), code_version='test')
            registration.assert_not_called()

    def test_native_selection_cannot_ignore_isolated_execution(self):
        with patch('fnit.recon_all.native_free.subprocess.run') as native:
            with self.assertRaisesRegex(ValueError, 'Torch GCA'):
                _run_native_em_register(Path('/native'), Path('/mri'), Path('/fixed.gca'), Path('/assets'),
                                        backend='original', execution='isolated')
            native.assert_not_called()

    def test_isolated_stage_only_passes_candidate_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory)
            mri = subject/'mri'
            (mri/'transforms').mkdir(parents=True)

            def worker(**kwargs):
                kwargs['output_path'].write_text('LTA')
                return {'isolation': 'fresh exec'}

            with patch('fnit.recon_all.gca_torch_worker.run_isolated_registration', side_effect=worker) as run, \
                    patch('fnit.recon_all.native_free.subprocess.run') as native:
                result = _run_native_em_register(
                    Path('/unused-native'), mri, Path('/declared.gca'), Path('/assets'), backend='torch',
                    device='cuda:1', threads=4, inverse_backend='torch', candidate_chunk=1024, execution='isolated')
            self.assertEqual(run.call_args.kwargs['nu_path'], mri/'nu.mgz')
            self.assertEqual(run.call_args.kwargs['mask_path'], mri/'brainmask.mgz')
            self.assertEqual(run.call_args.kwargs['atlas_path'], Path('/declared.gca'))
            self.assertEqual(run.call_args.kwargs['report_path'], subject/'scripts/gca-isolated.json')
            self.assertEqual(result['isolation'], 'fresh exec')
            native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
