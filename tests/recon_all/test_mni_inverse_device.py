"""双 GPU 公共 fill API 同输入与异常设备恢复合同；真实阶段另报。"""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fnit.recon_all.mni_warp_inverse import fill_inverse_fields


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "requires two visible CUDA devices")
class MNIInverseDeviceTests(unittest.TestCase):
    def test_noncurrent_device_matches_existing_algorithm(self):
        old_path = Path(os.environ["FNIT_OLD_MNI_INVERSE"])
        spec = importlib.util.spec_from_file_location("fnit.recon_all._mni_inverse_reference", old_path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        ctrl = np.zeros((7, 8, 9), dtype=bool)
        ctrl[2:5, 3:6, 2:7] = True
        coords = np.indices(ctrl.shape).astype(np.float32)
        coords[:, ~ctrl] = 0
        before_env = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with torch.cuda.device(1):
            expected, old_stats = old.fill_inverse_fields(coords, ctrl, device="cuda:1")
        with torch.cuda.device(0):
            actual, new_stats = fill_inverse_fields(coords, ctrl, device="cuda:1")
            self.assertEqual(torch.cuda.current_device(), 0)
        self.assertTrue(np.array_equal(expected, actual))
        self.assertEqual(old_stats, new_stats)
        self.assertEqual(before_env, os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"))
        self.assertEqual(precision, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_launch_failure_restores_current_device(self):
        class FailedKernel:
            def __getitem__(self, grid):
                def fail(*args, **kwargs): raise RuntimeError("injected launch failure")
                return fail
        ctrl = np.zeros((5, 6, 7), dtype=bool)
        ctrl[2, 2, 3] = True
        with torch.cuda.device(0), patch("fnit.recon_all.mni_warp_kernels._expand", FailedKernel()):
            with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
                fill_inverse_fields(np.zeros((3, *ctrl.shape), dtype=np.float32), ctrl, device="cuda:1")
            self.assertEqual(torch.cuda.current_device(), 0)


if __name__ == "__main__": unittest.main()
