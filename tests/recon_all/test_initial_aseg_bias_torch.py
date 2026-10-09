"""初始aseg偏场复用回归；真实数据另外通过完整文件API验证。"""
import os
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from fnit.recon_all.normalization.normalize_aseg_source import apply_initial_aseg_bias

if os.environ.get("FNIT_NORMALIZATION_TEST_OVERLAY"):
    # 测试候选只上传的单模块；原包与其余依赖仍从冻结源码只读加载。
    path = Path(os.environ["FNIT_NORMALIZATION_TEST_OVERLAY"]) / "normalize_aseg_source.py"
    spec = importlib.util.spec_from_file_location("fnit.recon_all.normalization.normalize_aseg_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    apply_initial_aseg_bias = module.apply_initial_aseg_bias


class InitialBiasGuardTests(unittest.TestCase):
    def test_explicit_backend_and_device(self):
        source = np.ones((4, 5, 6), np.float32)
        control = np.ones_like(source, np.uint8)
        for keywords in ({"backend": "other"}, {"backend": "torch"},
                         {"backend": "torch", "device": "cpu"}):
            with self.assertRaises(ValueError):
                apply_initial_aseg_bias(source=source, controls=control, **keywords)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class InitialBiasCudaTests(unittest.TestCase):
    device = os.environ.get("FNIT_TEST_GPU_DEVICE", "cuda:0")

    def test_same_input_float32_and_parent_device(self):
        previous = torch.cuda.current_device()
        try:
            parent_device = 0
            torch.cuda.set_device(parent_device)
            parent = torch.tensor([19.], device="cuda:0")
            rng = np.random.default_rng(71)
            source = (rng.random((9, 8, 7)) * 210).astype(np.float32)
            control = np.zeros(source.shape, np.uint8)
            control[2:7:2, 1:7:2, 1:6:2] = 1
            source_before, control_before = source.copy(), control.copy()
            tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            reference = apply_initial_aseg_bias(source=source, controls=control)
            actual = apply_initial_aseg_bias(source=source, controls=control,
                                           backend="torch", device=self.device)
            np.testing.assert_array_equal(actual, reference)
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(torch.cuda.current_device(), parent_device)
            self.assertEqual(parent.item(), 19.)
            self.assertEqual(tf32, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
            np.testing.assert_array_equal(source, source_before)
            np.testing.assert_array_equal(control, control_before)
        finally:
            torch.cuda.set_device(previous)


if __name__ == "__main__":
    unittest.main()
