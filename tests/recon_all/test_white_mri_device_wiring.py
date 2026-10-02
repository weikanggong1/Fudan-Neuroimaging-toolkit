"""回归真实整例暴露的 MNI 主设备漏传；不执行影像算法。"""

from pathlib import Path
import unittest
import torch

from fnit.recon_all.native_free import _run_white_mri_chain


class WhiteMriDeviceWiringTest(unittest.TestCase):
    def test_nonlinear_receives_cpu_and_nondefault_cuda_device(self):
        for device in ("cpu", "cuda:1"):
            with self.subTest(device=device):
                calls = {}

                def stage(name, function, *args, **kwargs):
                    calls[name] = kwargs
                    calls[name]["actual_cudnn_tf32"] = torch.backends.cudnn.allow_tf32
                    if name == "mni_aux":
                        return {"runtime": {"device": device}}

                with torch.backends.cudnn.flags(allow_tf32=True):
                    result = _run_white_mri_chain(
                        subject=Path("subject"), weights=Path("weights"),
                        assets=Path("assets"), threads=4,
                        warp_binaries=tuple(Path(name) for name in ("convert", "inverse", "resample")),
                        stage=stage, device=device)
                    self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertEqual(result, {"device": device})
                self.assertEqual(calls["mni_nonlinear"]["device"], device)
                self.assertEqual(calls["mni_aux"]["device"], device)
                self.assertEqual(calls["brain_finalsurfs"]["device"], "cpu")
                self.assertFalse(calls["mni_aux"]["actual_cudnn_tf32"])
                self.assertTrue(calls["mni_nonlinear"]["actual_cudnn_tf32"])
                self.assertTrue(calls["brain_finalsurfs"]["actual_cudnn_tf32"])

    def test_aux_failure_restores_precision_and_stops_downstream(self):
        def stage(name, function, *args, **kwargs):
            self.assertEqual(name, "mni_aux")
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            raise RuntimeError("inference failure")

        with torch.backends.cudnn.flags(allow_tf32=True):
            old_matmul = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                with self.assertRaisesRegex(RuntimeError, "inference failure"):
                    _run_white_mri_chain(
                        subject=Path("subject"), weights=Path("weights"),
                        assets=Path("assets"), threads=4,
                        warp_binaries=tuple(Path(name) for name in ("convert", "inverse", "resample")),
                        stage=stage, device="cuda:1")
                self.assertTrue(torch.backends.cudnn.allow_tf32)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul


if __name__ == "__main__":
    unittest.main()
