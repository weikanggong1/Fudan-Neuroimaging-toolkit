"""回归真实整例暴露的 MNI 主设备漏传；不执行影像算法。"""

from pathlib import Path
import unittest

from fnit.recon_all.native_free import _run_white_mri_chain


class WhiteMriDeviceWiringTest(unittest.TestCase):
    def test_nonlinear_receives_cpu_and_nondefault_cuda_device(self):
        for device in ("cpu", "cuda:1"):
            with self.subTest(device=device):
                calls = {}

                def stage(name, function, *args, **kwargs):
                    calls[name] = kwargs

                _run_white_mri_chain(
                    subject=Path("subject"), weights=Path("weights"),
                    assets=Path("assets"), threads=4,
                    warp_binaries=tuple(Path(name) for name in ("convert", "inverse", "resample")),
                    stage=stage, device=device)
                self.assertEqual(calls["mni_nonlinear"]["device"], device)
                self.assertEqual(calls["mni_aux"]["device"], "cpu")
                self.assertEqual(calls["brain_finalsurfs"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
