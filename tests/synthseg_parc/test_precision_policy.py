"""CPU/fake-model 回归：CUDA 精度作用域、真实前向记录和异常恢复。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from fnit._dmri import configure_device
from fnit.synthseg_parc.segment import SynthSegSegmenter, _blur
from fnit.synthseg_parc.synthseg import SynthSeg


class FakeCUDAInput:
    """只模拟设备选择，前向张量留在 CPU，不初始化或占用 GPU。"""

    ndim = 3

    def to(self, *, device, dtype):
        assert device.type == "cuda"
        return torch.zeros((2, 2, 2), dtype=dtype)


class RecordingModel(torch.nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 33, 1)
        self.fail = fail
        self.seen = []

    def forward(self, image):
        self.seen.append((torch.backends.cuda.matmul.allow_tf32,
                          torch.backends.cudnn.allow_tf32))
        if self.fail:
            raise RuntimeError("forward failed")
        return torch.softmax(self.conv(image), dim=1)


class SynthSegPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.matmul = torch.backends.cuda.matmul.allow_tf32
        self.cudnn = torch.backends.cudnn.allow_tf32
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.matmul
        torch.backends.cudnn.allow_tf32 = self.cudnn
        torch.set_num_threads(self.threads)

    def segmenter(self, policy, *, fail=False, device="cuda"):
        segmenter = object.__new__(SynthSegSegmenter)
        segmenter.device = torch.device(device)
        segmenter.cudnn_tf32 = policy
        segmenter.model = RecordingModel(fail=fail)
        segmenter.flip_indices = torch.arange(33)
        return segmenter

    def test_device_selection_can_leave_precision_untouched(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(configure_device("cuda", configure_precision=False),
                             torch.device("cuda"))
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            configure_device("cuda")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_both_constructors_select_device_without_resetting_precision(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        selected = []

        def select(device, *, configure_precision):
            selected.append((str(device), configure_precision))
            # 模拟请求 CUDA 的构造路径，所有张量/模型实际留在 CPU。
            return torch.device("cpu")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            labels = np.concatenate((np.arange(33), np.arange(22)))
            for name, values in (
                ("synthseg_segmentation_labels_2.0.npy", labels),
                ("synthseg_segmentation_names_2.0.npy", np.array([str(n) for n in labels])),
                ("synthseg_topological_classes_2.0.npy", np.zeros(55, dtype=np.int64)),
            ):
                np.save(directory / name, values)
            model = directory / "synthseg_2.0.h5"
            model.write_bytes(b"fake weight file")
            fake_model = RecordingModel()
            fake_model.load_h5 = lambda path: fake_model
            with mock.patch("fnit.synthseg_parc.synthseg.configure_device", side_effect=select), \
                    mock.patch("fnit.synthseg_parc.segment.configure_device", side_effect=select), \
                    mock.patch("fnit.synthseg_parc.segment.SegmentUNet", return_value=fake_model):
                instance = SynthSeg(weights=model, device="cuda:0", cudnn_tf32=False)
            self.assertEqual(selected, [("cuda:0", False), ("cpu", False)])
            self.assertFalse(instance.segmenter.cudnn_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_invalid_precision_policy_fails_before_loading_weights(self):
        with self.assertRaisesRegex(ValueError, "cudnn_tf32"):
            SynthSeg(weights="unused", cudnn_tf32="false")
        with self.assertRaisesRegex(ValueError, "cudnn_tf32"):
            SynthSegSegmenter("unused", "unused", cudnn_tf32="false")

    def test_explicit_cudnn_policy_applies_to_both_forwards_and_restores(self):
        for policy in (False, True, None):
            with self.subTest(policy=policy):
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = True
                segmenter = self.segmenter(policy)
                output = segmenter.posterior(FakeCUDAInput())
                expected = True if policy is None else policy
                self.assertEqual(segmenter.model.seen, [(True, expected)] * 2)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertEqual(output.shape, (33, 2, 2, 2))
                self.assertEqual(output.dtype, torch.float32)
                records = segmenter.precision["forwards"]
                self.assertEqual([row["pass"] for row in records], ["original", "flipped"])
                for row in records:
                    self.assertTrue(row["matmul_tf32"])
                    self.assertEqual(row["cudnn_tf32"], expected)
                    self.assertEqual(row["model_dtypes"], ["torch.float32"])
                    self.assertEqual(row["input_dtype"], "torch.float32")
                    self.assertEqual(row["output_dtype"], "torch.float32")
                    self.assertFalse(row["autocast"]["cpu"]["enabled"])
                    self.assertFalse(row["autocast"]["cuda"]["enabled"])

    def test_forward_failure_restores_and_keeps_entry_precision(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        segmenter = self.segmenter(False, fail=True)
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            segmenter.posterior(FakeCUDAInput())
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        row = segmenter.precision["forwards"][0]
        self.assertFalse(row["cudnn_tf32"])
        self.assertNotIn("output_dtype", row)

    def test_fp32_ensemble_reuses_buffer_and_matches_previous_expression(self):
        """非对称 CPU 输入逐值对照旧集成，检查返回值复用相加目标存储。"""
        segmenter = self.segmenter(False, device="cpu")
        segmenter.flip_indices = torch.arange(32, -1, -1)
        image = torch.arange(60, dtype=torch.float32).reshape(5, 4, 3) / 59
        with torch.inference_mode():
            x = image[None, None]
            original = _blur(segmenter.model(x))
            flipped = _blur(segmenter.model(torch.flip(x, (2,))))
            flipped = torch.flip(flipped, (2,))[:, segmenter.flip_indices]
            expected = (0.5 * (original + flipped))[0]
        additions = []
        previous_add = torch.Tensor.add_

        def record_add(tensor, other, *args, **kwargs):
            pointer = tensor.data_ptr()
            result = previous_add(tensor, other, *args, **kwargs)
            additions.append((pointer, result.data_ptr()))
            return result

        with mock.patch.object(torch.Tensor, "add_", record_add):
            actual = segmenter.posterior(image=image, flip=True, smooth=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())
        self.assertEqual(len(additions), 1)
        self.assertEqual(additions[0], (actual.data_ptr(), actual.data_ptr()))
        self.assertNotEqual(actual.data_ptr(), image.data_ptr())
        self.assertEqual(len(segmenter.precision["forwards"]), 2)
        self.assertTrue(segmenter.precision["posterior_buffer_reused"])
        self.assertEqual(segmenter.precision["posterior_ensemble_dtype"], "torch.float32")

    def test_second_forward_failure_restores_without_claiming_buffer_reuse(self):
        """第二轮失败仍恢复调用方精度，并保留真实前向及未完成复用记录。"""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        segmenter = self.segmenter(False)
        previous_forward = segmenter.model.forward

        def fail_second(image):
            if segmenter.model.seen:
                raise RuntimeError("second forward failed")
            return previous_forward(image)

        segmenter.model.forward = fail_second
        with self.assertRaisesRegex(RuntimeError, "second forward failed"):
            segmenter.posterior(FakeCUDAInput())
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        records = segmenter.precision["forwards"]
        self.assertEqual([row["pass"] for row in records], ["original", "flipped"])
        self.assertIn("output_dtype", records[0])
        self.assertNotIn("output_dtype", records[1])
        self.assertFalse(segmenter.precision["posterior_buffer_reused"])

    def test_cpu_forward_preserves_settings_and_reports_caller_autocast(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        segmenter = self.segmenter(False, device="cpu")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = segmenter.posterior(torch.zeros((2, 2, 2)), flip=False, smooth=False)
        self.assertEqual(segmenter.model.seen, [(False, True)])
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertEqual(output.dtype, torch.bfloat16)
        row = segmenter.precision["forwards"][0]
        self.assertEqual(row["input_dtype"], "torch.float32")
        self.assertEqual(row["output_dtype"], "torch.bfloat16")
        self.assertEqual(row["autocast"]["cpu"],
                         {"enabled": True, "dtype": "torch.bfloat16"})
        self.assertFalse(segmenter.precision["posterior_buffer_reused"])


if __name__ == "__main__":
    unittest.main()
