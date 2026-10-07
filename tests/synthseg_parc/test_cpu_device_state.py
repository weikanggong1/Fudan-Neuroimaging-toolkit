"""CPU 调用保留同进程 CUDA 策略；小模型仅用于状态合同，不作 benchmark。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.synthseg_parc.pipeline import SynthSegParc
from fnit.synthseg_parc.segment import SynthSegSegmenter
from fnit.synthseg_parc.synthseg import SynthSeg
from fnit.wmh_synthseg.pipeline import WMHSynthSeg


def state():
    return (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.mkldnn.enabled)


class TinyModel(torch.nn.Module):
    def __init__(self, channels, *, fail=False):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.channels, self.fail = channels, fail
        # Match the mature WMH pipeline's inference-mode state contract.
        self._memory_efficient_inference = True

    def load_h5(self, path):
        return self

    def load_state_dict(self, values, strict=True):
        assert strict

    def forward(self, image):
        if self.fail:
            raise RuntimeError("test forward failure")
        shape = (image.shape[0], self.channels, *image.shape[2:])
        return torch.full(shape, 1 / self.channels, dtype=image.dtype)


@pytest.mark.parametrize("matmul,cudnn", [(False, False), (False, True),
                                        (True, False), (True, True)])
def test_cpu_constructors_and_forwards_preserve_cuda_and_backend_state(matmul, cudnn):
    previous, threads = state(), torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        expected = state()
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            labels = np.concatenate((np.arange(33), np.arange(22)))
            for name, values in (
                ("synthseg_segmentation_labels_2.0.npy", labels),
                ("synthseg_segmentation_names_2.0.npy", labels.astype(str)),
                ("synthseg_topological_classes_2.0.npy", np.zeros(55)),
            ):
                np.save(directory / name, values)
            model_path = directory / "synthseg_2.0.h5"
            model_path.write_bytes(b"state test fixture")
            with mock.patch("fnit.synthseg_parc.segment.SegmentUNet",
                            side_effect=lambda: TinyModel(33)):
                synthseg = SynthSeg(model_path, device="cpu", cudnn_tf32=False)
            assert state() == expected
            synthseg.segmenter.posterior(torch.ones((3, 3, 3)))
            assert state() == expected

            with mock.patch("fnit.synthseg_parc.pipeline.ParcUNet",
                            side_effect=lambda: TinyModel(69)):
                parc = SynthSegParc("state-only.h5", np.arange(69), "cpu")
            assert state() == expected
            parc(torch.ones((3, 3, 3)), torch.full((3, 3, 3), 3))
            assert state() == expected

            with mock.patch("fnit.wmh_synthseg.pipeline.resolve_weights", return_value="state-only.pth"), \
                    mock.patch("fnit.wmh_synthseg.pipeline.UNet3D", side_effect=lambda: TinyModel(39)), \
                    mock.patch("torch.load", return_value={"model_state_dict": {}}):
                wmh = WMHSynthSeg(device="cpu")
            assert state() == expected
            image = nib.Nifti1Image(np.ones((3, 3, 3), np.float32), np.eye(4))
            wmh(image, save_lesion_probabilities=True)
            assert state() == expected
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.mkldnn.enabled) = previous
        torch.set_num_threads(threads)


def test_cpu_failed_segmenter_forward_preserves_caller_flags():
    previous = state()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        segmenter = object.__new__(SynthSegSegmenter)
        segmenter.device = torch.device("cpu")
        segmenter.cudnn_tf32 = False
        segmenter.model = TinyModel(33, fail=True)
        expected = state()
        with pytest.raises(RuntimeError, match="test forward failure"):
            segmenter.posterior(torch.ones((2, 2, 2)))
        assert state() == expected
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.mkldnn.enabled) = previous
