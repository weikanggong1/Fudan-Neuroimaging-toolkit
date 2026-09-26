"""CUDA entry points enable TF32 while retaining float32 tensors."""

import pytest
import torch

from fnit.applywarp import TorchApplyWarp
from fnit.fast import TorchFAST
from fnit.flirt import TorchFLIRT
from fnit.fnirt import TorchFNIRT
from fnit.topup import TorchTOPUP


@pytest.mark.parametrize(
    "constructor",
    (TorchApplyWarp, TorchFAST, TorchFLIRT, TorchFNIRT, TorchTOPUP),
)
def test_cuda_registration_components_enable_tf32(monkeypatch, constructor):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    constructor(device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
