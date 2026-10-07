"""CPU API lifecycle and failure contracts; not a real-data benchmark."""
import importlib
import sys

import pytest
import torch

from fnit.robust_register import CPURegistration, load_cpu_registration
from fnit.robust_register.cli import main, parser


def test_closed_context_rejects_before_image_io():
    registration = load_cpu_registration()
    registration.close()
    registration.close()
    with pytest.raises(RuntimeError, match="closed"):
        registration.cpu_robust_register("missing-source", "missing-target")
    with pytest.raises(RuntimeError, match="closed"):
        registration.cpu_robust_rigid_affine(
            "missing-source", "missing-target", stage_directory="unused")


def test_context_manager_closes_on_error():
    registration = CPURegistration()
    with pytest.raises(LookupError):
        with registration:
            raise LookupError("caller failure")
    with pytest.raises(RuntimeError, match="closed"):
        registration.__enter__()


def test_cpu_only_rejection_preserves_cuda_precision():
    original = torch.backends.cuda.matmul.allow_tf32
    initialized = torch.cuda.is_initialized()
    with load_cpu_registration() as registration:
        with pytest.raises(ValueError, match="only device=cpu"):
            registration.cpu_robust_register(
                "missing-source", "missing-target", device="cuda:0")
        with pytest.raises(ValueError, match="only device=cpu"):
            registration.cpu_robust_rigid_affine(
                "missing-source", "missing-target", stage_directory="unused", device="cuda:0")
    assert torch.backends.cuda.matmul.allow_tf32 == original
    assert torch.cuda.is_initialized() == initialized


def test_instances_share_normal_module_but_close_independently(monkeypatch):
    module = importlib.import_module("fnit.robust_register._cpu_engine.registration")
    sentinel = object()
    calls = []

    def no_image_computation(source, target, **parameters):
        calls.append((source, target, parameters))
        return sentinel

    monkeypatch.setattr(module, "robust_register", no_image_computation)
    first, second = load_cpu_registration(), load_cpu_registration()
    assert first.cpu_robust_register("source", "target", mode="affine") is sentinel
    first.close()
    assert second.cpu_robust_register("other", "target") is sentinel
    assert sys.modules[module.__name__] is module
    assert len(calls) == 2 and calls[0][2] == {"device": "cpu", "mode": "affine"}
    second.close()


def test_existing_cli_output_rejected_before_input_io(tmp_path):
    with pytest.raises(FileExistsError):
        main(["--source", "missing", "--target", "missing",
              "--output-directory", str(tmp_path)])


def test_cli_thread_budget_is_optional_and_positive():
    arguments = ["--source", "source.mgz", "--target", "target.mgz",
                 "--output-directory", "registration"]
    assert parser().parse_args(arguments).threads is None
    assert parser().parse_args(arguments + ["--threads", "8"]).threads == 8
    with pytest.raises(SystemExit):
        parser().parse_args(arguments + ["--threads", "0"])


def test_rank_deficient_cpu_qr_is_not_regularized():
    from fnit.robust_register._cpu_linpack_qr import try_cpu_float_linpack
    with pytest.raises(ValueError, match="rank"):
        try_cpu_float_linpack(torch.zeros(20, 6), torch.ones(20))


def test_unsupported_cpu_helper_input_returns_original_path():
    from fnit.robust_register._cpu_linpack_qr import try_cpu_float_linpack
    assert try_cpu_float_linpack(torch.ones(20, 13), torch.ones(20)) is None
    assert try_cpu_float_linpack(
        torch.ones(20, 6, dtype=torch.float64), torch.ones(20, dtype=torch.float64)) is None
