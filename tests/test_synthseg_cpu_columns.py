"""Guard/cache contracts only; no C++ compiler, SGEMM or MRI benchmark."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fnit.synthseg_parc import cpu_columns as columns
from fnit.synthseg_parc import _cpu_columns_build as builds
from fnit.synthseg_parc import cpu_conv


def full_geometry_metadata():
    # One physical FP32 element, a metadata-only logical shape. Never convolved.
    return torch.empty(1).as_strided((1, 72, 192, 224, 256), (0, 0, 0, 0, 0))


@pytest.fixture
def eligible_state(monkeypatch):
    layer = cpu_conv.CPUInferenceConv3d(72, 24, 3, padding=1).eval()
    layer._fnit_columns_reuse = True
    image = full_geometry_metadata()
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    # No private MRI/weight arrays are in this test. Accepted parameter hash
    # qualification is mocked here; its real-byte negative gate is tested below.
    monkeypatch.setattr(columns, "_parameter_sha", lambda value:
                        columns._WEIGHT_SHA if value is layer.weight else columns._BIAS_SHA)
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        assert columns._eligible(layer, image)
        yield layer, image


def test_cpp_bytes_and_numerical_body_preserved():
    repository = Path(__file__).resolve().parents[1]
    source = repository / "src/fnit/synthseg_parc/_columns_reuse.cpp"
    original = repository / "validation/smri_cpu/seg_columns_reuse_20261006/columns_reuse.cpp"
    assert source.read_bytes() == original.read_bytes()
    assert len(source.read_bytes()) == 4385
    assert hashlib.sha256(source.read_bytes()).hexdigest() == builds._SOURCE_SHA

    def tail(path, class_name):
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        fun = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        first = next(index for index, node in enumerate(fun.body)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "_provider_still_matches")
        return ast.dump(ast.Module(body=fun.body[first:], type_ignores=[]), include_attributes=False)
    assert tail(Path(columns.__file__), "_Columns") == tail(
        repository / "validation/smri_cpu/seg_columns_reuse_v2_20261006/prototype.py", "ColumnsReuse")


@pytest.mark.parametrize("change", [
    "marker", "threads", "training", "grad", "mkldnn", "autocast", "stride",
    "dilation", "padding", "groups", "padding_mode", "bias_none", "weight_dtype",
    "weight_shape", "image_dtype", "image_shape", "requires_grad", "tracing", "parameters",
])
def test_unsupported_guards_never_prepare(eligible_state, monkeypatch, change):
    layer, image = eligible_state
    if change == "marker": layer._fnit_columns_reuse = False
    elif change == "threads": monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    elif change == "training": layer.train()
    elif change == "grad": monkeypatch.setattr(torch, "is_grad_enabled", lambda: True)
    elif change == "mkldnn":
        with torch.backends.mkldnn.flags(enabled=True):
            assert columns.try_columns_reuse(layer, image) is None
        return
    elif change == "autocast": monkeypatch.setattr(cpu_conv, "cpu_autocast_enabled", lambda: True)
    elif change in ("stride", "dilation", "padding"): setattr(layer, change, (2, 2, 2))
    elif change == "groups": layer.groups = 2
    elif change == "padding_mode": layer.padding_mode = "reflect"
    elif change == "bias_none": layer.bias = None
    elif change == "weight_dtype": layer.weight = torch.nn.Parameter(torch.empty(24, 72, 3, 3, 3, dtype=torch.float64))
    elif change == "weight_shape": layer.weight = torch.nn.Parameter(torch.empty(24, 71, 3, 3, 3))
    elif change == "image_dtype": image = torch.empty(1, dtype=torch.float64).as_strided(image.shape, (0,) * 5)
    elif change == "image_shape": image = torch.empty(1, 72, 2, 2, 2)
    elif change == "requires_grad": image = torch.empty(1).as_strided(image.shape, (0,) * 5).requires_grad_(True)
    elif change == "tracing": monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    elif change == "parameters": monkeypatch.setattr(columns, "_parameter_sha", lambda value: "0" * 64)
    loader = Mock(side_effect=AssertionError("unsupported state attempted loader"))
    monkeypatch.setattr(columns, "_engine", loader)
    assert columns.try_columns_reuse(layer, image) is None
    loader.assert_not_called()


def test_cuda_guard_does_not_read_parameters_or_loader(monkeypatch):
    layer = SimpleNamespace()
    cuda_metadata = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    loader = Mock(side_effect=AssertionError("CUDA reached loader"))
    monkeypatch.setattr(columns, "_engine", loader)
    assert columns.try_columns_reuse(layer, cuda_metadata) is None
    loader.assert_not_called()
    assert not torch.cuda.is_initialized()


def test_tensor_subclass_and_forward_ad_are_rejected():
    class SubTensor(torch.Tensor):
        pass
    assert not columns.ordinary_tensor_without_forward_ad(torch.empty(2).as_subclass(SubTensor))
    with torch.autograd.forward_ad.dual_level():
        value = torch.autograd.forward_ad.make_dual(torch.empty(2), torch.empty(2))
        assert not columns.ordinary_tensor_without_forward_ad(value)
    assert columns.ordinary_tensor_without_forward_ad(torch.empty(2))


def test_lazy_negative_or_conjugate_views_are_rejected():
    # These views may be contiguous while their pointer exposes unresolved bits.
    negative = torch._neg_view(torch.zeros(2))
    assert negative.is_contiguous() and negative.is_neg()
    assert not columns.ordinary_tensor_without_forward_ad(negative)
    conjugate = torch.ones(2, dtype=torch.complex64).conj()
    assert conjugate.is_contiguous() and conjugate.is_conj()
    assert not columns.ordinary_tensor_without_forward_ad(conjugate)


@pytest.mark.parametrize("kind", ["forward", "forward_pre", "backward", "global_forward", "global_forward_pre"])
def test_hook_guards_preserve_native_dispatch(eligible_state, monkeypatch, kind):
    layer, image = eligible_state
    from torch.nn.modules import module
    attributes = {"forward": "_forward_hooks", "forward_pre": "_forward_pre_hooks", "backward": "_backward_hooks"}
    if kind.startswith("global"):
        attribute = "_global_forward_hooks" if kind == "global_forward" else "_global_forward_pre_hooks"
        monkeypatch.setattr(module, attribute, {1: lambda *args: None})
    else:
        monkeypatch.setattr(layer, attributes[kind], {1: lambda *args: None})
    loader = Mock(side_effect=AssertionError("hook state attempted loader"))
    monkeypatch.setattr(columns, "_engine", loader)
    assert columns.try_columns_reuse(layer, image) is None
    loader.assert_not_called()


def test_unknown_actual_parameter_bytes_fallback_before_loader(eligible_state, monkeypatch):
    layer, image = eligible_state
    monkeypatch.setattr(columns, "_parameter_sha", lambda value:
                        hashlib.sha256(memoryview(value.detach().numpy()).cast("B")).hexdigest())
    loader = Mock(side_effect=AssertionError("unaccepted weights attempted loader"))
    monkeypatch.setattr(columns, "_engine", loader)
    assert columns.try_columns_reuse(layer, image) is None
    loader.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("provider"), OSError("compiler"), TimeoutError("lock")])
def test_preparation_failure_fallback_before_math(eligible_state, monkeypatch, error):
    layer, image = eligible_state
    monkeypatch.setattr(columns, "_engine", Mock(side_effect=error))
    assert columns.try_columns_reuse(layer, image) is None


def test_numerical_error_is_propagated_no_second_convolution(eligible_state, monkeypatch):
    layer, image = eligible_state
    engine = SimpleNamespace(forward=Mock(side_effect=RuntimeError("copy started then failed")))
    monkeypatch.setattr(columns, "_engine", lambda: engine)
    mature = Mock(side_effect=AssertionError("hidden fallback after math"))
    monkeypatch.setattr(cpu_conv, "convolution_slabs", mature)
    with pytest.raises(RuntimeError, match="copy started"):
        layer(image)
    engine.forward.assert_called_once_with(layer, image)
    mature.assert_not_called()


def test_cpu_forward_none_uses_mature_slabs_once(eligible_state, monkeypatch):
    layer, image = eligible_state
    monkeypatch.setattr(columns, "try_columns_reuse", lambda *values: None)
    result = object()
    mature = Mock(return_value=result)
    monkeypatch.setattr(cpu_conv, "convolution_slabs", mature)
    assert layer(image) is result
    mature.assert_called_once()


@pytest.fixture
def fake_build(monkeypatch, tmp_path):
    # Exercise cache/locking/atomic receipts with stub bytes, never a compiler
    # or dlopen. Real interface build/short numeric tests remain a later gate.
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    torch_root = tmp_path / "torch"
    source = tmp_path / "source.cpp"
    source.write_bytes(b"own source placeholder for cache contract")
    compiler = {"command": "/not/invoked/compiler"}
    key = "a" * 64
    monkeypatch.setattr(builds, "_build_inputs", lambda actual_torch: (torch_root, source, compiler, {"test": True}, key))
    monkeypatch.setattr(builds, "_cache_root", lambda: root)
    monkeypatch.setattr(builds, "_BUILDS", {})
    monkeypatch.setattr(builds, "_FAILED", {})
    compiles = []
    def compile_stub(command):
        compiles.append(command)
        Path(command[-1]).write_bytes(b"not a dynamic library; mock-only")
    monkeypatch.setattr(builds, "_compile", compile_stub)
    monkeypatch.setattr(columns, "_Columns", lambda *args, **kwargs:
                        SimpleNamespace(_provider_still_matches=lambda: None))
    return root, key, compiles


def test_cache_private_atomic_reuse_and_corruption_rebuild(fake_build):
    root, key, compiles = fake_build
    first = builds.build_artifact(torch)
    second = builds.build_artifact(torch)
    assert first == second and len(compiles) == 1
    assert builds._valid_artifact(root / (key + ".so"), root / (key + ".json"), key)
    assert (root / (key + ".so")).stat().st_mode & 0o777 == 0o600
    assert (root / (key + ".json")).stat().st_mode & 0o777 == 0o600
    (root / (key + ".so")).write_bytes(b"changed")
    builds._BUILDS.clear()
    assert not builds._valid_artifact(root / (key + ".so"), root / (key + ".json"), key)
    builds.build_artifact(torch)
    assert len(compiles) == 2


def test_build_failure_cached_without_retry(fake_build, monkeypatch):
    root, key, compiles = fake_build
    failure = Mock(side_effect=RuntimeError("compiler failed before math"))
    monkeypatch.setattr(builds, "_compile", failure)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="compiler failed"):
            builds.build_artifact(torch)
    failure.assert_called_once()
    assert not (root / (key + ".so")).exists()


def test_cache_symlink_and_nonprivate_modes_rejected(tmp_path, monkeypatch):
    source = tmp_path / "payload"
    source.write_bytes(b"data")
    source.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(OSError):
        builds._private_bytes(alias)
    root = tmp_path / "cache"
    root.mkdir(mode=0o755)
    monkeypatch.setenv("FNIT_SYNTHSEG_CPU_CACHE", str(root))
    with pytest.raises(RuntimeError, match="private directory"):
        builds._cache_root()


def test_cxx_argument_list_is_not_executed(monkeypatch):
    monkeypatch.setenv("CXX", "compiler --unexpected-argument")
    child = Mock(side_effect=AssertionError("compiler command executed"))
    monkeypatch.setattr(builds.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="one compiler executable"):
        builds._compiler()
    child.assert_not_called()


def test_unaccepted_compiler_version_fallback(tmp_path, monkeypatch):
    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"not invoked")
    monkeypatch.setenv("CXX", str(compiler))
    monkeypatch.setattr(builds.shutil, "which", lambda value: str(compiler))
    monkeypatch.setattr(builds.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="GNU C++ 12.2.0"))
    with pytest.raises(RuntimeError, match="GCC11"):
        builds._compiler()
