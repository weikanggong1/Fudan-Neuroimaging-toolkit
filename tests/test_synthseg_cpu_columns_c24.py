"""C24 dispatch/cache contracts; no MRI, full CNN, compiler or SGEMM execution."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fnit.synthseg_parc import cpu_columns_c24 as columns
from fnit.synthseg_parc import _cpu_columns_c24_build as builds
from fnit.synthseg_parc import _cpu_columns_build as mature
from fnit.synthseg_parc import cpu_conv


@pytest.fixture
def qualified(monkeypatch):
    # Tiny physical storage with full logical geometry; no convolution runs.
    image = torch.empty(1).as_strided((1, 24, 192, 224, 256), (0,) * 5)
    layer = cpu_conv.CPUInferenceConv3d(24, 24, 3, padding=1).eval()
    layer._fnit_columns_c24 = True
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(columns, "_parameter_sha", lambda value:
                        columns._WEIGHT_SHA if value is layer.weight else columns._BIAS_SHA)
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        assert columns._eligible(layer, image)
        yield layer, image


def test_cpp_and_mathematics_match_frozen_real_validated_prototype():
    repository = Path(__file__).resolve().parents[1]
    frozen = repository / "validation/smri_cpu/seg_columns_c24_prepare_20261006"
    source = repository / "src/fnit/synthseg_parc/_columns_c24.cpp"
    assert source.read_bytes() == (frozen / "columns_c24.cpp").read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == builds._SOURCE_SHA
    def numerical_tail(path, class_name):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        first = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                     and n.value.func.attr == "_provider_still_matches")
        return ast.dump(ast.Module(body=forward.body[first:], type_ignores=[]), include_attributes=False)
    assert numerical_tail(Path(columns.__file__), "_ColumnsC24") == numerical_tail(frozen / "prototype.py", "ColumnsC24")
    # C72 builder/code identities remain exactly the accepted bytes.
    assert hashlib.sha256(Path(mature.__file__).read_bytes()).hexdigest() == "31237d750678098a417390bf6a8ac1cc448f3293a90bf1fcd879b1f0a8fd293b"
    assert hashlib.sha256((repository / "src/fnit/synthseg_parc/cpu_columns.py").read_bytes()).hexdigest() == "9cc29bc05600b0115d2ac559da362d03b72042156dd2747e6a0237f836ffedff"


@pytest.mark.parametrize("change", ["marker", "threads", "training", "grad", "oneDNN", "autocast",
                                   "stride", "bias_none", "other_shape", "parameters", "hooks"])
def test_uncovered_states_never_prepare(qualified, monkeypatch, change):
    layer, image = qualified
    if change == "marker": layer._fnit_columns_c24 = False
    elif change == "threads": monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    elif change == "training": layer.train()
    elif change == "grad": monkeypatch.setattr(torch, "is_grad_enabled", lambda: True)
    elif change == "oneDNN": monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)
    elif change == "autocast": monkeypatch.setattr(cpu_conv, "cpu_autocast_enabled", lambda: True)
    elif change == "stride": layer.stride = (2, 2, 2)
    elif change == "bias_none": layer.bias = None
    elif change == "other_shape": image = torch.empty(1, 24, 2, 2, 2)
    elif change == "parameters": monkeypatch.setattr(columns, "_parameter_sha", lambda value: "0" * 64)
    elif change == "hooks": layer._forward_pre_hooks[1] = lambda *values: None
    prepare = Mock(side_effect=AssertionError("uncovered state reached preparation"))
    monkeypatch.setattr(columns, "_engine", prepare)
    assert columns.try_columns_c24(layer, image) is None
    prepare.assert_not_called()


def test_foreign_device_does_not_inspect_parameters_or_prepare(monkeypatch):
    image = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    prepare = Mock(side_effect=AssertionError("CUDA reached C24 preparation"))
    monkeypatch.setattr(columns, "_engine", prepare)
    monkeypatch.setattr(columns, "_parameter_sha", Mock(side_effect=AssertionError("CUDA read parameters")))
    assert columns.try_columns_c24(object(), image) is None
    prepare.assert_not_called()
    assert not torch.cuda.is_initialized()


def test_failed_prepare_falls_back_once_but_math_error_propagates(qualified, monkeypatch):
    layer, image = qualified
    monkeypatch.setattr(columns, "_engine", Mock(side_effect=RuntimeError("provider gate")))
    old_result = object()
    fallback = Mock(return_value=old_result)
    monkeypatch.setattr(cpu_conv, "convolution_slabs", fallback)
    assert layer(image) is old_result
    fallback.assert_called_once()
    fallback.reset_mock()
    engine = SimpleNamespace(forward=Mock(side_effect=RuntimeError("copy began")))
    monkeypatch.setattr(columns, "_engine", lambda: engine)
    with pytest.raises(RuntimeError, match="copy began"):
        layer(image)
    fallback.assert_not_called()


def test_build_identity_preserves_C72_and_isolates_C24(monkeypatch, tmp_path):
    base_identity = {"source_sha256": "old", "flags": list(mature._FLAGS), "provider": "accepted"}
    base_key = "a" * 64
    compiler = {"command": "never-invoked"}
    monkeypatch.setattr(mature, "_build_inputs", lambda actual_torch: (tmp_path, tmp_path / "old.cpp", compiler, base_identity, base_key))
    _, source, actual_compiler, identity, key = builds._build_inputs(torch)
    assert base_identity["source_sha256"] == "old"
    assert identity["accepted_C72_build_key"] == base_key and key != base_key
    assert identity["source_sha256"] == builds._SOURCE_SHA
    assert source.name == "_columns_c24.cpp" and actual_compiler is compiler
    assert identity["flags"] == list(mature._FLAGS)


def test_C24_cache_does_not_rewrite_C72_and_reuses_own_artifact(monkeypatch, tmp_path):
    old = tmp_path / "old"
    old.mkdir(mode=0o700)
    sentinel = old / "accepted_c72.so"
    sentinel.write_bytes(b"accepted unchanged C72 mock bytes")
    cache = tmp_path / "new"
    cache.mkdir(mode=0o700)
    identity = {"specialization": "C24_depth32"}
    key = "b" * 64
    monkeypatch.setattr(builds, "_cache_root", lambda: cache)
    monkeypatch.setattr(builds, "_build_inputs", lambda t: (tmp_path, tmp_path / "source.cpp", {"command": "mock-only"}, identity, key))
    monkeypatch.setattr(builds, "_BUILDS", {})
    monkeypatch.setattr(builds, "_FAILED", {})
    calls = []
    def compile_stub(command):
        calls.append(command)
        Path(command[-1]).write_bytes(b"not a dynamic library; software contract only")
    monkeypatch.setattr(mature, "_compile", compile_stub)
    monkeypatch.setattr(columns, "_ColumnsC24", lambda *a, **kw: SimpleNamespace(_provider_still_matches=lambda: None))
    a = builds.build_artifact(torch)
    b = builds.build_artifact(torch)
    assert a == b and len(calls) == 1
    assert sentinel.read_bytes() == b"accepted unchanged C72 mock bytes"
    assert set(old.iterdir()) == {sentinel}
    assert mature._valid_artifact(cache / (key + ".so"), cache / (key + ".json"), key)
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in cache.iterdir())
    assert tuple(calls[0][1:1 + len(mature._FLAGS)]) == mature._FLAGS


def test_C24_cache_rejects_symlink_and_public_permissions(monkeypatch, tmp_path):
    destination = tmp_path / "actual"
    destination.mkdir(mode=0o700)
    selected = tmp_path / "symlink"
    selected.symlink_to(destination, target_is_directory=True)
    monkeypatch.setenv("FNIT_SYNTHSEG_C24_CPU_CACHE", str(selected))
    with pytest.raises(RuntimeError, match="owned private"):
        builds._cache_root()
    monkeypatch.setenv("FNIT_SYNTHSEG_C24_CPU_CACHE", str(destination))
    os.chmod(destination, 0o755)
    with pytest.raises(RuntimeError, match="owned private"):
        builds._cache_root()
