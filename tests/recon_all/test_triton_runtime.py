"""Test compiler selection without initializing CUDA or compiling a kernel."""

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / "src/fnit/recon_all/triton_runtime.py"


def _fresh_runtime():
    # Each test receives the same clean process-local selection state as a new CLI.
    spec = importlib.util.spec_from_file_location("fnit_triton_runtime_test", SOURCE)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def _compiler_module(*, path: Path, release: str = "11.8"):
    """Provide the two cached Triton queries used by configure_triton_compiler."""
    @lru_cache(maxsize=None)
    def path_to_binary(name):
        assert name == "ptxas"
        return str(path), release

    @lru_cache(maxsize=None)
    def get_ptxas_version():
        return "ptxas release 11.8, V11.8.89"

    return SimpleNamespace(_path_to_binary=path_to_binary,
                           get_ptxas_version=get_ptxas_version)


class TritonRuntimeTests(unittest.TestCase):
    def _fake_torch(self):
        # There is deliberately no cuda attribute: any CUDA runtime access fails.
        return SimpleNamespace(version=SimpleNamespace(cuda="11.8"))

    def _ptxas(self, directory: Path, name: str = "ptxas") -> Path:
        path = directory / name
        path.write_bytes(b"test compiler, never executed\n")
        path.chmod(0o755)
        return path

    def test_cpu_is_noop_even_with_invalid_explicit_compiler(self):
        runtime = _fresh_runtime()
        with mock.patch.dict(os.environ, {"TRITON_PTXAS_PATH": "unchanged"}), \
             mock.patch.object(runtime, "_probe", side_effect=AssertionError("CPU probed ptxas")), \
             mock.patch.object(runtime.importlib, "import_module",
                               side_effect=AssertionError("CPU imported Triton")):
            self.assertEqual(runtime.inspect_triton_compiler(
                device="cpu", compiler_path="/missing/ptxas"),
                {"status": "not_required", "device_type": "cpu"})
            self.assertEqual(runtime.configure_triton_compiler(
                device="cpu", compiler_path="/missing/ptxas"),
                {"status": "not_required", "device_type": "cpu"})
            self.assertEqual(os.environ["TRITON_PTXAS_PATH"], "unchanged")
            self.assertIsNone(runtime._last)

    def test_invalid_explicit_path_does_not_fall_back_to_declared_environment(self):
        runtime = _fresh_runtime()
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "bin").mkdir()
            self._ptxas(prefix / "bin")
            missing = prefix / "missing-ptxas"
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}), \
                 mock.patch.object(runtime.sys, "prefix", str(prefix)), \
                 mock.patch.object(runtime.subprocess, "run",
                                   return_value=SimpleNamespace(stdout="ptxas release 11.8, V11.8.89")) as run, \
                 mock.patch.dict(os.environ):
                os.environ.pop("TRITON_PTXAS_PATH", None)
                for select in (runtime.inspect_triton_compiler,
                               runtime.configure_triton_compiler):
                    with self.assertRaisesRegex(ValueError, "不存在或不可执行"):
                        select(device="cuda:0", compiler_path=missing)
                run.assert_not_called()
                self.assertNotIn("TRITON_PTXAS_PATH", os.environ)
                self.assertIsNone(runtime._last)

    def test_repeated_selection_is_idempotent_and_preserves_first_source(self):
        runtime = _fresh_runtime()
        with tempfile.TemporaryDirectory() as directory:
            candidate = self._ptxas(Path(directory))
            compiler = _compiler_module(path=candidate)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}), \
                 mock.patch.object(runtime.subprocess, "run",
                                   return_value=SimpleNamespace(stdout="ptxas release 11.8, V11.8.89")), \
                 mock.patch.object(runtime.importlib, "import_module",
                                   return_value=compiler) as imported, \
                 mock.patch.dict(os.environ):
                os.environ.pop("TRITON_PTXAS_PATH", None)
                first = runtime.configure_triton_compiler(
                    device="cuda:0", compiler_path=candidate)
                second = runtime.configure_triton_compiler(
                    device="cuda:0", compiler_path=candidate)
                self.assertTrue(first["configured"])
                self.assertFalse(first["idempotent_reuse"])
                self.assertTrue(second["idempotent_reuse"])
                self.assertFalse(first["preexisting_path_cache"])
                self.assertTrue(second["preexisting_path_cache"])
                self.assertEqual(first["selection_source"], "explicit_parameter")
                self.assertEqual(second["selection_source"], "explicit_parameter")
                self.assertEqual(first["compiler_sha256"], second["compiler_sha256"])
                self.assertEqual(os.environ["TRITON_PTXAS_PATH"], str(candidate.resolve()))
                self.assertEqual(compiler._path_to_binary.cache_info().currsize, 1)
                imported.assert_called_with("triton.backends.nvidia.compiler")

    def test_cached_different_compiler_path_fails_without_changing_environment(self):
        runtime = _fresh_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._ptxas(root, "candidate-ptxas")
            cached = self._ptxas(root, "cached-ptxas")
            compiler = _compiler_module(path=cached)
            compiler._path_to_binary("ptxas")  # Simulate an earlier Triton JIT.
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}), \
                 mock.patch.object(runtime.subprocess, "run",
                                   return_value=SimpleNamespace(stdout="ptxas release 11.8, V11.8.89")), \
                 mock.patch.object(runtime.importlib, "import_module",
                                   return_value=compiler), \
                 mock.patch.dict(os.environ, {"TRITON_PTXAS_PATH": "original-setting"}):
                with self.assertRaisesRegex(RuntimeError, "已缓存其他ptxas"):
                    runtime.configure_triton_compiler(
                        device="cuda:0", compiler_path=candidate)
                self.assertEqual(os.environ["TRITON_PTXAS_PATH"], "original-setting")
                self.assertEqual(compiler._path_to_binary.cache_info().currsize, 1)
                self.assertIsNone(runtime._last)


if __name__ == "__main__":
    unittest.main()
