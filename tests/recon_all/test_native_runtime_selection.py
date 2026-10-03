"""能力、线程预算与失败策略的 CPU 回归；不是影像 benchmark。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import json
import tempfile
import unittest


path = Path(__file__).resolve().parents[2] / "src/fnit/recon_all/native_runtime_selection.py"
spec = importlib.util.spec_from_file_location("fnit_native_selection_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeSelectionTests(unittest.TestCase):
    def test_original_does_not_probe_and_keeps_pial(self):
        with patch.object(module.subprocess, "run") as run:
            result = module.select_native_optimizations("/tmp/fnit-native", threads=4, mode="original")
        run.assert_not_called()
        self.assertEqual(result["white_binary"], result["pial_binary"])
        self.assertEqual(result["em_backend"], "original")

    def test_old_or_partial_capability_cannot_enable_cache(self):
        for capability in (None, {}, {"version": 2, "fnit_gca_cached_search": True},
                           {"version": 2, "fnit_gca_cached_search": True,
                            "full_native_em": True, "reduction": "fast"}):
            with self.subTest(capability=capability), patch.object(module.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=json.dumps(capability))):
                result = module.select_native_optimizations("/tmp/fnit-native", threads=4)
            self.assertEqual(result["em_backend"], "original")

    def test_reproducible_cache_requires_validated_thread_budget(self):
        capability = {"version": 2, "fnit_gca_cached_search": True,
                      "full_native_em": True, "reduction": "upstream_ROMP_partials"}
        for threads, expected in ((4, "cpu_cached"), (2, "original"), (8, "original")):
            with self.subTest(threads=threads), patch.object(module.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=json.dumps(capability))):
                result = module.select_native_optimizations("/tmp/fnit-native", threads=threads)
            self.assertEqual(result["em_backend"], expected)

    def test_white_optimization_never_replaces_pial(self):
        capability = {"schema_version": 1,
                      "upstream_commit": "d932c45b7941662ea380a05efef580568b98d41a",
                      "features": ["skip-unconsumed-repulse-face-table"]}
        with tempfile.TemporaryDirectory() as temporary:
            fast = Path(temporary) / "mris_place_surface_white_fast"
            fast.touch()
            results = [SimpleNamespace(returncode=1, stdout=""),
                       SimpleNamespace(returncode=0, stdout=json.dumps(capability))]
            with patch.object(module.subprocess, "run", side_effect=results):
                result = module.select_native_optimizations(temporary, threads=4)
            self.assertEqual(result["white_binary"], str(fast))
            self.assertEqual(result["pial_binary"], str(Path(temporary) / "mris_place_surface"))

    def test_bad_installed_white_capability_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "mris_place_surface_white_fast").touch()
            results = [SimpleNamespace(returncode=1, stdout=""),
                       SimpleNamespace(returncode=0, stdout='{"features": []}')]
            with patch.object(module.subprocess, "run", side_effect=results):
                with self.assertRaisesRegex(ValueError, "capability mismatch"):
                    module.select_native_optimizations(temporary, threads=4)


if __name__ == "__main__":
    unittest.main()
