"""线程预算的状态恢复与子进程环境回归；无需 GPU 或额外测试依赖。"""

from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch


class ThreadBudgetTests(unittest.TestCase):
    def setUp(self):
        self.state = {"torch": 8, "numba": 12}
        self.calls = []

        def setter(name):
            def set_threads(value):
                self.calls.append((name, value))
                self.state[name] = value
            return set_threads

        fake_torch = types.SimpleNamespace(
            get_num_threads=lambda: self.state["torch"],
            set_num_threads=setter("torch"), get_num_interop_threads=lambda: 16)
        fake_numba = types.SimpleNamespace(
            config=types.SimpleNamespace(NUMBA_NUM_THREADS=16),
            get_num_threads=lambda: self.state["numba"],
            set_num_threads=setter("numba"), threading_layer=lambda: "omp")
        path = Path(__file__).resolve().parents[2] / "src/fnit/recon_all/thread_budget.py"
        spec = spec_from_file_location("thread_budget_under_test", path)
        self.module = module_from_spec(spec)
        with patch.dict("sys.modules", {"torch": fake_torch, "numba": fake_numba}):
            spec.loader.exec_module(self.module)

    def test_records_effective_mask_and_restores_caller(self):
        before_env = dict(os.environ)
        with self.module.thread_budget(threads=4) as report:
            self.assertEqual(self.state, {"torch": 4, "numba": 4})
            self.assertEqual(report["numba"]["initial_capacity"], 16)
            self.assertEqual(report["torch"]["interop_unchanged"], 16)
            self.assertEqual(report["numba"]["effective"], 4)
            self.assertFalse(report["restoration_complete"])
        self.assertEqual(self.state, {"torch": 8, "numba": 12})
        self.assertTrue(report["restoration_complete"])
        self.assertEqual(report["numba"]["restored"], 12)
        self.assertEqual(dict(os.environ), before_env)

    def test_exception_restores_settings_and_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            with self.module.thread_budget(threads=4) as report:
                raise RuntimeError("stage failed")
        self.assertTrue(report["restoration_complete"])
        self.assertEqual(self.state, {"torch": 8, "numba": 12})

    def test_rejects_above_capacity_before_mutation(self):
        with self.assertRaisesRegex(ValueError, "initial capacity 16"):
            with self.module.thread_budget(threads=17):
                self.fail("invalid scope was entered")
        self.assertEqual(self.calls, [])

    def test_positive_integer_only(self):
        for value in (0, -1, True, False, 4.0, "4", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with self.module.thread_budget(threads=value):
                    pass
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.native_thread_environment(threads=value)
        self.assertEqual(self.calls, [])

    def test_nested_scopes_restore_each_boundary(self):
        with self.module.thread_budget(threads=4) as outer:
            with self.module.thread_budget(threads=2) as inner:
                self.assertEqual(self.state, {"torch": 2, "numba": 2})
            self.assertEqual(self.state, {"torch": 4, "numba": 4})
            self.assertTrue(inner["restoration_complete"])
            self.assertEqual(inner["torch"]["restored"], 4)
        self.assertTrue(outer["restoration_complete"])
        self.assertEqual(self.state, {"torch": 8, "numba": 12})

    def test_numba_set_failure_also_restores_torch(self):
        original_setter = self.module.numba.set_num_threads

        def reject_four(value):
            if value == 4:
                raise RuntimeError("Numba backend error")
            original_setter(value)

        self.module.numba.set_num_threads = reject_four
        with self.assertRaisesRegex(RuntimeError, "Numba backend error"):
            with self.module.thread_budget(threads=4):
                pass
        self.assertEqual(self.state, {"torch": 8, "numba": 12})

    def test_detects_ignored_setting_and_restores(self):
        self.module.numba.set_num_threads = lambda value: None
        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            with self.module.thread_budget(threads=4):
                pass
        self.assertEqual(self.state, {"torch": 8, "numba": 12})

    def test_environment_copy_has_explicit_scope_and_preserves_other_values(self):
        source = {"OMP_NUM_THREADS": "128", "NUMBA_NUM_THREADS": "128",
                  "PYTORCH_NO_CUDA_MEMORY_CACHING": "1", "CUDA_VISIBLE_DEVICES": "GPU-example"}
        before = source.copy()
        child, report = self.module.native_thread_environment(threads=4, environ=source)
        self.assertEqual(source, before)
        for name in self.module.NATIVE_THREAD_VARIABLES:
            self.assertEqual(child[name], "4")
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "GPU-example")
        self.assertEqual(child["PYTORCH_NO_CUDA_MEMORY_CACHING"], "1")
        self.assertEqual(report["environment_before"]["OMP_NUM_THREADS"], "128")
        self.assertFalse(report["active_worker_counts_verified"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", report["environment_after"])

    def test_default_environment_does_not_modify_parent(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "32"}):
            before = dict(os.environ)
            child, report = self.module.native_thread_environment(threads=4)
            self.assertEqual(dict(os.environ), before)
            self.assertEqual(child["OMP_NUM_THREADS"], "4")
            self.assertEqual(report["environment_before"]["OMP_NUM_THREADS"], "32")


if __name__ == "__main__":
    unittest.main()
