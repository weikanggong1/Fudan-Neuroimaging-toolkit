"""整例比较driver的资源锁、失败归档与来源绑定；不计算MRI指标。"""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[2] / "validation/recon_all/optimizations/20261002_parallel/compare_whole_cases.py"
SPEC = importlib.util.spec_from_file_location("whole_comparison_driver", PATH)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class WholeComparisonDriverTests(unittest.TestCase):
    def test_common_lock_blocks_an_independent_file_description(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "common.lock"
            with DRIVER.common_lock(path):
                with path.open("a") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with path.open("a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_failed_subprocess_is_recorded_before_exception(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            commands = []
            with self.assertRaises(RuntimeError):
                DRIVER.execute([sys.executable, "-c", "raise SystemExit(3)"], folder / "commands.log", commands)
            recorded = DRIVER.read(folder / "commands.json")
            self.assertEqual(recorded[0]["exit_code"], 3)
            self.assertEqual(recorded[0]["command"], commands[0]["command"])
            self.assertGreaterEqual(recorded[0]["seconds"], 0)

    def test_interop_configuration_is_once_for_two_cases(self):
        calls = []
        numba = types.SimpleNamespace(set_num_threads=lambda n: calls.append(("numba", n)))
        torch = types.SimpleNamespace(set_num_threads=lambda n: calls.append(("torch", n)),
                                      set_num_interop_threads=lambda n: calls.append(("interop", n)))
        DRIVER.configure_runtime.cache_clear()
        with patch.dict(sys.modules, {"numba": numba, "torch": torch}):
            DRIVER.configure_runtime()
            DRIVER.configure_runtime()
        DRIVER.configure_runtime.cache_clear()
        self.assertEqual(calls, [("numba", 4), ("torch", 4), ("interop", 1)])

    def test_failed_completion_cannot_be_treated_as_ready(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            config = folder / "config.json"
            DRIVER.write(config, {"diagnostic_root": str(folder)})
            case = {"id": "sub01", "baseline_config": str(config), "candidate_config": str(config)}
            self.assertFalse(DRIVER.ready(case))
            DRIVER.write(folder / "completion.json", {"execution_status": "failed", "exit_code": 1})
            with self.assertRaises(RuntimeError):
                DRIVER.ready(case)

    def test_receipt_rejects_source_change_after_launch(self):
        # 仅构造编排所需metadata/文本文件，不替代真实影像benchmark。
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            subject, diagnostic = folder / "subject", folder / "launch"
            source = folder / "code/src/fnit/recon_all"
            subject.mkdir(); diagnostic.mkdir(); source.mkdir(parents=True)
            t1 = folder / "input.txt"; t1.write_text("unit-test provenance only")
            for name in ("native_free.py", "surface_stats_cache.py", "anatomical_stats_file.py"):
                (source / name).write_text("# source binding test\n")
            config = {"input": str(t1), "output": str(subject), "diagnostic_root": str(diagnostic),
                      "code_root": str(folder / "code"), "code_commit": "commit", "source_archive_sha256": "archive"}
            config_path = folder / "config.json"; DRIVER.write(config_path, config)
            DRIVER.write(diagnostic / "launch.json", {"config_sha256": DRIVER.digest(config_path),
                "code_commit": "commit", "source_archive_sha256": "archive",
                "candidate_native_free_sha256": DRIVER.digest(source / "native_free.py")})
            DRIVER.write(diagnostic / "completion.json", {"code_commit": "commit", "source_archive_sha256": "archive",
                "execution_status": "complete", "exit_code": 0, "pipeline_total_seconds": 1.0, "command_seconds": 2.0})
            DRIVER.write(subject / "fnit-native-free-run.json", {"status": "complete", "threads": 4,
                "output_validation": {"expected": 138, "present": 138, "missing": []}})
            case = {"baseline_config": str(config_path), "candidate_config": str(config_path),
                    "official": "official", "official_code_version": "archived"}
            self.assertEqual(DRIVER.receipts(case)["baseline"]["code_commit"], "commit")
            (source / "native_free.py").write_text("# changed after execution\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                DRIVER.receipts(case)

    def test_partial_quality_cannot_report_zero_change(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            metric = {"per_region": {"lh/region": {"absolute_error": 1.0}}}
            regions = {"aparc_68": {"ThickAvg": metric}, "aseg": metric, "wmparc": metric}
            for kind in ("baseline", "candidate"):
                DRIVER.write(folder / f"region_{kind}_vs_official.json", regions)
                DRIVER.write(folder / f"no_th3_{kind}_vs_official.json", {"atlases": {"aparc": metric}})
                DRIVER.write(folder / f"dice_{kind}_vs_official.json", {"files": {"aseg.mgz": {"per_label": {"2": {"dice": .9}}}}})
                quality = {"hemispheres": {}}
                for hemi in ("lh", "rh"):
                    quality["hemispheres"][hemi] = {
                        "topology": {}, "vertex_links": {},
                        "white_pial_crossings": {"status": "complete" if kind == "baseline" else "incomplete_timeout", "proper_transverse_pairs": 0},
                        "sphere_orientation": {stage: {"negative_faces": 0, "zero_area_faces": 0, "nonfinite_areas": 0} for stage in ("sphere", "sphere.reg")}}
                (folder / f"quality_{kind}").mkdir()
                DRIVER.write(folder / f"quality_{kind}/report.json", quality)
            DRIVER.write(folder / "execution_binding.json", {kind: {"mesh_validation": {}} for kind in ("baseline", "candidate")})
            DRIVER.write(folder / "geometry_candidate_vs_baseline.json", {"surfaces": {}})
            strict = {"baseline_vs_official": {"failed_files": ["inherited"]},
                      "candidate_vs_official": {"failed_files": ["inherited", "new"]},
                      "candidate_vs_baseline": {"failed_files": ["changed"]}}
            result = DRIVER.observations(folder, strict)
            self.assertEqual(result["inherited_strict_official_failures"], ["inherited"])
            self.assertEqual(result["new_strict_official_failures"], ["new"])
            crossing = [row for row in result["quality_count_changes"] if row["section"] == "white_pial_crossings"]
            self.assertEqual(len(crossing), 2)
            self.assertTrue(all(row["difference"] is None and not row["coverage_complete"] for row in crossing))

    def test_distance_cache_requires_identical_shape_dtype_and_all_bytes(self):
        # 标准库数组协议stub，仅验证缓存控制；不替代真实几何回归。
        class Array:
            def __init__(self, data, shape=(1, 3), dtype="<f8"):
                self.data, self.shape, self.dtype = data, shape, types.SimpleNamespace(str=dtype)
            def tobytes(self, order):
                self.asserted_order = order
                return self.data
        class Distances:
            def setflags(self, *, write):
                self.writeable = write
        calls = []
        def operation(source, target, faces):
            calls.append((source, target, faces))
            return Distances()
        cache = DRIVER.ExactDistanceCache()
        source, target, faces = Array(b"source"), Array(b"target"), Array(b"faces", dtype="<i4")
        first = cache.evaluate(operation, source, target, faces)
        second = cache.evaluate(operation, Array(b"source"), Array(b"target"), Array(b"faces", dtype="<i4"))
        self.assertIs(first, second)
        self.assertFalse(first.writeable)
        for changed in (Array(b"changed"), Array(b"source", shape=(3, 1)), Array(b"source", dtype=">f8")):
            self.assertIsNot(cache.evaluate(operation, changed, target, faces), first)
        cache.evaluate(operation, source, Array(b"changed target"), faces)
        cache.evaluate(operation, source, target, Array(b"changed faces", dtype="<i4"))
        self.assertEqual(len(calls), 6)
        self.assertEqual(cache.report()["cache_hits"], 1)
        cache.clear()
        self.assertEqual(cache.report()["cached_result_count"], 0)


if __name__ == "__main__":
    unittest.main()
