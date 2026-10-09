"""仅验证诊断的状态、SHA与文件存在性契约；不替代真实T1指标。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("diagnose_failed_outputs.py")
SPEC = importlib.util.spec_from_file_location("_failed_output_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FailedProducerContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.subject = self.root / "subject"
        (self.subject / "surf").mkdir(parents=True)
        (self.subject / "scripts").mkdir()
        source = self.root / "src/fnit/recon_all/native_free.py"
        source.parent.mkdir(parents=True)
        source.write_text("# provenance only, no scientific data\n")
        self.raw = self.root / "raw.fixture"
        self.raw.write_text("nonmedical provenance fixture\n")
        self.benchmark = self.root / "benchmark.json"
        self.report = {"execution": "failed", "exit_code": 1, "input_sha256": MODULE.sha256(self.raw),
                       "source_sha256": {"recon_all/native_free.py": MODULE.sha256(source)}}
        self.runtime = {"input": str(self.raw), "status": "failed", "failed_stage": "mni_mesh_parallel"}
        self.mesh = {"status": "failed", "mesh_validation": {"status": "failed"}, "mesh_inputs_sha256": {}}
        for hemi in ("lh", "rh"):
            for name in ("orig", "white", "pial", "sphere.reg"):
                path = self.subject / "surf" / f"{hemi}.{name}"
                path.write_text("nonmedical surface binding fixture\n")
                self.mesh["mesh_inputs_sha256"][path.name] = MODULE.sha256(path)
        self.manifest = {"cases": {"public-case": {"input_sha256": MODULE.sha256(self.raw)}}}
        self.save()

    def save(self):
        self.benchmark.write_text(json.dumps(self.report))
        (self.subject / "fnit-native-free-run.json").write_text(json.dumps(self.runtime))
        (self.subject / "scripts/mni-mesh-parallel.json").write_text(json.dumps(self.mesh))

    def validate(self):
        return MODULE.validate_failed_producer(benchmark_path=self.benchmark, source_root=self.root,
                                               case="public-case", reference_manifest=self.manifest)

    def test_preserved_failure_remains_failed_and_readonly(self):
        paths = (self.benchmark, self.subject / "fnit-native-free-run.json",
                 self.subject / "scripts/mni-mesh-parallel.json")
        before = [path.read_bytes() for path in paths]
        report, runtime, mesh = self.validate()
        self.assertEqual((report["execution"], runtime["status"], mesh["status"]), ("failed",) * 3)
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_rejects_complete_running_zero_and_boolean_exit_codes(self):
        for execution, code in (("complete", 0), ("running", None), ("failed", 0), ("failed", True)):
            with self.subTest(execution=execution, code=code):
                self.report.update(execution=execution, exit_code=code)
                self.save()
                with self.assertRaisesRegex(ValueError, "requires preserved"):
                    self.validate()

    def test_rejects_other_stage_or_changed_runtime_state(self):
        self.runtime["failed_stage"] = "some_other_stage"
        self.save()
        with self.assertRaisesRegex(ValueError, "mni_mesh_parallel"):
            self.validate()
        self.runtime.update(status="complete", failed_stage="mni_mesh_parallel")
        self.save()
        with self.assertRaisesRegex(ValueError, "requires preserved"):
            self.validate()

    def test_rejects_source_input_and_mesh_mutation(self):
        for path, message in ((self.root / "src/fnit/recon_all/native_free.py", "source changed"),
                              (self.raw, "runtime original T1"),
                              (self.subject / "surf/rh.white", "mesh changed")):
            with self.subTest(path=path.name):
                data = path.read_bytes()
                path.write_bytes(data + b"changed")
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                path.write_bytes(data)

    def test_rejects_bad_reference_and_incomplete_source_mesh_bindings(self):
        self.manifest["cases"]["public-case"]["input_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "reference original"):
            self.validate()
        self.manifest["cases"]["public-case"]["input_sha256"] = self.report["input_sha256"]
        self.report["source_sha256"] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, "source binding"):
            self.validate()
        self.report["source_sha256"] = {"recon_all/native_free.py": MODULE.sha256(self.root / "src/fnit/recon_all/native_free.py")}
        self.mesh["mesh_inputs_sha256"].pop("rh.white")
        self.save()
        with self.assertRaisesRegex(ValueError, "mesh input binding"):
            self.validate()

    def test_rejects_source_path_traversal(self):
        self.report["source_sha256"]["../escape.py"] = "bad"
        self.save()
        with self.assertRaisesRegex(ValueError, "relative path"):
            self.validate()

    def test_existing_complete_evaluator_still_rejects_failed_producer(self):
        root = SCRIPT.parents[4]
        path = Path(os.environ.get("FNIT_COMPLETE_VALIDATOR", root / "tools/evaluate_recon_torch_run.py"))
        complete = MODULE.load_module(path=path, name="_unchanged_complete_guard")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            complete.validate_candidate(benchmark_path=self.benchmark, source_root=self.root,
                                        case="public-case", reference_manifest=self.manifest)


class InventoryAndBudgetContracts(unittest.TestCase):
    def test_reference_qc_cache_is_new_and_child_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_cache, parent_cache = root / "reference_cache", str(root / "parent_cache")
            command = [sys.executable, "-c", "import os; print(os.environ['NUMBA_CACHE_DIR'])"]
            commands = []
            write = lambda path, value: path.write_text(json.dumps(value))
            with patch.dict(os.environ, {"NUMBA_CACHE_DIR": parent_cache}):
                MODULE.execute_quality_reference(command=command, log=root / "commands.log",
                    commands=commands, cache_directory=child_cache, write=write)
                self.assertEqual(os.environ["NUMBA_CACHE_DIR"], parent_cache)
                self.assertEqual((root / "commands.log").read_text().strip(), str(child_cache))
                self.assertTrue(commands[0]["parent_numba_cache_unchanged"])
                self.assertEqual(commands[0]["exit_code"], 0)
                with self.assertRaises(FileExistsError):
                    MODULE.execute_quality_reference(command=command, log=root / "commands.log",
                        commands=commands, cache_directory=child_cache, write=write)

    def test_reference_qc_failure_keeps_parent_environment_and_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, commands = Path(temporary), []
            command = [sys.executable, "-c", "raise SystemExit(7)"]
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "reference quality command failed"):
                    MODULE.execute_quality_reference(command=command, log=root / "commands.log",
                        commands=commands, cache_directory=root / "isolated_cache",
                        write=lambda path, value: path.write_text(json.dumps(value)))
                self.assertNotIn("NUMBA_CACHE_DIR", os.environ)
                self.assertEqual(json.loads((root / "commands.json").read_text())[0]["exit_code"], 7)

    def test_missing_output_is_existence_only_and_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = tuple(f"file_{index}" for index in range(138))
            (root / paths[0]).write_bytes(b"bound bytes")
            report = MODULE.inventory_outputs(subject=root, expected_paths=paths)
            self.assertEqual((report["expected"], report["present"], len(report["missing"])), (138, 1, 137))
            self.assertIn("producer execution and mesh gate remain failed", report["interpretation"])
            self.assertEqual(report["files"][paths[0]]["sha256"], MODULE.sha256(root / paths[0]))
            with self.assertRaisesRegex(ValueError, "fixed 138"):
                MODULE.inventory_outputs(subject=root, expected_paths=paths[:-1])
            with self.assertRaisesRegex(ValueError, "relative path"):
                MODULE.inventory_outputs(subject=root, expected_paths=("../escape", *paths[1:]))

    def test_resource_budget_and_actual_target_gpu_must_match(self):
        control = {key: "same" for key in ("input_sha256", "host", "cpu", "cpu_affinity", "device",
                                          "resource_sha256", "native_program_sha256", "source_sha256")}
        control.update(threads=4, environment={"CUDA_VISIBLE_DEVICES": "7"})
        candidate = dict(control)
        MODULE.require_same_resources(control=control, candidate=candidate, threads=4)
        candidate["resource_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "resource_sha256"):
            MODULE.require_same_resources(control=control, candidate=candidate, threads=4)
        candidate["resource_sha256"] = control["resource_sha256"]
        candidate["environment"] = {"CUDA_VISIBLE_DEVICES": "6"}
        with self.assertRaisesRegex(ValueError, "GPU visibility"):
            MODULE.require_same_resources(control=control, candidate=candidate, threads=4)


if __name__ == "__main__":
    unittest.main()
