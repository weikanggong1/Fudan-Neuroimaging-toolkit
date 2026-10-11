"""GA选择、完整门与发布合同；小文件夹具不是MRI benchmark。"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from fnit.recon_all.batch import run_recon_all_python_batch
from fnit.recon_all.native_free import (
    _prepare_native_topology, _run_and_publish_genetic_topology,
    main, run_recon_all_python,
)
from fnit.recon_all.python_profile import python_profile_options


class PythonGeneticProfileTest(unittest.TestCase):
    def test_opt_in_changes_only_topology_and_preserves_original_program_exclusion(self):
        baseline = python_profile_options(device="cuda:1", threads=4)
        candidate = python_profile_options(device="cuda:1", threads=4,
            topology_backend="python-ga")
        self.assertEqual(candidate, dict(baseline, topology_backend="python-ga"))
        self.assertEqual(len(candidate), 22)
        self.assertEqual(candidate['input_execution'], 'serial')
        self.assertNotIn("native", candidate.values())
        for invalid in ("native", "ga", None, True, ["python-ga"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                python_profile_options(topology_backend=invalid)

    def test_public_api_does_not_overwrite_explicit_genetic_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "subject").mkdir()
            with patch("fnit.recon_all.native_free._run_recon_all_python",
                       return_value={"status": "complete", "total_seconds": 0.0}) as pipeline:
                run_recon_all_python(t1="input", subject_dir=Path(folder) / "subject",
                    weights_dir="weights", assets_dir="assets", device="cuda:1",
                    threads=4, backend="python", topology_backend="python-ga")
        for key, value in python_profile_options(device="cuda:1", threads=4,
                topology_backend="python-ga").items():
            self.assertEqual(pipeline.call_args.kwargs.get(key, "in-process"), value)

    def test_batch_and_real_cli_parser_preserve_genetic_recipe(self):
        # Resolve dependency CPU-feature probes before mocking the FNIT child
        # dispatcher; NumPy's first import may itself invoke lscpu.
        from fnit.recon_all.input_n4_chain import validate_n4_execution
        validate_n4_execution(n4_backend="simpleitk", n4_execution="in-process", device="cuda:1")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights, assets = root / "weights", root / "assets"
            weights.mkdir(); assets.mkdir()
            source = root / "input.nii.gz"
            source.write_bytes(b"interface fixture only")
            captured = []
            def pipeline(**kwargs):
                captured.append(kwargs)
                Path(kwargs["subject_dir"]).mkdir()
                return {"status": "complete", "total_seconds": 0.0}
            def child(command, **kwargs):
                self.assertEqual(command[command.index("--topology-backend") + 1], "python-ga")
                with patch("fnit.recon_all.native_free._run_recon_all_python", side_effect=pipeline), \
                        redirect_stdout(io.StringIO()):
                    main(command[3:])
                return type("Completed", (), {"returncode": 0})()
            with patch("fnit.recon_all.batch.subprocess.run", side_effect=child):
                reports = run_recon_all_python_batch(
                    jobs=[{"t1": source, "subject_dir": root / "subject"}],
                    weights_dir=weights, assets_dir=assets, devices=("cuda:1",),
                    threads=4, backend="python", topology_backend="python-ga")
        self.assertEqual(len(reports), 1)
        self.assertEqual(captured[0]["topology_backend"], "python-ga")
        self.assertEqual(captured[0]["hemisphere_workers"], 2)

    def _fixture_adapter(self, *, tamper=None):
        def adapter(**kwargs):
            self.assertEqual(kwargs["threads"], 2)
            self.assertEqual(kwargs["seed"], 1234)
            self.assertEqual(kwargs["niters"], -1)
            self.assertTrue(kwargs["final_orig"])
            self.assertIs(kwargs["face_detection_cache"], True)
            self.assertEqual(kwargs["blocker_scan_backend"], "numba")
            self.assertEqual(kwargs["brain"].name, "brain.mgz")
            self.assertEqual(kwargs["wm"].name, "wm.mgz")
            output = Path(kwargs["output_directory"])
            output.mkdir()
            outputs, hashes = {}, {}
            for name, data in (("premesh", b"own-original-slot"),
                               ("defect_labels", b"raw-nofix-domain"),
                               ("orig", b"own-remesh-and-cleanup")):
                path = output / name
                path.write_bytes(data)
                outputs[name] = str(path)
                hashes[name] = hashlib.sha256(data).hexdigest()
            report = output / "report.json"
            parameters = {"face_detection_cache": True, "blocker_scan_backend": "numba"}
            report.write_text(json.dumps({"status": "complete", "final_orig": True,
                "parameters": parameters,
                "quality": {"status": "passed"}, "output_sha256": hashes}))
            completion = output / "COMPLETE.json"
            completion.write_text(json.dumps({"status": "complete", "final_orig": True,
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "output_sha256": hashes}))
            result = {"status": "complete", "final_orig": True,
                "parameters": parameters,
                "quality": {"status": "passed"},
                "outputs": outputs, "output_sha256": hashes,
                "report_path": str(report), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "completion_path": str(completion)}
            if tamper == "artifact":
                Path(outputs["orig"]).write_bytes(b"changed")
            elif tamper == "report":
                report.write_text("changed")
            elif tamper == "quality":
                result["quality"]["status"] = "failed"
            elif tamper == "cache_policy":
                result["parameters"]["face_detection_cache"] = False
            elif tamper == "blocker_policy":
                result["parameters"]["blocker_scan_backend"] = "python"
            elif tamper == "blocker_policy_missing":
                result["parameters"].pop("blocker_scan_backend")
            elif tamper == "recorded_policy":
                content = json.loads(report.read_text())
                content["parameters"]["face_detection_cache"] = False
                report.write_text(json.dumps(content))
                result["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
            elif tamper == "completion":
                completion.write_text(json.dumps({"status": "failed"}))
            elif tamper == "completion_binding":
                receipt = json.loads(completion.read_text())
                receipt["report_sha256"] = "0" * 64
                completion.write_text(json.dumps(receipt))
            elif tamper == "scope":
                path = output.parent / "outside-orig"
                path.write_bytes(b"own-remesh-and-cleanup")
                outputs["orig"] = str(path)
            return result
        module = ModuleType("fnit.recon_all.topology_genetic_files")
        module.run_genetic_topology_files = adapter
        return module

    def test_publication_requires_complete_quality_and_all_artifact_hashes(self):
        for tamper in (None, "artifact", "report", "quality", "cache_policy", "recorded_policy", "blocker_policy", "blocker_policy_missing", "completion", "completion_binding", "scope"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as folder:
                subject = Path(folder)
                for name in ("surf", "scripts", "mri"):
                    (subject / name).mkdir()
                nofix = subject / "surf/lh.orig.nofix"
                nofix.write_bytes(b"immutable-input")
                with patch.dict(sys.modules, {
                    "fnit.recon_all.topology_genetic_files": self._fixture_adapter(tamper=tamper)
                }):
                    if tamper is None:
                        _run_and_publish_genetic_topology(subject=subject, hemi="lh",
                            threads=2, remesh_scalar_storage="python")
                        self.assertEqual((subject / "surf/lh.orig").read_bytes(), b"own-remesh-and-cleanup")
                        self.assertEqual((subject / "surf/lh.defect_labels").read_bytes(), b"raw-nofix-domain")
                        publication = json.loads((subject / "scripts/lh.topology.genetic.publish.json").read_text())
                        self.assertIs(publication["face_detection_cache"], True)
                        self.assertEqual(publication["blocker_scan_backend"], "numba")
                    else:
                        with self.assertRaises(RuntimeError):
                            _run_and_publish_genetic_topology(subject=subject, hemi="lh",
                                threads=2, remesh_scalar_storage="python")
                        for name in ("orig", "orig.premesh", "defect_labels"):
                            self.assertFalse((subject / "surf" / f"lh.{name}").exists())
                self.assertEqual(nofix.read_bytes(), b"immutable-input")

    def test_prepare_uses_worker_budget_and_does_not_repeat_complete_orig(self):
        with tempfile.TemporaryDirectory() as folder:
            subject = Path(folder)
            (subject / "surf").mkdir()
            (subject / "surf/lh.orig.nofix").write_bytes(b"own-input")
            def copy(source, output, **kwargs):
                shutil.copyfile(source, output)
            def inflate(**kwargs):
                copy(kwargs["input_surface"], kwargs["inflated_output"])
                return {"total_seconds_including_io": 0.1}
            with patch("fnit.recon_all.smooth_surface_python.smooth_surface", side_effect=copy), \
                 patch("fnit.recon_all.inflate_standard_run.run_standard_inflate", side_effect=inflate), \
                 patch("fnit.recon_all.sphere_quick_python.write_quick_sphere", side_effect=copy), \
                 patch("fnit.recon_all.native_free._run_and_publish_genetic_topology", return_value={
                     "seconds": {"remesh": 0.2, "original_intersection_cleanup": 0.3}}) as adapter, \
                 patch("fnit.recon_all.mris_remesh_python.remesh_surface", side_effect=AssertionError("duplicate remesh")), \
                 patch("fnit.recon_all.mris_remove_intersection_python.remove_intersection_surface", side_effect=AssertionError("duplicate cleanup")), \
                 patch("fnit.recon_all.native_free.subprocess.run", side_effect=AssertionError("original stage called")):
                result = _prepare_native_topology(None, subject, "lh", subject, "cpu", None, None,
                    python_auxiliary=True, topology_backend="python-ga", threads=2,
                    remesh_scalar_storage="python")
            self.assertEqual(adapter.call_args.kwargs["threads"], 2)
            self.assertEqual(result[1], 0.0)
            self.assertEqual(result[-2:], (0.2, 0.3))

    def test_invalid_worker_budget_fails_before_any_stage(self):
        for invalid in (0, -1, True, 2.5):
            with self.subTest(invalid=invalid), patch("fnit.recon_all.smooth_surface_python.smooth_surface") as smooth:
                with self.assertRaises(ValueError):
                    _prepare_native_topology(None, Path("unused"), "lh", Path("assets"),
                        "cpu", None, None, topology_backend="python-ga", threads=invalid)
                smooth.assert_not_called()


if __name__ == "__main__":
    unittest.main()
