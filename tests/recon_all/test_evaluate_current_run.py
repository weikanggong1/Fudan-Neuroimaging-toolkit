"""整例比较前拒绝失败执行、输入错配与改动后的冻结源码。"""
import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.expected_outputs import paths as expected_paths

PATH = Path(__file__).resolve().parents[2] / "tools/evaluate_recon_torch_run.py"
SPEC = importlib.util.spec_from_file_location("_evaluate_current_run", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_candidate_outputs(subject: Path) -> dict[str, str]:
    """建立真实138路径结构的微型合同夹具；不充当影像 benchmark。"""
    outputs = {}
    for relative in expected_paths():
        path = subject / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        outputs[relative] = str(path)
    return outputs


class EvaluationBindingTests(unittest.TestCase):
    @staticmethod
    def native_fixture(root):
        """构造源码绑定阶段收据，验证拒绝规则；不作为MRI数值benchmark。"""
        subject = root / "subject"; subject.mkdir()
        (subject / "surf").mkdir(); (subject / "scripts").mkdir()
        directory = root / "native-bin"; directory.mkdir()
        names = ("mris_fix_topology_fnit", "mris_remesh", "mris_remove_intersection",
                 "mris_inflate", "mris_place_surface", "mris_place_surface_white_fast",
                 "mrisp_paint", "mris_curvature_stats")
        expected = {}
        for name in names:
            binary = directory / name; binary.write_text("test-only program " + name)
            binary.chmod(0o755); expected[name] = MODULE.sha(binary)
        recipe = {"backend": "native", "threads": 4, "topology_backend": "native",
            "topology_finish_backend": "conda-cpp", "surface_placement_backend": "native",
            "n4_backend": "simpleitk", "n4_execution": "in-process",
            "wm_backend": "torch-optimized", "wm_execution": "isolated", "wm_edit_backend": "python",
            "defects_backend": "torch", "inflate_backend": "torch", "sphere_finish_backend": "torch",
            "annotation_gibbs_backend": "numba", "gca_inverse_backend": "torch",
            "gca_candidate_chunk": 4096, "gca_reuse_atlas_context": True, "gca_execution": "isolated",
            "fill_backend": "numba", "normalization_controls_backend": "torch",
            "normalization_initial_bias_backend": "torch", "mni_execution": "parallel-late",
            "hemisphere_execution": "groups", "topology_distance_backend": "serial",
            "topology_state_backend": "legacy", "remesh_scalar_storage": "numpy"}
        command = [sys.executable, "-X", "faulthandler", "-u", "-m",
                   "fnit.recon_all.native_free", str(root / "raw.txt"), str(subject)]
        for name, value in recipe.items():
            command += ["--" + name.replace("_", "-")]
            if name != "gca_reuse_atlas_context":
                command += [str(value)]
        command += ["--native-optimizations", "torch", "--hemisphere-workers", "2",
                    "--native-bin-dir", str(directory), "--profile-stages", "--device", "cuda:0"]
        report = {**recipe, "device": "cuda:0", "cli_command": command,
                  "native_program_sha256": expected.copy()}
        binding = lambda name: {"binary": str(directory / name), "sha256": expected[name]}
        runtime = {"backend": "native", "n4_binary": None,
            "native_optimizations": {"mode": "torch", "em_backend": "torch",
                                     "white_binary": str(directory / names[5])},
            "topology_repair": {**binding(names[0]), "intersection_binary": str(directory / names[2]),
                                "intersection_sha256": expected[names[2]]},
            "remesh": binding(names[1]), "white_preaparc": binding(names[5]),
            "surface_metrics": binding(names[4]), "sphere_generation": {"mris_inflate": binding(names[3])},
            "extra_curvature": {"paint_backend": "native", "statistics_backend": "native",
                "mrisp_paint_sha256": expected[names[6]], "mris_curvature_stats_sha256": expected[names[7]]},
            "hemisphere_scheduling": {"mode": "independent-exec-private-subjects", "workers": 2,
                "total_thread_budget": 4, "execution": "groups", "groups": [
                    {"operation": name, "status": "complete"} for name in
                    ("surface", "register", "annotation", "finish_surface")]},
            "mesh_validation": {"status": "passed"}, "surfaces": {}, "stages": []}
        runtime["hemisphere_scheduling"]["groups"][1]["values"] = {}
        for hemi in ("lh", "rh"):
            runtime["surfaces"][hemi] = {"topology_backend": "native", "topology_finish_backend": "conda-cpp",
                "topology_native_seconds": 1., "topology_remesh_seconds": 1., "topology_intersection_seconds": 1.,
                "white_preaparc_report": {"place_seconds": 1.}, "final_white_report": {"seconds": 1.},
                "pial_report": {"implementation": "conda-source-cpp", "seconds": 1.},
                "sphere_seconds": {"inflate_nofix": 1.}}
            runtime["stages"].append({"name": "stats_" + hemi + "_curv", "seconds": 1.})
            runtime["hemisphere_scheduling"]["groups"][1]["values"][hemi] = {
                "stages": [{"name": "avg_curv_" + hemi, "seconds": 1.}]}
            for suffix in ("orig.premesh", "orig"):
                (subject / "surf" / f"{hemi}.{suffix}").write_text("test surface identity " + suffix)
            premesh = MODULE.sha(subject / "surf" / f"{hemi}.orig.premesh")
            orig = MODULE.sha(subject / "surf" / f"{hemi}.orig")
            pre = {"status": "complete", "binary": str(directory / names[2]), "binary_sha256": expected[names[2]],
                   "input_sha256": premesh, "candidate_sha256": "1" * 64}
            remesh = {"status": "complete", "binary": str(directory / names[1]), "binary_sha256": expected[names[1]],
                      "input_sha256": "1" * 64, "candidate_sha256": "2" * 64, "iterations": 3}
            post = {**pre, "input_sha256": "2" * 64, "candidate_sha256": orig,
                    "mesh_structure": {"status": "passed"},
                    "source_intersecting_faces_after_independent_recheck": 0,
                    "geometric_intersecting_faces_after_independent_recheck": 0}
            finish = {"status": "complete", "input_sha256": premesh, "output_sha256": orig,
                      "published_to_orig": True}
            for name, row in zip(("preclean", "remesh", "intersection", "topology-finish"),
                                 (pre, remesh, post, finish)):
                (subject / "scripts" / f"surface.{hemi}.{hemi}.native-{name}.json").write_text(json.dumps(row))
        return report, runtime, subject, expected

    def test_native_surface_profile_binds_programs_completed_steps_and_content_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, runtime, subject, expected = self.native_fixture(Path(temporary))
            kwargs = dict(report=report, runtime=runtime, subject=subject, expected_programs=expected)
            accepted = MODULE.validate_native_surface_programs(**kwargs)
            self.assertEqual(accepted["os_exec_audit"], "not_assessed")
            self.assertEqual(set(accepted["hemispheres"]), {"lh", "rh"})
            for field, invalid in (("surface_placement_backend", "python"), ("gca_candidate_chunk", 2048),
                                   ("gca_reuse_atlas_context", False)):
                changed = {**report, field: invalid}
                with self.assertRaisesRegex(ValueError, "fixed recipe"):
                    MODULE.validate_native_surface_programs(**{**kwargs, "report": changed})
            for modify, message in (
                (lambda value: value["surfaces"]["rh"]["pial_report"].update(seconds=0), "placement/topology"),
                (lambda value: value["stages"][0].update(error="failed"), "paint/curvature"),
                (lambda value: value["hemisphere_scheduling"]["groups"].pop(), "four completed"),
                (lambda value: value["mesh_validation"].update(status="failed"), "production mesh"),
            ):
                changed = copy.deepcopy(runtime); modify(changed)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.validate_native_surface_programs(**{**kwargs, "runtime": changed})
            path = subject / "scripts/surface.rh.rh.native-intersection.json"
            original = json.loads(path.read_text())
            for field, invalid in (("candidate_sha256", "0" * 64),
                    ("source_intersecting_faces_after_independent_recheck", 1),
                    ("geometric_intersecting_faces_after_independent_recheck", 1)):
                path.write_text(json.dumps({**original, field: invalid}))
                with self.assertRaisesRegex(ValueError, "content chain or strict mesh"):
                    MODULE.validate_native_surface_programs(**kwargs)
            path.write_text(json.dumps(original))
            (subject.parent / "native-bin/mris_place_surface").write_text("changed program")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                MODULE.validate_native_surface_programs(**kwargs)

    def test_native_program_manifest_requires_predeclared_hash_and_unique_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bin.sha256"
            path.write_text("a" * 64 + "  /private/conda/bin/mris_remesh\n")
            result = MODULE.validate_native_program_manifest(path=path, expected_sha256=MODULE.sha(path))
            self.assertEqual(result, {"mris_remesh": "a" * 64})
            with self.assertRaisesRegex(ValueError, "previously fixed"):
                MODULE.validate_native_program_manifest(path=path, expected_sha256=None)
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                MODULE.validate_native_program_manifest(path=path, expected_sha256="0" * 64)
            path.write_text(path.read_text() * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.validate_native_program_manifest(path=path, expected_sha256=MODULE.sha(path))

    def test_native_candidate_keeps_frozen_source_raw_input_and_all_output_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, runtime, subject, expected = self.native_fixture(root)
            source = root / "src/fnit/recon_all"; source.mkdir(parents=True)
            generator = source / "native_free.py"; generator.write_text("# source fixture\n")
            raw = root / "raw.txt"; raw.write_text("input identity fixture\n")
            outputs = {}
            for relative in expected_paths():
                path = subject / relative; path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(relative.encode())
                outputs[relative] = str(path)
            report.update(execution="complete", exit_code=0, runtime_status="complete",
                input_sha256=MODULE.sha(raw), cli_wall_seconds=1., full_harness_seconds=1.1,
                source_sha256={"recon_all/native_free.py": MODULE.sha(generator)})
            runtime.update(input=str(raw), subject_dir=str(subject), status="complete", outputs=outputs,
                output_validation={"status": "passed", "expected": 138, "present": 138, "missing": []})
            benchmark = root / "benchmark.json"; benchmark.write_text(json.dumps(report))
            (subject / "fnit-native-free-run.json").write_text(json.dumps(runtime))
            kwargs = dict(benchmark_path=benchmark, source_root=root, case="public-case",
                reference_manifest={"cases": {"public-case": {"input_sha256": MODULE.sha(raw)}}},
                candidate_profile="mixed-native-surfaces", expected_native_programs=expected)
            accepted, _ = MODULE.validate_candidate(**kwargs)
            self.assertEqual(accepted["candidate_output_binding"]["verified"], 138)
            with self.assertRaisesRegex(ValueError, "complete fixed program manifest"):
                MODULE.validate_candidate(**{**kwargs, "expected_native_programs": None})
            generator.write_text("# changed frozen source\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                MODULE.validate_candidate(**kwargs)

    def test_receipt_source_and_input_must_all_bind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "subject"; subject.mkdir()
            source = root / "src/fnit/recon_all"; source.mkdir(parents=True)
            generator = source / "native_free.py"; generator.write_text("# binding fixture\n")
            raw = root / "raw.txt"; raw.write_text("provenance fixture, not real T1\n")
            report = {"execution": "complete", "exit_code": 0, "input_sha256": MODULE.sha(raw),
                      "backend": "python", "topology_backend": "python-ga",
                      "native_program_sha256": {}, "threads": 4,
                      "cli_command": [sys.executable, "-X", "faulthandler", "-u", "-m",
                                      "fnit.recon_all.native_free", str(raw), str(subject)],
                      "execution_audit": {"status": "complete", "passed": True,
                                          "original_program_attempts": []},
                      "runtime_status": "complete", "cli_wall_seconds": 1.0,
                      "full_harness_seconds": 1.1,
                      "source_sha256": {"recon_all/native_free.py": MODULE.sha(generator)}}
            runtime = {"input": str(raw), "subject_dir": str(subject), "status": "complete",
                       "outputs": make_candidate_outputs(subject), "output_validation":
                       {"status": "passed", "expected": 138, "present": 138, "missing": []}}
            benchmark = root / "benchmark.json"; benchmark.write_text(json.dumps(report))
            runtime_path = subject / "fnit-native-free-run.json"; runtime_path.write_text(json.dumps(runtime))
            manifest = {"cases": {"public-case": {"input_sha256": MODULE.sha(raw)}}}
            kwargs = {"benchmark_path": benchmark, "source_root": root,
                      "case": "public-case", "reference_manifest": manifest}
            MODULE.validate_candidate(**kwargs)
            report["execution"] = "running"; benchmark.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                MODULE.validate_candidate(**kwargs)
            report["execution"] = "complete"; benchmark.write_text(json.dumps(report))
            generator.write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                MODULE.validate_candidate(**kwargs)
            generator.write_text("# binding fixture\n")
            raw.write_text("changed raw data\n")
            with self.assertRaisesRegex(ValueError, "runtime original T1"):
                MODULE.validate_candidate(**kwargs)
            raw.write_text("provenance fixture, not real T1\n")
            report["cli_command"][6] = str(root / "other-raw.txt")
            (root / "other-raw.txt").write_bytes(raw.read_bytes())
            benchmark.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "CLI original T1"):
                MODULE.validate_candidate(**kwargs)
            report["cli_command"][6] = str(raw)
            report["cli_command"][7] = str(root / "other-subject")
            benchmark.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "CLI subject"):
                MODULE.validate_candidate(**kwargs)
            report["cli_command"][7] = str(subject)
            benchmark.write_text(json.dumps(report))
            runtime["subject_dir"] = str(root / "other-subject")
            runtime_path.write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "CLI subject"):
                MODULE.validate_candidate(**kwargs)
            runtime["subject_dir"] = str(subject)
            runtime_path.write_text(json.dumps(runtime))
            runtime["output_validation"]["present"] = 137
            runtime_path.write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.validate_candidate(**kwargs)
            runtime["output_validation"]["present"] = 138
            runtime_path.write_text(json.dumps(runtime))
            missing = subject / expected_paths()[-1]
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "output missing"):
                MODULE.validate_candidate(**kwargs)
            missing.symlink_to(subject / expected_paths()[0])
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                MODULE.validate_candidate(**kwargs)
            missing.unlink()
            missing.write_bytes(expected_paths()[-1].encode())
            runtime["outputs"][expected_paths()[-1]] = str(root / "other-raw.txt")
            runtime_path.write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "runtime output path differs"):
                MODULE.validate_candidate(**kwargs)

    def test_mixed_topology_binds_only_the_fixed_conda_program(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "subject"; subject.mkdir()
            source = root / "src/fnit/recon_all"; source.mkdir(parents=True)
            generator = source / "native_free.py"; generator.write_text("# fixed source\n")
            raw = root / "raw.txt"; raw.write_text("public T1 fixture identity\n")
            conda_bin = root / "conda/bin"; conda_bin.mkdir(parents=True)
            interpreter = conda_bin / "python"; interpreter.write_text("test interpreter\n")
            binary = conda_bin / "mris_fix_topology_fnit"
            binary.write_text("test pinned source build\n"); binary.chmod(0o755)
            native_bin = root / "native-bin"; native_bin.mkdir()
            selected = native_bin / binary.name; selected.symlink_to(binary)
            checksum = MODULE.sha(binary)
            benchmark = root / "benchmark.json"
            report = {
                "execution": "complete", "exit_code": 0, "input_sha256": MODULE.sha(raw),
                "backend": "native", "topology_backend": "native", "threads": 4,
                "native_program_sha256": {binary.name: checksum},
                "cli_command": [str(interpreter), "-X", "faulthandler", "-u", "-m",
                                "fnit.recon_all.native_free", str(raw), str(subject),
                                "--backend", "native", "--topology-backend", "native",
                                "--native-optimizations", "torch", "--native-bin-dir", str(native_bin)],
                "runtime_status": "complete", "cli_wall_seconds": 1.0,
                "full_harness_seconds": 1.1,
                "source_sha256": {"recon_all/native_free.py": MODULE.sha(generator)},
                "hemisphere_execution": "groups", "n4_backend": "simpleitk",
                "n4_execution": "in-process", "wm_backend": "torch-optimized",
                "wm_execution": "isolated", "wm_edit_backend": "python",
                "defects_backend": "torch", "inflate_backend": "torch",
                "sphere_finish_backend": "torch", "surface_placement_backend": "python",
                "normalization_controls_backend": "torch",
                "normalization_initial_bias_backend": "torch", "gca_inverse_backend": "torch",
                "gca_execution": "isolated", "fill_backend": "numba",
                "annotation_gibbs_backend": "numba", "remesh_scalar_storage": "python",
                "mni_execution": "parallel-late", "topology_distance_backend": "serial",
                "topology_state_backend": "legacy",
            }
            runtime = {
                "input": str(raw), "subject_dir": str(subject),
                "outputs": make_candidate_outputs(subject),
                "status": "complete", "backend": "native",
                "n4_binary": None, "native_optimizations": {"em_backend": "torch"},
                "hemisphere_scheduling": {
                    "workers": 2, "total_thread_budget": 4,
                    "mode": "independent-exec-private-subjects", "execution": "groups",
                    "groups": [{"operation": operation, "status": "complete"} for operation in
                               ("surface", "register", "annotation", "final_white", "pial_surface")],
                },
                "surfaces": {"lh": {"topology_backend": "native", "topology_native_seconds": 1.0},
                             "rh": {"topology_backend": "native", "topology_native_seconds": 1.1}},
                "topology_repair": {"backend": "native", "binary": str(selected),
                                    "sha256": checksum},
                "output_validation": {"status": "passed", "expected": 138,
                                      "present": 138, "missing": []},
            }
            benchmark.write_text(json.dumps(report))
            runtime_path = subject / "fnit-native-free-run.json"
            runtime_path.write_text(json.dumps(runtime))
            manifest = {"cases": {"public-case": {"input_sha256": MODULE.sha(raw)}}}
            kwargs = {"benchmark_path": benchmark, "source_root": root,
                      "case": "public-case", "reference_manifest": manifest,
                      "candidate_profile": "mixed-topology-cpp",
                      "expected_topology_sha256": checksum}
            accepted, _ = MODULE.validate_candidate(**kwargs)
            self.assertEqual(accepted["mixed_topology_program_validation"]["sha256"], checksum)
            self.assertEqual(accepted["candidate_output_binding"]["verified"], 138)
            with self.assertRaisesRegex(ValueError, "Python GA"):
                MODULE.validate_candidate(**{**kwargs, "candidate_profile": "python-ga"})
            with self.assertRaisesRegex(ValueError, "previously fixed"):
                MODULE.validate_candidate(**{**kwargs, "expected_topology_sha256": None})
            with self.assertRaisesRegex(ValueError, "only the fixed topology"):
                report["native_program_sha256"]["mri_segment"] = checksum
                benchmark.write_text(json.dumps(report))
                MODULE.validate_candidate(**kwargs)
            report["native_program_sha256"].pop("mri_segment")
            benchmark.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                binary.write_text("modified program\n")
                MODULE.validate_candidate(**kwargs)
            binary.write_text("test pinned source build\n")
            with self.assertRaisesRegex(ValueError, "five completed hemisphere groups"):
                runtime["hemisphere_scheduling"]["groups"].pop()
                runtime_path.write_text(json.dumps(runtime))
                MODULE.validate_candidate(**kwargs)
            runtime["hemisphere_scheduling"]["groups"].append(
                {"operation": "pial_surface", "status": "complete"})
            with self.assertRaisesRegex(ValueError, "completed native surface stage"):
                runtime["surfaces"]["rh"]["topology_native_seconds"] = 0.0
                runtime_path.write_text(json.dumps(runtime))
                MODULE.validate_candidate(**kwargs)
            runtime["surfaces"]["rh"]["topology_native_seconds"] = 1.1
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runtime["output_validation"]["present"] = 137
                runtime_path.write_text(json.dumps(runtime))
                MODULE.validate_candidate(**kwargs)

    def test_mixed_cli_requires_predeclared_reference_manifest_hash(self):
        argv = ["evaluate_recon_torch_run.py", "--benchmark", "unused.json",
                "--source-root", "unused-source", "--reference-root", "unused-reference",
                "--case", "public-case", "--driver", "unused-driver.py",
                "--scripts-dir", "unused-scripts", "--label-table", "unused-lut",
                "--output", "unused-output", "--candidate-profile", "mixed-topology-cpp"]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "previously fixed reference manifest"):
                MODULE.main()


if __name__ == "__main__":
    unittest.main()
