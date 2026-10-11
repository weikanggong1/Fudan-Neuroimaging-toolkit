"""文件编号域/头/严格失败契约；小文件只用于接口测试，不是 MRI benchmark。"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
from nibabel.freesurfer import io as fsio
import numba
import numpy as np
import torch

from fnit.recon_all import topology_genetic_files as api
from fnit.recon_all import mris_remesh_python as mature_remesh


class GeneticFilesContracts(unittest.TestCase):
    def fixture(self, directory):
        xyz = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], np.float32)
        faces = np.array([[0, 2, 3], [0, 3, 4], [0, 4, 2],
                          [1, 3, 2], [1, 4, 3], [1, 2, 4]], np.int32)
        surface = directory / "lh.orig.nofix"
        sphere = directory / "lh.qsphere.nofix"
        fsio.write_geometry(str(surface), xyz, faces)
        fsio.write_geometry(str(sphere), xyz * 100, faces)
        brain = directory / "brain.mgz"
        wm = directory / "wm.mgz"
        nib.save(nib.MGHImage(np.ones((3, 4, 5), np.uint8), np.eye(4)), brain)
        affine = np.array([[0, 0, -1.5, 20], [-2, 0, 0, 30], [0, 2.5, 0, 40], [0, 0, 0, 1]], float)
        nib.save(nib.MGHImage(np.ones((4, 5, 6), np.uint8), affine), wm)
        return dict(input_surface=surface, input_qsphere=sphere, brain=brain, wm=wm,
                    hemisphere="lh", output_directory=directory / "result", threads=2)

    def result(self):
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32)
        return dict(execution_status="complete", complete_ga=True, native_program_called=False,
            surface_state=dict(original_vertices=xyz, current_vertices=xyz.copy(), faces=faces, nfaces=4),
            quality=dict(status="passed", final_mesh_quality="not_assessed",
                         original_geometry_requires_standard_remesh_and_intersection=True),
            raw_to_base_vertex_translation=np.array([2, 4, 0, 3, 1], np.int32),
            base_to_compact_vertex_translation=np.array([1, 0, 2, -1, 3], np.int32),
            original_domain_defect_labels=np.array([0, 1, 1, 0, 0], np.int32),
            current_cleanup_vertices=xyz + 100, per_defect=[dict(defect_number=0, generations=28)],
            required_next_steps=("standard_remesh", "original_intersection_cleanup", "final_mesh_quality"),
            seconds={"complete_api_including_validation": 0.001}, random_draws=10, standard_policy=True,
            blocker_scan_backend="python", distance_backend="serial")

    @contextmanager
    def budget(self, *, threads):
        report = dict(requested_threads=threads, restoration_complete=False)
        try:
            yield report
        finally:
            report["restoration_complete"] = True

    def cleanup(self, **kwargs):
        self.assertEqual(kwargs["marking_backend"], "source_numba")
        self.assertIsNone(kwargs["device"])
        self.assertIs(kwargs["require_zero"], True)
        self.assertEqual(kwargs["input_path"], kwargs["output_path"])
        report = dict(execution_status="complete", output_written=True, require_zero=True,
            output_sha256=api._sha256(kwargs["output_path"]),
            source_intersecting_faces_after_independent_recheck=0,
            geometric_intersecting_faces_after_independent_recheck=0)
        api._write_json(kwargs["report_path"], report)
        return report

    def remesh(self, **kwargs):
        self.assertEqual(kwargs["iterations"], 3)
        self.assertEqual(kwargs["scalar_storage"], "numpy")
        shutil.copyfile(kwargs["input_path"], kwargs["output_path"])

    def run_fake(self, kwargs, result=None, **changes):
        with patch.object(api, "repair_genetic_topology", return_value=result or self.result()), \
                patch.object(api, "thread_budget", side_effect=self.budget):
            return api.run_genetic_topology_files(**{**kwargs, **changes})

    def test_premesh_is_compact_original_labels_stay_raw_and_footer_is_wm(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            def core(**kwargs):
                self.assertEqual(kwargs["threads"], 2)
                self.assertEqual(kwargs["seed"], 1234)
                self.assertEqual((kwargs["population_size"], kwargs["max_unchanged"], kwargs["niters"]), (10, 10, -1))
                self.assertIs(kwargs["initial_selection"], True)
                self.assertIs(kwargs["vertex_eliminate"], True)
                self.assertIs(kwargs["use_overlap"], False)
                self.assertIs(kwargs["face_detection_cache"], False)
                expected_matrix = api.source_topology_ras_to_voxel(volume_header=nib.load(args["brain"]).header)
                np.testing.assert_array_equal(kwargs["ras_to_voxel"], expected_matrix)
                kwargs["progress_callback"](defect_number=0, result=dict(search=dict(best_fitness=-10,
                    counts={"evaluated": 1}, generations=28, termination="no_improvement"),
                    finalization={"fitness": -10}))
                return self.result()
            with patch.object(api, "repair_genetic_topology", side_effect=core), \
                    patch.object(api, "thread_budget", side_effect=self.budget), \
                    patch.object(api, "remesh_surface") as remesh:
                report = api.run_genetic_topology_files(**args)
            remesh.assert_not_called()
            self.assertEqual(report["status"], "complete")
            self.assertIs(report["final_orig"], False)
            self.assertIsNone(report["outputs"]["orig"])
            self.assertEqual(report["quality"]["status"], "not_assessed")
            self.assertTrue(report["thread_budget"]["restoration_complete"])
            xyz, faces, geometry = fsio.read_geometry(report["outputs"]["premesh"], read_metadata=True)
            np.testing.assert_array_equal(xyz, self.result()["surface_state"]["original_vertices"])
            self.assertEqual((len(xyz), len(faces)), (4, 4))
            expected = api._wm_geometry(nib.load(args["wm"]), args["wm"].resolve())
            self.assertTrue(api._same_geometry(geometry, expected))
            with Path(report["outputs"]["defect_labels"]).open("rb") as stream:
                self.assertEqual(stream.read(3), b"\xff\xff\xff")
                self.assertEqual(struct.unpack(">iii", stream.read(12)), (5, 6, 1))
            np.testing.assert_array_equal(fsio.read_morph_data(report["outputs"]["defect_labels"]), [0, 1, 1, 0, 0])
            with np.load(report["maps_path"]) as mapping:
                np.testing.assert_array_equal(mapping["raw_to_compact_vertex_translation"], [2, 3, 1, -1, 0])
            stored = json.loads(Path(report["report_path"]).read_text())
            self.assertEqual(stored["outputs"]["premesh"], "lh.orig.premesh")
            self.assertEqual(stored["maps_path"], "lh.genetic.maps.private.npz")
            self.assertEqual(report["report_sha256"], api._sha256(report["report_path"]))
            complete = json.loads(Path(report["completion_path"]).read_text())
            self.assertEqual(complete["report_sha256"], report["report_sha256"])
            self.assertIs(report["parameters"]["face_detection_cache"], False)
            self.assertNotIn("all_algorithm_inputs_self_generated", report)
            self.assertIs(report["internal_ga_inputs_self_generated"], True)
            self.assertEqual(report["input_file_upstream_provenance"], "caller_supplied_not_verified")
            self.assertIsNone(report["input_file_upstream_self_generated"])
            self.assertIs(report["declared_inputs_only"], True)
            self.assertIs(report["reference_comparison_performed"], False)
            self.assertIs(report["reference_inputs_used"], False)
            self.assertEqual(report["reference_inputs_used_scope"],
                             "no_additional_reference_inputs_read_by_this_stage")
            json.dumps(report, allow_nan=False)

    def test_face_cache_forwards_actual_core_signature_and_records_normalized_bool(self):
        signature = inspect.signature(api.repair_genetic_topology)
        for selected in (True, np.bool_(True), np.bool_(False)):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                def core(**kwargs):
                    signature.bind(**kwargs)  # Validate against the actual packaged core API.
                    self.assertIs(kwargs["face_detection_cache"], bool(selected))
                    self.assertEqual(kwargs["threads"], 2)
                    return self.result()
                with patch.object(api, "repair_genetic_topology", side_effect=core) as run, \
                        patch.object(api, "thread_budget", side_effect=self.budget):
                    report = api.run_genetic_topology_files(**args, face_detection_cache=selected)
                run.assert_called_once()
                self.assertIs(report["parameters"]["face_detection_cache"], bool(selected))
                stored = json.loads(Path(report["report_path"]).read_text())
                self.assertIs(stored["parameters"]["face_detection_cache"], bool(selected))
                self.assertEqual(report["domains"]["defect_labels"], "raw_nofix_vertices_F32_with_raw_face_count")
                json.dumps(report, allow_nan=False)

    def test_invalid_face_cache_fails_before_directory_creation(self):
        for selected in (None, 0, 1, "true", [], np.array(True)):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                with patch.object(api, "repair_genetic_topology") as core:
                    with self.assertRaisesRegex(ValueError, "face_detection_cache.*bool"):
                        api.run_genetic_topology_files(**args, face_detection_cache=selected)
                core.assert_not_called()
                self.assertFalse(args["output_directory"].exists())

    def test_different_ordered_faces_fail_before_core(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            xyz, faces = fsio.read_geometry(args["input_qsphere"])
            fsio.write_geometry(args["input_qsphere"], xyz, faces[::-1])
            with patch.object(api, "repair_genetic_topology") as core:
                with self.assertRaisesRegex(ValueError, "ordered faces"):
                    api.run_genetic_topology_files(**args)
            core.assert_not_called()
            self.assertTrue((args["output_directory"] / "failure.json").is_file())
            self.assertFalse((args["output_directory"] / "COMPLETE.json").exists())

    def test_existing_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            args["output_directory"].mkdir()
            sentinel = args["output_directory"] / "keep.txt"
            sentinel.write_text("owned before call")
            with patch.object(api, "repair_genetic_topology") as core:
                with self.assertRaisesRegex(ValueError, "new directory"):
                    api.run_genetic_topology_files(**args)
            core.assert_not_called()
            self.assertEqual(sentinel.read_text(), "owned before call")

    def test_no_quantization_or_invalid_wm_geometry(self):
        for bad in ("brain", "wm"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                if bad == "brain":
                    nib.save(nib.MGHImage(np.ones((3, 4, 5), np.float32), np.eye(4)), args["brain"])
                else:
                    with patch.object(api, "_wm_geometry", side_effect=ValueError("invalid WM geometry")):
                        with self.assertRaisesRegex(ValueError, "WM geometry"):
                            api.run_genetic_topology_files(**args)
                    continue
                with self.assertRaisesRegex(ValueError, "uint8"):
                    api.run_genetic_topology_files(**args)

    def test_bad_core_slots_maps_or_labels_never_publish_completion(self):
        for kind in ("tmp_slot", "permutation", "labels", "incomplete", "quality"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                result = self.result()
                if kind == "tmp_slot":
                    result["surface_state"]["current_vertices"] += 100
                elif kind == "permutation":
                    result["raw_to_base_vertex_translation"][0] = 4
                elif kind == "labels":
                    result["original_domain_defect_labels"] = np.zeros(4, np.int32)
                elif kind == "incomplete":
                    result["complete_ga"] = False
                else:
                    result["quality"]["status"] = "failed"
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_fake(args, result=result)
                self.assertFalse((args["output_directory"] / "COMPLETE.json").exists())

    def test_final_orig_reuses_once_remesh_and_strict_same_sha_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with patch.object(api, "remesh_surface", side_effect=self.remesh) as remesh, \
                    patch.object(api, "run_remove_intersection", side_effect=self.cleanup) as cleanup:
                report = self.run_fake(args, final_orig=True)
            self.assertEqual((remesh.call_count, cleanup.call_count), (1, 1))
            self.assertEqual(report["quality"]["status"], "passed")
            self.assertTrue(report["quality"]["wm_geometry_matches"])
            self.assertTrue(report["quality"]["footer_bytes_preserved"])
            self.assertEqual(report["quality"]["intersection_counts_bound_to_output_sha256"],
                             report["output_sha256"]["orig"])
            self.assertTrue(all(report["seconds"][name] is not None for name in
                ("remesh", "original_intersection_cleanup", "final_mesh_quality", "complete_file_api")))

    def test_mature_remesh_file_api_accepts_profile_python_and_forwards_both_selectors(self):
        for storage in ("numpy", "python"):
            with self.subTest(storage=storage), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                def geometry(vertices, faces, iterations, *, scalar_storage):
                    self.assertEqual(iterations, 3)
                    self.assertEqual(scalar_storage, storage)
                    return vertices, faces
                # Actual mature file API performs enum validation, IO and raw footer copy.
                with patch.object(mature_remesh, "remesh_geometry", side_effect=geometry) as kernel, \
                        patch.object(api, "run_remove_intersection", side_effect=self.cleanup):
                    report = self.run_fake(args, final_orig=True, remesh_scalar_storage=storage)
                kernel.assert_called_once()
                self.assertEqual(report["parameters"]["remesh_scalar_storage"], storage)
                self.assertEqual(report["quality"]["status"], "passed")
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with patch.object(api, "repair_genetic_topology") as core:
                with self.assertRaisesRegex(ValueError, "numpy or python"):
                    api.run_genetic_topology_files(**args, remesh_scalar_storage="torch")
            core.assert_not_called()
            self.assertFalse(args["output_directory"].exists())

    def test_final_orig_failure_and_wrong_report_sha_are_preserved(self):
        for bad in ("cleanup_exception", "remaining", "wrong_sha", "wrong_footer"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                def cleanup(**kwargs):
                    if bad == "cleanup_exception":
                        raise RuntimeError("residual intersects")
                    if bad == "wrong_footer":
                        xyz, faces, geom = fsio.read_geometry(kwargs["output_path"], read_metadata=True)
                        geom["cras"] += 5
                        fsio.write_geometry(kwargs["output_path"], xyz, faces, volume_info=geom)
                    report = self.cleanup(**kwargs)
                    if bad == "remaining":
                        report["source_intersecting_faces_after_independent_recheck"] = 1
                    elif bad == "wrong_sha":
                        report["output_sha256"] = "0" * 64
                    return report
                with patch.object(api, "remesh_surface", side_effect=self.remesh), \
                        patch.object(api, "run_remove_intersection", side_effect=cleanup):
                    with self.assertRaises((RuntimeError, ValueError)):
                        self.run_fake(args, final_orig=True)
                self.assertTrue((args["output_directory"] / "failure.json").is_file())
                self.assertFalse((args["output_directory"] / "COMPLETE.json").exists())

    def test_thread_restoration_success_and_exception(self):
        before_torch = torch.get_num_threads()
        before_numba = numba.get_num_threads()
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                def core(**kwargs):
                    self.assertEqual((torch.get_num_threads(), numba.get_num_threads()), (2, 2))
                    if fail:
                        raise RuntimeError("failure after applying budget")
                    return self.result()
                with patch.object(api, "repair_genetic_topology", side_effect=core):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "applying budget"):
                            api.run_genetic_topology_files(**args)
                    else:
                        report = api.run_genetic_topology_files(**args)
                        self.assertTrue(report["thread_budget"]["restoration_complete"])
                self.assertEqual((torch.get_num_threads(), numba.get_num_threads()), (before_torch, before_numba))

    def test_changed_input_fails_even_if_core_succeeded(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            def core(**kwargs):
                args["input_surface"].write_bytes(args["input_surface"].read_bytes() + b"changed")
                return self.result()
            with patch.object(api, "repair_genetic_topology", side_effect=core), \
                    patch.object(api, "thread_budget", side_effect=self.budget):
                with self.assertRaisesRegex(RuntimeError, "input files changed"):
                    api.run_genetic_topology_files(**args)
            self.assertFalse((args["output_directory"] / "COMPLETE.json").exists())

    def test_cli_forwards_explicit_policy_and_cpu_budget(self):
        values = ["--input-surface", "own.nofix", "--input-qsphere", "own.qsphere", "--brain", "brain.mgz",
                  "--wm", "wm.mgz", "--hemisphere", "rh", "--output-directory", "new-stage", "--threads", "2",
                  "--seed", "77", "--no-initial-selection", "--use-overlap", "--final-orig",
                  "--remesh-scalar-storage", "python", "--face-detection-cache"]
        with patch.object(api, "run_genetic_topology_files", return_value={"status": "complete"}) as run, \
                patch("builtins.print"):
            self.assertEqual(api.main(values), 0)
        kwargs = run.call_args.kwargs
        self.assertEqual((kwargs["hemisphere"], kwargs["threads"], kwargs["seed"]), ("rh", 2, 77))
        self.assertIs(kwargs["initial_selection"], False)
        self.assertIs(kwargs["use_overlap"], True)
        self.assertIs(kwargs["final_orig"], True)
        self.assertEqual(kwargs["remesh_scalar_storage"], "python")
        self.assertIs(kwargs["face_detection_cache"], True)
        for flags in ([], ["--no-face-detection-cache"]):
            with self.subTest(flags=flags), \
                    patch.object(api, "run_genetic_topology_files", return_value={"status": "complete"}) as run, \
                    patch("builtins.print"):
                self.assertEqual(api.main(values[:-1] + flags), 0)
                self.assertIs(run.call_args.kwargs["face_detection_cache"], False)


if __name__ == "__main__":
    unittest.main()
