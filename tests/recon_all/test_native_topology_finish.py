"""Conda source-built remesh/cleanup opt-in and strict publication tests."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel.freesurfer.io as fs
import numpy as np

from fnit.recon_all.native_free import (
    _bounded_geometric_intersections, _finish_native_topology, _native_final_mesh_structure,
    _prepare_native_topology, _run_native_intersection, _run_native_remesh,
    _validate_topology_finish_backend, main,
)


# Real fresh-exec workers exercise the production publisher, while native CLI
# copies and intersection counts stay controlled scheduling fixtures.
NATIVE_FINISH_WORKER = r'''
def execute(subject, hemi, device, threads, operation, binary, fail_lh=False):
    from pathlib import Path
    import shutil
    from unittest.mock import patch
    from fnit.recon_all.native_free import _finish_native_topology
    subject = Path(subject)
    def native(command, **kwargs):
        if "-i" in command:
            shutil.copyfile(command[2], command[4])
        else:
            shutil.copyfile(command[1], command[2])
    geometric_count = int(fail_lh and hemi == "lh")
    with patch("fnit.recon_all.native_free.subprocess.run", side_effect=native), \
         patch("fnit.recon_all.place_surface_intersection_marking.mark_source_intersections",
               return_value=(None, 0)), \
         patch("fnit.recon_all.native_free._bounded_geometric_intersections",
               return_value=(geometric_count, {"test_fixture": True})):
        _finish_native_topology(subject=subject, hemi=hemi, assets=subject.parent,
                                remesh_binary=Path(binary), intersection_binary=Path(binary))
    artifacts = {name: str(subject / "surf" / (hemi + ".orig.native-" + name + ".candidate"))
                 for name in ("preclean", "remesh", "intersection")}
    artifacts["orig"] = str(subject / "surf" / (hemi + ".orig"))
    artifacts.update({name + "_receipt": str(subject / "scripts" / (hemi + ".native-" + name + ".json"))
                      for name in ("preclean", "remesh", "intersection", "topology-finish")})
    return {"artifacts": artifacts}
'''


class NativeTopologyFinishTest(unittest.TestCase):
    def _subject(self, root):
        subject = Path(root) / "subjects" / "sub01"
        for name in ("surf", "scripts"):
            (subject / name).mkdir(parents=True)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        source = subject / "surf" / "lh.orig.premesh"
        volume_info = {"head": np.array([2, 0, 20], dtype=np.int32),
                       "valid": "1", "filename": "brain.mgz",
                       "volume": np.array([256, 256, 256]),
                       "voxelsize": np.array([1.0, 1.0, 1.0]),
                       "xras": np.array([-1.0, 0.0, 0.0]),
                       "yras": np.array([0.0, 0.0, -1.0]),
                       "zras": np.array([0.0, 1.0, 0.0]),
                       "cras": np.array([0.625, -17.25, 12.5])}
        fs.write_geometry(str(source), vertices, faces, volume_info=volume_info)
        return subject, source, vertices, faces

    def _actual_publication_group(self, directory, subject, *, fail_lh=False):
        from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
        root = Path(directory)
        (root / "native_finish_publication_fixture.py").write_text(NATIVE_FINISH_WORKER)
        binary = root / "mock-native"
        binary.write_bytes(b"controlled native CLI fixture")
        pythonpath = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
        with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
            return run_hemisphere_group(
                subject=subject, operation="nativefinish", device="cpu", threads=4, workers=2,
                callable_path="native_finish_publication_fixture:execute",
                kwargs={"binary": str(binary), "fail_lh": fail_lh})

    def test_native_candidate_names_obey_existing_hemisphere_ownership(self):
        from fnit.recon_all.hemisphere_parallel import _owned
        for hemi, other in (("lh", "rh"), ("rh", "lh")):
            for name in ("preclean", "remesh", "intersection", "publish"):
                relative = f"surf/{hemi}.orig.native-{name}.candidate"
                self.assertTrue(_owned(relative, hemi), relative)
                self.assertFalse(_owned(relative, other), relative)
                # Shared-write rules stay strict: hidden candidates are not owned.
                self.assertFalse(_owned(f"surf/.{hemi}.orig.native-{name}.candidate", hemi))

    def test_real_publisher_retains_all_native_artifacts_for_both_hemispheres(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, source, _, _ = self._subject(directory)
            shutil.copyfile(source, subject / "surf/rh.orig.premesh")
            before = source.read_bytes()
            report = self._actual_publication_group(directory, subject)
            self.assertEqual(report["status"], "complete")
            for hemi in ("lh", "rh"):
                artifacts = report["values"][hemi]["artifacts"]
                for name in ("preclean", "remesh", "intersection", "orig"):
                    path = Path(artifacts[name])
                    self.assertTrue(path.is_file(), path)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertIn(str(path.relative_to(subject)), report["published"])
                for name in ("preclean", "remesh", "intersection", "topology-finish"):
                    path = Path(artifacts[name + "_receipt"])
                    self.assertEqual(path.parent, subject / "scripts")
                    self.assertEqual(path.name, f"nativefinish.{hemi}.{hemi}.native-{name}.json")
                    self.assertEqual(json.loads(path.read_text())["status"], "complete")
                    self.assertIn(str(path.relative_to(subject)), report["published"])
                self.assertEqual((subject / f"surf/{hemi}.orig.premesh").read_bytes(), before)
                # Atomic publication consumes only its staging file.
                self.assertFalse((subject / f"surf/{hemi}.orig.native-publish.candidate").exists())
            self.assertFalse(list((subject / "surf").glob(".*.orig.native-*.candidate")))
            self.assertFalse(list((subject / "scripts").glob("hemi-*")))

    def test_real_publisher_quality_failure_leaves_both_public_orig_unchanged(self):
        from fnit.recon_all.hemisphere_parallel import HemisphereGroupError
        with tempfile.TemporaryDirectory() as directory:
            subject, source, _, _ = self._subject(directory)
            shutil.copyfile(source, subject / "surf/rh.orig.premesh")
            for hemi in ("lh", "rh"):
                (subject / f"surf/{hemi}.orig").write_bytes((hemi + " accepted orig").encode())
            with self.assertRaises(HemisphereGroupError) as caught:
                self._actual_publication_group(directory, subject, fail_lh=True)
            self.assertEqual(caught.exception.report["status"], "failed")
            self.assertEqual(caught.exception.report["published"], [])
            self.assertIn("native intersection left 0 source / 1 geometric",
                          caught.exception.report["workers"]["lh"]["error"])
            for hemi in ("lh", "rh"):
                self.assertEqual((subject / f"surf/{hemi}.orig").read_bytes(),
                                 (hemi + " accepted orig").encode())
            self.assertFalse(list((subject / "surf").glob("*.orig.native-*.candidate")))
            self.assertFalse(list((subject / "scripts").glob("nativefinish.*.native-*.json")))

    def test_opt_in_rejects_pure_python_and_non_native_ga(self):
        _validate_topology_finish_backend("python", topology_backend="python-ga", backend="python")
        _validate_topology_finish_backend("conda-cpp", topology_backend="native", backend="native")
        for value, topology, backend in (
            ("conda-cpp", "python-ga", "native"),
            ("conda-cpp", "native", "python"),
            ("invalid", "native", "native"),
        ):
            with self.assertRaises(ValueError):
                _validate_topology_finish_backend(value, topology_backend=topology, backend=backend)

    def test_native_remesh_consumes_preclean_and_keeps_private_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, source, _, faces = self._subject(directory)
            preclean = subject / "surf/lh.orig.native-preclean.candidate"
            shutil.copyfile(source, preclean)
            binary = Path(directory) / "mris_remesh"
            binary.write_bytes(b"mock native executable")
            def remesh(command, **kwargs):
                self.assertEqual(command[:3], [str(binary), "-i", str(preclean)])
                self.assertEqual(command[3], "-o")
                self.assertEqual(command[5:], ["--remesh", "--iters", "3"])
                self.assertEqual(kwargs["env"]["SUBJECTS_DIR"], str(subject.parent))
                shutil.copyfile(command[2], command[4])
            with patch("fnit.recon_all.native_free.subprocess.run", side_effect=remesh):
                seconds = _run_native_remesh(binary, subject, "lh", Path(directory),
                                             input_path=preclean)
            self.assertGreaterEqual(seconds, 0)
            candidate = subject / "surf/lh.orig.native-remesh.candidate"
            xyz, output_faces, metadata = fs.read_geometry(str(candidate), read_metadata=True)
            self.assertTrue(np.isfinite(xyz).all())
            np.testing.assert_array_equal(output_faces, faces)
            np.testing.assert_array_equal(metadata["cras"], [0.625, -17.25, 12.5])
            self.assertFalse((subject / "surf/lh.orig").exists())

    def _copy_native(self, command, **kwargs):
        if "-i" in command:
            shutil.copyfile(command[2], command[4])
        else:
            shutil.copyfile(command[1], command[2])

    def test_final_publication_requires_both_zero_counts_and_keeps_all_candidates(self):
        for existing_orig in (False, True):
            for source_count, geometric_count in ((0, 0), (11, 11), (0, 1), (1, 0)):
                with self.subTest(existing_orig=existing_orig, source=source_count,
                                  geometric=geometric_count):
                    with tempfile.TemporaryDirectory() as directory:
                        subject, source, _, _ = self._subject(directory)
                        orig = subject / "surf/lh.orig"
                        if existing_orig:
                            orig.write_bytes(b"previous accepted orig")
                        before = orig.read_bytes() if existing_orig else None
                        premesh_before = source.read_bytes()
                        binary = Path(directory) / "mock-native"
                        binary.write_bytes(b"mock native executable")
                        with patch("fnit.recon_all.native_free.subprocess.run", side_effect=self._copy_native) as run, \
                             patch("fnit.recon_all.place_surface_intersection_marking.mark_source_intersections",
                                   return_value=(None, source_count)), \
                             patch("fnit.recon_all.native_free._bounded_geometric_intersections",
                                   return_value=(geometric_count, {"max_face_radius_mm": 12.0})):
                            arguments = dict(subject=subject, hemi="lh", assets=Path(directory),
                                             remesh_binary=binary, intersection_binary=binary)
                            if source_count or geometric_count:
                                with self.assertRaisesRegex(RuntimeError, "native intersection left"):
                                    _finish_native_topology(**arguments)
                            else:
                                remesh_seconds, cleanup_seconds = _finish_native_topology(**arguments)
                                self.assertGreaterEqual(remesh_seconds, 0)
                                self.assertGreaterEqual(cleanup_seconds, 0)
                        self.assertEqual(len(run.call_args_list), 3)
                        commands = [call.args[0] for call in run.call_args_list]
                        self.assertEqual(Path(commands[0][1]), source)
                        self.assertEqual(commands[1][2], commands[0][2])
                        self.assertEqual(commands[2][1], commands[1][4])
                        self.assertEqual(source.read_bytes(), premesh_before)
                        for name in ("preclean", "remesh", "intersection"):
                            self.assertTrue((subject / f"surf/lh.orig.native-{name}.candidate").exists())
                        receipt = json.loads((subject / "scripts/lh.native-intersection.json").read_text())
                        publication = json.loads((subject / "scripts/lh.native-topology-finish.json").read_text())
                        failed = bool(source_count or geometric_count)
                        self.assertEqual(receipt["source_intersecting_faces_after_independent_recheck"], source_count)
                        self.assertEqual(receipt["geometric_intersecting_faces_after_independent_recheck"], geometric_count)
                        self.assertEqual(publication["published_to_orig"], not failed)
                        self.assertEqual(publication["status"], "failed" if failed else "complete")
                        if failed:
                            self.assertEqual(orig.read_bytes() if orig.exists() else None, before)
                        else:
                            self.assertEqual(orig.read_bytes(), premesh_before)
                            _, _, metadata = fs.read_geometry(str(orig), read_metadata=True)
                            np.testing.assert_array_equal(metadata["cras"], [0.625, -17.25, 12.5])

    def test_native_cleanup_rejects_reordered_faces_and_changed_mri_geometry(self):
        for change in ("faces", "geometry"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                subject, source, vertices, faces = self._subject(directory)
                remesh = subject / "surf/lh.orig.native-remesh.candidate"
                shutil.copyfile(source, remesh)
                orig = subject / "surf/lh.orig"
                orig.write_bytes(b"previous accepted orig")
                binary = Path(directory) / "mock-native"
                binary.write_bytes(b"mock native executable")
                def native(command, **kwargs):
                    _, _, metadata = fs.read_geometry(str(source), read_metadata=True)
                    if change == "geometry":
                        metadata["cras"] = metadata["cras"] + [1, 0, 0]
                    fs.write_geometry(command[2], vertices, faces[::-1] if change == "faces" else faces,
                                      volume_info=metadata)
                with patch("fnit.recon_all.native_free.subprocess.run", side_effect=native), \
                     patch("fnit.recon_all.place_surface_intersection_marking.mark_source_intersections",
                           side_effect=AssertionError("must reject before marking")):
                    with self.assertRaisesRegex(RuntimeError, "ordered mesh, MRI geometry"):
                        _run_native_intersection(binary, subject, "lh", Path(directory), phase="postclean")
                self.assertEqual(orig.read_bytes(), b"previous accepted orig")

    def test_native_failure_does_not_publish_or_modify_existing_orig(self):
        for failed_step in (0, 1, 2):
            with self.subTest(failed_step=failed_step), tempfile.TemporaryDirectory() as directory:
                subject, source, _, _ = self._subject(directory)
                orig = subject / "surf/lh.orig"
                orig.write_bytes(b"previous accepted orig")
                binary = Path(directory) / "mock-native"
                binary.write_bytes(b"mock native executable")
                calls = []
                def native(command, **kwargs):
                    self._copy_native(command, **kwargs)
                    calls.append(command)
                    if len(calls) == failed_step + 1:
                        raise RuntimeError("command failed")
                with patch("fnit.recon_all.native_free.subprocess.run", side_effect=native):
                    with self.assertRaisesRegex(RuntimeError, "command failed"):
                        _finish_native_topology(subject=subject, hemi="lh", assets=Path(directory),
                                                remesh_binary=binary, intersection_binary=binary)
                self.assertEqual(orig.read_bytes(), b"previous accepted orig")
                self.assertEqual(len(calls), failed_step + 1)
                receipt = json.loads((subject / "scripts/lh.native-topology-finish.json").read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse(receipt["published_to_orig"])

    def test_open_mesh_never_publishes_even_when_programs_exit_successfully(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, source, vertices, faces = self._subject(directory)
            binary = Path(directory) / "mock-native"
            binary.write_bytes(b"mock native executable")
            def native(command, **kwargs):
                if "-i" in command:
                    _, _, metadata = fs.read_geometry(str(source), read_metadata=True)
                    fs.write_geometry(command[4], vertices, faces[:-1], volume_info=metadata)
                else:
                    self._copy_native(command, **kwargs)
            with patch("fnit.recon_all.native_free.subprocess.run", side_effect=native), \
                 patch("fnit.recon_all.place_surface_intersection_marking.mark_source_intersections",
                       side_effect=AssertionError("must reject before intersection marking")):
                with self.assertRaisesRegex(RuntimeError, "structural mesh gate"):
                    _finish_native_topology(subject=subject, hemi="lh", assets=Path(directory),
                                            remesh_binary=binary, intersection_binary=binary)
            self.assertFalse((subject / "surf/lh.orig").exists())
            receipt = json.loads((subject / "scripts/lh.native-intersection.json").read_text())
            self.assertEqual(receipt["mesh_structure"]["status"], "failed")
            self.assertEqual(receipt["mesh_structure"]["topology"]["boundary_edges"], 3)

    def test_invalid_structural_mesh_is_rejected_before_vertex_link_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, vertices, faces = self._subject(directory)
            faces[0] = [0, 0, 1]
            with patch("fnit.recon_all.topology_volume_python._nonmanifold_vertex_links",
                       side_effect=AssertionError("unsafe kernel must not see degenerate indices")):
                self.assertEqual(_native_final_mesh_structure(vertices, faces)["status"], "failed")

    def test_prepare_forwards_conda_programs_to_complete_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, source, _, _ = self._subject(directory)
            shutil.copyfile(source, subject / "surf/lh.orig.nofix")
            binaries = [Path(directory) / name for name in ("ga", "inflate", "cleanup", "remesh")]
            with patch("fnit.recon_all.smooth_surface_python.smooth_surface"), \
                 patch("fnit.recon_all.sphere_quick_python.write_quick_sphere"), \
                 patch("fnit.recon_all.native_free._run_native_sphere_step", return_value=1.0), \
                 patch("fnit.recon_all.native_free._run_native_topology"), \
                 patch("fnit.recon_all.native_free._finish_native_topology", return_value=(2.0, 3.0)) as finish:
                result = _prepare_native_topology(binaries[0], subject, "lh", Path(directory), "cpu",
                                                  binaries[1], binaries[2], remesh_binary=binaries[3],
                                                  topology_finish_backend="conda-cpp")
            self.assertEqual(result[-2:], (2.0, 3.0))
            finish.assert_called_once_with(subject=subject, hemi="lh", assets=Path(directory),
                                            remesh_binary=binaries[3], intersection_binary=binaries[2])

    def test_bounded_geometric_candidates_match_original_with_long_face(self):
        from fnit.recon_all.mris_remove_intersection_python import mark_intersections
        vertices = np.array([
            [0, 0, 0], [10, 0, 0], [0, 10, 0],
            [2, 2, -1], [2, 2, 1], [3, 3, 0],
            [100, 100, 100], [101, 100, 100], [100, 101, 100],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        original = mark_intersections(vertices, faces)[1]
        bounded, diagnostics = _bounded_geometric_intersections(
            vertices, faces, large_face_radius_mm=2.0)
        self.assertEqual(bounded, original)
        self.assertGreater(diagnostics["max_face_radius_mm"], 2)
        self.assertGreater(diagnostics["large_faces"], 0)

    def test_cli_forwards_explicit_finish_backend(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python", return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--topology-finish-backend", "conda-cpp"])
        self.assertEqual(run.call_args.kwargs["topology_finish_backend"], "conda-cpp")

    def test_whole_harness_records_and_forwards_finish_backend(self):
        tool = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("fnit_native_finish_harness", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t1 = root / "input.nii.gz"
            t1.write_bytes(b"contract input")
            for name in ("weights", "assets", "bin"):
                (root / name).mkdir()
            output = root / "whole"
            argv = [str(tool), "--t1", str(t1), "--output-root", str(output),
                    "--weights-dir", str(root / "weights"),
                    "--assets-dir", str(root / "assets"),
                    "--native-bin-dir", str(root / "bin"),
                    "--device", "cuda:0", "--backend", "native",
                    "--topology-backend", "native",
                    "--topology-finish-backend", "conda-cpp",
                    "--surface-placement-backend", "native",
                    "--code-version", "wiring-test"]
            process = unittest.mock.MagicMock()
            process.pid = 12345
            process.returncode = 0
            process.poll.return_value = 0
            sampler = unittest.mock.MagicMock()
            sampler.report.return_value = {"scope": "mocked sampler"}
            with patch.object(sys, "argv", argv), \
                 patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies", return_value={}), \
                 patch.object(module, "ProcessTreeDeviceSampler", return_value=sampler), \
                 patch.object(module.subprocess, "Popen", return_value=process):
                self.assertEqual(module.main(), 0)
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(report["topology_finish_backend"], "conda-cpp")
            child = report["cli_command"]
            self.assertEqual(child[child.index("--topology-finish-backend") + 1], "conda-cpp")
            self.assertEqual(report["execution"], "complete")


if __name__ == "__main__":
    unittest.main()
