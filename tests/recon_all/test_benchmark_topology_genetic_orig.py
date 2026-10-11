"""完整回执与同版本QC复用合同；模拟文件不是真实T1 benchmark。"""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import nibabel.freesurfer.io as fsio
import numpy as np

import benchmark_recon_topology_genetic_orig as tool


class FinalOrigContracts(unittest.TestCase):
    def fixture(self, root):
        directory = root / "premesh"
        directory.mkdir()
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32)
        geometry = dict(head=np.array([2, 0, 20]), valid="1  # volume info valid", filename="own.mgz",
            volume=np.array([2, 2, 2]), voxelsize=np.ones(3), xras=np.array([-1, 0, 0]),
            yras=np.array([0, 0, -1]), zras=np.array([0, 1, 0]), cras=np.zeros(3))
        surface = directory / "lh.orig.premesh"
        fsio.write_geometry(str(surface), xyz, faces, volume_info=geometry)
        for name in ("premesh_state.private.npz", "mappings.private.npz"):
            np.savez(directory / name, control_only=np.array([1], np.int32))
        input_sha = tool.sha(surface)  # Simulated gate contracts, not MRI inputs.
        report = dict(status="complete_premesh_stage", execution_status="complete",
            complete_ga=True, all_algorithm_inputs_self_generated=True,
            reference_inputs_used=False, native_program_called=False,
            quality={"status": "passed"}, input_sha256={name: input_sha
                for name in ("nofix", "qsphere", "brain")},
            actual_module_sha256={"fnit.recon_all.topology_genetic_repair":
                tool.sha(Path(tool.__file__))},
            output_sha256={name: tool.sha(directory / name) for name in
                (surface.name, "premesh_state.private.npz", "mappings.private.npz")})
        self.receipt(directory, report)
        return directory, surface, report

    def receipt(self, directory, report):
        tool._write(directory / "report.json", report)
        tool._write(directory / "COMPLETE.json", dict(status="complete_premesh_stage",
            report_sha256=tool.sha(directory / "report.json")))

    def cleanup(self, path):
        return dict(execution_status="complete", output_written=True, require_zero=True,
            output_sha256=tool.sha(path),
            source_intersecting_faces_after_independent_recheck=0,
            geometric_intersecting_faces_after_independent_recheck=0)

    def test_receipt_and_surface_hash_failure_before_remesh(self):
        changes = ({"complete_ga": False}, {"native_program_called": True},
                   {"reference_inputs_used": True}, {"complete_ga": 1})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory, _, report = self.fixture(root)
                self.receipt(directory, {**report, **change})
                with patch.object(tool, "remesh_surface") as remesh:
                    with self.assertRaises(ValueError):
                        tool.run(premesh_directory=directory, hemisphere="lh", output=root / "result")
                    remesh.assert_not_called()
                self.assertTrue((root / "result/failure.json").is_file())
                self.assertFalse((root / "result/COMPLETE.json").exists())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, surface, _ = self.fixture(root)
            surface.write_bytes(surface.read_bytes() + b"wrong-version")
            with self.assertRaisesRegex(ValueError, "surface SHA"):
                tool.verified_premesh(premesh_directory=directory, hemisphere="lh")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, _, report = self.fixture(root)
            tool._write(directory / "report.json", {**report, "extra": True})
            with self.assertRaisesRegex(ValueError, "report SHA"):
                tool.verified_premesh(premesh_directory=directory, hemisphere="lh")

    def test_same_sha_complete_cleanup_counts_bind_final_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, surface, _ = self.fixture(Path(temporary))
            report = self.cleanup(surface)
            quality = tool.final_quality(surface_path=surface, cleanup_report=report)
            self.assertEqual(quality["status"], "passed")
            self.assertTrue(quality["geometry_footer_present"])
            self.assertTrue(quality["geometry"]["valid"])
            for name in ("source_intersecting_faces_after_independent_recheck",
                         "geometric_intersecting_faces_after_independent_recheck"):
                with self.subTest(count=name):
                    bad = {**report, name: 1}
                    self.assertEqual(tool.final_quality(surface_path=surface,
                                                       cleanup_report=bad)["status"], "failed")
            with self.assertRaises(ValueError):
                tool.final_quality(surface_path=surface, cleanup_report={**report, "require_zero": False})
            surface.write_bytes(surface.read_bytes() + b"different-file-version")
            with self.assertRaisesRegex(ValueError, "exact successful"):
                tool.final_quality(surface_path=surface, cleanup_report=report)

    def test_missing_or_changed_declared_outputs_fail_before_remesh(self):
        for name in ("premesh_state.private.npz", "mappings.private.npz"):
            for change in ("missing", "changed", "undeclared"):
                with self.subTest(name=name, change=change), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    directory, _, report = self.fixture(root)
                    if change == "missing":
                        (directory / name).unlink()
                    elif change == "changed":
                        (directory / name).write_bytes(b"different-checkpoint")
                    else:
                        del report["output_sha256"][name]
                        self.receipt(directory, report)
                    with patch.object(tool, "remesh_surface") as remesh:
                        with self.assertRaises((FileNotFoundError, ValueError)):
                            tool.run(premesh_directory=directory, hemisphere="lh", output=root / "failed")
                        remesh.assert_not_called()
                    self.assertFalse((root / "failed/COMPLETE.json").exists())

    def test_missing_input_or_source_provenance_and_nonlocal_output_fail(self):
        changes = ({"input_sha256": {}}, {"actual_module_sha256": {}},
                   {"input_sha256": {"nofix": "not-a-hash", "qsphere": "a" * 64,
                                     "brain": "b" * 64}})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                directory, _, report = self.fixture(Path(temporary))
                self.receipt(directory, {**report, **change})
                with self.assertRaisesRegex(ValueError, "provenance"):
                    tool.verified_premesh(premesh_directory=directory, hemisphere="lh")
        with tempfile.TemporaryDirectory() as temporary:
            directory, _, report = self.fixture(Path(temporary))
            report["output_sha256"]["../other-file"] = "a" * 64
            self.receipt(directory, report)
            with self.assertRaisesRegex(ValueError, "relative output"):
                tool.verified_premesh(premesh_directory=directory, hemisphere="lh")

    def test_missing_and_invalid_geometry_cannot_pass_final_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, surface, _ = self.fixture(root)
            xyz, faces, geometry = fsio.read_geometry(surface, read_metadata=True)
            missing = root / "missing.surface"
            fsio.write_geometry(missing, xyz, faces)
            with self.assertWarns(UserWarning):
                quality = tool.final_quality(surface_path=missing, cleanup_report=self.cleanup(missing))
            self.assertEqual(quality["status"], "failed")
            self.assertFalse(quality["geometry"]["valid"])
            for name, value in (("valid", "0"), ("voxelsize", np.array([1., 0., 1.])),
                                ("xras", np.zeros(3)), ("cras", np.array([np.nan, 0., 0.]))):
                with self.subTest(name=name):
                    bad = root / (name + ".surface")
                    fsio.write_geometry(bad, xyz, faces, volume_info={**geometry, name: value})
                    self.assertEqual(tool.final_quality(surface_path=bad,
                        cleanup_report=self.cleanup(bad))["status"], "failed")

    def test_complete_chain_file_scope_and_failed_cleanup_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, surface, _ = self.fixture(root)
            before = tool.sha(surface)
            def remesh(**kwargs):
                self.assertEqual(kwargs["iterations"], 3)
                shutil.copyfile(kwargs["input_path"], kwargs["output_path"])
            def cleanup(**kwargs):
                self.assertEqual(kwargs["input_path"], kwargs["output_path"])
                self.assertTrue(kwargs["require_zero"])
                return self.cleanup(kwargs["output_path"])
            with patch.object(tool, "remesh_surface", side_effect=remesh), \
                    patch.object(tool, "run_remove_intersection", side_effect=cleanup):
                report = tool.run(premesh_directory=directory, hemisphere="lh",
                                  output=root / "result", threads=1)
            self.assertEqual(report["status"], "complete_orig_stage")
            self.assertEqual(report["whole_recon_all"], "not_assessed")
            self.assertIsNone(report["gpu_memory_bytes"])
            self.assertEqual(before, tool.sha(surface))
            self.assertTrue((root / "result/COMPLETE.json").is_file())
            with patch.object(tool, "remesh_surface", side_effect=remesh), \
                    patch.object(tool, "run_remove_intersection", side_effect=RuntimeError("residual")):
                with self.assertRaisesRegex(RuntimeError, "residual"):
                    tool.run(premesh_directory=directory, hemisphere="lh",
                             output=root / "failed", threads=1)
            self.assertTrue((root / "failed/failure.json").is_file())
            self.assertFalse((root / "failed/COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
