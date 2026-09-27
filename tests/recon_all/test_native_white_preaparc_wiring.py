"""Check opt-in pre-aparc stage order and public flag propagation."""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.batch import run_recon_all_python_batch
from fnit.recon_all.native_free import (
    _place_preaparc_and_smooth, _run_white_mri_chain,
    _write_principal_curvature_maps, main,
    run_recon_all_python,
)


class WhitePreaparcWiringTest(unittest.TestCase):
    def test_requires_topology_in_single_and_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t1.nii.gz").write_bytes(b"image")
            for name in ("weights", "assets"):
                (root / name).mkdir()
            with self.assertRaisesRegex(ValueError, "native_topology"):
                run_recon_all_python(
                    root / "t1.nii.gz", root / "sub", root / "weights",
                    root / "assets", native_bin_dir=root,
                    native_white_preaparc=True)
            with self.assertRaisesRegex(ValueError, "native_topology"):
                run_recon_all_python_batch(
                    [], root / "weights", root / "assets",
                    native_bin_dir=root, native_white_preaparc=True)

    def test_required_external_assets_and_placement_binary_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t1.nii.gz").write_bytes(b"image")
            for name in ("weights", "assets", "bin"):
                (root / name).mkdir()
            with self.assertRaisesRegex(FileNotFoundError,
                                        "synthmorph.affine.2.h5"):
                run_recon_all_python(
                    root / "t1.nii.gz", root / "sub", root / "weights",
                    root / "assets", native_bin_dir=root / "bin",
                    native_topology=True, native_white_preaparc=True)
            from fnit.recon_all.mni_aux_chain import validate_mni_aux_assets
            weights = ("synthmorph.affine.2.h5",
                       "mca-dura.both-lh.nstd21.fhs.h5",
                       "vsinus.no-sp.m.all.nstd10-070.h5")
            assets = (
                "average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.cropped.nii.gz",
                "average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.nii.gz",
                "average/mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz",
                "average/mca-dura.prior.warp.mni152.1.0mm.rh.nii.gz",
                "average/vsinus.no-sp.prior.mni152.1.0mm.mgz",
            )
            for root_dir, names in ((root / "weights", weights),
                                    (root / "assets", assets)):
                for name in names:
                    path = root_dir / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"asset")
            validate_mni_aux_assets(root / "weights", root / "assets")
            for name in ("mri_em_register", "mris_fix_topology_fnit",
                         "mri_segment", "mri_edit_wm_with_aseg"):
                binary = root / "bin" / name
                binary.write_text("#!/bin/sh\nexit 0\n")
                binary.chmod(0o755)
            with self.assertRaisesRegex(FileNotFoundError,
                                        "mris_place_surface"):
                run_recon_all_python(
                    root / "t1.nii.gz", root / "sub", root / "weights",
                    root / "assets", native_bin_dir=root / "bin",
                    native_topology=True, native_white_preaparc=True)

    def test_cpu_mri_chain_precedes_surface_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "sub"
            calls = []

            def stage(name, function, *args, **kwargs):
                calls.append((name, function.__name__, args, kwargs))

            _run_white_mri_chain(subject, Path("weights"), Path("assets"), 4, stage)
            self.assertEqual([call[0] for call in calls],
                             ["mni_aux", "brain_finalsurfs"])
            self.assertEqual([call[1] for call in calls],
                             ["run_mni_aux_chain", "run_finalsurfs"])
            self.assertEqual([call[3]["device"] for call in calls],
                             ["cpu", "cpu"])
            self.assertEqual(calls[0][3]["threads"], 4)

    def test_placed_surface_is_three_pass_smoothwm_input(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory)
            calls = []

            def place(*args, **kwargs):
                calls.append(("place", args, kwargs))
                return {"place_seconds": 1.0}

            def smooth(*args, **kwargs):
                calls.append(("smooth", args, kwargs))

            with patch("fnit.recon_all.white_preaparc_conda.run_white_preaparc",
                       side_effect=place), patch(
                           "fnit.recon_all.smooth_surface_python.smooth_surface",
                           side_effect=smooth):
                result = _place_preaparc_and_smooth(
                    subject, "lh", Path("binary"), Path("assets"), 4)
            self.assertEqual([call[0] for call in calls], ["place", "smooth"])
            self.assertEqual(calls[0][1][1], "lh")
            self.assertEqual(calls[0][2], {"threads": 4})
            self.assertEqual(calls[1][1],
                             (subject / "surf/lh.white.preaparc",
                              subject / "surf/lh.smoothwm"))
            self.assertEqual(calls[1][2],
                             {"iterations": 3, "device": "cpu"})
            self.assertIn("smoothwm_seconds", result)

    def test_placed_curvatures_use_preaparc_and_smoothwm_meshes(self):
        source_value = {"white.preaparc": 2, "inflated": 3, "smoothwm": 4}
        output = {}
        reads = []

        def read(path):
            surface = Path(path).name.split(".", 1)[1]
            reads.append(surface)
            return np.full((3, 3), source_value[surface], np.float32), np.zeros((1, 3), np.int32)

        def curv(vertices, _faces, *, device):
            self.assertEqual(device, "cpu")
            value = vertices[0, 0]
            return np.full(3, value, np.float32), np.full(3, value * 10, np.float32)

        with patch("fnit.recon_all.native_free.fs.read_geometry", side_effect=read), patch(
                "fnit.recon_all.native_free.fs.write_morph_data",
                side_effect=lambda path, values: output.__setitem__(Path(path).name, values)), patch(
                "fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                side_effect=curv):
            for hemi in ("lh", "rh"):
                _write_principal_curvature_maps(Path("surf"), hemi, "cpu", True)
        self.assertEqual(reads, ["white.preaparc", "inflated", "smoothwm"] * 2)
        self.assertEqual(len(output), 16)
        for hemi in ("lh", "rh"):
            for name in ("H", "K", "K1", "K2"):
                self.assertIn(f"{hemi}.smoothwm.{name}.crv", output)
            np.testing.assert_array_equal(output[f"{hemi}.white.preaparc.H"],
                                          np.full(3, 11, np.float32))
            np.testing.assert_array_equal(output[f"{hemi}.smoothwm.K1.crv"],
                                          np.full(3, 4, np.float32))

    def test_default_curvature_path_reuses_white_smoothwm(self):
        output = {}
        reads = []

        def read(path):
            surface = Path(path).name.split(".", 1)[1]
            reads.append(surface)
            value = {"white": 1, "inflated": 3}[surface]
            return np.full((3, 3), value, np.float32), np.zeros((1, 3), np.int32)

        def curv(vertices, _faces, *, device):
            value = vertices[0, 0]
            return np.full(3, value, np.float32), np.full(3, value * 10, np.float32)

        with patch("fnit.recon_all.native_free.fs.read_geometry", side_effect=read), patch(
                "fnit.recon_all.native_free.fs.write_morph_data",
                side_effect=lambda path, values: output.__setitem__(Path(path).name, values)), patch(
                "fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                side_effect=curv):
            for hemi in ("lh", "rh"):
                _write_principal_curvature_maps(Path("surf"), hemi, "cpu", False)
        self.assertEqual(reads, ["white", "inflated"] * 2)
        self.assertEqual(len(output), 16)
        for hemi in ("lh", "rh"):
            np.testing.assert_array_equal(output[f"{hemi}.white.preaparc.H"],
                                          output[f"{hemi}.smoothwm.H.crv"])

    def test_single_cli_and_batch_forward_flag(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["t1.nii.gz", "sub", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin",
                      "--native-topology", "--native-white-preaparc"])
            self.assertTrue(run.call_args.kwargs["native_white_preaparc"])
            self.assertTrue(run.call_args.kwargs["native_topology"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets", "bin"):
                (root / name).mkdir()
            (root / "t1.nii.gz").write_bytes(b"image")
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                subject = Path(command[4])
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(
                    json.dumps({"status": "complete"}))
                return type("Completed", (), {"returncode": 0})()

            with patch("fnit.recon_all.batch.subprocess.run",
                       side_effect=fake_run):
                reports = run_recon_all_python_batch(
                    [{"t1": root / "t1.nii.gz", "subject_dir": root / "sub"}],
                    root / "weights", root / "assets", devices=("cpu",),
                    native_bin_dir=root / "bin", native_topology=True,
                    native_white_preaparc=True)
            self.assertEqual(reports[0]["status"], "complete")
            self.assertEqual(commands[0][-2:],
                             ["--native-topology", "--native-white-preaparc"])


if __name__ == "__main__":
    unittest.main()
