"""早期曲率写出及最终八图的调度契约；不代替真实影像 benchmark。

只隔离昂贵的主曲率拟合及原生命令，真实使用 nibabel 读写表面和 morph。
native 路径可延迟 smoothwm 四图；Python 路径和其余表面的原行为保留。
"""

import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.native_free import (
    _run_curvature_stats, _surface_pair, _write_principal_curvature_maps,
)


PRINCIPAL_MAPS = ("H", "K", "K1", "K2")
DISCRETE_MAPS = (*PRINCIPAL_MAPS, "BE", "C", "FI", "S")


class PrincipalCurvatureScheduleTest(unittest.TestCase):
    def _subject(self, directory, hemi="lh"):
        subject = Path(directory)
        for name in ("surf", "mri", "label", "scripts", "stats"):
            (subject / name).mkdir(parents=True, exist_ok=True)
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        for offset, name in enumerate(("white.preaparc", "inflated", "smoothwm")):
            fsio.write_geometry(str(subject / "surf" / f"{hemi}.{name}"),
                                vertices + offset, faces)
        return subject, vertices, faces

    @staticmethod
    def _principal(vertices, faces, *, device):
        # 与文件内容相关的确定性数列，检查写出公式和顶点顺序；不模拟拟合精度。
        assert device == "cpu"
        first = np.asarray(vertices[:, 0] + 2, np.float32)
        return first, np.asarray(first / 2, np.float32)

    @staticmethod
    def _early_paths(subject, hemi="lh", *, smoothwm=True):
        surf = subject / "surf"
        result = [surf / f"{hemi}.{surface}.{name}"
                  for surface in ("white.preaparc", "inflated")
                  for name in ("H", "K")]
        if smoothwm:
            result.extend(surf / f"{hemi}.smoothwm.{name}.crv"
                          for name in PRINCIPAL_MAPS)
        return result

    def test_default_preserves_all_early_maps_and_explicit_true_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, _, _ = self._subject(directory)
            with patch("fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                       side_effect=self._principal) as calculate:
                _write_principal_curvature_maps(surf=subject / "surf", hemi="lh", device="cpu")
            self.assertEqual(calculate.call_count, 3)
            before = {path: path.read_bytes() for path in self._early_paths(subject)}
            with patch("fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                       side_effect=self._principal) as calculate:
                _write_principal_curvature_maps(surf=subject / "surf", hemi="lh", device="cpu",
                                                include_smoothwm=True)
            self.assertEqual(calculate.call_count, 3)
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            xyz, _ = fsio.read_geometry(str(subject / "surf/lh.smoothwm"))
            k1, k2 = self._principal(xyz, None, device="cpu")
            expected = {"H": (k1 + k2) / 2, "K": k1 * k2, "K1": k1, "K2": k2}
            for name, values in expected.items():
                np.testing.assert_array_equal(
                    fsio.read_morph_data(str(subject / "surf" / f"lh.smoothwm.{name}.crv")), values)

    def test_deferred_maps_are_neither_read_nor_computed_and_other_bytes_match(self):
        for hemi in ("lh", "rh"):
            with self.subTest(hemi=hemi), tempfile.TemporaryDirectory() as directory:
                subject, _, _ = self._subject(directory, hemi)
                with patch("fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                           side_effect=self._principal):
                    _write_principal_curvature_maps(surf=subject / "surf", hemi=hemi, device="cpu")
                kept = {path: path.read_bytes() for path in self._early_paths(subject, hemi, smoothwm=False)}
                for path in self._early_paths(subject, hemi)[4:]:
                    path.unlink()
                # An unused smoothwm input may be absent; the other two surfaces must still work.
                (subject / "surf" / f"{hemi}.smoothwm").unlink()
                read_geometry = fsio.read_geometry
                with patch("fnit.recon_all.native_free.fs.read_geometry", wraps=read_geometry) as read, \
                     patch("fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                           side_effect=self._principal) as calculate:
                    _write_principal_curvature_maps(surf=subject / "surf", hemi=hemi,
                                                    device="cpu", include_smoothwm=False)
                self.assertEqual(calculate.call_count, 2)
                self.assertEqual([Path(call.args[0]).name for call in read.call_args_list],
                                 [f"{hemi}.white.preaparc", f"{hemi}.inflated"])
                self.assertEqual(kept, {path: path.read_bytes() for path in kept})
                self.assertTrue(all(not path.exists() for path in self._early_paths(subject, hemi)[4:]))

    def test_deferred_maps_do_not_delete_or_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, _, _ = self._subject(directory)
            existing = {}
            for path in self._early_paths(subject)[4:]:
                path.write_bytes(("existing " + path.name).encode())
                existing[path] = path.read_bytes()
            with patch("fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                       side_effect=self._principal) as calculate:
                _write_principal_curvature_maps(surf=subject / "surf", hemi="lh",
                                                device="cpu", include_smoothwm=False)
            self.assertEqual(calculate.call_count, 2)
            self.assertEqual(existing, {path: path.read_bytes() for path in existing})

    def test_curvature_failure_propagates_instead_of_publishing_smoothwm_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, _, _ = self._subject(directory)
            with patch("fnit.recon_all.surface_roi_curvature_gpu.principal_curvatures",
                       side_effect=RuntimeError("principal calculation failed")):
                with self.assertRaisesRegex(RuntimeError, "principal calculation failed"):
                    _write_principal_curvature_maps(surf=subject / "surf", hemi="lh",
                                                    device="cpu", include_smoothwm=False)
            self.assertFalse(any(path.exists() for path in self._early_paths(subject)))

    def test_surface_pair_only_defers_maps_for_native_placement(self):
        for placement in ("native", "python"):
            with self.subTest(placement=placement), tempfile.TemporaryDirectory() as directory:
                subject, vertices, faces = self._subject(directory)
                with contextlib.ExitStack() as stack:
                    replacements = {
                        "fnit.recon_all.pretess_python.pretess_mgh": None,
                        "fnit.recon_all.tessellate_gpu.tessellate_mgh": (vertices, faces),
                        "fnit.recon_all.tessellate_gpu.write_quad_surface": None,
                        "fnit.recon_all.extract_main_component_python.extract_main_component": 1,
                        "fnit.recon_all.label_cortex_python.label_cortex": None,
                        "fnit.recon_all.label_cortex_fix_ga_python.label_cortex_fix_ga":
                            (np.array([0, 1], np.int32), np.array([2], np.int32)),
                        "fnit.recon_all.native_free.nib.load": object(),
                        "fnit.recon_all.native_free._prepare_native_topology": (0, 0, {}, 0, 0),
                        "fnit.recon_all.native_free._place_preaparc_and_smooth": {},
                        "fnit.recon_all.native_free._run_accurate_sphere_pair": ({}, {}),
                    }
                    for target, value in replacements.items():
                        stack.enter_context(patch(target, return_value=value))
                    write = stack.enter_context(patch(
                        "fnit.recon_all.native_free._write_principal_curvature_maps"))
                    result = _surface_pair(
                        subject=subject, hemi="lh", filled=subject / "mri/filled.mgz",
                        norm=subject / "mri/norm.mgz", device="cpu", threads=2,
                        topology_binary=None, inflate_binary=None, intersection_binary=None,
                        place_binary=Path("declared-native"), defect_binary=None,
                        assets=subject, defer_defects=True, surface_placement_backend=placement)
                self.assertEqual(write.call_count, 1)
                self.assertEqual(write.call_args.kwargs["include_smoothwm"], placement == "python")
                self.assertEqual(result["vertices"], len(vertices))
                self.assertTrue(result["placement_pending"])
                early = result["early_curvature_maps"]
                self.assertEqual(early["include_smoothwm"], placement == "python")
                self.assertEqual(len(early["written_outputs"]), 8 if placement == "python" else 4)
                self.assertGreaterEqual(early["seconds_including_io"], 0)
                self.assertEqual(early["smoothwm_final_stage"], "stats_lh_curv")
                self.assertEqual(early["smoothwm_final_producer"],
                    "fnit.recon_all.curvature_stats_python.write_curvature_stats"
                    if placement == "python" else "conda-source-cpp:mris_curvature_stats")

    def _native_stats(self, directory, *, omitted=None, empty=None):
        subject, _, _ = self._subject(directory)
        output = subject / "stats/lh.curv.stats"
        paths = {"stats": output,
                 **{name: subject / "surf" / f"lh.smoothwm.{name}.crv" for name in DISCRETE_MAPS}}
        def native(command, **kwargs):
            self.assertEqual(command[1:], ["-m", "--writeCurvatureFiles", "-G", "-o",
                str(output), "-F", "smoothwm", subject.name, "lh", "curv", "sulc"])
            self.assertTrue(kwargs["check"])
            for name, path in paths.items():
                if name != omitted:
                    path.write_bytes(b"" if name == empty else b"complete native stage output")
        with patch("fnit.recon_all.native_free.subprocess.run", side_effect=native):
            return _run_curvature_stats(binary=Path("declared-native-stats"), subject=subject,
                                        hemi="lh", assets=subject, device="cpu", python_only=False)

    def test_native_stats_requires_all_eight_maps_and_stats_file(self):
        for missing in ("stats", *DISCRETE_MAPS):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileNotFoundError):
                    self._native_stats(directory, omitted=missing)

    def test_native_stats_rejects_empty_outputs(self):
        for empty in ("stats", *DISCRETE_MAPS):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileNotFoundError):
                    self._native_stats(directory, empty=empty)

    def test_native_stats_with_complete_outputs_preserves_return_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(self._native_stats(directory))


if __name__ == "__main__":
    unittest.main()
