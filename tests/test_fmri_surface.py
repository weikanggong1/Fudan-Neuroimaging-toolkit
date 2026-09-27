"""Output and coordinate contracts for the Workbench surface projection stage."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri import surface


def _gifti(path, data, intent):
    arrays = [nib.gifti.GiftiDataArray(data, intent=intent)]
    if intent == "NIFTI_INTENT_POINTSET":
        arrays.append(nib.gifti.GiftiDataArray(
            np.array([[0, 1, 2]], np.int32), intent="NIFTI_INTENT_TRIANGLE"))
    image = nib.GiftiImage(darrays=arrays)
    nib.save(image, str(path))
    return path


def _case(tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    data = np.ones((6, 7, 8, 4), np.float32)
    bold = nib.Nifti1Image(data, affine)
    bold.header.set_zooms((2, 2, 2, 0.735))
    bold.header.set_xyzt_units(t="sec")
    paths = {"bold": tmp_path / "clean_MNI.nii.gz"}
    nib.save(bold, str(paths["bold"]))
    for name, array in (
        ("reference", np.ones((6, 7, 8), np.float32)),
        ("goodvoxels", np.ones((6, 7, 8), np.uint8)),
        ("subject_rois", np.ones((6, 7, 8), np.uint8)),
        ("atlas_rois", np.ones((6, 7, 8), np.uint8)),
    ):
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), str(paths[name]))
    hemispheres = {}
    for name in ("L", "R"):
        native = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
        atlas = np.zeros((32492, 3), np.float32)
        fields = {}
        for field in ("white", "pial", "midthickness", "registered_sphere"):
            fields[field] = _gifti(tmp_path / f"{name}.{field}.surf.gii", native,
                                   "NIFTI_INTENT_POINTSET")
        for field in ("atlas_sphere", "atlas_midthickness"):
            fields[field] = _gifti(tmp_path / f"{name}.{field}.surf.gii", atlas,
                                   "NIFTI_INTENT_POINTSET")
        fields["native_roi"] = _gifti(tmp_path / f"{name}.native_roi.shape.gii",
                                      np.ones(4, np.float32), "NIFTI_INTENT_SHAPE")
        fields["atlas_roi"] = _gifti(tmp_path / f"{name}.atlas_roi.shape.gii",
                                     np.ones(32492, np.float32), "NIFTI_INTENT_SHAPE")
        hemispheres[name] = surface.SurfaceHemisphere(**fields)
    return paths, hemispheres


def _call(paths, hemispheres, output):
    return surface.run_surface_projection(
        clean_mni=paths["bold"], mni_reference=paths["reference"],
        goodvoxels=paths["goodvoxels"], subject_rois=paths["subject_rois"],
        atlas_rois=paths["atlas_rois"], left=hemispheres["L"],
        right=hemispheres["R"], output_dir=output,
    )


def test_rejects_mismatched_volume_grid_before_workbench(tmp_path, monkeypatch):
    paths, hemispheres = _case(tmp_path)
    mismatch = nib.Nifti1Image(np.ones((6, 7, 8), np.uint8), np.eye(4))
    nib.save(mismatch, str(paths["goodvoxels"]))
    monkeypatch.setattr(surface.shutil, "which", lambda _: pytest.fail("Workbench was invoked"))
    with pytest.raises(ValueError, match="goodvoxels must match"):
        _call(paths, hemispheres, tmp_path / "out")


def test_projection_uses_registered_sphere_and_writes_time_correct_cifti(tmp_path, monkeypatch):
    paths, hemispheres = _case(tmp_path)
    calls = []
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(surface.shutil, "which", lambda _: "/usr/bin/wb_command")

    def fake_workbench(argv, **kwargs):
        assert 1 <= int(kwargs["env"]["OMP_NUM_THREADS"]) <= 8
        calls.append(argv)
        option = argv[1]
        if option in ("-volume-to-surface-mapping", "-metric-mask"):
            destination = argv[4]
        elif option in ("-metric-dilate", "-metric-smoothing"):
            destination = argv[5]
        elif option == "-metric-resample":
            destination = argv[6]
        elif option in ("-cifti-create-dense-timeseries", "-cifti-create-label"):
            destination = argv[2]
        elif option in ("-cifti-dilate", "-cifti-smoothing"):
            destination = argv[6]
        elif option == "-cifti-resample":
            destination = argv[8]
        elif option == "-cifti-separate":
            destination = argv[argv.index("-volume-all") + 1]
        else:
            pytest.fail(f"Unexpected Workbench operation: {option}")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if option == "-cifti-separate":
            nib.save(nib.Nifti1Image(np.ones((6, 7, 8, 4), np.float32),
                                          np.diag([2, 2, 2, 1])), str(path))
        elif option == "-cifti-create-dense-timeseries" and path.name.endswith(
            "registered_sphere_s2.dtseries.nii"
        ):
            axes = nib.cifti2.cifti2_axes
            left = axes.BrainModelAxis.from_mask(np.ones(32492, bool), name="CortexLeft")
            right = axes.BrainModelAxis.from_mask(np.ones(32492, bool), name="CortexRight")
            volume = axes.BrainModelAxis.from_mask(
                np.ones((1, 1, 1), bool), name="ThalamusLeft", affine=np.eye(4)
            )
            series = axes.SeriesAxis(0, 0.735, 4)
            image = nib.Cifti2Image(np.ones((4, 64985), np.float32),
                                    nib.Cifti2Header.from_axes((series, left + right + volume)))
            nib.save(image, str(path))
        else:
            path.write_bytes(b"workbench-mock")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(surface.subprocess, "run", fake_workbench)
    result = _call(paths, hemispheres, tmp_path / "out")
    assert nib.load(str(result.dtseries)).shape == (4, 64985)
    assert result.coverage_report.is_file()
    assert result.subcortical_volume.is_file()
    assert result.left_metric.is_file() and result.right_metric.is_file()
    ribbon_calls = [call for call in calls if call[1] == "-volume-to-surface-mapping"]
    assert len(ribbon_calls) == 2
    assert all("-ribbon-constrained" in call and "-volume-roi" in call for call in ribbon_calls)
    assert all(str(paths["goodvoxels"]) in call for call in ribbon_calls)
    resample_calls = [call for call in calls if call[1] == "-metric-resample"]
    assert len(resample_calls) == 2
    assert all("ADAP_BARY_AREA" in call and "-area-surfs" in call
               and "-current-roi" in call for call in resample_calls)
    assert any(call[1] == "-cifti-resample" and "CUBIC" in call for call in calls)
    final = calls[-1]
    assert final[1] == "-cifti-create-dense-timeseries"
    assert float(final[final.index("-timestep") + 1]) == pytest.approx(0.735, abs=1e-6)
