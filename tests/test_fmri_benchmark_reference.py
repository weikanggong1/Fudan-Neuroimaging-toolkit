"""Benchmark validation must use the same native reference as the public API."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


_spec = importlib.util.spec_from_file_location(
    "fmri_benchmark_bids", Path(__file__).parents[1] / "validation/fmri/benchmark_bids.py"
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def _save(path, shape, affine):
    image = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm", "sec")
    if len(shape) == 4:
        image.header.set_zooms((*image.header.get_zooms()[:3], .8))
    nib.save(image, path)
    return path


def test_native_benchmark_accepts_selected_sbref_grid(tmp_path):
    bold = _save(tmp_path / "bold.nii.gz", (4, 5, 6, 8), np.eye(4))
    sbref_affine = np.diag([2., 2., 2., 1.])
    sbref_affine[:3, 3] = [-4., 5., 8.]
    sbref = _save(tmp_path / "sbref.nii.gz", (5, 6, 7), sbref_affine)
    native = _save(tmp_path / "clean_native.nii.gz", (5, 6, 7, 8), sbref_affine)
    inputs = SimpleNamespace(bold=bold, sbref=sbref)

    checks = benchmark.check_native_volume(native, inputs)
    assert checks["shape"] == [5, 6, 7, 8]
    assert checks["nonfinite_values"] == 0
    with pytest.raises(ValueError, match="reference grid"):
        benchmark.check_volume(native, nib.load(bold))


def test_native_benchmark_without_sbref_uses_bold_grid(tmp_path):
    bold = _save(tmp_path / "bold.nii.gz", (4, 5, 6, 8), np.eye(4))
    native = _save(tmp_path / "clean_native.nii.gz", (4, 5, 6, 8), np.eye(4))
    checks = benchmark.check_native_volume(native, SimpleNamespace(bold=bold, sbref=None))
    assert checks["shape"] == [4, 5, 6, 8]


@pytest.mark.parametrize("backend", ["fnirt", "synthmorph"])
def test_benchmark_capture_follows_final_public_route_without_changing_calls(
    tmp_path, monkeypatch, backend,
):
    """Capture the current route while preserving each output's exact arguments."""
    from fnit.fmri import end_to_end, pipeline

    bold = _save(tmp_path / "bold.nii.gz", (4, 5, 6, 2), np.eye(4))
    t1 = _save(tmp_path / "t1.nii.gz", (4, 5, 6), np.eye(4))
    mask = _save(tmp_path / "mask.nii.gz", (4, 5, 6), np.eye(4))
    pull = _save(tmp_path / "pull.nii.gz", (4, 5, 6, 3), np.eye(4))
    for path in (bold, mask):
        image = nib.load(path)
        values = np.asarray(image.dataobj).copy()
        values[0, 0, 0] = 0
        nib.save(nib.Nifti1Image(values, image.affine, image.header), path)
    matrix = np.eye(4)
    matrix[:3, 3] = [0.25, -0.5, 0.75]
    motion = np.repeat(np.eye(4)[None], 2, axis=0)
    motion[1, :3, 3] = [0.1, 0.2, 0.3]
    bbr = tmp_path / "bbr.txt"
    np.savetxt(bbr, matrix)
    motion_file = tmp_path / "motion.npy"
    np.save(motion_file, motion)
    capture = tmp_path / "captured-resampling"
    report = tmp_path / "report.json"
    outputs = {
        name: tmp_path / f"{name}.nii.gz"
        for name in ("mask_mni", "clean_mni", "preproc_t1w", "preproc_mni")
    }
    calls = []

    def original_sampler(*positional, **keywords):
        calls.append((positional, keywords))
        nib.save(nib.load(str(positional[0])), str(positional[3]))
        return positional[3]

    # The current volume route has no old direct alias. Also emulate removal
    # of the unused standalone motion-warp hook without changing MCFLIRT.
    monkeypatch.delattr(end_to_end, "resample_world", raising=False)
    monkeypatch.delattr(pipeline, "apply_motion_warp", raising=False)
    monkeypatch.setattr(end_to_end, "_resample_final_volume", original_sampler)
    for name in ("run_feat_core", "prepare_anatomical", "run_aroma_pipeline"):
        monkeypatch.setattr(end_to_end, name, getattr(end_to_end, name))
    monkeypatch.setattr(pipeline.TorchMCFLIRT, "run", pipeline.TorchMCFLIRT.run)
    monkeypatch.setattr(benchmark.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(benchmark, "source_hashes", lambda _: {})
    monkeypatch.setattr(
        benchmark, "locate_bids_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            bold=bold, sbref=None, t1w_images=(t1,),
        ),
    )
    expected = [
        ((mask, t1, matrix, outputs["mask_mni"]), dict(
            backend=backend, pre_affine_pull_ras=pull,
            interpolation="nearest", device="cpu")),
        ((bold, t1, matrix, outputs["clean_mni"]), dict(
            backend=backend, pre_affine_pull_ras=pull,
            output_mask=outputs["mask_mni"], interpolation="spline",
            boundary="periodic", batch_size=8, device="cpu")),
        ((bold, t1, matrix, outputs["preproc_t1w"]), dict(
            backend=backend, motion_pull_world=motion,
            interpolation="spline", boundary="grid-constant",
            coordinate_precision="fmriprep", batch_size=8, device="cpu")),
        ((bold, t1, matrix, outputs["preproc_mni"]), dict(
            backend=backend, pre_affine_pull_ras=pull,
            motion_pull_world=motion, interpolation="spline",
            boundary="grid-constant", coordinate_precision="fmriprep",
            batch_size=8, device="cpu")),
    ]

    def fake_volume(*args, **kwargs):
        for positional, keywords in expected:
            assert end_to_end._resample_final_volume(
                *positional, **keywords,
            ) is positional[3]
        metadata = tmp_path / "metadata.json"
        metadata.write_text(json.dumps({"FNIT": {"Report": {
            "registration_backend": backend,
            "mni_interpolation": "cubic-bspline-periodic",
            "ica_components": 2, "ica_converged": True, "ica_iterations": 1,
            "aroma_noise_components": 0, "wm_csf_motion_regression": True,
        }}}))
        return SimpleNamespace(
            **outputs, clean_native=bold, bbr_matrix=bbr,
            motion_pull=motion_file, mni_pull=pull, metadata=metadata,
            timing_seconds={"total": 1.0, "mni_resampling": 1.0},
        )

    monkeypatch.setattr(benchmark, "fMRIVolume_pipeline", fake_volume)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_bids.py", "volume", "--bids-root", str(tmp_path),
        "--derivatives-root", str(tmp_path / "derivatives"),
        "--subject", "test", "--source-root", str(tmp_path),
        "--source-revision", "test-revision", "--report-out", str(report),
        "--device", "cpu", "--mni-template", str(t1),
        "--registration-backend", backend,
        "--capture-intermediates", str(tmp_path / "intermediates"),
        "--capture-resampling-inputs", str(capture),
    ])
    benchmark.main()

    assert len(calls) == len(expected)
    for (actual_args, actual_keywords), (expected_args, expected_keywords) in zip(
        calls, expected,
    ):
        assert len(actual_args) == len(expected_args)
        assert all(actual is wanted for actual, wanted in zip(actual_args, expected_args))
        assert actual_keywords.keys() == expected_keywords.keys()
        assert all(actual_keywords[key] is value for key, value in expected_keywords.items())
    assert (capture / "mni_to_t1_pull_ras.nii.gz").read_bytes() == pull.read_bytes()
    np.testing.assert_array_equal(np.loadtxt(capture / "reference_to_source_world.txt"), matrix)
    recorded = json.loads(report.read_text())
    assert recorded["algorithm"]["registration_backend"] == backend
    assert recorded["source_revision"] == "test-revision"
    assert recorded["timing_seconds"]["private_capture_seconds_by_stage"].keys() == {"mni_resampling"}
