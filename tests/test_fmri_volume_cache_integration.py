"""Controlled integration gate, not a substitute for the real-data benchmark.

Only learned/statistical model fits are replaced. BIDS resolution, FEAT
motion resampling/filtering, AROMA classification/cleanup, coordinate
composition, final cubic resampling and derivative/cache I/O execute normally.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from fnit._nib import FNITNifti1Image
from fnit.flirt.coordinates import world_to_flirt_affine
from fnit.fmri import _anatomical, aroma_pipeline, bbr, end_to_end, normalization


def _write_image(path, data, affine, *, tr=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = FNITNifti1Image(np.asarray(data), affine)
    if tr is not None:
        image.header.set_zooms((*image.header.get_zooms()[:3], tr))
        image.header.set_xyzt_units(xyz="mm", t="sec")
    nib.save(image, path)
    return path


def test_multiple_bids_runs_reuse_anatomy_and_execute_run_specific_projection(tmp_path, monkeypatch):
    shape, frames = (8, 9, 10), 8
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    bids, derivatives = tmp_path / "bids", tmp_path / "derivatives"
    bids.mkdir()
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "Controlled unit fixture", "BIDSVersion": "1.9.0"}))
    (bids / "task-rest_bold.json").write_text(json.dumps({"TaskName": "rest", "RepetitionTime": 1.0}))
    grid = np.indices(shape, dtype=np.float32)
    t1 = _write_image(bids / "sub-01/anat/sub-01_T1w.nii.gz", (50 + grid.sum(0)).astype(np.int16), affine)
    template = _write_image(tmp_path / "template.nii.gz", (100 + grid.sum(0)).astype(np.float32), affine)
    template_mask = _write_image(tmp_path / "template_mask.nii.gz", np.ones(shape, dtype=np.uint8), affine)
    for run in (1, 2):
        temporal = np.sin(np.arange(frames) * 2 * np.pi / frames).astype(np.float32)
        data = (40 + run * 5 + grid[0] * run + grid[1])[..., None] + temporal
        stem = f"sub-01_task-rest_run-{run}"
        _write_image(bids / f"sub-01/func/{stem}_bold.nii.gz", data, affine, tr=1.0)
        _write_image(bids / f"sub-01/func/{stem}_sbref.nii.gz", data[..., 0], affine)

    # Small, aligned classifier masks keep the integration test independent of
    # the full atlas grid. No official-mask accuracy claim follows from them.
    controlled_module = tmp_path / "controlled_fmri" / "end_to_end.py"
    monkeypatch.setattr(end_to_end, "__file__", str(controlled_module))
    for name in ("csf", "edge", "out"):
        classifier_mask = np.zeros(shape, dtype=np.uint8)
        classifier_mask[0, 0, 0] = 1
        _write_image(controlled_module.parent / f"assets/mask_{name}.nii.gz", classifier_mask, affine)
    weights = tmp_path / "strip.pt"
    weights.write_bytes(b"controlled estimator identity")
    counts = {"strip_t1": 0, "strip_epi": 0, "fast": 0, "affine": 0, "fnirt": 0, "bbr": 0, "ica": 0}
    executions = []

    class Strip:
        def __init__(self, **kwargs):
            self.model_path = weights

        def __call__(self, path):
            source = nib.load(path)
            counts["strip_t1" if Path(path) == t1 else "strip_epi"] += 1
            return SimpleNamespace(image=FNITNifti1Image(np.asarray(source.dataobj), source.affine, source.header),
                                   mask=FNITNifti1Image(np.ones(source.shape, dtype=np.uint8), source.affine))

    class FAST:
        def __init__(self, **kwargs):
            pass

        def __call__(self, path, **kwargs):
            counts["fast"] += 1
            source = nib.load(path)
            white = np.full(shape, .1, dtype=np.float32)
            csf = white.copy()
            white[1:4], csf[4:7] = .9, .9
            return SimpleNamespace(pve_wm=FNITNifti1Image(white, source.affine), pve_csf=FNITNifti1Image(csf, source.affine))

    class FLIRT:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *args):
            counts["affine"] += 1
            return SimpleNamespace(matrix=np.eye(4), moving_to_fixed_world=np.eye(4))

    class FNIRT:
        def __init__(self, *, execution, **kwargs):
            executions.append(("fnirt", execution))

        def __call__(self, moving, fixed, initial, **kwargs):
            counts["fnirt"] += 1
            return SimpleNamespace(pull_transform=FNITNifti1Image(np.zeros((*fixed.shape, 3), dtype=np.float32), fixed.affine), qc={"controlled_estimator": True})

    def boundary_fit(epi, t1, wmseg, *, execution, **kwargs):
        counts["bbr"] += 1
        executions.append(("bbr", execution))
        world = np.eye(4)
        world[0, 3] = 2.0 * (counts["bbr"] - 1)
        epi_image, t1_image = nib.load(epi), nib.load(t1)
        matrix = world_to_flirt_affine(world, epi_image.affine, t1_image.affine,
                                      epi_image.shape, t1_image.shape,
                                      epi_image.header.get_zooms()[:3], t1_image.header.get_zooms()[:3])

        def save(*, output, omat):
            normalization.resample_world(epi, t1, np.linalg.inv(world), output, device="cpu")
            np.savetxt(omat, matrix)

        return SimpleNamespace(moving_to_fixed_world=world, matrix=matrix, save=save,
                               phase_timings={"initial_flirt": .01, "boundary_preparation": .02,
                                              "coarse_bbr": .03, "local_bbr": .04, "final_resampling": .05})

    def spatial_ica(source, mask, output, **kwargs):
        counts["ica"] += 1
        output = Path(output)
        output.mkdir(parents=True)
        image = nib.load(source)
        maps = _write_image(output / "thresholded.nii.gz", np.ones((*shape, 1), dtype=np.float32), image.affine)
        mix = np.sin(np.arange(frames)[:, None] * 2 * np.pi / frames)
        return SimpleNamespace(thresholded_maps=maps, mixing=mix,
                               frequency_power=np.array([[1], [0], [0], [0]]),
                               n_components=1, converged=True, n_iterations=1)

    monkeypatch.setattr(end_to_end, "SynthStrip", Strip)
    monkeypatch.setattr(_anatomical, "TorchFAST", FAST)
    monkeypatch.setattr(normalization, "TorchFLIRT", FLIRT)
    monkeypatch.setattr("fnit.fnirt.TorchFNIRT", FNIRT)
    monkeypatch.setattr(bbr, "register_bbr", boundary_fit)
    monkeypatch.setattr(aroma_pipeline, "decompose_spatial_ica", spatial_ica)
    # Zero motion-fit iterations execute the real motion resampler at identity.
    options = dict(subject="01", mni_template=template, mni_brain_mask=template_mask,
                   registration_backend="fnirt", fnirt_execution="reference",
                   bbr_execution="reference", motion_iterations=(0, 0, 0),
                   n_splits=2, device="cpu", batch_size=2)
    first = end_to_end.fMRIVolume_pipeline(bids, derivatives, run="1", **options)
    second = end_to_end.fMRIVolume_pipeline(bids, derivatives, run="2", **options)
    assert counts == {"strip_t1": 1, "strip_epi": 2, "fast": 1, "affine": 1, "fnirt": 1, "bbr": 2, "ica": 2}
    assert executions == [("fnirt", "reference"), ("bbr", "reference"), ("bbr", "reference")]
    first_report = json.loads(first.metadata.read_text())["FNIT"]["Report"]
    second_report = json.loads(second.metadata.read_text())["FNIT"]["Report"]
    assert not first_report["anatomical_cache"]["reused"]
    assert second_report["anatomical_cache"]["reused"]
    assert first_report["anatomical_cache"]["fingerprint"] == second_report["anatomical_cache"]["fingerprint"]
    for report in (first_report, second_report):
        assert report["configuration"]["reuse_anatomical"] is True
        assert report["configuration"]["bbr_execution"] == "reference"
        assert report["configuration"]["fnirt_execution"] == "reference"
        assert report["configuration"]["anatomical_cache"] == report["anatomical_cache"]
    for phase in ("t1_synthstrip", "template_preparation", "fast", "t1_to_mni_affine", "t1_to_mni_nonlinear", "warp_conversion"):
        assert second.timing_seconds[phase] == 0
    for result in (first, second):
        for phase in ("bbr_initial_flirt", "bbr_refinement", "bbr_final_resampling", "mni_resampling"):
            assert result.timing_seconds[phase] > 0
        image = nib.load(result.clean_mni)
        assert image.shape == (*shape, frames)
        assert image.get_data_dtype() == np.dtype("float32")
        assert image.header.get_zooms() == (2, 2, 2, 1)
    native, mni = np.asarray(nib.load(second.clean_native).dataobj), np.asarray(nib.load(second.clean_mni).dataobj)
    assert np.count_nonzero(mni[0]) == 0
    np.testing.assert_allclose(mni[1:], native[:-1], atol=.01, rtol=1e-6)
    assert np.linalg.norm(np.loadtxt(first.bbr_matrix) - np.loadtxt(second.bbr_matrix)) > 0
    assert nib.load(first.t1_brain).get_data_dtype() == np.dtype("float32")
