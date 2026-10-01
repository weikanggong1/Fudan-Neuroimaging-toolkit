"""Regressions for invalid surface data, time axes, and failed publication."""

from pathlib import Path
import os
import subprocess

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.surface import SurfaceHemisphere, _gifti_count
from fnit.fmri import surface_fmriprep as surface


def _metric(path, values, structure=None):
    array = nib.gifti.GiftiDataArray(np.asarray(values, dtype=np.float32))
    metadata = {"AnatomicalStructurePrimary": structure} if structure else {}
    nib.save(nib.GiftiImage(darrays=[array], meta=nib.gifti.GiftiMetaData(metadata)), path)
    return path


def _bold(path, affine=None, shape=(2, 2, 2, 2), tr=0.8):
    image = nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm", "sec")
    image.header.set_zooms((*nib.affines.voxel_sizes(image.affine), tr))
    nib.save(image, path)
    return path


@pytest.fixture
def official_assets():
    configured = os.environ.get("FNIT_TEST_FMRI_SURFACE_ASSETS")
    if not configured:
        pytest.skip("Set FNIT_TEST_FMRI_SURFACE_ASSETS to verified public templates")
    root = Path(configured)
    mesh = root / "global/templates/standard_mesh_atlases"
    return (mesh / "L.atlasroi.32k_fs_LR.shape.gii",
            mesh / "R.atlasroi.32k_fs_LR.shape.gii",
            root / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz")


@pytest.fixture
def cifti_inputs(tmp_path, official_assets):
    atlas = nib.load(official_assets[2])
    bold = _bold(tmp_path / "mni.nii.gz", atlas.affine, (*atlas.shape, 2))
    metrics = []
    for hemi in ("L", "R"):
        path = tmp_path / f"{hemi}.func.gii"
        frames = [nib.gifti.GiftiDataArray(np.full(32492, t + 1, dtype=np.float32)) for t in range(2)]
        nib.save(nib.GiftiImage(darrays=frames), path)
        metrics.append(path)
    return (bold, *metrics, *official_assets, tmp_path / "output.dtseries.nii")


def test_negative_roi_is_rejected(tmp_path):
    path = _metric(tmp_path / "negative.shape.gii", [1, -1, 0])
    with pytest.raises(ValueError, match="nonnegative"):
        _gifti_count(path, "native ROI", metric=True)


def test_opposite_hemisphere_metadata_is_rejected(tmp_path):
    path = _metric(tmp_path / "roi.shape.gii", [1, 1, 0], "CortexRight")
    with pytest.raises(ValueError, match="opposite hemisphere"):
        _gifti_count(path, "left ROI", metric=True, hemisphere="LEFT")


@pytest.mark.parametrize("values", [np.ones(5, dtype=np.float32),
                                   np.full(32492, np.nan, dtype=np.float32)])
def test_truncated_or_nonfinite_metric_is_rejected(tmp_path, values):
    path = _metric(tmp_path / "invalid.func.gii", values)
    with pytest.raises(ValueError, match="32,492 finite"):
        surface._metric_frames(path, "LEFT", 1)


@pytest.mark.parametrize("tr", [0.0, -1.0, np.nan, np.inf])
def test_invalid_tr_is_rejected(tr):
    image = nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm", "sec")
    image.header["pixdim"][4] = tr
    with pytest.raises(ValueError, match="positive and finite"):
        surface._tr_seconds(image, "BOLD")


def test_time_units_match_raw_tr_but_inconsistent_axis_is_rejected():
    image = nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm", "msec")
    image.header.set_zooms((1, 1, 1, 800))
    assert surface._tr_seconds(image, "BOLD", 0.8) == 0.8
    with pytest.raises(ValueError, match="differs"):
        surface._tr_seconds(image, "BOLD", 1.0)


@pytest.mark.parametrize("artifact", ["L.32k.func.gii", "R.32k.func.gii", "coverage.json"])
def test_partial_existing_outputs_are_protected_before_work(tmp_path, artifact):
    output = tmp_path / "existing"
    output.mkdir()
    path = output / artifact
    path.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        surface.run_fmriprep_surface_projection(
            "absent.nii.gz", "absent.nii.gz", None, None, "L", "R", "dseg", output,
        )
    assert path.read_bytes() == b"previous result"


def test_dangling_output_symlink_is_protected_before_loading_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    path = output / "coverage.json"
    path.symlink_to(output / "missing-target")
    with pytest.raises(FileExistsError):
        surface.run_fmriprep_surface_projection(
            "absent.nii.gz", "absent.nii.gz", None, None, "L", "R", "dseg", output,
        )
    assert path.is_symlink() and not path.exists()


def test_low_level_projection_rejects_nonmillimeter_t1w_geometry(tmp_path):
    native = _bold(tmp_path / "t1w.nii.gz")
    mni = _bold(tmp_path / "mni.nii.gz")
    image = nib.load(native)
    image.header.set_xyzt_units("meter", "sec")
    nib.save(image, native)
    with pytest.raises(ValueError, match="T1w BOLD.*millimeter"):
        surface.run_fmriprep_surface_projection(
            native, mni, None, None, "L", "R", "dseg", tmp_path / "output", tr_seconds=0.8,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("damage", ["empty", "nan", "fractional"])
def test_modified_dseg_cannot_emit_mislabeled_91k(tmp_path, cifti_inputs, damage):
    arguments = list(cifti_inputs)
    image = nib.load(arguments[5])
    values = np.zeros(image.shape, dtype=np.float32)
    if damage == "nan":
        values[0, 0, 0] = np.nan
    elif damage == "fractional":
        values[0, 0, 0] = 26.5
    arguments[5] = tmp_path / "damaged_dseg.nii.gz"
    nib.save(nib.Nifti1Image(values, image.affine), arguments[5])
    with pytest.raises(ValueError, match="SHA-256"):
        surface.create_fmriprep_cifti(*arguments)
    assert not arguments[-1].exists()


def test_nonfinite_mni_bold_cannot_emit_a_cifti(cifti_inputs):
    arguments = list(cifti_inputs)
    image = nib.load(arguments[0])
    values = np.asarray(image.dataobj).copy()
    values[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, image.affine, image.header), arguments[0])
    with pytest.raises(ValueError, match="MNI BOLD contains nonfinite"):
        surface.create_fmriprep_cifti(*arguments)
    assert not arguments[-1].exists()


def test_cifti_embeds_same_metadata_and_original_tr_as_public_sidecar(cifti_inputs):
    output = surface.create_fmriprep_cifti(*cifti_inputs, tr_seconds=0.8)
    image = nib.load(output)
    metadata = surface.fmriprep_cifti_metadata()
    assert image.header.get_axis(0).step == 0.8
    assert image.header.matrix.metadata["Density"] == metadata["Density"]
    # NiWorkflows passes the nested dictionary directly to Cifti2MetaData.
    assert image.header.matrix.metadata["SpatialReference"] == str(metadata["SpatialReference"])


def _hemisphere(tmp_path, hemisphere, atlas_roi):
    native_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], dtype=np.float32)
    native_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    files = {}
    for field in ("white", "pial", "midthickness", "registered_sphere", "atlas_sphere", "atlas_midthickness"):
        is_atlas = field.startswith("atlas")
        points = np.resize(native_points, (32492, 3)) if is_atlas else native_points
        image = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(points, intent="NIFTI_INTENT_POINTSET"),
                                       nib.gifti.GiftiDataArray(native_faces, intent="NIFTI_INTENT_TRIANGLE")])
        path = tmp_path / f"{hemisphere}.{field}.surf.gii"
        nib.save(image, path)
        files[field] = path
    files["native_roi"] = _metric(tmp_path / f"{hemisphere}.native_roi.shape.gii", np.ones(4))
    files["atlas_roi"] = atlas_roi
    return SurfaceHemisphere(**files)


def test_failed_workbench_leaves_previous_outputs_intact(tmp_path, cifti_inputs, monkeypatch):
    bold, _, _, left_roi, right_roi, dseg, _ = cifti_inputs
    left = _hemisphere(tmp_path, "L", left_roi)
    right = _hemisphere(tmp_path, "R", right_roi)
    output = tmp_path / "published"
    output.mkdir()
    previous = output / "L.32k.func.gii"
    previous.write_bytes(b"previous result")
    monkeypatch.setattr(surface.shutil, "which", lambda _: "/mock/wb_command")

    def fail_after_partial_write(arguments, **kwargs):
        Path(arguments[4]).write_bytes(b"partial Workbench output")
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(surface.subprocess, "run", fail_after_partial_write)
    with pytest.raises(subprocess.CalledProcessError):
        surface.run_fmriprep_surface_projection(
            bold, bold, left, right, left_roi, right_roi, dseg, output,
            tr_seconds=0.8, overwrite=True,
        )
    assert previous.read_bytes() == b"previous result"
    assert sorted(path.name for path in output.iterdir()) == [previous.name]
    assert not list(tmp_path.glob(".fnit-surface-*"))


@pytest.mark.parametrize("names", [("first", "second"), ("native/left", "native/right")])
def test_publication_rolls_back_if_second_file_cannot_be_replaced(tmp_path, monkeypatch, names):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    for name in names:
        (staging / name).parent.mkdir(parents=True, exist_ok=True)
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        (staging / name).write_bytes(b"new")
        (output / name).write_bytes(b"old")
    original = surface.os.replace

    def fail_second(source, target):
        if Path(source) == staging / names[1]:
            raise OSError("simulated disk failure")
        return original(source, target)

    monkeypatch.setattr(surface.os, "replace", fail_second)
    with pytest.raises(OSError, match="disk failure"):
        surface._publish_projection(staging, output, names, True)
    assert [(output / name).read_bytes() for name in names] == [b"old", b"old"]


def test_no_overwrite_publication_keeps_file_created_after_initial_checks(tmp_path, monkeypatch):
    staging, output = tmp_path / "staging", tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    for name in ("first", "second"):
        (staging / name).write_bytes(b"new result")
    original_link = surface.os.link

    def concurrent_file(source, target):
        if Path(target).name == "second":
            Path(target).write_bytes(b"concurrent result")
        return original_link(source, target)

    monkeypatch.setattr(surface.os, "link", concurrent_file)
    with pytest.raises(FileExistsError):
        surface._publish_projection(staging, output, ("first", "second"), False)
    assert not (output / "first").exists()
    assert (output / "second").read_bytes() == b"concurrent result"


def test_shared_publisher_preserves_dangling_link(tmp_path):
    staging, output = tmp_path / "staging", tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (staging / "result").write_bytes(b"new result")
    destination = output / "result"
    destination.symlink_to(output / "absent")
    with pytest.raises(FileExistsError):
        surface._publish_projection(staging, output, ("result",), False)
    assert destination.is_symlink()


def test_matching_frame_counts_do_not_hide_t1w_mni_tr_conflict(tmp_path):
    native = _bold(tmp_path / "t1w.nii.gz", tr=0.9)
    mni = _bold(tmp_path / "mni.nii.gz", tr=0.8)
    with pytest.raises(ValueError, match="T1w BOLD TR differs"):
        surface.run_fmriprep_surface_projection(
            native, mni, None, None, "L", "R", "dseg", tmp_path / "output", tr_seconds=0.8,
        )
    assert not (tmp_path / "output").exists()


def test_wrong_standard_space_grid_is_rejected(cifti_inputs):
    arguments = list(cifti_inputs)
    image = nib.load(arguments[0])
    affine = image.affine.copy()
    affine[:3, :3] /= 2
    image = nib.Nifti1Image(np.asarray(image.dataobj), affine, image.header)
    nib.save(image, arguments[0])
    with pytest.raises(ValueError, match="HCP dseg affine"):
        surface.create_fmriprep_cifti(*arguments)
    assert not arguments[-1].exists()


@pytest.mark.parametrize("force_default_mmap", [False, True])
def test_generated_cifti_has_no_live_mapping_at_temp_cleanup(
    tmp_path, cifti_inputs, monkeypatch, force_default_mmap,
):
    """Model NFS EBUSY using actual nibabel mmap ownership, not a benchmark."""
    import errno
    import mmap
    import shutil
    import weakref
    from tempfile import TemporaryDirectory
    from nibabel.arrayproxy import ArrayProxy

    bold, left_metric, right_metric, left_roi, right_roi, dseg, _ = cifti_inputs
    left = _hemisphere(tmp_path, "L", left_roi)
    right = _hemisphere(tmp_path, "R", right_roi)
    output = tmp_path / "published"
    mappings = []
    original_array = ArrayProxy.__array__
    original_load = nib.load

    def generated_cifti(path):
        path = Path(path)
        return path.name.endswith(".dtseries.nii") and path.parent.name.startswith(".fnit-surface-")

    def observe_array(proxy, *args, **kwargs):
        result = original_array(proxy, *args, **kwargs)
        if generated_cifti(proxy.file_like):
            owner = result
            while owner is not None:
                if isinstance(owner, mmap.mmap):
                    mappings.append(weakref.ref(owner))
                    break
                owner = getattr(owner, "base", None)
        return result

    def load(path, *args, **kwargs):
        if force_default_mmap and generated_cifti(path):
            kwargs["mmap"] = True
        return original_load(path, *args, **kwargs)

    class NFSCheckingTemporaryDirectory(TemporaryDirectory):
        def __exit__(self, exc, value, traceback):
            if any(ref() is not None and not ref().closed for ref in mappings):
                raise OSError(errno.EBUSY, "NFS still owns a generated CIFTI mapping")
            return super().__exit__(exc, value, traceback)

    def workbench(arguments, **kwargs):
        command = arguments[1]
        index = {"-volume-to-surface-mapping": 4, "-metric-dilate": 5,
                 "-metric-mask": 4, "-metric-resample": 6}[command]
        destination = Path(arguments[index])
        shutil.copyfile(left_metric if destination.name.startswith("L.") else right_metric, destination)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(ArrayProxy, "__array__", observe_array)
    monkeypatch.setattr(surface.nib, "load", load)
    monkeypatch.setattr(surface, "TemporaryDirectory", NFSCheckingTemporaryDirectory)
    monkeypatch.setattr(surface.shutil, "which", lambda _: "/mock/wb_command")
    monkeypatch.setattr(surface.subprocess, "run", workbench)
    arguments = (bold, bold, left, right, left_roi, right_roi, dseg, output)
    if force_default_mmap:
        # Independent failure control: restoring nibabel's mmap recreates the owner.
        with pytest.raises(OSError, match="NFS still owns"):
            surface.run_fmriprep_surface_projection(*arguments, tr_seconds=0.8)
        assert mappings
        return
    result = surface.run_fmriprep_surface_projection(*arguments, tr_seconds=0.8)
    assert not mappings
    assert result.dtseries.is_file() and result.coverage_report.is_file()
    image = original_load(result.dtseries, mmap=False, keep_file_open=False)
    assert image.shape == (2, 91282)
    assert image.header.get_axis(0).step == 0.8
    assert np.isfinite(np.asarray(image.dataobj)).all()
    assert not list(tmp_path.glob(".fnit-surface-*"))
