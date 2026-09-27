"""Geometry checks for inverse FNIT pull and FreeSurfer tkRAS conversion."""

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

from fnit.fmri.surface_prepare import invert_mni_to_t1_pull, prepare_mni_surface_geometry


def _pull(path, shape=(20, 20, 20), nonlinear=False):
    xyz = np.indices(shape, dtype=np.float32) * 2
    field = np.zeros((*shape, 3), np.float32)
    if nonlinear:
        field[..., 0] = 0.8 * np.sin(xyz[0] / 6)
        field[..., 1] = 0.5 * np.cos(xyz[1] / 7)
        field[..., 2] = 0.4 * np.sin(xyz[2] / 5)
    else:
        field[..., 0] = 1.0
    nib.save(nib.Nifti1Image(field, np.diag([2, 2, 2, 1])), str(path))
    return path


def test_inverse_uses_nonlinear_pull_and_checks_residual(tmp_path):
    pull = _pull(tmp_path / "pull.nii.gz", nonlinear=True)
    mni = np.array([[8, 10, 12], [16, 20, 14], [26, 22, 30]], np.float64)
    shift = np.column_stack((0.8 * np.sin(mni[:, 0] / 6),
                             0.5 * np.cos(mni[:, 1] / 7),
                             0.4 * np.sin(mni[:, 2] / 5)))
    t1 = mni + shift
    recovered, residual = invert_mni_to_t1_pull(
        t1, pull, np.eye(4), tolerance_mm=0.02, max_iterations=80,
    )
    assert np.max(np.linalg.norm(recovered - mni, axis=1)) < 0.03
    assert residual <= 0.02
    assert np.max(np.linalg.norm(t1 - mni, axis=1)) > 0.4
    with pytest.raises(RuntimeError, match="did not converge"):
        invert_mni_to_t1_pull(t1, pull, np.eye(4),
                              tolerance_mm=1e-4, max_iterations=1)


def test_tkras_to_mni_preserves_native_topology_and_midpoints(tmp_path):
    subject = tmp_path / "subject"
    (subject / "mri/orig").mkdir(parents=True)
    (subject / "surf").mkdir()
    orig = nib.MGHImage(np.zeros((16, 16, 16), np.float32), np.eye(4))
    nib.save(orig, str(subject / "mri/orig.mgz"))
    nib.save(orig, str(subject / "mri/orig/001.mgz"))
    tk_to_world = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    world_to_tk = np.linalg.inv(tk_to_world)
    white_world = np.array([[8, 8, 8], [10, 8, 8], [8, 10, 8], [8, 8, 10]], np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    for hemi in ("lh", "rh"):
        offset = 0 if hemi == "lh" else 2
        for name, extra in (("white", 0), ("pial", 0.6)):
            points = white_world + offset + extra
            tk = points @ world_to_tk[:3, :3].T + world_to_tk[:3, 3]
            fsio.write_geometry(str(subject / "surf" / f"{hemi}.{name}"),
                                tk.astype(np.float32), faces)
    pull = _pull(tmp_path / "pull.nii.gz")
    result = prepare_mni_surface_geometry(subject, pull, np.eye(4), tmp_path / "out")
    for pair, offset in ((result.left, 0), (result.right, 2)):
        white = nib.load(str(pair.white))
        pial = nib.load(str(pair.pial))
        mid = nib.load(str(pair.midthickness))
        xyz = white.darrays[0].data
        np.testing.assert_allclose(xyz, white_world + offset - [1, 0, 0], atol=0.02)
        np.testing.assert_allclose(mid.darrays[0].data,
                                   (xyz + pial.darrays[0].data) / 2, atol=1e-5)
        np.testing.assert_array_equal(white.darrays[1].data, faces)
        assert pair.vertex_count == 4
        assert pair.max_inverse_residual_mm <= 0.05


def test_scanner_original_is_not_accepted_as_surface_geometry(tmp_path):
    subject = tmp_path / "subject"
    (subject / "mri/orig").mkdir(parents=True)
    nib.save(nib.MGHImage(np.zeros((4, 4, 4), np.float32), np.eye(4)),
             str(subject / "mri/orig/001.mgz"))
    with pytest.raises(FileNotFoundError, match="mri/orig.mgz"):
        prepare_mni_surface_geometry(subject, tmp_path / "missing.nii.gz",
                                     np.eye(4), tmp_path / "out")


def test_wmparc_labels_follow_nonlinear_pull_not_initial_affine(tmp_path):
    from fnit.fmri.surface_prepare import _resample_wmparc_to_mni

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    labels = np.zeros((8, 8, 8), np.int32)
    labels[3, 2, 2] = 17
    wmparc = tmp_path / "wmparc.nii.gz"
    reference = tmp_path / "MNI_2mm.nii.gz"
    pull = tmp_path / "pull.nii.gz"
    output = tmp_path / "wmparc_MNI.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), str(wmparc))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), affine), str(reference))
    displacement = np.zeros((8, 8, 8, 3), np.float32)
    displacement[..., 0] = 2.0
    nib.save(nib.Nifti1Image(displacement, affine), str(pull))
    _resample_wmparc_to_mni(wmparc, pull, reference, output, "cpu")
    warped = np.asarray(nib.load(str(output)).dataobj)
    assert warped[2, 2, 2] == 17
    assert warped[3, 2, 2] == 0
    assert np.count_nonzero(warped) == 1


def test_fs_sphere_preparation_uses_thickness_roi_cleanup_and_subject_labels(
    tmp_path, monkeypatch,
):
    import shutil
    import subprocess
    from fnit.fmri.surface_prepare import prepare_fs_sphere_projection_inputs

    subject = tmp_path / "subject"
    (subject / "mri").mkdir(parents=True)
    (subject / "surf").mkdir()
    original = nib.MGHImage(np.zeros((16, 16, 16), np.float32), np.eye(4))
    nib.save(original, str(subject / "mri/orig.mgz"))
    segmentation = np.zeros((16, 16, 16), np.float32)
    segmentation[5:10, 5:10, 5:10] = 17
    nib.save(nib.MGHImage(segmentation, np.eye(4)), str(subject / "mri/wmparc.mgz"))
    tk_to_world = original.affine @ np.linalg.inv(original.header.get_vox2ras_tkr())
    world_to_tk = np.linalg.inv(tk_to_world)
    points = np.array([[8, 8, 8], [10, 8, 8], [8, 10, 8], [8, 8, 10]], np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    for hemi in ("lh", "rh"):
        for name, offset in (("white", 0.0), ("pial", 0.5), ("sphere.reg", 0.0)):
            world = points + offset
            tk = world @ world_to_tk[:3, :3].T + world_to_tk[:3, 3]
            fsio.write_geometry(str(subject / "surf" / f"{hemi}.{name}"),
                                tk.astype(np.float32), faces)
        fsio.write_morph_data(str(subject / "surf" / f"{hemi}.thickness"),
                              np.array([-1, 0, 1, 0], np.float32))
    assets = tmp_path / "assets"
    mesh = assets / "global/templates/standard_mesh_atlases"
    for hemi in ("L", "R"):
        needed = (
            f"fs_{hemi}/fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii",
            f"fs_{hemi}/fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR.spherical_std.164k_fs_{hemi}.surf.gii",
            f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii",
            f"{hemi}.atlasroi.164k_fs_LR.shape.gii",
            f"{hemi}.sphere.32k_fs_LR.surf.gii",
            f"{hemi}.atlasroi.32k_fs_LR.shape.gii",
        )
        for name in needed:
            path = mesh / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    for name in ("FreeSurferAllLut.txt", "FreeSurferSubcorticalLabelTableLut.txt"):
        path = assets / "global/config" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    atlas = assets / "global/templates/91282_Greyordinates/Atlas_ROIs.2.nii.gz"
    atlas.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((20, 20, 20), np.int16), affine), str(atlas))
    reference = tmp_path / "MNI_2mm.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), np.float32), affine),
             str(reference))
    pull = _pull(tmp_path / "pull.nii.gz")
    monkeypatch.setattr(shutil, "which", lambda command: "/fake/wb_command")
    commands = []

    def fake_wb(argv, **kwargs):
        commands.append(argv)
        operation = argv[1]
        if operation == "-volume-label-import":
            destination = argv[-2]
        elif operation == "-metric-resample":
            destination = argv[6]
        elif operation == "-metric-math":
            destination = argv[3]
        else:
            destination = argv[-1]
        if operation in ("-metric-fill-holes", "-metric-remove-islands"):
            shutil.copyfile(argv[3], destination)
        elif operation == "-metric-resample":
            nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
                np.array([0, 1, 0, 0], np.float32), intent="NIFTI_INTENT_SHAPE")]), destination)
        elif operation == "-metric-math":
            individual = np.asarray(nib.load(argv[-1]).darrays[0].data)
            atlas_roi = np.asarray(nib.load(argv[6]).darrays[0].data)
            nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
                ((individual + atlas_roi) > 0).astype(np.float32),
                intent="NIFTI_INTENT_SHAPE")]), destination)
        elif operation == "-surface-resample":
            fake_mid = nib.GiftiImage(darrays=[
                nib.gifti.GiftiDataArray(
                    np.zeros((32492, 3), np.float32),
                    intent="NIFTI_INTENT_POINTSET"),
                nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
            ])
            nib.save(fake_mid, destination)
        elif operation == "-volume-label-import":
            shutil.copyfile(argv[2], destination)
        else:
            shutil.copyfile(argv[2], destination)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_wb)
    result = prepare_fs_sphere_projection_inputs(
        subject, pull, np.eye(4), reference, assets, tmp_path / "prepared",
    )
    for hemi in (result.left, result.right):
        np.testing.assert_array_equal(
            nib.load(str(hemi.native_roi)).darrays[0].data,
            [1, 1, 1, 0],
        )
        assert nib.load(str(hemi.atlas_midthickness)).darrays[0].data.shape == (32492, 3)
    assert result.subject_rois.is_file()
    assert result.atlas_rois == atlas
    operations = [command[1] for command in commands]
    assert operations == (
        ["-surface-sphere-project-unproject", "-metric-fill-holes",
         "-metric-remove-islands", "-metric-resample",
         "-metric-math", "-surface-resample"] * 2
        + ["-volume-label-import"] * 2
    )
    assert commands[4][2] == "(atlas + individual) > 0"
    assert commands[-1][-1] == "-discard-others"
