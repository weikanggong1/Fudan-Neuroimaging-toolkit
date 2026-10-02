"""Geometry preparation and publication contracts, not accuracy benchmarks."""

from pathlib import Path
import subprocess

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

from fnit.fmri import surface_prepare as prepare


@pytest.fixture
def reconstruction(tmp_path):
    subject = tmp_path / "recon"
    (subject / "mri").mkdir(parents=True)
    (subject / "surf").mkdir()
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = (10., 20., 30.)
    nib.save(nib.MGHImage(np.zeros((3, 4, 5), dtype=np.float32), affine), subject / "mri/orig.mgz")
    white = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    for hemi in ("lh", "rh"):
        for name, points in (("white", white), ("pial", white * 2),
                             ("midthickness", white * 1.3), ("graymid", white * 1.7),
                             ("sphere.reg", white * 100)):
            fsio.write_geometry(str(subject / "surf" / f"{hemi}.{name}"), points, faces)
        fsio.write_morph_data(str(subject / "surf" / f"{hemi}.thickness"), np.ones(4, np.float32))
    return subject


def _geometry_names():
    return tuple(f"{hemi}.{name}.T1w.native.surf.gii" for hemi in ("lh", "rh")
                 for name in ("white", "pial", "midthickness"))


def _previous(output, names):
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous result")


def test_right_existing_geometry_is_protected_before_left_is_written(reconstruction, tmp_path):
    output = tmp_path / "geometry"
    _previous(output, ["rh.midthickness.T1w.native.surf.gii"])
    with pytest.raises(FileExistsError):
        prepare.prepare_t1w_surface_geometry(reconstruction, output)
    assert sorted(path.name for path in output.iterdir()) == ["rh.midthickness.T1w.native.surf.gii"]


def test_missing_right_middle_does_not_overwrite_previous_left_geometry(reconstruction, tmp_path):
    for name in ("midthickness", "graymid"):
        (reconstruction / "surf" / f"rh.{name}").unlink()
    output = tmp_path / "geometry"
    _previous(output, _geometry_names())
    with pytest.raises(FileNotFoundError, match="rh.midthickness"):
        prepare.prepare_t1w_surface_geometry(reconstruction, output, overwrite=True)
    assert all((output / name).read_bytes() == b"previous result" for name in _geometry_names())


def test_geometry_serialization_failure_keeps_complete_previous_result(reconstruction, tmp_path, monkeypatch):
    output = tmp_path / "geometry"
    _previous(output, _geometry_names())
    original = prepare._write_gifti

    def fail_right_mid(path, *args):
        if Path(path).name == "rh.midthickness.T1w.native.surf.gii":
            raise OSError("controlled write failure")
        return original(path, *args)

    monkeypatch.setattr(prepare, "_write_gifti", fail_right_mid)
    with pytest.raises(OSError, match="controlled write failure"):
        prepare.prepare_t1w_surface_geometry(reconstruction, output, overwrite=True)
    assert all((output / name).read_bytes() == b"previous result" for name in _geometry_names())
    assert not list(tmp_path.glob(".fnit-t1-surfaces-*"))


def test_existing_middle_keeps_exact_world_transform_and_native_topology(reconstruction, tmp_path):
    world = np.eye(4)
    world[0, 3] = 5.0
    result = prepare.prepare_t1w_surface_geometry(reconstruction, tmp_path / "geometry",
                                                 fsnative_to_t1w=world)
    orig = nib.load(reconstruction / "mri/orig.mgz")
    transform = world @ orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    for hemi, pair in (("lh", result.left), ("rh", result.right)):
        points, faces = fsio.read_geometry(str(reconstruction / "surf" / f"{hemi}.midthickness"))
        expected = np.asarray(points @ transform[:3, :3].T + transform[:3, 3], np.float32)
        image = nib.load(pair.midthickness)
        np.testing.assert_array_equal(image.darrays[0].data, expected)
        np.testing.assert_array_equal(image.darrays[1].data, faces)
        assert pair.midthickness_source.name == f"{hemi}.midthickness"
        assert len(list(pair.white.parent.glob("*.surf.gii"))) == 6


@pytest.mark.parametrize("dangling", [False, True])
def test_existing_preparation_sphere_is_protected_before_geometry(reconstruction, tmp_path, dangling):
    output = tmp_path / "prepared"
    output.mkdir()
    sphere = output / "R.sphere.FS_to_fsLR.native.surf.gii"
    if dangling:
        sphere.symlink_to(output / "absent")
    else:
        sphere.write_bytes(b"previous sphere")
    with pytest.raises(FileExistsError):
        prepare.prepare_fmriprep_surface_inputs(reconstruction, tmp_path / "assets", output)
    assert not (output / "native").exists()
    assert sphere.is_symlink() if dangling else sphere.read_bytes() == b"previous sphere"


@pytest.mark.parametrize("previous", [False, True])
def test_workbench_failure_does_not_publish_partial_preparation(reconstruction, tmp_path, monkeypatch, previous):
    output = tmp_path / "prepared"
    if previous:
        _previous(output, prepare._preparation_names())
    monkeypatch.setattr("shutil.which", lambda _: "/mock/wb_command")

    def command(arguments, **kwargs):
        target = Path(arguments[-1])
        if target.name.startswith("R."):
            raise subprocess.CalledProcessError(1, arguments)
        target.write_bytes(b"staged workbench result")

    monkeypatch.setattr("subprocess.run", command)
    with pytest.raises(subprocess.CalledProcessError):
        prepare.prepare_fmriprep_surface_inputs(reconstruction, tmp_path / "assets", output,
                                               overwrite=previous)
    if previous:
        assert all((output / name).read_bytes() == b"previous result" for name in prepare._preparation_names())
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".fnit-surface-preparation-*"))


def test_complete_preparation_returns_persistent_paths(reconstruction, tmp_path, monkeypatch):
    output = tmp_path / "prepared"
    monkeypatch.setattr("shutil.which", lambda _: "/mock/wb_command")
    monkeypatch.setattr("subprocess.run", lambda args, **kwargs: Path(args[-1]).write_bytes(b"workbench result"))
    result = prepare.prepare_fmriprep_surface_inputs(reconstruction, tmp_path / "assets", output)
    assert len([p for p in output.rglob("*") if p.is_file()]) == 16
    paths = (*result.initial_spheres, *result.individual_rois,
             result.geometry.left.white, result.geometry.left.midthickness,
             result.geometry.right.white, result.geometry.right.midthickness)
    assert all(path.is_file() and path.is_relative_to(output) for path in paths)
