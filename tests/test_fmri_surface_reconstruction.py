"""重建路由与文件契约回归；native 重计算被 mock，这些小 fixture 不是 benchmark。"""

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import zipfile

import nibabel as nib
from nibabel.freesurfer import io as fsio
import numpy as np
import pytest

from fnit.fmri import surface_reconstruction as adapter


def _source(tmp_path, *, dtype=np.float32, units=True, scaled=False):
    values = np.arange(64).reshape(4, 4, 4).astype(dtype)
    image = nib.Nifti1Image(values, np.diag([1, 1, 1, 1]))
    image.set_sform(image.affine, code=1)
    if units:
        image.header.set_xyzt_units("mm", "sec")
    if scaled:
        image.header.set_slope_inter(2, 3)
    path = tmp_path / "raw T1w.nii.gz"
    nib.save(image, path)
    return path


def _subject(path, source, *, middle=False):
    path = Path(path)
    (path / "mri/orig").mkdir(parents=True)
    (path / "surf").mkdir()
    image = nib.load(source)
    for name in ("mri/orig.mgz", "mri/orig/001.mgz"):
        nib.save(nib.MGHImage(np.asarray(image.dataobj, dtype=np.float32), image.affine), path / name)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    for hemi in ("lh", "rh"):
        for name in ("white", "pial", "sphere", "sphere.reg"):
            fsio.write_geometry(path / "surf" / f"{hemi}.{name}", vertices + [0, 0, name == "pial"], faces)
        for name in ("thickness", "sulc"):
            fsio.write_morph_data(path / "surf" / f"{hemi}.{name}", np.ones(3, dtype=np.float32))
        if middle:
            fsio.write_geometry(path / "surf" / f"{hemi}.graymid", vertices + [0, 0, 0.5], faces)
    return path


def _tree_hash(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def _binary(tmp_path, name="mris_expand"):
    path = tmp_path / "native" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def native(tmp_path, monkeypatch):
    executable = _binary(tmp_path)
    monkeypatch.setenv("FNIT_RECON_ALL_BIN_DIR", str(executable.parent))
    calls = []

    def run(argv, *, cwd, env, log):
        calls.append({"argv": argv, "cwd": cwd, "env": dict(env)})
        log.write_text("mock native contract check\n")
        if "-thickness" in argv:
            # Fixture output checks argv/topology plumbing only, not the native algorithm.
            vertices, faces = fsio.read_geometry(argv[2])
            fsio.write_geometry(argv[-1], vertices + [0, 0, 0.5], faces)
        else:
            _subject(Path(argv[argv.index("-sd") + 1]) / argv[argv.index("-s") + 1],
                     argv[argv.index("-i") + 1])

    monkeypatch.setattr(adapter, "_run_command", run)
    return executable, calls


def test_provided_complete_is_read_only_and_needs_no_native(tmp_path, monkeypatch):
    source = _source(tmp_path)
    subject = _subject(tmp_path / "original", source, middle=True)
    before = _tree_hash(subject)
    monkeypatch.setenv("FNIT_RECON_ALL_BIN_DIR", str(tmp_path / "absent"))
    result = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=subject)
    assert result.backend == "provided" and result.reused
    assert result.subject_dir == subject and result.source_t1w == source
    assert not (tmp_path / "work").exists()
    assert _tree_hash(subject) == before
    assert result.metadata["commands"] == []


def test_missing_middle_is_expanded_in_owned_copy(tmp_path, native):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source)
    before = _tree_hash(original)
    output = tmp_path / "derivatives"
    result = adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=original, output_dir=output)
    executable, calls = native
    assert result.subject_dir == output / "subject"
    assert not result.reused and _tree_hash(original) == before
    assert len(calls) == 2
    for hemi, call in zip(("lh", "rh"), calls):
        assert call["argv"] == [str(executable), "-thickness", str(result.subject_dir / "surf" / f"{hemi}.white"),
                                "0.5", str(result.subject_dir / "surf" / f"{hemi}.graymid")]
        assert call["cwd"] == result.subject_dir / "surf"
    assert result.metadata["commands"][0]["sha256"] == adapter._sha256(executable)
    assert result.metadata["status"] == "complete"
    assert all((result.subject_dir / "surf" / f"{hemi}.graymid").is_file() for hemi in ("lh", "rh"))


def _zip_subject(path, subject, *, prefix="FreeSurfer/"):
    with zipfile.ZipFile(path, "w") as archive:
        for file in subject.rglob("*"):
            if file.is_file():
                archive.write(file, prefix + file.relative_to(subject).as_posix())
    return path


def test_standard_zip_without_middle_uses_native_and_preserves_archive(tmp_path, native):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source)
    archive = _zip_subject(tmp_path / "recon.zip", original)
    digest = adapter._sha256(archive)
    result = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=archive)
    assert result.backend == "provided" and not result.reused
    assert len(native[1]) == 2 and adapter._sha256(archive) == digest
    assert not (original / "surf/lh.graymid").exists()


def test_zip_with_middle_does_not_require_expand_binary(tmp_path, monkeypatch):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source, middle=True)
    archive = _zip_subject(tmp_path / "recon.zip", original, prefix="subject-a/")
    monkeypatch.setenv("FNIT_RECON_ALL_BIN_DIR", str(tmp_path / "absent"))
    result = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=archive)
    assert result.metadata["commands"] == []
    second = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=archive)
    assert second.reused


@pytest.mark.parametrize("bad_name", ["../outside", "/absolute", "C:/escape", "dir\\escape", "."])
def test_zip_rejects_unsafe_members_before_creating_outputs(tmp_path, bad_name):
    source = _source(tmp_path)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(bad_name, "unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=archive)
    assert not (tmp_path / "work").exists()


def test_zip_rejects_symlink_and_multiple_subjects(tmp_path):
    source = _source(tmp_path)
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("linked")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(info, "../other")
    with pytest.raises(ValueError, match="unsafe"):
        adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=archive)
    subject = _subject(tmp_path / "provided", source)
    archive = _zip_subject(tmp_path / "ambiguous.zip", subject, prefix="a/")
    with zipfile.ZipFile(archive, "a") as writer:
        for file in subject.rglob("*"):
            if file.is_file():
                writer.write(file, "b/" + file.relative_to(subject).as_posix())
    with pytest.raises(ValueError, match="exactly one"):
        adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=archive)


def test_fnit_reuses_api_and_converts_scaled_integer_input_without_resampling(tmp_path, native, monkeypatch):
    source = _source(tmp_path, dtype=np.int16, units=False, scaled=True)
    source_sha = adapter._sha256(source)
    weights, assets = tmp_path / "weights", tmp_path / "assets"
    weights.mkdir()
    assets.mkdir()
    (weights / "synthstrip.1.pt").write_bytes(b"mock weight resolver; no model execution")
    monkeypatch.setenv("FNIT_WEIGHTS", str(weights))
    monkeypatch.setenv("FNIT_ASSETS", str(assets))
    monkeypatch.setattr(adapter, "_fnit_resource_fingerprint", lambda *args: {"fixture": "mock heavy model validation"})
    calls = []

    def run_fnit(**kwargs):
        calls.append(kwargs)
        image = nib.load(kwargs["t1"])
        assert isinstance(image, nib.Nifti1Image) and image.get_data_dtype() == np.dtype("float32")
        assert image.header.get_xyzt_units() == ("mm", "sec") and image.header["sform_code"] != 0
        np.testing.assert_array_equal(np.asarray(image.dataobj), np.asarray(nib.load(source).dataobj, dtype=np.float32))
        np.testing.assert_array_equal(image.affine, nib.load(source).affine)
        _subject(kwargs["subject_dir"], kwargs["t1"])
        return {"status": "complete", "stages": [], "outputs": {}}

    monkeypatch.setattr(adapter, "_run_fnit", run_fnit)
    result = adapter.prepare_surface_reconstruction(
        source, tmp_path / "work", device="cpu", options={"threads": 2, "profile_stages": True})
    assert result.backend == "fnit" and result.source_t1w == source
    assert calls[0]["device"] == "cpu" and calls[0]["threads"] == 2 and calls[0]["profile_stages"]
    assert calls[0]["subject_dir"] == result.subject_dir and calls[0]["weights_dir"] == weights
    assert calls[0]["assets_dir"] == assets and calls[0]["t1"] != source
    assert adapter._sha256(source) == source_sha
    assert result.metadata["input_preparation"]["converted"]
    assert len(native[1]) == 2


@pytest.mark.parametrize("key", ["t1", "subject_dir", "device", "arbitrary_flag"])
def test_options_cannot_override_locked_inputs(tmp_path, key):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        adapter.prepare_surface_reconstruction(source, tmp_path, options={key: "/other"})


@pytest.mark.parametrize("prior_home", [None, "/other/freesurfer/version"])
def test_explicit_freesurfer_uses_safe_argv_and_selected_license(tmp_path, native, monkeypatch, prior_home):
    source = _source(tmp_path)
    command = _binary(tmp_path, "recon-all")
    license_path = tmp_path / "personal-license.txt"
    license_path.write_text("fixture only")
    for name in ("FREESURFER", "FREESURFER_HOME"):
        if prior_home is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, prior_home)
    result = adapter.prepare_surface_reconstruction(
        source, tmp_path / "work", backend="freesurfer",
        options={"command": command, "threads": 2, "fs_license": license_path})
    calls = native[1]
    assert calls[0]["argv"] == [str(command), "-all", "-i", str(source), "-s", "subject", "-sd",
                                 str(tmp_path / "work/reconstruction"), "-parallel", "-openmp", "2"]
    assert calls[0]["env"]["FS_LICENSE"] == str(license_path)
    assert calls[0]["env"]["OMP_NUM_THREADS"] == "2"
    # Real recon-all 8.2 reads $FREESURFER/etc/global-expert-options.v8.txt
    # before importing T1w. Both names must follow the explicitly selected installation.
    assert calls[0]["env"]["FREESURFER"] == str(command.parent.parent)
    assert calls[0]["env"]["FREESURFER_HOME"] == str(command.parent.parent)
    assert calls[0]["env"]["SUBJECTS_DIR"] == str(result.subject_dir.parent)
    assert calls[0]["env"]["PATH"].split(":")[0] == str(command.parent)
    assert str(command.parent.parent / "mni/bin") in calls[0]["env"]["PATH"].split(":")
    assert os.environ.get("FREESURFER") == prior_home
    assert os.environ.get("FREESURFER_HOME") == prior_home
    assert len(calls) == 3 and result.backend == "freesurfer"
    assert "fixture only" not in json.dumps(result.metadata)


def test_cache_checks_source_options_binary_and_actual_output(tmp_path, native):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source)
    options = {"native_bin_dir": native[0].parent}
    kwargs = dict(recon_all=original, options=options)
    first = adapter.prepare_surface_reconstruction(source, tmp_path / "work", **kwargs)
    second = adapter.prepare_surface_reconstruction(source, tmp_path / "work", **kwargs)
    assert second.reused and len(native[1]) == 2
    vertices, faces = fsio.read_geometry(first.subject_dir / "surf/lh.graymid")
    fsio.write_geometry(first.subject_dir / "surf/lh.graymid", vertices + 0.1, faces)
    third = adapter.prepare_surface_reconstruction(source, tmp_path / "work", **kwargs)
    assert not third.reused and len(native[1]) == 4
    native[0].write_text("#!/bin/sh\n# new independently built program\nexit 0\n")
    fourth = adapter.prepare_surface_reconstruction(source, tmp_path / "work", **kwargs)
    assert not fourth.reused and len(native[1]) == 6
    fsio.write_geometry(original / "surf/lh.pial", vertices + [0, 0, 1.1], faces)
    fifth = adapter.prepare_surface_reconstruction(source, tmp_path / "work", **kwargs)
    assert not fifth.reused and len(native[1]) == 8
    sixth = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=original,
                                                  options={**options, "mris_expand_command": native[0]})
    assert not sixth.reused and len(native[1]) == 10


def test_middle_failure_records_failed_command_and_retry_rebuilds_owned_output(tmp_path, native, monkeypatch):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source)
    successful_runner = adapter._run_command

    def fail_right(argv, **kwargs):
        if argv[-1].endswith("rh.graymid"):
            raise subprocess.CalledProcessError(1, argv)
        successful_runner(argv, **kwargs)

    monkeypatch.setattr(adapter, "_run_command", fail_right)
    with pytest.raises(subprocess.CalledProcessError):
        adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=original)
    report = json.loads((tmp_path / "work/reconstruction" / adapter._MANIFEST).read_text())
    assert report["status"] == "failed" and len(report["commands"]) == 2
    assert report["commands"][1]["seconds"] >= 0
    assert not (tmp_path / "work/reconstruction/.fnit-surface-reconstruction.lock").exists()
    monkeypatch.setattr(adapter, "_run_command", successful_runner)
    result = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=original)
    assert result.metadata["status"] == "complete" and not result.reused


def test_generated_directory_needs_own_manifest_and_cannot_overlap_provided(tmp_path, native):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source)
    output = tmp_path / "output"
    (output / "subject").mkdir(parents=True)
    user_file = output / "subject/user.txt"
    user_file.write_text("preserve me")
    with pytest.raises(FileExistsError, match="no FNIT adapter manifest"):
        adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=original, output_dir=output)
    assert user_file.read_text() == "preserve me"
    with pytest.raises(ValueError, match="separate"):
        adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=original, output_dir=original / "nested")
    assert not (original / "nested").exists()


def test_provided_identity_is_checked_and_affine_is_explicit(tmp_path, native):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source, middle=True)
    changed = nib.load(source)
    nib.save(nib.Nifti1Image(np.asarray(changed.dataobj) + 10, changed.affine), source)
    with pytest.raises(ValueError, match="voxel content"):
        adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=original)
    result = adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=original,
                                                    options={"fsnative_to_t1w": np.eye(4)})
    assert result.metadata["identity"]["OriginalT1Identity"] == "explicit fsnative-to-T1w world affine"
    with pytest.raises(ValueError, match="4x4"):
        adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=original,
                                                options={"fsnative_to_t1w": np.zeros((4, 4))})


def test_existing_input_cannot_be_reinterpreted_as_generation_output(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="existing input"):
        adapter.prepare_surface_reconstruction(source, tmp_path, recon_all=tmp_path / "provided", backend="fnit")


def test_resource_fingerprint_checks_bytes_against_fixed_manifest(tmp_path, monkeypatch):
    from fnit import weights as weight_module
    from fnit.recon_all import assets as asset_module

    weight_dir, assets_dir = tmp_path / "weights", tmp_path / "assets"
    weight_dir.mkdir()
    assets_dir.mkdir()
    payload = b"small resource verification fixture"
    digest = hashlib.sha256(payload).hexdigest()
    (weight_dir / "model.pt").write_bytes(payload)
    (assets_dir / "atlas.dat").write_bytes(payload)
    monkeypatch.setitem(weight_module.MODEL_FILES, "recon-all", ("model.pt",))
    monkeypatch.setitem(weight_module.WEIGHT_FILES, "model.pt", ("https://original.example/model", len(payload), digest))
    monkeypatch.setattr(asset_module, "CORE_ASSETS", ("atlas.dat",))
    monkeypatch.setitem(asset_module.ASSET_FILES, "atlas.dat", (len(payload), digest, ""))
    fingerprint = adapter._fnit_resource_fingerprint(weight_dir, assets_dir)
    assert fingerprint["weights"]["model.pt"]["sha256"] == digest
    (assets_dir / "atlas.dat").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        adapter._fnit_resource_fingerprint(weight_dir, assets_dir)


def test_producer_change_invalidates_owned_cache(tmp_path, native, monkeypatch):
    source = _source(tmp_path)
    original = _subject(tmp_path / "provided", source)
    adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=original)
    monkeypatch.setattr(adapter, "_producer_fingerprint", lambda backend: {"adapter": "new source digest"})
    result = adapter.prepare_surface_reconstruction(source, tmp_path / "work", recon_all=original)
    assert not result.reused and len(native[1]) == 4
