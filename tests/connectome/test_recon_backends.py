"""Anatomy provenance, complete-run publishing and explicit backend selection."""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fnit.connectome import recon_backend as recon
from fnit.recon_all.expected_outputs import paths as expected_paths


def subject_fixture(root: Path, *, full_fnit=False) -> Path:
    (root / "mri").mkdir(parents=True)
    (root / "surf").mkdir()
    for name in ("brain", "aparc+aseg", "ribbon"):
        nib.save(nib.MGHImage(np.ones((4, 4, 4), np.float32), np.eye(4)), root / f"mri/{name}.mgz")
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], int)
    for hemi in ("lh", "rh"):
        for kind in ("white", "pial"):
            nib.freesurfer.write_geometry(str(root / f"surf/{hemi}.{kind}"), vertices, faces)
    if full_fnit:
        for name in expected_paths():
            path = root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unit fixture only")
        report = {"status": "complete", "output_validation": {
            "status": "passed", "expected": 138, "present": 138, "missing": []},
            "mesh_validation": {"status": "passed"}, "stages": [{"name": "unit_fixture", "status": "complete"}],
            "numeric_validation": {"status": "not_run"}}
        (root / "fnit-native-free-run.json").write_text(json.dumps(report))
    return root


def raw_t1(tmp_path):
    path = tmp_path / "T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), path)
    return path


def fnit_options(tmp_path):
    weights, assets, binaries = [tmp_path / name for name in ("weights", "assets", "bin")]
    for directory in (weights, assets, binaries):
        directory.mkdir()
    (weights / "unit-weight").write_bytes(b"unit fixture")
    (assets / "unit-atlas").write_bytes(b"unit fixture")
    return {"weights_dir": weights, "assets_dir": assets, "native_bin_dir": binaries}


@pytest.mark.parametrize("name,provided,expected", [("auto", "subject", "provided"),
    ("auto", None, "fnit"), ("fnit", None, "fnit"), ("freesurfer", None, "freesurfer")])
def test_explicit_backend_resolution(name, provided, expected):
    assert recon.resolve_recon_backend(name, provided) == expected


@pytest.mark.parametrize("backend,subject", [("provided", None), ("fnit", "subject"),
    ("freesurfer", "subject"), ("unknown", None)])
def test_backend_conflicts_rejected(backend, subject):
    with pytest.raises(ValueError):
        recon.resolve_recon_backend(backend, subject)


def test_auto_does_not_silently_use_official_when_fnit_resources_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(recon.shutil, "which", lambda _: "/official/recon-all")
    with pytest.raises(ValueError, match="weights_dir"):
        recon.prepare_recon_subject(raw_t1(tmp_path), tmp_path / "out", subject_name="sub-01")
    assert not (tmp_path / "out").exists()


def test_content_identity_detects_same_size_same_mtime_edit(tmp_path):
    file = tmp_path / "data"; file.write_bytes(b"first")
    before = recon.file_fingerprint(file); stat = file.stat()
    file.write_bytes(b"other"); os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = recon.file_fingerprint(file)
    assert before["size"] == after["size"] and before["sha256"] != after["sha256"]


def test_supplied_subject_is_read_only_and_actual_content_hashed(tmp_path):
    subject = subject_fixture(tmp_path / "provided")
    before = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in subject.rglob("*") if p.is_file()}
    result = recon.prepare_recon_subject(None, tmp_path / "out", subject_name="sub-01",
                                        freesurfer_subject_dir=subject)
    after = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in subject.rglob("*") if p.is_file()}
    assert before == after and result.stage == "supplied"
    assert result.metadata["resolved_backend"] == "provided" and not (tmp_path / "out").exists()
    assert len(result.metadata["anatomy"]["content_sha256"]) == 64


def test_corrupt_surface_file_is_not_complete(tmp_path):
    subject = subject_fixture(tmp_path / "provided")
    (subject / "surf/lh.pial").write_bytes(b"not a surface")
    with pytest.raises((ValueError, OSError)):
        recon.inspect_recon_subject(subject)


def test_white_pial_different_vertex_correspondence_rejected(tmp_path):
    subject = subject_fixture(tmp_path / "provided")
    vertices, faces = nib.freesurfer.read_geometry(str(subject / "surf/lh.pial"))
    nib.freesurfer.write_geometry(str(subject / "surf/lh.pial"), vertices, faces[::-1])
    with pytest.raises(ValueError, match="correspondence"):
        recon.inspect_recon_subject(subject)


def test_mismatched_volume_geometry_rejected(tmp_path):
    subject = subject_fixture(tmp_path / "provided")
    affine = np.eye(4); affine[0, 3] = 3
    nib.save(nib.MGHImage(np.ones((4, 4, 4), np.float32), affine), subject / "mri/ribbon.mgz")
    with pytest.raises(ValueError, match="geometry mismatch"):
        recon.inspect_recon_subject(subject)


def test_fnit_uses_complete_one_job_batch_and_reuses_unchanged_result(tmp_path, monkeypatch):
    import fnit.recon_all.batch as batch
    image = raw_t1(tmp_path); options = fnit_options(tmp_path); calls = []
    def run(*, jobs, weights_dir, assets_dir, devices, **kwargs):
        calls.append((jobs, devices, kwargs))
        assert weights_dir == str(options["weights_dir"]) and assets_dir == str(options["assets_dir"])
        subject_fixture(jobs[0]["subject_dir"], full_fnit=True)
        return [{}]
    monkeypatch.setattr(batch, "run_recon_all_python_batch", run)
    first = recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01",
                                        recon_options=options, device="cpu")
    second = recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01",
                                         recon_options=options, device="cpu")
    assert first.stage == "completed" and second.stage == "skipped"
    assert len(calls) == 1 and calls[0][1] == ("cpu",) and calls[0][2]["threads"] == 4
    assert first.subject_dir == second.subject_dir
    # A resource edit creates a new generation while retaining the completed old subject.
    (options["weights_dir"] / "unit-weight").write_bytes(b"edited weight")
    third = recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01",
                                        recon_options=options, device="cpu")
    assert third.stage == "completed" and first.subject_dir.exists()
    assert third.subject_dir != first.subject_dir and len(calls) == 2


def test_fnit_failed_api_never_publishes_completed_state(tmp_path, monkeypatch):
    import fnit.recon_all.batch as batch
    image = raw_t1(tmp_path); options = fnit_options(tmp_path)
    def fail(*, jobs, **kwargs):
        subject_fixture(jobs[0]["subject_dir"], full_fnit=True)
        raise RuntimeError("failure after output publication")
    monkeypatch.setattr(batch, "run_recon_all_python_batch", fail)
    with pytest.raises(RuntimeError, match="failure after"):
        recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01", recon_options=options, device="cpu")
    assert not list((tmp_path / "out/anatomy/state").glob("*.json"))


def test_incomplete_fnit_report_rejected_even_when_outputs_exist(tmp_path, monkeypatch):
    import fnit.recon_all.batch as batch
    image = raw_t1(tmp_path); options = fnit_options(tmp_path)
    def fake(*, jobs, **kwargs):
        subject = subject_fixture(jobs[0]["subject_dir"], full_fnit=True)
        report = json.loads((subject / "fnit-native-free-run.json").read_text())
        report["mesh_validation"]["status"] = "failed"
        (subject / "fnit-native-free-run.json").write_text(json.dumps(report))
    monkeypatch.setattr(batch, "run_recon_all_python_batch", fake)
    with pytest.raises(ValueError, match="not complete"):
        recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01", recon_options=options, device="cpu")
    assert not list((tmp_path / "out/anatomy/state").glob("*.json"))


def test_edited_output_forces_new_attempt_without_overwriting_old_subject(tmp_path, monkeypatch):
    import fnit.recon_all.batch as batch
    image = raw_t1(tmp_path); options = fnit_options(tmp_path)
    monkeypatch.setattr(batch, "run_recon_all_python_batch", lambda **kw: subject_fixture(kw["jobs"][0]["subject_dir"], full_fnit=True))
    first = recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01", recon_options=options, device="cpu")
    (first.subject_dir / "stats/aseg.stats").write_bytes(b"edited")
    # Both the complete FNIT output contract and anatomy define generation identity.
    (first.subject_dir / "label/lh.aparc.annot").write_bytes(b"edited annotation")
    second = recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01", recon_options=options, device="cpu")
    assert second.stage == "completed" and second.subject_dir != first.subject_dir
    assert (first.subject_dir / "label/lh.aparc.annot").read_bytes() == b"edited annotation"


def test_input_changes_during_execution_not_published(tmp_path, monkeypatch):
    import fnit.recon_all.batch as batch
    image = raw_t1(tmp_path); options = fnit_options(tmp_path)
    def fake(**kwargs):
        subject_fixture(kwargs["jobs"][0]["subject_dir"], full_fnit=True)
        image.write_bytes(b"changed input")
    monkeypatch.setattr(batch, "run_recon_all_python_batch", fake)
    with pytest.raises(RuntimeError, match="changed during execution"):
        recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01", recon_options=options, device="cpu")
    assert not list((tmp_path / "out/anatomy/state").glob("*.json"))


def test_official_home_bin_precedes_fnit_native_bins(tmp_path, monkeypatch):
    image = raw_t1(tmp_path)
    home = tmp_path / "official-freesurfer"; (home / "bin").mkdir(parents=True)
    (home / "SetUpFreeSurfer.sh").write_text('. "$FREESURFER_HOME/FreeSurferEnv.sh"\n')
    (home / "FreeSurferEnv.sh").write_text('export FREESURFER="$FREESURFER_HOME"\n')
    executable = home / "bin/recon-all"
    executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o755)
    monkeypatch.setenv("PATH", "/fnit-native/bin:/usr/bin")
    environments = []
    def run(command, *, check, env):
        environments.append(env)
        subject = subject_fixture(Path(command[command.index("-sd") + 1]) / command[command.index("-s") + 1])
        (subject / "scripts").mkdir()
        (subject / "scripts/recon-all.done").write_text("done")
    monkeypatch.setattr(recon.subprocess, "run", run)
    result = recon.prepare_recon_subject(image, tmp_path / "out", subject_name="sub-01", recon_backend="freesurfer",
        recon_options={"executable": executable, "freesurfer_home": home})
    assert result.stage == "completed"
    assert environments[0]["FREESURFER_HOME"] == str(home)
    assert environments[0]["PATH"].split(os.pathsep)[0] == str(home / "bin")
    assert environments[0]["OMP_NUM_THREADS"] == "4"


@pytest.mark.parametrize("inherited_home", [False, True])
def test_official_full_setup_real_child_preserves_parent_and_literal_args(tmp_path, monkeypatch, inherited_home):
    import sys
    image = tmp_path / "T1 with space; $(touch SHELL_EXPANDED).nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), image)
    home = tmp_path / "official home with space"; (home / "bin").mkdir(parents=True)
    setup = home / "SetUpFreeSurfer.sh"
    setup.write_text('. "$FREESURFER_HOME/FreeSurferEnv.sh"\nexport FNIT_SETUP_SEEN="complete"\n')
    (home / "FreeSurferEnv.sh").write_text('export FREESURFER="$FREESURFER_HOME"\n')
    executable = home / "bin/recon-all"
    # A controlled executable records selected variables and argv. It never
    # invokes MRI software; this tests the real Bash setup/exec mechanism.
    executable.write_text(f"#!{sys.executable}\n" + '''import json,os,sys
from pathlib import Path
subject=Path(sys.argv[sys.argv.index('-sd')+1])/sys.argv[sys.argv.index('-s')+1]
(subject/'scripts').mkdir(parents=True)
(subject/'scripts/recon-all.done').write_text('done')
(subject/'observed.json').write_text(json.dumps({'argv':sys.argv[1:],
    'setup':os.environ.get('FNIT_SETUP_SEEN'),'parent':os.environ.get('FNIT_PARENT_KEEP'),
    'freesurfer':os.environ.get('FREESURFER'),'home':os.environ.get('FREESURFER_HOME'),
    'path_first':os.environ['PATH'].split(os.pathsep)[0],'threads':os.environ.get('OMP_NUM_THREADS')}))
''')
    executable.chmod(0o755)
    monkeypatch.setenv("FNIT_PARENT_KEEP", "inherited-value")
    monkeypatch.setenv("PATH", "/fnit-native/bin:/usr/bin")
    monkeypatch.chdir(tmp_path)
    options = {"executable": str(executable), "threads": 4}
    if inherited_home:
        monkeypatch.setenv("FREESURFER_HOME", str(home))
    else:
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        options["freesurfer_home"] = str(home)
    # Independent output geometry tests above exercise the inspector itself.
    monkeypatch.setattr(recon, "inspect_recon_subject", lambda *args, **kwargs: {"content_sha256": "fixture"})
    first = recon.prepare_recon_subject(image, tmp_path / "output with space", subject_name="sub-01",
                                       recon_backend="freesurfer", recon_options=options, device="cpu")
    observed = json.loads((first.subject_dir / "observed.json").read_text())
    assert first.stage == "completed" and observed["setup"] == "complete"
    assert observed["parent"] == "inherited-value" and observed["threads"] == "4"
    assert observed["home"] == observed["freesurfer"] == str(home)
    assert observed["path_first"] == str(home / "bin")
    assert observed["argv"][observed["argv"].index("-i") + 1] == str(image)
    assert not (tmp_path / "SHELL_EXPANDED").exists()
    second = recon.prepare_recon_subject(image, tmp_path / "output with space", subject_name="sub-01",
                                        recon_backend="freesurfer", recon_options=options, device="cpu")
    assert second.stage == "skipped" and second.subject_dir == first.subject_dir
    before = setup.stat()
    setup.write_text(setup.read_text().replace('"complete"', '"modified"'))
    os.utime(setup, ns=(before.st_atime_ns, before.st_mtime_ns))
    third = recon.prepare_recon_subject(image, tmp_path / "output with space", subject_name="sub-01",
                                       recon_backend="freesurfer", recon_options=options, device="cpu")
    assert third.stage == "completed" and third.subject_dir != first.subject_dir
    assert first.subject_dir.is_dir() and json.loads((third.subject_dir / "observed.json").read_text())["setup"] == "modified"


def test_official_setup_failure_never_runs_program_or_publishes_completion(tmp_path, monkeypatch):
    home = tmp_path / "official"; (home / "bin").mkdir(parents=True)
    (home / "SetUpFreeSurfer.sh").write_text("exit 7\n")
    (home / "FreeSurferEnv.sh").write_text("true\n")
    executable = home / "bin/recon-all"
    sentinel = tmp_path / "PROGRAM_RAN"
    executable.write_text(f"#!/bin/sh\ntouch '{sentinel}'\n"); executable.chmod(0o755)
    with pytest.raises(recon.subprocess.CalledProcessError) as error:
        recon.prepare_recon_subject(raw_t1(tmp_path), tmp_path / "out", subject_name="sub-01", device="cpu",
            recon_backend="freesurfer", recon_options={"executable": executable, "freesurfer_home": home})
    assert error.value.returncode == 7 and not sentinel.exists()
    assert not list((tmp_path / "out").rglob("state/*.json"))


def test_official_missing_setup_is_rejected_before_reconstruction(tmp_path, monkeypatch):
    home = tmp_path / "official"; home.mkdir()
    executable = home / "recon-all"
    executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o755)
    with pytest.raises(FileNotFoundError, match="environment script missing"):
        recon.prepare_recon_subject(raw_t1(tmp_path), tmp_path / "out", subject_name="sub-01",
            recon_backend="freesurfer", recon_options={"executable": executable, "freesurfer_home": home})
    assert not (tmp_path / "out").exists()
