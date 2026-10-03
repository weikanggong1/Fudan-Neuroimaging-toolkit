"""CPU provenance regressions; byte fixtures are not MRI or speed benchmarks."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import benchmark_connectome_accuracy_cohort as driver
from tools import benchmark_connectome_raw_cohort as cohort


def frozen_tools(tmp_path):
    wall = tmp_path / "wall.py"
    wall.write_text("# frozen wall producer\n")
    return {"accuracy_coordinator_sha256": cohort.sha256(driver.__file__),
            "worker_script": cohort.__file__,
            "worker_script_sha256": cohort.sha256(cohort.__file__),
            "wall_script": str(wall), "wall_script_sha256": cohort.sha256(wall)}


def file_identity(path, kind=None):
    record = {"path": str(path), "sha256": cohort.sha256(path)}
    if kind is not None:
        record["kind"] = kind
    return record


def wall_receipt(config, case, job, anatomy):
    raw = next(record for record in case["input_files"] if record["kind"] == "raw_dwi")
    return {
        "mode": "wall", "status": "completed", "exit_code": 0,
        "total_runtime_seconds": 2.0,
        "initial_output_state": {"output_directory_existed": False,
                                 "preexisting_state_files": [], "preexisting_run_state": False},
        "preprocessing": [{"topup": "no_reverse_pe", "eddy": "completed", "recon_all": "supplied"}],
        "actual_eddy_gp_seeds": [config["eddy_gp_seed"]], "official_recon_all_calls": [],
        "outputs": {"status": "complete"}, "inputs": {"raw/image": raw},
        "cli_arguments": cohort.cli_command(config, case, job, anatomy_subject=anatomy),
        "selected_inputs": {"freesurfer_subject_dir": str(anatomy)},
        "gpu_process_memory": {"peak_process_tree_bytes": 1000},
        "cuda_allocator": {"allocated_bytes": 500, "reserved_bytes": 750},
    }


def run_fixture(tmp_path, version="baseline"):
    from tools import analyze_connectome_accuracy_cohort as analysis
    raw = tmp_path / "raw_dwi.nii.gz"
    raw.write_bytes(b"raw acquisition identity fixture")
    anatomy = tmp_path / "official_subject"
    case = {"case_id": "sub-CON03", "subject": "CON03", "bids_root": str(tmp_path),
            "input_files": [file_identity(raw, "raw_dwi")]}
    fingerprint = "1" * 64
    config = {
        "run_root": str(tmp_path / "phase"),
        "execution_order": [{"version": version, "case_id": case["case_id"]}],
        "declared_source_manifests": {version: {"source_fingerprint": fingerprint}},
        "eddy_gp_seed": 12345, "device": "cuda:0", "n_seeds": 100000, "seed": 0,
        "atlases": ["fs-aparc"], "atlas_options": [],
    }
    job = Path(config["run_root"]) / version / case["case_id"]
    job.mkdir(parents=True)
    gpu_path, wall_path = job / "gpu_report.json", job / "raw_bids_wall.json"
    anatomy_record = {"mri/brain.mgz": {"path": str(anatomy / "mri/brain.mgz"), "sha256": "2" * 64}}
    gpu = {"status": "completed", "version": version, "case_id": case["case_id"],
           "source_before": {"source_fingerprint": fingerprint},
           "source_after": {"source_fingerprint": fingerprint},
           "wall_report": str(wall_path), "anatomy": anatomy_record,
           "anatomy_after": copy.deepcopy(anatomy_record), "raw_dwi_cli_total_runtime_seconds": 2.0}
    wall = wall_receipt(config, case, job, anatomy)

    def save():
        gpu_path.write_text(json.dumps(gpu))
        wall_path.write_text(json.dumps(wall))
    save()
    return analysis, config, case, anatomy, gpu, wall, gpu_path, save


def test_declared_wall_tool_is_verified_without_replacing_its_hash(tmp_path):
    config = frozen_tools(tmp_path)
    before = copy.deepcopy(config)
    driver.verify_tools(config)
    assert config == before
    Path(config["wall_script"]).write_text("# changed after phase declaration\n")
    with pytest.raises(ValueError, match="wall tool changed"):
        driver.verify_tools(config)
    assert config["wall_script_sha256"] == before["wall_script_sha256"]


@pytest.mark.parametrize("field,message", [
    ("wall_script_sha256", "wall tool changed"),
    ("accuracy_coordinator_sha256", "coordinator changed"),
    ("worker_script_sha256", "worker differs"),
])
def test_declared_tool_hash_tampering_is_rejected(tmp_path, field, message):
    config = frozen_tools(tmp_path)
    config[field] = "0" * 64
    with pytest.raises(ValueError, match=message):
        driver.verify_tools(config)


def test_same_bytes_at_another_worker_path_do_not_bind_the_loaded_worker(tmp_path):
    config = frozen_tools(tmp_path)
    other = tmp_path / "unloaded_worker.py"
    other.write_bytes(Path(cohort.__file__).read_bytes())
    config["worker_script"] = str(other)
    assert cohort.sha256(other) == config["worker_script_sha256"]
    with pytest.raises(ValueError, match="actual worker differs"):
        driver.verify_tools(config)


def test_changed_wall_is_rejected_before_claim_or_any_worker(tmp_path, monkeypatch):
    config = frozen_tools(tmp_path)
    cases = [{"case_id": f"case-{number}"} for number in range(10)]
    monkeypatch.setattr(cohort, "validate_manifest", lambda manifest: manifest["cases"])
    config.update(sources={"baseline": "/frozen/old", "candidate": "/frozen/new"},
                  execution_order=[{"version": "candidate", "case_id": case["case_id"]} for case in cases],
                  run_root=str(tmp_path / "unclaimed_phase"))
    for name, payload in (("raw_manifest", {"cases": cases}),
                          ("input_bindings", {"cases": {case["case_id"]: {} for case in cases}})):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(payload))
        config[name] = file_identity(path)
    configuration = tmp_path / "configuration.json"
    configuration.write_text(json.dumps(config))
    Path(config["wall_script"]).write_text("# forbidden tool replacement\n")
    def forbidden(*args, **kwargs):
        pytest.fail("changed declared tool reached source claiming or a worker")
    monkeypatch.setattr(cohort, "source_manifest", forbidden)
    monkeypatch.setattr(cohort, "worker", forbidden)
    with pytest.raises(ValueError, match="wall tool changed"):
        driver.execute(configuration)
    assert not Path(config["run_root"]).exists()


@pytest.mark.parametrize("version", ["baseline", "candidate"])
def test_completed_run_accepts_bound_source_case_plan_and_wall(tmp_path, version):
    analysis, config, case, anatomy, gpu, wall, gpu_path, _ = run_fixture(tmp_path, version)
    actual_gpu, actual_wall, actual_path = analysis.completed_run(config, case, version, anatomy)
    assert actual_gpu == gpu and actual_wall == wall and actual_path == gpu_path


@pytest.mark.parametrize("corruption,message", [
    ("case_id", "same-case same-version"), ("version", "same-case same-version"),
    ("source_before", "before/after source differs"),
    ("source_after", "before/after source differs"),
    ("wall_path", "wall binding differs"),
    ("cli_seed_attempts", "CLI parameters differ"),
    ("cli_anatomy", "CLI parameters differ"),
    ("outside_plan", "outside the frozen phase plan"),
    ("anatomy_after", "anatomy before/after differs"),
    ("timer", "GPU/wall timer binding differs"),
])
def test_paired_baseline_rejects_corrupted_receipt(tmp_path, corruption, message):
    analysis, config, case, anatomy, gpu, wall, _, save = run_fixture(tmp_path)
    if corruption == "case_id":
        gpu["case_id"] = "sub-CON01"
    elif corruption == "version":
        gpu["version"] = "candidate"
    elif corruption in ("source_before", "source_after"):
        gpu[corruption]["source_fingerprint"] = "9" * 64
    elif corruption == "wall_path":
        gpu["wall_report"] = str(tmp_path / "unrelated_wall.json")
    elif corruption == "cli_seed_attempts":
        wall["cli_arguments"][wall["cli_arguments"].index("--n-seeds") + 1] = "200000"
    elif corruption == "cli_anatomy":
        wall["cli_arguments"][wall["cli_arguments"].index("--freesurfer-subject-dir") + 1] = str(tmp_path / "other_subject")
    elif corruption == "outside_plan":
        config["execution_order"] = [{"version": "candidate", "case_id": case["case_id"]}]
    elif corruption == "anatomy_after":
        gpu["anatomy_after"]["mri/brain.mgz"]["sha256"] = "3" * 64
    elif corruption == "timer":
        gpu["raw_dwi_cli_total_runtime_seconds"] = 0.5
    save()
    with pytest.raises(ValueError, match=message):
        analysis.completed_run(config, case, "baseline", anatomy)


@pytest.mark.parametrize("change_anatomy", [False, True])
def test_worker_rehashes_real_anatomy_files_after_child_run(tmp_path, monkeypatch, change_anatomy):
    # Only the scientific child process is mocked. Real byte hashes, fresh
    # reports, raw verification, worker control flow and its private lock run.
    anatomy = tmp_path / "official_subject"
    for name in (*cohort.ANATOMY, "scripts/recon-all.done"):
        path = anatomy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("frozen file identity: " + name).encode())
    before = cohort.check_anatomy(anatomy, ["fs-aparc"])
    source = tmp_path / "source"
    (source / "src/fnit").mkdir(parents=True)
    (source / "src/fnit/cli.py").write_text("# source inventory fixture\n")
    raw = tmp_path / "raw_dwi.nii.gz"
    raw.write_bytes(b"unchanged raw identity fixture")
    case = {"case_id": "sub-CON03", "subject": "CON03", "bids_root": str(tmp_path),
            "input_files": [file_identity(raw, "raw_dwi")]}
    config = frozen_tools(tmp_path)
    config.update(run_root=str(tmp_path / "phase"), sources={"baseline": str(source)},
                  frozen_sources={"baseline": cohort.source_manifest(source)},
                  gpu_lock=str(tmp_path / "private_cpu_test.lock"), gpu_python=sys.executable,
                  gpu_cpu_threads=1, device="cuda:0", n_seeds=100000, seed=0,
                  eddy_gp_seed=12345, atlases=["fs-aparc"], atlas_options=[])
    real_run = subprocess.run
    scientific_calls = []
    def cpu_child(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        assert command[:2] == [sys.executable, config["wall_script"]]
        scientific_calls.append(command)
        job = Path(config["run_root"]) / "baseline" / case["case_id"]
        (job / "raw_bids_wall.json").write_text(json.dumps(wall_receipt(config, case, job, anatomy)))
        if change_anatomy:
            (anatomy / "mri/aparc+aseg.mgz").write_bytes(b"changed during child execution")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(cohort.subprocess, "run", cpu_child)
    monkeypatch.setattr(cohort, "check_outputs", lambda *args: {"files": {}})
    report = cohort.worker(
        {"action": "gpu", "config": config, "case": case, "version": "baseline"},
        anatomy_loader=lambda *args: {"anatomy": before}, anatomy_subject=lambda *args: anatomy)
    assert len(scientific_calls) == 1
    after = cohort.check_anatomy(anatomy, ["fs-aparc"])
    assert report["anatomy"] == before and report["anatomy_after"] == after
    if change_anatomy:
        assert report["status"] == "failed"
        assert "anatomy changed during" in report["error"]["message"]
        assert after != before
    else:
        assert report["status"] == "completed"
        assert after == before
    saved = json.loads((Path(config["run_root"]) / "baseline" / case["case_id"] / "gpu_report.json").read_text())
    assert saved["status"] == report["status"]
