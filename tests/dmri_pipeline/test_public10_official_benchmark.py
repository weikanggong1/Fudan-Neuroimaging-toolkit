"""Independent original-reference wiring and real process-lock regression checks."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "validation/dmri_pipeline/public10_20261002/benchmark_official.py"


def load_driver():
    specification = importlib.util.spec_from_file_location("public10_official_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def base_cli(root):
    return ["--case-id", "case01", "--raw-dir", str(root / "raw"),
            "--output-dir", str(root / "output"), "--fsl-dir", str(root / "fsl"),
            "--topup-config", str(root / "topup.cnf"),
            "--fa-reference", str(root / "FA.nii.gz"),
            "--amico-runner", str(root / "amico.py"),
            "--report", str(root / "report.json")]


def test_branch_preflight_rejects_missing_t1_before_creating_outputs(tmp_path):
    driver = load_driver()
    with pytest.raises(ValueError, match="required argument"):
        driver.main(base_cli(tmp_path) + ["--registration-backend", "mmorf"])
    assert not (tmp_path / "output").exists()
    defaults = driver.parser().parse_args(base_cli(tmp_path))
    assert defaults.registration_backend == "tbss"
    assert defaults.brain_extractor == "synthstrip"


def test_mmorf_map_contract_has_no_skeleton(tmp_path):
    driver = load_driver()
    mmorf_standard = [driver.map_output_path(tmp_path, "mmorf", "standard", name)
                      for name in driver.MAP_NAMES]
    assert len(set(mmorf_standard)) == 9
    assert all(path.parent == tmp_path / "mmorf/standard" for path in mmorf_standard)
    assert driver.map_output_path(tmp_path, "mmorf", "native", "FA") == tmp_path / "native/dti_FA.nii.gz"
    assert driver.map_output_path(tmp_path, "tbss", "standard", "OD") == tmp_path / "tbss/stats/all_OD.nii.gz"
    with pytest.raises(ValueError, match="no skeleton"):
        driver.map_output_path(tmp_path, "mmorf", "skeleton", "FA")


def test_real_gpu_flock_wait_does_not_inflate_subprocess_clock(tmp_path):
    driver = load_driver()
    lock, ready = tmp_path / "gpu.lock", tmp_path / "ready"
    holder_code = (
        "import fcntl,pathlib,sys,time; "
        "stream=open(sys.argv[1],'a'); fcntl.flock(stream,fcntl.LOCK_EX); "
        "pathlib.Path(sys.argv[2]).touch(); time.sleep(0.5)"
    )
    holder = subprocess.Popen([sys.executable, "-c", holder_code, str(lock), str(ready)])
    try:
        deadline = time.perf_counter() + 4
        while not ready.exists() and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert ready.exists(), "independent GPU-lock holder did not start"
        output = tmp_path / "output"
        output.mkdir()
        run = driver.ReferenceRun(output, os.environ.copy(), lock)
        with run.stage("native_gpu_stage"):
            text = run.run([sys.executable, "-c", "print('independent native GPU step')"],
                           name="native_gpu", capture=True, gpu=True)
        assert "independent native GPU step" in text
        command = run.commands[0]
        assert command["gpu_queue_seconds"] > 0.20
        assert command["seconds"] < command["gpu_queue_seconds"]
        assert run.stages["native_gpu_stage"] > command["gpu_queue_seconds"]
        assert run.stages_gpu_queue_seconds["native_gpu_stage"] == command["gpu_queue_seconds"]
        assert command["uses_gpu"] is True
    finally:
        holder.wait(timeout=5)


def test_failed_native_gpu_command_releases_lock_and_records_exit(tmp_path):
    driver = load_driver()
    output = tmp_path / "output"
    output.mkdir()
    lock = tmp_path / "gpu.lock"
    run = driver.ReferenceRun(output, os.environ.copy(), lock)
    with pytest.raises(RuntimeError, match="exit code 7"):
        run.run([sys.executable, "-c", "raise SystemExit(7)"], name="failed_gpu", gpu=True)
    assert run.commands[0]["returncode"] == 7
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)
    command_file = output / "private_logs/001_failed_gpu.command.json"
    assert json.loads(command_file.read_text())[0] == sys.executable
