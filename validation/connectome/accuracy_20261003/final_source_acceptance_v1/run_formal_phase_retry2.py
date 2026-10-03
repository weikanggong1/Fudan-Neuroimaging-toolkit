"""Private launcher: CUDA regression gate, then the frozen raw accuracy plan."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time

root = Path("/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1")
frozen = root / "formal_frozen_v1"
config_path = frozen / "accuracy_configuration.json"
if hashlib.sha256(config_path.read_bytes()).hexdigest() != "f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c":
    raise ValueError("frozen formal configuration changed")
config = json.loads(config_path.read_bytes())
tests = root / "root_integrated_tests_v3"
job = frozen / "execution_launcher_retry2"
job.mkdir(exist_ok=False)
state = {"status": "waiting_for_cuda_test_lock", "pid": os.getpid(), "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
def write():
    (job / "status.json").write_text(json.dumps(state, indent=2))
write()
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES=config["gpu_uuid"], PYTHONPATH=str(tests / "src"),
                   FNIT_WEIGHTS=config["fnit_weights"], PYTORCH_CUDA_ALLOC_CONF=config["cuda_alloc_conf"])
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    environment[key] = "8"
test_command = [config["gpu_python"], "-m", "pytest", "-q", "tests/connectome"]
with Path(config["gpu_lock"]).open("a") as lock:
    queue = time.perf_counter()
    fcntl.flock(lock, fcntl.LOCK_EX)
    state.update(status="running_cuda_regression", cuda_test_lock_queue_seconds=time.perf_counter()-queue,
                 cuda_test_command=test_command)
    write()
    started = time.perf_counter()
    with (job / "cuda_tests.log").open("xb") as log:
        result = subprocess.run(test_command, cwd=tests, env=environment, stdout=log, stderr=subprocess.STDOUT)
    state.update(cuda_test_exit_code=result.returncode, cuda_test_seconds=time.perf_counter()-started)
    write()
if result.returncode:
    state.update(status="cuda_regression_failed; raw phase not started")
    write()
    raise SystemExit(result.returncode)
state.update(status="executing_raw_accuracy_phase")
write()
command = [config["gpu_python"], str(frozen / "tools_source/tools/benchmark_connectome_accuracy_cohort.py"),
           "--configuration", str(config_path)]
state["raw_coordinator_command"] = command
write()
with (job / "raw_coordinator.log").open("xb") as log:
    result = subprocess.run(command, cwd=frozen / "tools_source", env=environment,
                            stdout=log, stderr=subprocess.STDOUT)
state.update(status="execution_completed" if not result.returncode else "raw_execution_failed",
             raw_coordinator_exit_code=result.returncode)
write()
raise SystemExit(result.returncode)
