"""Private CPU regression; preserve failed collection and launcher evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path("/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1/root_integrated_tests_v3")
job = root / "cpu_tests_retry2"
job.mkdir(exist_ok=False)
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root / "src"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    environment[key] = "8"
command = ["/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/bin/python", "-m", "pytest", "-q",
           "tests/connectome", "tests/eddy", "tests/topup"]
started = time.perf_counter()
with (job / "pytest.log").open("xb") as log:
    result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
(job / "status.json").write_text(json.dumps({
    "status": "completed" if not result.returncode else "failed", "returncode": result.returncode,
    "seconds": time.perf_counter()-started, "command": command,
    "CUDA_VISIBLE_DEVICES": "", "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "science_origin_commit": "1fe86ab8347b29d9c47be8109627736576222912",
}, indent=2))
raise SystemExit(result.returncode)
