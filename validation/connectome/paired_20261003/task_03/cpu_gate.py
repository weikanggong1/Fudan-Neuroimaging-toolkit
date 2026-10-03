"""CPU integrity/orchestration regression, not a scientific benchmark."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
TESTS = ["tests/connectome/test_checkpoints.py",
         "tests/connectome/test_pipeline_checkpoints.py",
         "tests/connectome/test_pipeline.py",
         "tests/connectome/test_bids_entry_options.py"]
SOURCES = ["src/fnit/connectome/checkpoints.py", "src/fnit/connectome/pipeline.py", *TESTS]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


class Outcomes:
    def __init__(self):
        self.passed, self.failed = [], []

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.append(report.nodeid + ":" + report.when)
        elif report.when == "call" and report.passed:
            self.passed.append(report.nodeid)


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run this unit-test gate with CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(4)
    os.chdir(ROOT)
    before = hashes()
    initialized_before = torch.cuda.is_initialized()
    observer = Outcomes()
    start = time.perf_counter()
    code = pytest.main(["-q", *TESTS], plugins=[observer])
    elapsed = time.perf_counter() - start
    after = hashes()
    initialized_after = torch.cuda.is_initialized()
    report = dict(kind="CPU simulated-stage integrity/orchestration regression",
                  scientific_benchmark=False, baseline="231dfaa16f479c8f076a8ce38d4bfe690768217b",
                  hostname=platform.node(), executable=sys.executable, prefix=sys.prefix,
                  source_root=str(ROOT), source_sha256_before=before, source_sha256_after=after,
                  source_unchanged=before == after, torch=str(torch.__version__),
                  cpu_threads=torch.get_num_threads(), cuda_initialized_before=initialized_before,
                  cuda_initialized_after=initialized_after, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                  wall_seconds=elapsed, exit_code=int(code), passed_tests=observer.passed,
                  failed_tests=observer.failed)
    destination = Path(sys.argv[1])
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("wall_seconds", "exit_code", "source_unchanged",
                                                  "cuda_initialized_after")}))
    if code or before != after or initialized_before or initialized_after:
        sys.exit(int(code) or 2)
