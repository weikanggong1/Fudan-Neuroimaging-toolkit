"""复跑 GA 排列的本地接口合同并绑定源文件；不是影像 benchmark。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest


def run(*, fixed_source, output):
    """本地固定实现/可选只读原源码 → JSON 合同报告。

    fixed_source 为可选固定 d932 mrisurf_defect.cpp；仅检验唯一插桩
    锚点，不编译/执行/上传原软件。output 为新 JSON 路径，必须不存在。
    输出测试数、失败数、合同运行时间、当前 HEAD、代码 SHA 和边界，
    不含服务器地址、原影像、完整私有收据；测试失败写报告后非零退出。
    """
    root = Path(__file__).resolve().parents[4]
    if output.exists():
        raise FileExistsError("new local contract report required")
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tools"))
    before_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    relative_files = [
        "src/fnit/recon_all/topology_ga_ordering.py",
        "src/fnit/recon_all/topology_fitness_search.py",
        "src/fnit/recon_all/gcsa_permutation.py",
        "tests/recon_all/test_topology_ga_ordering.py",
        "tests/recon_all/test_topology_ga_ordering_capture.py",
        "tools/benchmark_recon_topology_ga_ordering.py",
        "validation/recon_all/optimizations/20261010_topology_ga_ordering/build_ordering_observer.py",
        "validation/recon_all/optimizations/20261010_topology_ga_ordering/run_local_contracts.py",
    ]
    source_files = {}
    for relative in relative_files:
        data = (root / relative).read_bytes()
        source_files[relative] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    anchor_passed = None
    reference_source_sha256 = None
    if fixed_source is not None:
        reference_source_sha256 = hashlib.sha256(fixed_source.read_bytes()).hexdigest()
        expected = "55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187"
        if reference_source_sha256 != expected:
            raise ValueError("fixed d932 reference SHA differs")
        observer = root / relative_files[6]
        spec = importlib.util.spec_from_file_location("ordering_observer", observer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.instrument_ordering_source(source_text=fixed_source.read_text())
        anchor_passed = True
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(root / "tests/recon_all"), pattern="test_topology_ga_ordering*.py"
    )
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    # 检查测试期间实现没有被并发改写；其他 scope 的 HEAD 提交单独记录。
    for relative, record in source_files.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("tested source changed during local contracts")
    report = dict(
        status="complete_local_interface_contracts_only",
        observed_utc=datetime.now(timezone.utc).isoformat(),
        git_head_before_tests=before_head,
        git_head_after_tests=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_files=source_files,
        python=platform.python_version(),
        tests_run=result.testsRun,
        tests_failed=len(result.failures) + len(result.errors),
        tests_skipped=len(result.skipped),
        contract_runner_seconds=elapsed,
        source_anchor_contract_passed=anchor_passed,
        reference_source_sha256=reference_source_sha256,
        original_random_call_sites_preserved=8 if anchor_passed else None,
        real_before_after_capture="pending_explicit_observer_upload_authorization",
        strict_real_stage="not_assessed",
        whole_ga=False,
        whole_recon_all=False,
        overall_equivalence="not_assessed",
        actual_cuda_measurement=False,
        process_gpu_memory_bytes=None,
        evidence="local array and capture-schema contracts; not real MRI benchmark",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(**vars(parser.parse_args())))
