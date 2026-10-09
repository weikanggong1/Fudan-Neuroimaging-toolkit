"""在实际Python环境验证归一化接线；不执行模拟影像benchmark。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest


def main() -> int:
    """具名参数指定冻结源码、版本和新JSON路径；返回0表示入口契约通过。

    source-root须包含src/tests/tools，code-version记录实际Git提交；
    output-report必须不存在。测试覆盖非法CPU/设备组合、单例、批量、
    CLI和benchmark选项转发，以及既有标准输出/计时契约。不测真实图像
    精度或速度；异常/失败保留测试详情，文件/导入错误直接传播。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    if args.output_report.exists():
        raise FileExistsError(args.output_report)
    sys.path.insert(0, str(root / "src"))
    tests = ("test_normalization_controls_wiring", "test_standard_recon_wiring",
             "test_batch_python", "test_stage_metadata")
    files = [root / "tests/recon_all" / (name + ".py") for name in tests]
    files += [root / path for path in ("src/fnit/recon_all/native_free.py",
              "src/fnit/recon_all/batch.py", "tools/benchmark_recon_torch_end_to_end.py",
              "src/fnit/recon_all/stage_metadata.py")]
    digest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    suite = unittest.TestSuite()
    for name in tests:
        spec = importlib.util.spec_from_file_location(name, root / "tests/recon_all" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    import torch
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"scope": "entry contracts only; not real-image performance or numerical equivalence",
              "code_version": args.code_version, "source_sha256": digest,
              "python": platform.python_version(), "torch": torch.__version__,
              "torch_threads": torch.get_num_threads(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "wall_seconds": time.perf_counter() - started,
              "status": "passed" if result.wasSuccessful() else "failed", "test_log": stream.getvalue()}
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("tests_run", "failures", "errors", "skipped", "wall_seconds", "status")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
