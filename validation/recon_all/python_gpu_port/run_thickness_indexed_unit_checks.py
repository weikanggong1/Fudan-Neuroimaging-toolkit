"""没有 pytest 的主页 Conda 环境：执行七项厚度 CPU 单元语义检查。

读取候选源码、冻结原 FNIT 源码及测试源码；通过标准库 AST 选取相同测试
函数和 patch 上下文执行，不模拟真实数据。输出含输入 SHA-256 的 JSON。
四个路径均显式指定；测试失败抛异常，不将单元测试记作真实 benchmark。
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
from unittest.mock import patch

import nibabel.freesurfer as fs
import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-source-file", "baseline-source-file", "tests", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    stage = load_module("fnit_thickness_unit", args.candidate_source_file)
    baseline = load_module("fnit_thickness_baseline_unit", args.baseline_source_file)
    names = ["test_compiled_reachability_keeps_twenty_hop_boundary",
             "test_compiled_reachability_checks_all_equal_distance_candidates",
             "test_reachability_workspace_does_not_leak_previous_vertices",
             "test_clip_mean_preserves_existing_float32_conversion",
             "test_radius_candidates_preserves_values_across_thread_budgets",
             "test_indexed_cpu_matches_existing_folded_surface",
             "test_indexed_search_includes_candidates_beyond_256"]
    selected = [node for node in ast.parse(args.tests.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    space = {"stage": stage, "np": np, "fs": fs, "torch": torch,
             "dense_thickness_map": baseline.thickness_map}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.tests), "exec"), space)
    rows = []
    for name in names:
        tick = time.perf_counter()
        if name == "test_indexed_search_includes_candidates_beyond_256":
            with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
                class Setter:
                    def setattr(self, target, key, value):
                        stack.enter_context(patch.object(target, key, value))
                space[name](Path(directory), Setter())
        elif name == "test_indexed_cpu_matches_existing_folded_surface":
            with tempfile.TemporaryDirectory() as directory:
                space[name](Path(directory))
        else:
            space[name]()
        rows.append({"test": name, "pass": True, "seconds": time.perf_counter() - tick})
    paths = {"candidate": args.candidate_source_file, "baseline": args.baseline_source_file,
             "tests": args.tests, "script": Path(__file__)}
    report = {"scope": "synthetic unit regression; not real-data benchmark",
              "host": socket.gethostname(), "device": "cpu",
              "execution": "selected test functions via AST; standard-library patch context",
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in paths.items()}, "runs": rows, "pass": True}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
