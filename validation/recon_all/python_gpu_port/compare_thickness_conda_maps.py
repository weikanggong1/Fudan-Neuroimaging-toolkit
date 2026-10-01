"""读取既有同输入厚度：旧/新 PyTorch 与 Conda 自编译程序的诊断。

pair-report 指向 benchmark_thickness_indexed 的报告，white/pial 的文件
SHA-256 必须对应报告；conda-map 是用这两个输入和 20/5 参数生成的图。
conda-binary 记录该已声明源码构建程序的哈希，不运行程序或初始化 CUDA。
输出 JSON 固定沿用绝对0.005mm+相对0.001门槛，记录各轮最大/P99/均值
绝对误差与越界顶点。官方新参考和整例等效均不由这个诊断判定。
"""

import argparse
import hashlib
import json
from pathlib import Path
import socket

import nibabel.freesurfer as fs
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair-report", "white", "pial", "conda-map", "conda-binary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pair = json.loads(args.pair_report.read_text())
    inputs = {"white": sha256(args.white), "pial": sha256(args.pial)}
    if inputs != pair["input_sha256"]:
        raise ValueError("white/pial SHA-256 与冻结配对报告不一致")
    reference = fs.read_morph_data(str(args.conda_map))
    if reference.shape != (pair["vertices"],) or not np.isfinite(reference).all():
        raise ValueError("Conda map 必须是同输入网格的有限同序顶点图")
    rows = []
    for run in pair["runs"]:
        for name, implementation in run["implementations"].items():
            path = args.pair_report.parent / f"{run['repeat']}.{name}.thickness"
            if sha256(path) != implementation["output_sha256"]:
                raise ValueError("厚度文件 SHA-256 与冻结配对报告不一致：" + str(path))
            candidate = fs.read_morph_data(str(path))
            if candidate.shape != reference.shape:
                raise ValueError("Python/Conda 厚度顶点必须同序且等长")
            delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
            outliers = ~np.isfinite(delta) | (delta > .005 + .001 * np.abs(reference))
            rows.append({"repeat": run["repeat"], "implementation": name,
                         "candidate_sha256": implementation["output_sha256"],
                         "different_values": int(np.count_nonzero(delta)),
                         "max_abs_mm": float(delta.max()), "p99_abs_mm": float(np.quantile(delta, .99)),
                         "mae_mm": float(delta.mean()), "outliers": int(outliers.sum()),
                         "pass": not bool(outliers.any())})
    report = {"scope": "frozen same-input maps; current Conda source-build, not a new official benchmark",
              "host": socket.gethostname(), "pair_code_commit": pair["code_commit"],
              "pair_source_sha256": pair["source_sha256"],
              "pair_report_sha256": sha256(args.pair_report), "input_sha256": inputs,
              "conda_program": {"path": str(args.conda_binary), "sha256": sha256(args.conda_binary)},
              "conda_map": {"path": str(args.conda_map), "sha256": sha256(args.conda_map)},
              "script_sha256": sha256(__file__), "vertices": pair["vertices"],
              "threshold": {"absolute_mm": .005, "relative": .001},
              "reference_command": [str(args.conda_binary), "--thickness", str(args.white),
                                    str(args.pial), "20", "5", str(args.conda_map)],
              "reference_command_status": "already generated separately; not executed by this comparator",
              "rows": rows, "pass": all(row["pass"] for row in rows),
              "new_official_validation": "not_run", "overall_equivalence": "not_assessed"}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
