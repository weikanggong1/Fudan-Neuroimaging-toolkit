"""完整同输入回归后核对输出；必须先证明顶点数、有序面和输入 SHA 相同。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True, help="旧实现已完成阶段目录")
    parser.add_argument("--candidate", type=Path, required=True, help="新实现已完成阶段目录")
    parser.add_argument("--output", type=Path, required=True, help="新的 JSON 比较收据")
    args = parser.parse_args()
    import numpy as np
    from nibabel.freesurfer.io import read_geometry
    old = json.loads((args.control / "report.json").read_bytes())
    new = json.loads((args.candidate / "report.json").read_bytes())
    if old["status"] != "complete" or new["status"] != "complete":
        raise ValueError("不能将失败或未完成阶段作为完整同索引对照")
    for report in (old, new):
        if report["input_sha256_before"] != report["input_sha256_after"]:
            raise ValueError("输入在计算期间发生改变")
    if old["input_sha256_before"] != new["input_sha256_before"]:
        raise ValueError("两个阶段没有使用相同输入")
    old_path, new_path = args.control / "sphere.reg", args.candidate / "sphere.reg"
    before = [sha(path) for path in (old_path, new_path)]
    a, af = read_geometry(str(old_path))
    b, bf = read_geometry(str(new_path))
    if a.shape != b.shape or not np.array_equal(af, bf):
        raise ValueError("顶点对应关系不成立，禁止逐索引比较")
    displacement = np.linalg.norm(a - b, axis=1)
    result = {"scope": "complete frozen same-input sphere registration; not whole recon-all",
              "status": "complete", "comparison_script_sha256": sha(Path(__file__)),
              "control_report_sha256": sha(args.control / "report.json"),
              "candidate_report_sha256": sha(args.candidate / "report.json"),
              "control_source_module_sha256": old["selected_module_sha256"],
              "candidate_source_module_sha256": new["selected_module_sha256"],
              "input_sha256": old["input_sha256_before"], "vertices": len(a), "faces": len(af),
              "ordered_faces_identical": bool(np.array_equal(af, bf)),
              "finite_coordinates": bool(np.isfinite(a).all() and np.isfinite(b).all()),
              "different_coordinates": int(np.count_nonzero(a != b)),
              "different_vertices": int(np.count_nonzero(np.any(a != b, axis=1))),
              "mean_displacement_mm": float(displacement.mean()),
              "max_displacement_mm": float(displacement.max()),
              "p99_displacement_mm": float(np.percentile(displacement, 99)),
              "control_output_sha256": before[0], "candidate_output_sha256": before[1],
              "byte_exact": before[0] == before[1], "overall_equivalence": "not_assessed"}
    if before != [sha(path) for path in (old_path, new_path)]:
        raise ValueError("输出在比较期间改变")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "byte_exact": result["byte_exact"],
                      "different_coordinates": result["different_coordinates"]}))


if __name__ == "__main__":
    main()
