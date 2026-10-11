"""复验自产表面头；不是完整orig阶段或MRI精度benchmark。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel.freesurfer.io as fsio
import benchmark_recon_topology_genetic_orig as original_tool


def run(*, input_surfaces, case_names, output):
    """多个同意使用的表面Path及case名称→新JSON几何/版本收据。

    input_surfaces与case_names均必填非空等长列表；表面单位surface
    RAS/mm，原网格与顶点次序不修改。output必填且不得存在。只保存
    case名、输入SHA、顶点/面数、几何门及工具版本；无原影像或私有路径。
    缺文件/非法输入传播；头无效保存failed报告并抛RuntimeError。
    计时含加载、哈希和检查，不包括模块导入或原始T1整例。
    """
    if not input_surfaces or len(input_surfaces) != len(case_names):
        raise ValueError("nonempty equally sized surface paths and case names required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("new geometry report required")
    start = time.perf_counter()
    rows = []
    for name, path in zip(case_names, input_surfaces):
        xyz, faces, geometry = fsio.read_geometry(path, read_metadata=True)
        rows.append(dict(case=name, input_sha256=original_tool.sha(path),
                         vertices=len(xyz), faces=len(faces),
                         geometry=original_tool._geometry_quality(geometry)))
    passed = all(row["geometry"]["valid"] for row in rows)
    report = dict(status="passed_geometry_read_validation" if passed else "failed_geometry_read_validation",
                  scope="own surface header only, no GA or mesh algorithm",
                  cases=rows, original_tool_sha256=original_tool.sha(original_tool.__file__),
                  reproduction_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  whole_recon_all="not_assessed", full_orig_stage="not_assessed",
                  api_seconds=time.perf_counter() - start)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise RuntimeError("surface volume geometry invalid")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surfaces", nargs="+", type=Path, required=True)
    parser.add_argument("--case-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
