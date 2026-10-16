"""用自产真实上游单独测量 Conda 源码编译的拓扑 GA。

输入是已有 FNIT 整例 subject 的 orig.nofix、inflated.nofix、
qsphere.nofix、brain.mgz 和 wm.mgz，分别位于 surface RAS/mm 与
conformed MRI 网格。输出为全新隔离 subject 的 orig.premesh、程序日志和
report.json；不会读取官方结果或改写原 subject。完整整例耗时另测。

示例：
    python run_native_stage.py --source-subject /private/run/subject --output-root /private/new-stage --hemisphere rh --binary /conda/bin/mris_fix_topology_fnit --assets /fnit/assets

--source-subject 是 FNIT 自产的完整被试目录；--output-root 是尚不存在的隔离输出目录；
--hemisphere 指定半球；--binary 指向 Conda 独立源码构建程序；--assets 指向 FNIT 已声明的数据资产。

对应 FreeSurfer 命令为 mris_fix_topology -ga -seed 1234；实际执行命令
由 topology_conda_ga.run_topology_ga_conda 记录在报告中。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_conda_ga import run_topology_ga_conda


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_summary(path: Path) -> dict:
    vertices, faces = fsio.read_geometry(str(path))
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    n = len(vertices)
    valid = bool(vertices.ndim == 2 and vertices.shape[1] == 3
                 and faces.ndim == 2 and faces.shape[1] == 3
                 and np.isfinite(vertices).all()
                 and (not faces.size or (faces.min() >= 0 and faces.max() < n)))
    if not valid:
        raise ValueError(f"invalid ordered output mesh: {path}")
    return {"vertices": n, "faces": len(faces), "valid_indices_and_finite_coords": valid,
            "vertex_dtype": str(vertices.dtype), "face_dtype": str(faces.dtype),
            "ordered_vertex_sha256": hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest(),
            "ordered_face_sha256": hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-subject", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_subject.resolve()
    root = args.output_root.resolve()
    binary = args.binary.resolve()
    assets = args.assets.resolve()
    if root.exists():
        raise FileExistsError(f"output-root must not exist: {root}")
    names = (f"surf/{args.hemisphere}.orig.nofix",
             f"surf/{args.hemisphere}.inflated.nofix",
             f"surf/{args.hemisphere}.qsphere.nofix", "mri/brain.mgz", "mri/wm.mgz")
    input_paths = {name: source / name for name in names}
    for path in (*input_paths.values(), binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    input_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    root.mkdir(parents=True, mode=0o700)
    subject = root / "subject"
    for name, path in input_paths.items():
        destination = subject / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if _sha256(destination) != input_hashes[name]:
            raise RuntimeError(f"input copy changed: {name}")
    report = {
        "scope": "same FNIT-produced upstream; isolated native topology stage, not whole T1",
        "status": "running", "host": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "hemisphere": args.hemisphere, "source_subject": str(source),
        "input_sha256": input_hashes, "binary": str(binary),
        "binary_sha256": _sha256(binary), "assets": str(assets),
        "script_sha256": _sha256(Path(__file__)),
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    try:
        result = run_topology_ga_conda(subject=subject, hemisphere=args.hemisphere,
                                       binary=binary, assets=assets)
        report.update(status="complete", complete_api_wall_seconds=time.perf_counter()-started,
                      preflight=result["preflight"], native_seconds=result["native_seconds"],
                      command=result["command"], output_mesh=_mesh_summary(Path(result["output"])),
                      output_sha256=_sha256(Path(result["output"])), log_sha256=_sha256(Path(result["log"])))
    except Exception as error:
        report.update(status="failed", complete_api_wall_seconds=time.perf_counter()-started,
                      error=repr(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "native_seconds": report["native_seconds"],
                      "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
