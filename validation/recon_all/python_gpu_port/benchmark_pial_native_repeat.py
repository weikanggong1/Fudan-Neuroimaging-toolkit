"""隔离目录中新建同输入 Conda pial 双重复；原生参考不供生产读取。"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import nibabel.freesurfer.io as fs
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=Path, required=True)
    p.add_argument("--native-binary", type=Path, required=True)
    p.add_argument("--assets-directory", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    os.chmod(args.output_directory,0o700)
    h, out = args.hemisphere, args.output_directory
    inputs=[Path(f"surf/{h}.white"),Path(f"surf/autodet.gw.stats.{h}.dat"),
        *[Path(f"label/{h}.{n}") for n in ("cortex.label","cortex+hipamyg.label","aparc.annot")],
        *[Path(f"mri/{n}.mgz") for n in ("brain.finalsurfs","wm","aseg.presurf")]]
    report={"scope":"same_host_same_input_Conda_pial_two_fresh_native_repetitions",
        "reference_kind":"independently Conda source-built executable, not system installed FreeSurfer",
        "hostname":platform.node(),"threads":args.threads,"cpu_affinity_count":len(os.sched_getaffinity(0)),
        "binary_sha256":sha(args.native_binary),"script_sha256":sha(__file__),
        "input_sha256":{str(p):sha(args.subject/p) for p in inputs},"runs":[],"status":"started"}
    def save():
        tmp=out/"report.tmp";tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");tmp.replace(out/"report.json")
    save()
    reference=None
    for index in range(2):
        subject=out/f"repeat{index}/subject"
        for path in inputs:
            dest=subject/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.subject/path,dest)
        log=out/f"native-{index}.private.log"
        command=[sys.executable,"-m","fnit.recon_all.pial_t1_conda",str(subject),h,
            "--binary",str(args.native_binary),"--assets-dir",str(args.assets_directory),"--threads",str(args.threads)]
        started=time.perf_counter()
        with log.open("w") as stream:
            process=subprocess.run(command,stdout=stream,stderr=stream)
        elapsed=time.perf_counter()-started
        row={"index":index,"returncode":process.returncode,"cold_wrapper_CLI_seconds":elapsed,
            "private_log_sha256":sha(log)}
        if process.returncode:
            report["runs"].append(row);report["status"]="failed";save();raise RuntimeError("native reference failed; inspect private log")
        # 只解析 wrapper 的最后一个返回字典，不输出许可证等上游日志文本。
        native_seconds=None
        for line in reversed(log.read_text(errors="replace").splitlines()):
            if line.startswith("{'output':"):
                native_seconds=float(ast.literal_eval(line)["seconds"]);break
        surface=subject/f"surf/{h}.pial.T1";xyz,faces=fs.read_geometry(str(surface))
        if reference is None:
            reference=(xyz,faces)
        row.update(native_binary_CLI_seconds=native_seconds,output_sha256=sha(surface),
            ordered_faces_same_as_first=bool(np.array_equal(faces,reference[1])),
            different_coordinate_elements_from_first=int(np.count_nonzero(xyz!=reference[0]))
                if xyz.shape==reference[0].shape else None,
            native_input_sha256={str(p):sha(subject/p) for p in inputs})
        report["runs"].append(row);save()
    report["input_sha256_after"]={str(p):sha(args.subject/p) for p in inputs}
    report["status"]="complete" if report["input_sha256"]==report["input_sha256_after"] else "input_changed"
    report["repeatability_exact_geometry"]=all(r["ordered_faces_same_as_first"] and
        r["different_coordinate_elements_from_first"]==0 for r in report["runs"])
    save();print(json.dumps({"status":report["status"],"repeatability_exact_geometry":report["repeatability_exact_geometry"],
        "seconds":[r["native_binary_CLI_seconds"] for r in report["runs"]]}))


if __name__=="__main__":
    main()
