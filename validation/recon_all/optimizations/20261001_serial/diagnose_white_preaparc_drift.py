"""在隔离目录交叉自产MRI/WM与目标强度统计，定位white.preaparc变化。

输入为两个完整FNIT subject、同一源码构建binary、声明assets、源码commit、
hemi（默认rh）及线程（默认4）。影像均在conform网格，网格坐标为surface RAS mm。
只允许orig有序面/坐标、aseg.presurf的数组/dtype/affine完全相同；不符即失败。
brain.finalsurfs与wm可有体素差异，但网格、dtype必须相同。output必须不存在。
复制最小输入后执行四组（brain.finalsurfs、wm）×autodet统计；
每组保存命令、日志、完整白质表面和JSON计时，最终报告比较原保存表面和四组几何。
这是隔离诊断，交叉输入不得接入生产。对应mris_place_surface --white的固定分支。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
from nibabel.freesurfer import io as fsio
import numpy as np


def sha(path):
    """对必填文件path分块计算SHA-256，返回十六进制字符串；读失败抛异常。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distance(first, second):
    """比较两个必填表面路径，返回同索引均值/P99/最大位移（mm）与差异计数。

    输入需有相同顶点顺序及有序面，否则抛ValueError，不用最近点冒充顶点对应。
    """
    a, fa = fsio.read_geometry(str(first))
    b, fb = fsio.read_geometry(str(second))
    if a.shape != b.shape or not np.array_equal(fa, fb):
        raise ValueError("ordered vertex correspondence unavailable")
    d = np.linalg.norm(a-b, axis=1)
    return {"mean_mm": float(d.mean()), "p99_mm": float(np.quantile(d, .99)),
            "max_mm": float(d.max()), "vertices_over_0_1_mm": int((d>.1).sum()),
            "different_coordinate_components": int(np.count_nonzero(a!=b))}


def main():
    """读取具名CLI参数，保存四组诊断及绑定代码/程序/输入哈希的report.json。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-subject", "candidate-subject", "output", "binary", "assets"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), default="rh")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    sources = {"baseline": args.baseline_subject, "candidate": args.candidate_subject}
    for name in ("wm", "aseg.presurf", "brain.finalsurfs"):
        a, b = [nib.load(str(root/"mri"/(name+".mgz"))) for root in sources.values()]
        if (a.get_data_dtype()!=b.get_data_dtype() or a.shape!=b.shape
                or not np.array_equal(a.affine,b.affine)
                or (name=="aseg.presurf" and
                    not np.array_equal(np.asarray(a.dataobj),np.asarray(b.dataobj)))):
            raise ValueError("different placement input: "+name)
    distance(args.baseline_subject/"surf"/(args.hemi+".orig"),
             args.candidate_subject/"surf"/(args.hemi+".orig"))
    a, fa = fsio.read_geometry(str(args.baseline_subject/"surf"/(args.hemi+".orig")))
    b, fb = fsio.read_geometry(str(args.candidate_subject/"surf"/(args.hemi+".orig")))
    if not np.array_equal(a,b):
        raise ValueError("different orig coordinates")
    args.output.mkdir(parents=True)
    report = {"code_commit": args.code_commit, "script_sha256": sha(__file__),
              "binary": str(args.binary.resolve()), "binary_sha256": sha(args.binary),
              "host": os.uname().nodename, "threads": args.threads, "hemi": args.hemi,
              "scope": "frozen_self_generated_cross_input_diagnostic; no production writes",
              "coordinate_unit": "surface RAS mm", "runs": [],
              "input_sha256": {kind: {name: sha(root/name) for name in
                  ("mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz",
                   "surf/"+args.hemi+".orig", "surf/autodet.gw.stats."+args.hemi+".dat")}
                  for kind,root in sources.items()}}
    env = dict(os.environ, FREESURFER_HOME=str(args.assets.resolve()),
               SUBJECTS_DIR=str(args.output.resolve()),
               OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), CUDA_VISIBLE_DEVICES="")
    for image_kind, stats_kind in (("baseline","baseline"), ("candidate","candidate"),
                                    ("baseline","candidate"), ("candidate","baseline")):
        folder = args.output/(image_kind+"_inputs_"+stats_kind+"_stats")
        (folder/"mri").mkdir(parents=True); (folder/"surf").mkdir()
        shutil.copyfile(args.baseline_subject/"mri/aseg.presurf.mgz", folder/"mri/aseg.presurf.mgz")
        shutil.copyfile(sources[image_kind]/"mri/wm.mgz", folder/"mri/wm.mgz")
        shutil.copyfile(sources[image_kind]/"mri/brain.finalsurfs.mgz", folder/"mri/brain.finalsurfs.mgz")
        shutil.copyfile(args.baseline_subject/"surf"/(args.hemi+".orig"), folder/"surf"/(args.hemi+".orig"))
        stats = folder/"surf"/("autodet.gw.stats."+args.hemi+".dat")
        shutil.copyfile(sources[stats_kind]/"surf"/stats.name, stats)
        surface = folder/"surf"/(args.hemi+".white.preaparc")
        command = [str(args.binary.resolve()), "--adgws-in", str(stats.resolve()),
                   "--wm", str((folder/"mri/wm.mgz").resolve()), "--threads", str(args.threads),
                   "--invol", str((folder/"mri/brain.finalsurfs.mgz").resolve()), "--"+args.hemi,
                   "--i", str((folder/"surf"/(args.hemi+".orig")).resolve()), "--o", str(surface.resolve()),
                   "--white", "--seg", str((folder/"mri/aseg.presurf.mgz").resolve()),
                   "--restore-255", "--nsmooth", "5", "--rip-bg-no-annot", "--rip-bg",
                   "--rip-bg-lof", "--restore-255", "--outvol", str((folder/"mri/mrisps.wpa.mgz").resolve())]
        start = time.perf_counter()
        with (folder/"command.log").open("w") as log:
            subprocess.run(command, cwd=folder/"mri", env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        row = {"volume_inputs": image_kind, "stats": stats_kind, "command": command,
               "seconds_including_process_and_output_io": time.perf_counter()-start,
               "checkpoint_copy_in_timing": False, "output_sha256": sha(surface),
               "against_saved": {kind: distance(root/"surf"/surface.name, surface)
                                 for kind,root in sources.items()}}
        (folder/"report.json").write_text(json.dumps(row, indent=2)+"\n")
        report["runs"].append(row)
        (args.output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
    report["status"] = "complete"
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
