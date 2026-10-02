"""冻结 FNIT 体积、white 与 cortex，仅换入候选厚度，诊断 w-g.pct 的差异来源。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all import vol2surf_contrast_python
from fnit.recon_all.vol2surf_contrast_python import contrast_percentage


def main() -> None:
    """具名目录和代码提交均必填；仅写隔离诊断目录，输入缺失时抛异常。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    subject = args.output_dir / "thickness-swap"
    (subject / "surf").mkdir(parents=True)
    for name in ("mri", "label"):
        (subject / name).symlink_to((args.baseline / name).resolve(), target_is_directory=True)
    torch.set_num_threads(4)
    source = Path(vol2surf_contrast_python.__file__)
    report = {"code_commit": args.code_commit, "scope": "FNIT frozen inputs; thickness-only swap",
              "stage_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": "cpu", "threads": 4, "hemispheres": {}}
    for hemi in ("lh", "rh"):
        inputs = {"mri/rawavg.mgz": args.baseline, "mri/orig.mgz": args.baseline,
                  f"surf/{hemi}.white": args.baseline,
                  f"label/{hemi}.cortex.label": args.baseline,
                  f"surf/{hemi}.thickness": args.candidate}
        for name in ("white", "thickness"):
            relative = f"surf/{hemi}.{name}"
            (subject / relative).symlink_to((inputs[relative] / relative).resolve())
        started = time.perf_counter()
        values = contrast_percentage(subject=subject, hemi=hemi, device="cpu")
        seconds = time.perf_counter() - started
        comparisons = {}
        for label, root in (("baseline", args.baseline), ("candidate", args.candidate)):
            reference = np.asarray(nib.load(str(root / f"surf/{hemi}.w-g.pct.mgh")).dataobj).ravel()
            error = np.abs(values.astype(np.float64) - reference)
            comparisons[label] = {"different_vertices": int(np.count_nonzero(error)),
                                  "maximum_absolute_percentage_points": float(error.max()),
                                  "p99_absolute_percentage_points": float(np.quantile(error, .99))}
        report["hemispheres"][hemi] = {
            "seconds_including_input_io": seconds, "comparisons": comparisons,
            "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name, root in inputs.items()}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
