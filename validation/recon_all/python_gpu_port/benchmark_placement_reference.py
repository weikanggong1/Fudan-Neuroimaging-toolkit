"""仅用于独立 benchmark 的固定 white 输入 pial 原生参考，生产禁止使用此入口。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel.freesurfer.io as fs
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--python-report", type=Path)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    if args.threads < 1 or args.repeat < 1:
        raise ValueError("threads/repeat must be positive")
    args.output_directory.mkdir(parents=True)
    hemi = args.hemisphere
    inputs = [args.subject / f"surf/{hemi}.white", args.subject / f"surf/autodet.gw.stats.{hemi}.dat"]
    inputs += [args.subject / f"label/{hemi}.{suffix}" for suffix in ("cortex.label", "cortex+hipamyg.label", "aparc.annot")]
    inputs += [args.subject / f"mri/{name}.mgz" for name in ("brain.finalsurfs", "wm", "aseg.presurf")]
    report = {"scope": "isolated_same_input_pial_native_reference_only", "hostname": platform.node(),
              "threads": args.threads, "repeat": args.repeat, "binary_sha256": sha256(args.binary),
              "input_sha256": {path.name: sha256(path) for path in inputs}, "runs": [],
              "overall_metric_equivalence": "not_assessed", "script_sha256": sha256(__file__)}
    env = dict(os.environ, FREESURFER_HOME=str(args.assets_directory),
               SUBJECTS_DIR=str(args.subject.parent), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    for index in range(args.repeat):
        output = args.output_directory / f"{hemi}.pial.reference-{index}"
        command = [str(args.binary), "--adgws-in", str(inputs[1]), "--seg", str(inputs[-1]),
            "--threads", str(args.threads), "--wm", str(inputs[-2]), "--invol", str(inputs[-3]),
            f"--{hemi}", "--i", str(inputs[0]), "--o", str(output), "--pial", "--nsmooth", "0",
            "--rip-label", str(inputs[3]), "--pin-medial-wall", str(inputs[2]), "--aparc", str(inputs[4]),
            "--repulse-surf", str(inputs[0]), "--white-surf", str(inputs[0]), "--restore-255"]
        tick = time.perf_counter()
        with (args.output_directory / f"reference-{index}.log").open("w") as stream:
            result = subprocess.run(command, cwd=args.output_directory, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        row = {"command": command, "exit_code": result.returncode,
               "wall_seconds": time.perf_counter()-tick, "output": str(output)}
        if output.is_file():
            row["output_sha256"] = sha256(output)
        report["runs"].append(row)
        (args.output_directory / "report.json").write_text(json.dumps(report, indent=2))
        if result.returncode:
            raise RuntimeError(f"Reference exit {result.returncode}; failure preserved")
    def compare(first, second):
        a, af = fs.read_geometry(str(first)); b, bf = fs.read_geometry(str(second))
        ordered = a.shape == b.shape and np.array_equal(af, bf)
        delta = None if not ordered else np.linalg.norm(a-b, axis=1)
        return {"same_vertex_count_and_ordered_faces": ordered,
                "different_coordinate_elements": None if not ordered else int(np.count_nonzero(a != b)),
                "mean_vertex_distance_mm": None if not ordered else float(delta.mean()),
                "p99_vertex_distance_mm": None if not ordered else float(np.percentile(delta, 99)),
                "max_vertex_distance_mm": None if not ordered else float(delta.max())}
    if args.repeat > 1:
        report["reference_reproducibility"] = [compare(report["runs"][0]["output"], row["output"])
                                                 for row in report["runs"][1:]]
    if args.python_report is not None:
        python_report = json.loads(args.python_report.read_text())
        common = set(python_report["input_sha256"]) & set(report["input_sha256"])
        matched = all(python_report["input_sha256"][name] == report["input_sha256"][name] for name in common)
        report["python_input_hashes_match"] = matched and len(common) == 7
        if not report["python_input_hashes_match"]:
            raise ValueError("All seven Python pial input hashes must match native reference")
        report["comparison_to_python"] = {backend: compare(args.python_report.parent / f"{hemi}.pial.{backend}",
                                            report["runs"][0]["output"]) for backend in ("cpu", "torch")}
    report["input_sha256_after"] = {path.name: sha256(path) for path in inputs}
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
