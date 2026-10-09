"""同一真实 white/MRI 输入的完整 Python pial CPU/Torch 正则梯度回归。"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
import traceback

import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--code-base-commit", required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import place_pial_python as stage
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hemi = args.hemisphere
    inputs = [args.subject / f"surf/{hemi}.white", args.subject / f"surf/autodet.gw.stats.{hemi}.dat",
              args.subject / f"label/{hemi}.cortex.label", args.subject / f"label/{hemi}.cortex+hipamyg.label"]
    inputs += [args.subject / f"mri/{name}.mgz" for name in ("brain.finalsurfs", "wm", "aseg.presurf")]
    report = {"scope": "frozen_same_input_complete_python_pial", "hostname": platform.node(),
              "code_base_commit": args.code_base_commit, "candidate_snapshot": "source_sha256",
              "torch": torch.__version__, "threads": args.threads, "device": args.device,
              "tf32_matmul": True, "tf32_cudnn": True, "half_precision": False,
              "gpu": None, "input_sha256": {path.name: sha256(path) for path in inputs},
              "stages": {}, "order": ["cpu", "torch"], "repetitions": 1,
              "admission_requirement": "identical ordered final geometry and pass/trial decisions",
              "overall_metric_equivalence": "not_assessed", "source_sha256": {},
              "process_gpu_memory_sampling": "not_measured"}
    # 所有实际参与模块分别绑定；不把服务器旧repo提交当成候选提交。
    import sys
    for name, module in list(sys.modules.items()):
        if name.startswith("fnit.recon_all.place_") and getattr(module, "__file__", None):
            report["source_sha256"][Path(module.__file__).name] = sha256(module.__file__)
    report["source_sha256"][Path(__file__).name] = sha256(__file__)
    def save():
        for name, module in list(sys.modules.items()):
            if name.startswith("fnit.recon_all.place_") and getattr(module, "__file__", None):
                report["source_sha256"][Path(module.__file__).name] = sha256(module.__file__)
        (args.output_directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save()
    outputs, traces = {}, {}
    for backend in report["order"]:
        output = args.output_directory / f"{hemi}.pial.{backend}"
        trace = []
        def callback(step, outer_pass, coordinates, diagnostics):
            row = {"step": step, "pass": outer_pass,
                   "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
                   "diagnostics": diagnostics}
            trace.append(row)
            print(json.dumps({"backend": backend, **row}, ensure_ascii=False), flush=True)
        tick = time.perf_counter()
        try:
            result = stage.place_pial_t1(subject=args.subject, hemisphere=hemi, output=output,
                max_steps=args.max_steps, sampling_backend="cpu", regularization_backend=backend,
                candidate_backend="tree", device=args.device if backend == "torch" else None,
                trace_callback=callback)
            if backend == "torch":
                torch.cuda.synchronize(torch.device(args.device))
                report["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
            report["stages"][backend] = {"status": "complete", "wall_seconds": time.perf_counter()-tick,
                "stage": result, "output_sha256": sha256(output), "trace": trace,
                "peak_allocated_bytes": None if backend == "cpu" else torch.cuda.max_memory_allocated(torch.device(args.device)),
                "peak_reserved_bytes": None if backend == "cpu" else torch.cuda.max_memory_reserved(torch.device(args.device))}
            outputs[backend] = output
            traces[backend] = trace
        except Exception as exc:
            report["stages"][backend] = {"status": "failed", "wall_seconds": time.perf_counter()-tick,
                                         "error": str(exc), "traceback": traceback.format_exc(), "trace": trace}
            save()
            raise
        save()
    a, af = fs.read_geometry(str(outputs["cpu"]))
    b, bf = fs.read_geometry(str(outputs["torch"]))
    ordered = a.shape == b.shape and np.array_equal(af, bf)
    difference = None if not ordered else np.linalg.norm(a-b, axis=1)
    report["comparison"] = {"same_vertex_count_and_ordered_faces": ordered,
        "same_file_bytes": sha256(outputs["cpu"]) == sha256(outputs["torch"]),
        "same_trace": traces["cpu"] == traces["torch"],
        "different_coordinate_elements": None if not ordered else int(np.count_nonzero(a != b)),
        "max_vertex_distance_mm": None if not ordered else float(difference.max()),
        "p99_vertex_distance_mm": None if not ordered else float(np.percentile(difference, 99)),
        "speed_ratio_cpu_over_torch": report["stages"]["cpu"]["wall_seconds"] / report["stages"]["torch"]["wall_seconds"]}
    save()


if __name__ == "__main__":
    main()
