"""真实同输入 white 首轮前缀的 CPU/Torch 优化回归；不是完整 white 验收。"""
from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--steps", type=int, choices=range(1, 18), default=1)
    parser.add_argument("--code-base-commit", required=True)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import place_white_preaparc_python as stage
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hemi = args.hemisphere
    inputs = [args.subject / f"surf/{hemi}.orig", args.subject / f"surf/autodet.gw.stats.{hemi}.dat"]
    inputs += [args.subject / f"mri/{name}.mgz" for name in ("brain.finalsurfs", "wm", "aseg.presurf")]
    report = {"scope": "frozen_same_input_first_white_pass_prefix_only", "hostname": platform.node(),
              "code_base_commit": args.code_base_commit, "torch": torch.__version__,
              "threads": args.threads, "device": args.device, "steps_requested": args.steps,
              "tf32_matmul": True, "tf32_cudnn": True, "half_precision": False,
              "input_sha256": {path.name: sha256(path) for path in inputs},
              "stages": {}, "source_sha256": {}, "order": ["cpu", "torch"], "repetitions": 1,
              "admission_requirement": "identical ordered prefix geometry and step decisions",
              "complete_white_validation": "not_assessed", "overall_metric_equivalence": "not_assessed",
              "process_gpu_memory_sampling": "not_measured"}
    def save():
        import sys
        for name, module in list(sys.modules.items()):
            if name.startswith("fnit.recon_all.place_") and getattr(module, "__file__", None):
                report["source_sha256"][Path(module.__file__).name] = sha256(module.__file__)
        report["source_sha256"][Path(__file__).name] = sha256(__file__)
        (args.output_directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save()
    for backend in report["order"]:
        output = args.output_directory / f"{hemi}.white-prefix.{backend}"
        tick = time.perf_counter()
        try:
            result = stage.place_white_preaparc_prefix(subject_dir=args.subject, hemi=hemi,
                output=output, steps=args.steps, diagnostics=None,
                regularization_backend=backend, device=args.device if backend == "torch" else None)
            if backend == "torch":
                torch.cuda.synchronize(torch.device(args.device))
            report["stages"][backend] = {"status": "complete", "wall_seconds": time.perf_counter()-tick,
                "stage": result, "output_sha256": sha256(output),
                "peak_allocated_bytes": None if backend == "cpu" else torch.cuda.max_memory_allocated(torch.device(args.device)),
                "peak_reserved_bytes": None if backend == "cpu" else torch.cuda.max_memory_reserved(torch.device(args.device))}
        except Exception as exc:
            report["stages"][backend] = {"status": "failed", "wall_seconds": time.perf_counter()-tick,
                                         "error": str(exc), "traceback": traceback.format_exc()}
            save()
            raise
        save()
    a, af = fs.read_geometry(str(args.output_directory / f"{hemi}.white-prefix.cpu"))
    b, bf = fs.read_geometry(str(args.output_directory / f"{hemi}.white-prefix.torch"))
    ordered = a.shape == b.shape and np.array_equal(af, bf)
    difference = None if not ordered else np.linalg.norm(a-b, axis=1)
    report["comparison"] = {"same_vertex_count_and_ordered_faces": ordered,
        "same_per_step": report["stages"]["cpu"]["stage"]["per_step"] == report["stages"]["torch"]["stage"]["per_step"],
        "different_coordinate_elements": None if not ordered else int(np.count_nonzero(a != b)),
        "max_vertex_distance_mm": None if not ordered else float(difference.max()),
        "p99_vertex_distance_mm": None if not ordered else float(np.percentile(difference, 99)),
        "speed_ratio_cpu_over_torch": report["stages"]["cpu"]["wall_seconds"] / report["stages"]["torch"]["wall_seconds"]}
    report["input_sha256_after"] = {path.name: sha256(path) for path in inputs}
    save()


if __name__ == "__main__":
    main()
