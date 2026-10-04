"""Real-input CPU stage worker; official scoring runs after the measured job."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("subregions", "recon"))
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--atlas-root", type=Path)
    parser.add_argument("--aseg", type=Path)
    parser.add_argument("--wmparc", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--structures", default="all")
    parser.add_argument("--optimization", choices=("fast", "balanced"), default="fast")
    parser.add_argument("--hemisphere-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--native-optimizations", choices=("auto", "original"), default="auto")
    parser.add_argument("--save-posteriors", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    import fnit
    import numba
    import torch

    torch.set_num_interop_threads(1)
    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    before = {"torch_intraop": torch.get_num_threads(),
              "torch_interop": torch.get_num_interop_threads(),
              "numba_mask": numba.get_num_threads()}
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    if args.task == "subregions":
        from fnit import segment_4_subregions
        result = segment_4_subregions(
            t1=args.t1, atlas_root=args.atlas_root, structures=args.structures,
            coarse_segmentation=args.aseg, wmparc=args.wmparc,
            synthseg_weights=args.weights_dir, synthseg_parc_weights=args.weights_dir,
            device=args.device, threads=args.threads, optimization=args.optimization,
            output_dir=args.output_dir, save_highres=True,
            save_posteriors=args.save_posteriors)
        timings = dict(result.timings)
        report_file = result.output_files["report"]
    else:
        from fnit.recon_all.native_free import run_recon_all_python
        report = run_recon_all_python(
            t1=args.t1, subject_dir=args.output_dir, weights_dir=args.weights_dir,
            assets_dir=args.assets_dir, device=args.device, threads=args.threads,
            native_bin_dir=args.native_bin_dir,
            hemisphere_workers=args.hemisphere_workers,
            native_optimizations=args.native_optimizations)
        timings = {"public_api_total_seconds": report["total_seconds"]}
        report_file = args.output_dir / "fnit-native-free-run.json"
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    api_seconds = time.perf_counter() - started
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.output_dir / "cpu.prof"))
        with (args.output_dir / "cpu_profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(100)
    after = {"torch_intraop": torch.get_num_threads(),
             "torch_interop": torch.get_num_interop_threads(),
             "numba_mask": numba.get_num_threads()}
    metadata = {"task": args.task, "device": args.device, "threads": args.threads,
                "api_total_seconds": api_seconds, "timings": timings,
                "profile_diagnostic_not_benchmark": args.profile,
                "threads_before": before, "threads_after": after,
                "fnit_import": str(Path(fnit.__file__).resolve()),
                "torch_version": torch.__version__, "numba_version": numba.__version__,
                "report": str(report_file), "reference_labels_used_for_fitting": False,
                "stage_input_checkpoint": bool(args.aseg)}
    metadata["worker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output_dir / "worker.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "complete", "task": args.task,
                      "api_seconds": api_seconds, "report": str(report_file)}))


if __name__ == "__main__":
    main()
