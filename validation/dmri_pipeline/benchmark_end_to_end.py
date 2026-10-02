"""Run the actual default FNIT raw-to-MNI pipeline with stage logging.

Use an external process timer for startup-inclusive wall time. The logger wraps
only complete public stage calls and does not replace any numerical operation.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fa-template", type=Path, required=True)
    parser.add_argument("--fa-skeleton", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eddy-gp-seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--synthstrip-weights", type=Path,
                        help="explicit verified SynthStrip weights for the b0 brain mask")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this paired GPU benchmark requires a CUDA device")
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_bytes / torch.cuda.get_device_properties(device).total_memory, device)
    import fnit
    import fnit.dmri_pipeline.pipeline as pipeline_module
    import fnit.topup.ukb as topup_preparation_module

    events = []
    stage_stack = []

    def instrument(label, original):
        @functools.wraps(original)
        def measured(*positional, **keywords):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            parent = stage_stack[-1] if stage_stack else None
            stage_stack.append(label)
            print(json.dumps({"event": "stage_start", "stage": label}), flush=True)
            try:
                return original(*positional, **keywords)
            finally:
                torch.cuda.synchronize(device)
                row = {"event": "stage_end", "stage": label,
                       "parent_stage": parent,
                       "seconds": time.perf_counter() - start,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                events.append(row)
                stage_stack.pop()
                print(json.dumps(row), flush=True)
        return measured

    pipeline_module.run_ukb_topup = instrument("topup_and_b0_selection", pipeline_module.run_ukb_topup)
    topup_preparation_module.prepare_ukb_topup = instrument(
        "b0_selection_and_pair_save", topup_preparation_module.prepare_ukb_topup)
    topup_preparation_module.TorchTOPUP.run = instrument(
        "topup_estimation_and_save", topup_preparation_module.TorchTOPUP.run)
    pipeline_module.prepare_ukb_eddy = instrument("eddy_preparation", pipeline_module.prepare_ukb_eddy)
    for label, cls in (("eddy", pipeline_module.TorchEDDY),
                       ("dtifit", pipeline_module.TorchDTIFIT),
                       ("noddi", pipeline_module.TorchAMICONODDI),
                       ("tbss_registration_and_nine_maps", pipeline_module.TorchTBSS)):
        cls.run = instrument(label, cls.run)
    runner = pipeline_module.DMRIPipeline(
        device=device, registration_backend="tbss", noddi_fit_method="amico",
        bvec_source="rotated", eddy_gp_seed=args.eddy_gp_seed,
        synthstrip_weights=args.synthstrip_weights,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = runner.run(args.raw_dir, args.output_dir,
                        fa_template=args.fa_template, fa_skeleton=args.fa_skeleton)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    root = Path(fnit.__file__).parent
    report = {
        "source_commit": args.source_commit, "torch": torch.__version__,
        "scope": "actual FNIT raw AP/PA to nine native/standard/skeleton maps; no stage reuse",
        "api_wall_seconds": elapsed, "threads": args.threads,
        "memory_limit_bytes": args.memory_limit_bytes,
        # Components reset CUDA peak counters. Preserve each completed stage's
        # peak before a later component resets it again.
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in events),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in events),
        "memory_scope": "maximum recorded public-stage CUDA allocator peak; components reset counters",
        "stage_events": events, "qc": result.qc,
        "timing_tree_note": "child stages are contained in parent stages; do not sum both",
        "source_python_sha256": {str(path.relative_to(root)):
                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(root.rglob("*.py"))},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", "api_wall_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
