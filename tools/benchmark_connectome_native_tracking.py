"""Compare the FNIT native adapter with an explicit independent tckgen build.

Real FOD/5TT/GMWMI files are read unchanged. Adapter timings include verification,
TCK reading and the packed H2D transfer; command timings exclude those operations.
The JSON contains private paths and requires a public whitelist before sharing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import probabilistic_tractography


def digest(paths):
    points, offsets = hashlib.sha256(), hashlib.sha256()
    offsets.update(np.asarray([0], dtype="<i8").tobytes())
    total = count = 0
    for path in paths:
        array = path.detach().cpu().numpy() if isinstance(path, torch.Tensor) else path
        array = np.asarray(array, dtype="<f4")
        total += len(array)
        count += 1
        points.update(array.tobytes())
        offsets.update(np.asarray([total], dtype="<i8").tobytes())
    return dict(streamlines=count, points=total, points_sha256=points.hexdigest(),
                offsets_sha256=offsets.hexdigest())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def execute(args):
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory")
    if min(args.n_seeds, args.threads, args.repeats, args.strict_seeds) < 1:
        raise ValueError("seed budgets, threads and repeat count must be positive")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    report = dict(parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  benchmark_sha256=sha256(__file__), status="running", rows=[])
    report["inputs"] = {key: dict(sha256=sha256(getattr(args, key)),
                                   bytes=getattr(args, key).stat().st_size)
                        for key in ("fod", "five_tissue", "gmwmi", "official_tckgen")}
    if device.type == "cuda":
        torch.cuda.init()
    modes = [("strict", args.strict_seeds, 1, 0)] + [
        ("threaded", args.n_seeds, args.threads, seed) for seed in range(args.repeats)]
    try:
        for index, (mode, seeds, threads, seed) in enumerate(modes):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            candidate = probabilistic_tractography(
                args.fod, five_tissue=args.five_tissue, gmwmi=args.gmwmi,
                n_seeds=seeds, seed=seed, tracking_threads=threads, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            adapter_seconds = time.perf_counter() - started
            target = args.output_dir / f"reference_{index}.tck"
            command = [str(args.official_tckgen), str(args.fod), str(target),
                       "-algorithm", "iFOD2", "-seed_gmwmi", str(args.gmwmi),
                       "-act", str(args.five_tissue), "-seeds", str(seeds), "-select", "0",
                       "-maxlength", "250", "-angle", "45", "-cutoff", ".1",
                       "-samples", "3", "-power", ".5", "-nthreads", str(threads),
                       "-config", "RealignTransform", "false", "-config", "NIfTIUseSform", "true",
                       "-config", "NIfTIAutoLoadJSON", "false", "-config", "TckgenEarlyExit", "false"]
            started = time.perf_counter()
            with (args.output_dir / f"reference_{index}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "MRTRIX_RNG_SEED": str(seed),
                                    "MRTRIX_CONFIGFILE": "/dev/null"})
            reference_seconds = time.perf_counter() - started
            actual = digest(candidate.paths)
            reference = digest(nib.streamlines.load(target, lazy_load=True).streamlines)
            row = dict(mode=mode, seed=seed, seeds=seeds, threads=threads,
                       adapter_seconds=adapter_seconds,
                       native_command_seconds=candidate.native_provenance["command_seconds"],
                       official_command_seconds=reference_seconds,
                       candidate_digest=actual, reference_digest=reference,
                       ordered_points_and_offsets_equal=actual == reference,
                       runtime=candidate.native_provenance["runtime"])
            if device.type == "cuda":
                row["cuda_allocator"] = dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                                             reserved_bytes=torch.cuda.max_memory_reserved(device))
            report["rows"].append(row)
            if mode == "strict" and actual != reference:
                raise RuntimeError("single-thread ordered float32 points and offsets differ")
            del candidate
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        (args.output_dir / "report.private.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fod", "five-tissue", "gmwmi", "official-tckgen", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=100000)
    parser.add_argument("--strict-seeds", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    execute(parser.parse_args())
