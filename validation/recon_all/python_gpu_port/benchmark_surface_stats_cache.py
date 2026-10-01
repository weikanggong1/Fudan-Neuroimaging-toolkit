"""冻结真实表面配对测试：旧统计实现与显式缓存；计时包含读写和 CUDA 同步。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import types

import numpy as np
import torch

from fnit.recon_all.anatomical_stats_file import write_anatomical_stats
from fnit.recon_all.anatomical_stats_global import read_brain_volume_stats
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache


CASES = (("aparc", "white"), ("aparc.a2009s", "white"),
         ("aparc.DKTatlas", "white"), ("aparc", "pial"),
         ("BA_exvivo", "white"), ("BA_exvivo.thresh", "white"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_module(source: Path):
    package = types.ModuleType("_fnit_stats_baseline")
    package.__path__ = [str(source / "fnit" / "recon_all")]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".anatomical_stats_file")


def numeric_rows(text: str) -> dict[str, np.ndarray]:
    return {parts[0]: np.asarray(parts[1:], dtype=float)
            for line in text.splitlines() if (parts := line.split())
            and not line.startswith("#") and len(parts) == 10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--official-subject", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.threads < 1:
        raise ValueError("repetitions and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
    old = baseline_module(args.baseline_source)
    volumes = read_brain_volume_stats(args.subject / "stats" / "brainvol.stats")
    inputs = [args.subject / "stats" / "brainvol.stats",
              args.subject / "mri" / "transforms" / "talairach.xfm"]
    for hemi in ("lh", "rh"):
        inputs.extend(args.subject / "surf" / f"{hemi}.{suffix}" for suffix in
                      ("white", "white.preaparc", "pial", "area", "area.pial", "thickness"))
        inputs.append(args.subject / "label" / f"{hemi}.cortex.label")
        inputs.extend(args.subject / "label" / f"{hemi}.{atlas}.annot"
                      for atlas in dict(CASES))
    source_names = ("anatomical_stats_file.py", "anatomical_stats_rows.py",
                    "anatomical_stats_global.py", "surface_roi_gpu.py",
                    "surface_roi_curvature_gpu.py", "surface_stats_cache.py")
    candidate_dir = Path(importlib.import_module("fnit.recon_all.anatomical_stats_file").__file__).parent
    report = {"scope": "frozen-real-surface-stage-including-io;not-end-to-end",
              "acceptance": "Entire generated .stats text must be unchanged versus frozen baseline; numerical deltas are diagnostics, not a relaxed gate.",
              "baseline_commit": args.baseline_commit, "candidate_commit": args.candidate_commit,
              "subject": str(args.subject), "hostname": platform.node(),
              "device": str(device), "threads": args.threads,
              "torch_version": torch.__version__, "python_version": platform.python_version(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "cuda_allocation_cache_disabled": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1",
              "memory_scope": "PyTorch target-device allocator only; parent/child simultaneous NVML occupancy must be measured by the enclosing monitor. Disabled allocator statistics are null, not zero GPU occupancy.",
              "half_precision": False, "script_sha256": sha256(Path(__file__)),
              "inputs_sha256": {str(path): sha256(path) for path in inputs},
              "candidate_source_sha256": {name: sha256(candidate_dir / name) for name in source_names},
              "baseline_source_sha256": {name: sha256(args.baseline_source / "fnit" / "recon_all" / name)
                                          for name in source_names if name != "surface_stats_cache.py"},
              "runs": []}
    for repeat in range(args.repetitions):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        result = {}
        for implementation in order:
            destination = args.output / f"repeat_{repeat}" / implementation
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            counters = {}
            for hemi in ("lh", "rh"):
                with SurfaceStatsCache(device=str(device)) as cache:
                    for atlas, surface in CASES:
                        suffix = "aparc.pial" if surface == "pial" else atlas
                        kwargs = dict(subject=args.subject, hemi=hemi, atlas=atlas,
                                      surface=surface, brainvol_stats=volumes,
                                      output=destination / f"{hemi}.{suffix}.stats",
                                      device=str(device))
                        if implementation == "candidate":
                            write_anatomical_stats(**kwargs, cache=cache)
                        else:
                            old.write_anatomical_stats(**kwargs)
                    counters[hemi] = dict(cache.counters)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            result[implementation] = {"wall_seconds_including_io": time.perf_counter() - tick,
                                      "cache_counters": counters,
                                      "torch_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                                     if device.type == "cuda" and not report["cuda_allocation_cache_disabled"] else None),
                                      "torch_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                                                    if device.type == "cuda" and not report["cuda_allocation_cache_disabled"] else None)}
        comparisons = []
        for hemi in ("lh", "rh"):
            for atlas, surface in CASES:
                suffix = "aparc.pial" if surface == "pial" else atlas
                filename = f"{hemi}.{suffix}.stats"
                before = (args.output / f"repeat_{repeat}" / "baseline" / filename).read_text()
                after = (args.output / f"repeat_{repeat}" / "candidate" / filename).read_text()
                old_rows, new_rows = numeric_rows(before), numeric_rows(after)
                labels_equal = set(old_rows) == set(new_rows)
                delta = (np.max(np.abs(np.stack([new_rows[name] - old_rows[name]
                                                 for name in old_rows])), axis=0).tolist()
                         if labels_equal and old_rows else None)
                item = {"file": filename, "text_equal": before == after,
                        "roi_names_equal": labels_equal, "roi_count": len(new_rows),
                        "max_abs_column_delta": delta}
                if args.official_subject is not None:
                    official_path = args.official_subject / "stats" / filename
                    if official_path.is_file():
                        official_rows = numeric_rows(official_path.read_text())
                        shared = sorted(set(official_rows) & set(new_rows))
                        item["official_diagnostic"] = {
                            "reference_sha256": sha256(official_path),
                            "roi_names_equal": set(official_rows) == set(new_rows),
                            "shared_roi_count": len(shared),
                            "max_abs_column_delta": (np.max(np.abs(np.stack([
                                new_rows[name] - official_rows[name] for name in shared])), axis=0).tolist()
                                                     if shared else None),
                            "equivalence": "not_assessed"}
                comparisons.append(item)
        report["runs"].append({"repeat": repeat, "order": list(order),
                                **result, "comparisons": comparisons,
                                "regression_passed": all(row["text_equal"] for row in comparisons)})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["regression_passed"] = all(row["regression_passed"] for row in report["runs"])
    report["speedup_by_repeat"] = [row["baseline"]["wall_seconds_including_io"] /
                                  row["candidate"]["wall_seconds_including_io"]
                                  for row in report["runs"]]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output / "report.json"),
                      "regression_passed": report["regression_passed"],
                      "speedup_by_repeat": report["speedup_by_repeat"]}, indent=2))
    if not report["regression_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
