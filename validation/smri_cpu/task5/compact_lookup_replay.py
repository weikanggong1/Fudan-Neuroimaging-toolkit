"""Replay private real atlas lookup captures; never benchmark the full pipeline."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from fnit.gems._raster_cpu import lookup_candidates_cpu
from fnit.gems._raster_cpu_compact import lookup_compact_cpu


def array_sha256(value):
    return hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("use a new report path")
    torch.set_num_threads(args.threads)
    geometry_path = args.capture_directory / "geometry.private.npz"
    shared_geometry = dict(np.load(geometry_path)) if geometry_path.exists() else None
    batches, references = [], []
    geometry = None
    capture_files = []
    for path in sorted(args.capture_directory.glob("batch_*.private.npz")):
        capture_files.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        current = shared_geometry if shared_geometry is not None else arrays
        current = tuple(torch.from_numpy(current[name]) for name in ("origins", "inverses", "singular"))
        if geometry is None:
            geometry = current
        elif not all(torch.equal(first, second) for first, second in zip(geometry, current)):
            raise ValueError("captures span different mesh evaluations; do not merge their geometry")
        points, ids, mask, rows = (torch.from_numpy(arrays[name]) for name in ("points", "ids", "mask", "rows"))
        batches.append((points, ids, mask, torch.arange(len(ids))[:, None], rows))
        reference_path = args.capture_directory / path.name.replace("batch_", "reference_")
        if reference_path.exists():
            with np.load(reference_path) as archive:
                references.append((torch.from_numpy(archive["selected"]), torch.from_numpy(archive["covered"])))
    batches = tuple(batches)
    if not batches:
        parser.error("capture directory contains no real batches")

    def baseline():
        selected, covered, points = [], [], []
        for point, ids, mask, _, rows in batches:
            owners, inside = lookup_candidates_cpu(point, ids, mask, *geometry, rows)
            selected.append(owners)
            covered.append(inside)
            points.append(point.reshape(-1, 3)[rows])
        return torch.cat(selected), torch.cat(points), torch.cat(covered)

    started = time.perf_counter()
    first_baseline = baseline()
    baseline_first_seconds = time.perf_counter() - started
    cache = {}
    started = time.perf_counter()
    first_candidate = lookup_compact_cpu(batches, *geometry, cache)
    candidate_first_seconds = time.perf_counter() - started
    if first_candidate is None:
        raise ValueError("real capture is unsupported by candidate")
    for candidate, reference in zip(first_candidate, first_baseline):
        if not torch.equal(candidate, reference):
            raise ValueError("candidate differs from ordered baseline")
    if references:
        if len(references) != len(batches):
            raise ValueError("partial per-batch reference set")
        captured = (torch.cat([row[0] for row in references]), torch.cat([row[1] for row in references]))
        if not torch.equal(first_baseline[0], captured[0]) or not torch.equal(first_baseline[2], captured[1]):
            raise ValueError("baseline differs from captured original Torch result")
    else:
        with np.load(args.capture_directory / "reference.private.npz") as archive:
            captured = tuple(torch.from_numpy(archive[name]) for name in ("selected", "points", "covered"))
        if not all(torch.equal(first_baseline[index], captured[index]) for index in range(3)):
            raise ValueError("baseline differs from complete captured original lookup")
    executions = []
    for index, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
        started = time.perf_counter()
        result = baseline() if arm == "baseline" else lookup_compact_cpu(batches, *geometry, cache)
        seconds = time.perf_counter() - started
        executions.append({"index": index, "arm": arm, "seconds": seconds,
                           "all_selected_points_covered_exact": all(torch.equal(first, second) for first, second in zip(result, first_baseline))})
    report = {"schema": "fnit.gems.cpu.compact_lookup_replay.v1", "diagnostic_only": True,
              "pipeline_completed": False, "threads": torch.get_num_threads(),
              "batch_count": len(batches), "real_point_count": first_baseline[0].numel(),
              "capture_file_sha256": capture_files,
              "cold_baseline_seconds_includes_jit": baseline_first_seconds,
              "cold_candidate_seconds_includes_jit_and_plan": candidate_first_seconds,
              "warm_replay_executions": executions,
              "selected_sha256": array_sha256(first_candidate[0]),
              "points_sha256": array_sha256(first_candidate[1]),
              "covered_sha256": array_sha256(first_candidate[2]),
              "gate": "All selected IDs, fixed points and coverage exactly equal to actual original capture",
              "gate_passed": all(row["all_selected_points_covered_exact"] for row in executions)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
