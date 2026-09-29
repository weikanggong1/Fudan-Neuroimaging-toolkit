"""真实 FOD 固定候选的 iFOD2 校准最大值和拒绝概率对照。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from fnit.connectome.fod import tracking_sh_precomputed
from fnit.connectome.tracking import (_ifod2_arc_probability, _ifod2_calibration,
                                      _rotate_ifod2_directions, _sample)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fod", type=Path, required=True, help="真实 WM FOD NIfTI")
    parser.add_argument("--cases", type=Path, required=True, help="80×9 固定世界坐标和方向")
    parser.add_argument("--official", type=Path, required=True, help="官方参考 80×4 数值")
    parser.add_argument("--official-two-arc", type=Path, help="官方连续两弧 80×9 数值")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON 和逐弧 CSV")
    parser.add_argument("--device", default="cuda:0", help="计算设备")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    image = nib.load(str(args.fod))
    fod = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32).copy(), device=device)
    inverse = torch.linalg.inv(torch.as_tensor(image.affine, dtype=torch.float64, device=device))
    cases = np.loadtxt(args.cases, dtype=np.float32)
    reference = np.loadtxt(args.official, dtype=np.float64)
    if cases.shape != (80, 9) or reference.shape != (80, 4):
        raise ValueError("expected fixed 80×9 inputs and 80×4 official output")
    rows = torch.as_tensor(cases, device=device)
    position = rows[:, :3]
    prior = F.normalize(rows[:, 3:6], dim=-1)
    proposal = F.normalize(rows[:, 6:9], dim=-1)[:, None]
    start = (_sample(fod, position, inverse) * tracking_sh_precomputed(prior, 8)).sum(-1)
    half_log = .5 * start.clamp_min(1e-20).log()
    voxel = float(np.prod(image.header.get_zooms()[:3]) ** (1 / 3))
    local, ratio = _ifod2_calibration(8, voxel / 2, voxel, 45., .5, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    arc_started = time.perf_counter()
    rotated = _rotate_ifod2_directions(prior, local[None].expand(len(rows), -1, -1))
    no_act = torch.zeros((1, 1, 1, 5), device=device)
    calibration = _ifod2_arc_probability(
        position, prior, rotated, half_log, fod, no_act, inverse,
        torch.eye(4, device=device, dtype=torch.float64),
        lmax=8, step_mm=voxel / 2, cutoff=.1, power=.5,
    )[0]
    first_probability, _, first_endpoint, _, first_end_amplitude = _ifod2_arc_probability(
        position, prior, proposal, half_log, fod, no_act, inverse,
        torch.eye(4, device=device, dtype=torch.float64),
        lmax=8, step_mm=voxel / 2, cutoff=.1, power=.5,
    )
    candidate = first_probability[:, 0]
    result = torch.stack((start, calibration.amax(-1) * ratio, candidate), -1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    arc_seconds = time.perf_counter() - arc_started
    elapsed = time.perf_counter() - started
    error = np.abs(result - reference[:, 1:])
    report = {
        "dataset": "OpenNeuro ds004666 real normalised WM FOD, 80 frozen arcs",
        "mrtrix_commit": "eeab681d3e0cb004cf1d1d31579d3892197ef5b6",
        "sha256": {"fod": _hash(args.fod), "cases": _hash(args.cases),
                   "official": _hash(args.official)},
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "n_cases": len(rows), "official_valid": int(reference[:, 0].sum()),
        "absolute_error_max": dict(zip(("start", "calibrated_max", "proposal"),
                                       error.max(axis=0).tolist())),
        "absolute_error_p95": dict(zip(("start", "calibrated_max", "proposal"),
                                       np.quantile(error, .95, axis=0).tolist())),
        "positive_decisions_equal": bool(np.array_equal(result[:, 2] > 0,
                                                          reference[:, 3] > 0)),
        "calibrated_max_and_proposal_core_seconds": arc_seconds,
        "wall_seconds_including_load": elapsed,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }
    if args.official_two_arc is not None:
        two_arc = np.loadtxt(args.official_two_arc, dtype=np.float64)
        if two_arc.shape != (80, 9):
            raise ValueError("expected official two-arc output [80,9]")
        second_probability, _, second_endpoint, _, _ = _ifod2_arc_probability(
            first_endpoint[:, 0], proposal[:, 0], proposal,
            .5 * first_end_amplitude[:, 0].clamp_min(1e-20).log(),
            fod, no_act, inverse, torch.eye(4, device=device, dtype=torch.float64),
            lmax=8, step_mm=voxel / 2, cutoff=.1, power=.5,
        )
        positive = two_arc[:, 0] == 1
        comparison = torch.cat((first_probability, second_probability,
                                first_endpoint[:, 0], second_endpoint[:, 0]), -1).cpu().numpy()
        distance = np.linalg.norm(comparison[positive, 2:5] - two_arc[positive, 3:6], axis=-1)
        distance_second = np.linalg.norm(comparison[positive, 5:8] - two_arc[positive, 6:9], axis=-1)
        report["cross_step"] = {
            "official_positive_first": int(positive.sum()),
            "first_positive_decisions_equal": bool(np.array_equal(candidate.cpu().numpy() > 0, positive)),
            "first_probability_max_abs_error": float(np.abs(comparison[positive, 0] - two_arc[positive, 1]).max()),
            "second_probability_max_abs_error": float(np.abs(comparison[positive, 1] - two_arc[positive, 2]).max()),
            "first_endpoint_max_error_mm": float(distance.max()),
            "second_endpoint_max_error_mm": float(distance_second.max()),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["wall_seconds_including_load"] = time.perf_counter() - started
        report["sha256"]["official_two_arc"] = _hash(args.official_two_arc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(args.output.with_suffix(".csv"), np.column_stack((reference[:, 1:], result)),
               delimiter=",", header="official_start,official_max,official_proposal,fnit_start,fnit_max,fnit_proposal",
               comments="")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
