"""同一真实 FOD 的 iFOD2 单弧坐标和概率对照。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from fnit.connectome.fod import real_sh, tracking_sh_precomputed
from fnit.connectome.tracking import _sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _calculate(cases, fod, inverse, step, cutoff, basis):
    position = cases[:, :3]
    start = F.normalize(cases[:, 3:6], dim=-1)
    end = F.normalize(cases[:, 6:9], dim=-1)
    cosine = (start * end).sum(-1).clamp(-1, 1)
    angle = cosine.acos()
    curvature = F.normalize(end - cosine[:, None] * start, dim=-1)
    radius = step / angle.clamp_min(1e-6)
    half = angle / 2
    midpoint = position + radius[:, None] * (
        half.sin()[:, None] * start + (1 - half.cos())[:, None] * curvature
    )
    endpoint = position + radius[:, None] * (
        angle.sin()[:, None] * start + (1 - cosine)[:, None] * curvature
    )
    mid_direction = half.cos()[:, None] * start + half.sin()[:, None] * curvature
    straight = angle < 1e-4
    midpoint = torch.where(straight[:, None], position + .5 * step * start, midpoint)
    endpoint = torch.where(straight[:, None], position + step * start, endpoint)
    mid_direction = torch.where(straight[:, None], start, mid_direction)
    amplitudes = torch.stack(tuple(
        (_sample(fod, point, inverse) * basis(direction, 8)).sum(-1)
        for point, direction in ((position, start), (midpoint, mid_direction), (endpoint, end))
    ), dim=-1)
    probability = torch.exp(.5 * (
        .5 * amplitudes[:, 0].clamp_min(1e-20).log() +
        amplitudes[:, 1].clamp_min(1e-20).log() +
        .5 * amplitudes[:, 2].clamp_min(1e-20).log()
    )) * ((amplitudes[:, 1] >= cutoff) & (amplitudes[:, 2] >= cutoff))
    return amplitudes[:, 0], probability, torch.stack((midpoint, endpoint), dim=1)


def _quantiles(values):
    return {key: float(np.quantile(values, q)) for key, q in
            (("median", .5), ("p95", .95), ("max", 1.0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fod", type=Path, required=True, help="真实 WM FOD NIfTI，float32 [X,Y,Z,45]")
    parser.add_argument("--cases", type=Path, required=True, help="每行位置、起始方向、末方向，共 9 列")
    parser.add_argument("--reference", type=Path, required=True, help="官方单弧程序的 18 列输出")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON；同名 .csv 保存逐弧概率")
    parser.add_argument("--device", default="cpu", help="PyTorch 设备，如 cpu 或 cuda:0")
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
    reference = np.loadtxt(args.reference, dtype=np.float64)
    if cases.ndim != 2 or cases.shape[1] != 9 or reference.shape != (len(cases), 18):
        raise ValueError("expected cases [N,9] and official output [N,18]")
    if fod.ndim != 4 or fod.shape[-1] != 45 or not np.all(reference[:, 0] == 1):
        raise ValueError("expected real lmax=8 FOD and valid official cases")
    tensor_cases = torch.as_tensor(cases, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    step, cutoff = float(reference[0, 1]), float(reference[0, 2])
    table_start = time.perf_counter()
    tracking_sh_precomputed(tensor_cases[:1, 3:6], 8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    table_seconds = time.perf_counter() - table_start
    for method in (real_sh, tracking_sh_precomputed):
        _calculate(tensor_cases[:1], fod, inverse, step, cutoff, method)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    results = {}
    probabilities = []
    official_probability = reference[:, 5]
    official_positions = np.stack((reference[:, 6:9], reference[:, 12:15]), axis=1)
    for name, method in (("direct_sh", real_sh), ("mrtrix_lookup", tracking_sh_precomputed)):
        start = time.perf_counter()
        amplitude, probability, positions = _calculate(tensor_cases, fod, inverse, step, cutoff, method)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        core_seconds = time.perf_counter() - start
        amplitude, probability, positions = (value.detach().cpu().numpy() for value in
                                            (amplitude, probability, positions))
        positive = official_probability > 0
        results[name] = {
            "core_seconds": core_seconds,
            "accepted_decisions_equal": bool(np.array_equal(probability > 0, positive)),
            "start_amplitude_abs_error": _quantiles(np.abs(amplitude - reference[:, 3])),
            "probability_abs_error": _quantiles(np.abs(probability - official_probability)),
            "positive_probability_relative_error": _quantiles(
                np.abs(probability[positive] - official_probability[positive]) / official_probability[positive]
            ),
            "arc_position_error_mm": _quantiles(np.linalg.norm(positions - official_positions, axis=-1)),
        }
        probabilities.append(probability)
    report = {
        "dataset": "OpenNeuro ds004666 corrected DWI and paired FreeSurfer-derived FOD",
        "mrtrix_commit": "eeab681d3e0cb004cf1d1d31579d3892197ef5b6",
        "inputs_sha256": {name: _sha256(path) for name, path in
                          (("fod", args.fod), ("cases", args.cases), ("reference", args.reference))},
        "device": str(device), "n_cases": len(cases), "official_positive": int((official_probability > 0).sum()),
        "input_load_seconds": load_seconds,
        "lookup_table_init_seconds": table_seconds,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                      if device.type == "cuda" else None),
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savetxt(args.output.with_suffix(".csv"),
               np.column_stack((official_probability, *probabilities)), delimiter=",",
               header="official,direct_sh,mrtrix_lookup", comments="")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
