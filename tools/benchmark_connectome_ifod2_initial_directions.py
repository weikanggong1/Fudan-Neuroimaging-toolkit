"""在真实 FOD 固定种子上比较 MRtrix 与 PyTorch 的 iFOD2 初始方向。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.stats import ks_2samp

from fnit.connectome.fod import tracking_sh_precomputed
from fnit.connectome.tracking import _initial_directions, _sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _range(values):
    return [float(np.min(values)), float(np.max(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fod", type=Path, required=True, help="真实归一化 WM FOD NIfTI，形状 [X,Y,Z,45]")
    parser.add_argument("--seeds", type=Path, required=True, help="固定种子的 RAS 世界毫米坐标，每行 x y z")
    parser.add_argument("--official", type=Path, action="append", required=True, help="官方 oracle 的五列输出；可重复五次")
    parser.add_argument("--output", type=Path, required=True, help="报告 JSON；同名 NPZ 保存 PyTorch 逐种子结果")
    parser.add_argument("--device", default="cpu", help="PyTorch 设备，如 cpu 或 cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    image = nib.load(str(args.fod))
    if image.ndim != 4 or image.shape[-1] != 45:
        raise ValueError("FOD must have 45 lmax=8 SH coefficients")
    seed_array = np.loadtxt(args.seeds, dtype=np.float32)
    if seed_array.ndim != 2 or seed_array.shape[1] != 3:
        raise ValueError("seeds must have shape [N,3]")
    official = [np.loadtxt(path) for path in args.official]
    if any(array.shape != (len(seed_array), 5) for array in official):
        raise ValueError("each official output must have shape [N,5]")
    fod = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32).copy(), device=device)
    seeds = torch.as_tensor(seed_array, device=device)
    inverse = torch.linalg.inv(torch.as_tensor(image.affine, dtype=torch.float64, device=device))
    coefficients = _sample(fod, seeds, inverse)
    official_amplitude_error = []
    for array in official:
        valid = array[:, 0] == 1
        direction = torch.as_tensor(array[valid, 1:4].astype(np.float32), device=device)
        predicted = (coefficients[valid] * tracking_sh_precomputed(direction, 8)).sum(-1)
        official_amplitude_error.append(float(np.max(np.abs(predicted.cpu().numpy() - array[valid, 4]))))
    candidate_directions, candidate_valid, candidate_attempts = [], [], []
    candidate_amplitudes, durations = [], []
    for seed in range(len(official)):
        generator = torch.Generator(device=device).manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        direction, valid, attempts = _initial_directions(
            coefficients, generator, lmax=8, cutoff=0.1,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        amplitude = (coefficients * tracking_sh_precomputed(direction, 8)).sum(-1)
        candidate_directions.append(direction.cpu().numpy())
        candidate_valid.append(valid.cpu().numpy())
        candidate_attempts.append(attempts.cpu().numpy())
        candidate_amplitudes.append(amplitude.cpu().numpy())
    official_valid = [array[:, 0] == 1 for array in official]
    own_ks, cross_ks, own_z_ks, cross_z_ks = [], [], [], []
    for left in range(len(official)):
        for right in range(left + 1, len(official)):
            own_ks.append(ks_2samp(official[left][official_valid[left], 4],
                                   official[right][official_valid[right], 4]).statistic)
            own_z_ks.append(ks_2samp(official[left][official_valid[left], 3],
                                     official[right][official_valid[right], 3]).statistic)
        for right in range(len(official)):
            cross_ks.append(ks_2samp(official[left][official_valid[left], 4],
                                     candidate_amplitudes[right][candidate_valid[right]]).statistic)
            cross_z_ks.append(ks_2samp(official[left][official_valid[left], 3],
                                       candidate_directions[right][candidate_valid[right], 2]).statistic)
    report = {
        "dataset": "OpenNeuro ds004666 real normalised WM FOD and 10000 fixed GMWMI seeds",
        "mrtrix_commit": "eeab681d3e0cb004cf1d1d31579d3892197ef5b6",
        "device": str(device), "torch_seeds": list(range(len(official))),
        "input_sha256": {"fod": _sha256(args.fod), "seeds": _sha256(args.seeds),
                         "official": [_sha256(path) for path in args.official]},
        "official_valid_counts": [int(mask.sum()) for mask in official_valid],
        "fnit_valid_counts": [int(mask.sum()) for mask in candidate_valid],
        "fnit_attempts_mean": [float(value.mean()) for value in candidate_attempts],
        "fnit_attempts_max": [int(value.max()) for value in candidate_attempts],
        "fnit_invalid_accepted_amplitudes": [int(np.sum(value[mask] <= 0.1))
                                             for value, mask in zip(candidate_amplitudes, candidate_valid)],
        "official_direction_fod_max_abs_error": max(official_amplitude_error),
        "official_pair_amplitude_ks_range": _range(own_ks),
        "cross_amplitude_ks_range": _range(cross_ks),
        "official_pair_direction_z_ks_range": _range(own_z_ks),
        "cross_direction_z_ks_range": _range(cross_z_ks),
        "fnit_core_seconds": durations,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), directions=np.stack(candidate_directions),
                        valid=np.stack(candidate_valid), attempts=np.stack(candidate_attempts),
                        amplitudes=np.stack(candidate_amplitudes))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
