"""真实 5TT 与固定初始方向下，对照 MRtrix ACT 单向播种规则。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import _act_seed_direction, _five_tissue_mrtrix


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-tissue", type=Path, required=True, help="真实五组织 NIfTI [X,Y,Z,5]")
    parser.add_argument("--seeds", type=Path, required=True, help="三列 RAS 世界毫米种子")
    parser.add_argument("--initial", type=Path, required=True, help="官方初始方向五列文本")
    parser.add_argument("--official", type=Path, required=True, help="官方 ACT 种子检查十一列文本")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON；同名 CSV 存逐种子结果")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    image = nib.load(str(args.five_tissue))
    if image.ndim != 4 or image.shape[-1] != 5:
        raise ValueError("five-tissue image must have five channels")
    seeds = np.loadtxt(args.seeds, dtype=np.float32)
    initial = np.loadtxt(args.initial)
    official = np.loadtxt(args.official)
    if (seeds.ndim != 2 or seeds.shape[1] != 3 or
            initial.shape != (len(seeds), 5) or official.shape != (len(seeds), 11)):
        raise ValueError("seed/oracle row count or column count differs")
    volume = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32).copy(), device=device)
    positions = torch.as_tensor(seeds, device=device)
    directions = np.nan_to_num(initial[:, 1:4].astype(np.float32), nan=0.)
    directions[initial[:, 0] == 0] = [1., 0., 0.]
    directions = torch.as_tensor(directions, device=device)
    affine = torch.as_tensor(image.affine, dtype=torch.float64, device=device)
    columns = affine[:3, :3]
    affine[:3, :3] = columns / torch.linalg.vector_norm(columns, dim=0) * torch.as_tensor(
        image.header.get_zooms()[:3], dtype=torch.float64, device=device,
    )
    inverse = torch.linalg.inv(affine)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    valid, one_way, oriented = _act_seed_direction(volume, positions, directions, inverse)
    tissue = _five_tissue_mrtrix(volume, positions, inverse)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    initial_valid = initial[:, 0] == 1
    valid = valid.cpu().numpy() & initial_valid
    one_way = one_way.cpu().numpy() & initial_valid
    oriented = oriented.cpu().numpy()
    tissue = tissue.cpu().numpy()
    reference_valid = official[:, 0] == 1
    reference_one_way = official[:, 1] == 1
    compared = reference_valid & valid
    absolute = np.abs(oriented[compared] - official[compared, 2:5])
    tissue_absolute = np.abs(tissue[compared] - official[compared, 5:10])
    direction_error = np.linalg.norm(oriented[compared] - official[compared, 2:5], axis=1)
    differing_rows = np.flatnonzero(compared)[direction_error > 1e-4]
    report = {
        "dataset": "OpenNeuro ds004666 real 5TT, 10000 fixed GMWMI points and official FOD initial directions",
        "mrtrix_commit": "eeab681d3e0cb004cf1d1d31579d3892197ef5b6",
        "input_sha256": {name: _sha256(path) for name, path in
                         (("five_tissue", args.five_tissue), ("seeds", args.seeds),
                          ("initial", args.initial), ("official", args.official))},
        "device": str(device), "initial_valid": int(initial_valid.sum()),
        "official_seed_valid": int(reference_valid.sum()),
        "fnit_seed_valid": int(valid.sum()),
        "valid_xor": int(np.logical_xor(reference_valid, valid).sum()),
        "official_one_way": int(reference_one_way.sum()),
        "fnit_one_way": int(one_way.sum()),
        "one_way_xor": int(np.logical_xor(reference_one_way, one_way).sum()),
        "oriented_direction_max_abs_error": float(absolute.max()),
        "oriented_direction_mean_abs_error": float(absolute.mean()),
        "oriented_direction_disagreement_count": int(len(differing_rows)),
        "oriented_direction_disagreement_rows": differing_rows.tolist(),
        "official_gradient_at_disagreements": official[differing_rows, 10].tolist(),
        "seed_tissue_max_abs_error": float(tissue_absolute.max()),
        "seed_tissue_mean_abs_error": float(tissue_absolute.mean()),
        "core_seconds": seconds,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savetxt(args.output.with_suffix(".csv"),
               np.column_stack((valid.astype(int), one_way.astype(int), oriented)),
               fmt=["%d", "%d", "%.9g", "%.9g", "%.9g"], delimiter=",")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
