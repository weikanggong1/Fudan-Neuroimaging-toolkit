"""真实 5TT 上 300 条固定路径的 ACT 逐点状态对照。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import (_act_seed_direction, _act_structural_step,
                                      _five_tissue_mrtrix)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-tissue", type=Path, required=True, help="真实 5TT NIfTI")
    parser.add_argument("--cases", type=Path, required=True, help="S/P/R 固定世界坐标路径")
    parser.add_argument("--official", type=Path, required=True, help="官方逐点状态文本")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cases = [line.split() for line in args.cases.read_text().splitlines() if line.strip()]
    official = [line.split() for line in args.official.read_text().splitlines() if line.strip()]
    if len(cases) != 300 * 42 or len(official) != len(cases):
        raise ValueError("expected 300 groups with S, 20 P, R, 20 P")
    kinds = np.asarray([row[0] for row in cases]).reshape(300, 42)
    if not (np.all(kinds[:, 0] == "S") and np.all(kinds[:, 1:21] == "P") and
            np.all(kinds[:, 21] == "R") and np.all(kinds[:, 22:] == "P")):
        raise ValueError("invalid fixed ACT state case structure")
    if not all(left[0] == right[0] for left, right in zip(cases, official)):
        raise ValueError("official row order differs")
    positions = np.asarray([([0., 0., 0.] if row[0] == "R" else
                             [float(value) for value in row[1:4]]) for row in cases],
                           dtype=np.float32).reshape(300, 42, 3)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    image = nib.load(str(args.five_tissue))
    volume = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32).copy(), device=device)
    affine = torch.as_tensor(image.affine, dtype=torch.float64, device=device)
    affine[:3, :3] *= torch.as_tensor(image.header.get_zooms()[:3], dtype=torch.float64,
                                    device=device) / torch.linalg.vector_norm(affine[:3, :3], dim=0)
    inverse = torch.linalg.inv(affine)
    point_tensor = torch.as_tensor(positions, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    samples = _five_tissue_mrtrix(volume, point_tensor.reshape(-1, 3), inverse).reshape(300, 42, 5)
    initial_direction = torch.zeros((300, 3), device=device)
    initial_direction[:, 0] = 1
    seed_valid, _, _ = _act_seed_direction(volume, point_tensor[:, 0], initial_direction, inverse)
    seed_tissue = samples[:, 0]
    cgm, sgm, wm, csf, path = seed_tissue.unbind(-1)
    seed_in_sgm = (sgm > cgm) & (sgm >= wm) & (sgm > csf) & (sgm > path)
    depth = torch.zeros(300, dtype=torch.int32, device=device)
    to_wm = torch.zeros(300, dtype=torch.bool, device=device)
    result = np.zeros((300, 42, 4), dtype=np.int32)
    result[:, 0, 0] = seed_valid.int().cpu().numpy()
    result[:, 0, 2] = seed_in_sgm.int().cpu().numpy()
    for column in range(1, 42):
        if column == 21:
            depth.zero_()
        else:
            term, depth, to_wm, _ = _act_structural_step(
                samples[:, column], depth, seed_in_sgm, to_wm)
            result[:, column, 0] = term.cpu().numpy()
        result[:, column, 1] = depth.cpu().numpy()
        result[:, column, 2] = seed_in_sgm.int().cpu().numpy()
        result[:, column, 3] = to_wm.int().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    expected = np.asarray([[int(value) for value in row[1:5]] for row in official],
                          dtype=np.int32).reshape(300, 42, 4)
    different = result != expected
    reference_tissue = np.asarray([[float(value) for value in row[5:10]]
                                   if row[0] != "R" else [0.] * 5
                                   for row in official], dtype=np.float32).reshape(300, 42, 5)
    tissue_error = np.abs(samples.cpu().numpy() - reference_tissue)
    tissue_error[:, 21] = 0
    report = {
        "dataset": "OpenNeuro ds004666 real 5TT, 300 real seed positions and official directions",
        "mrtrix_commit": "eeab681d3e0cb004cf1d1d31579d3892197ef5b6",
        "input_sha256": {"five_tissue": _sha256(args.five_tissue),
                         "cases": _sha256(args.cases), "official": _sha256(args.official)},
        "point_count": len(cases), "seed_count": 300,
        "term_xor": int(different[:, :, 0].sum()),
        "depth_xor": int(different[:, :, 1].sum()),
        "seed_in_sgm_xor": int(different[:, :, 2].sum()),
        "seed_to_wm_xor": int(different[:, :, 3].sum()),
        "tissue_max_abs_error": float(tissue_error.max()),
        "core_seconds": elapsed,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
