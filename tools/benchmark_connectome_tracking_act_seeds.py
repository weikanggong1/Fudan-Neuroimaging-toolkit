"""用真实 5TT 对照 FNIT GPU GMWMI 种子与 MRtrix Seedtest 种子分布。

官方参考命令::

     tckgen -algorithm Seedtest -seed_gmwmi gmwmi.mif -act 5tt.mif \
      -seeds N -select 0 -nthreads 0 -output_seeds seeds.txt \
      wm_fod_norm.mif seedtest.tck

FNIT 对照命令::

     python tools/benchmark_connectome_tracking_act_seeds.py \
      --five-tissue five_tissue.nii.gz --gmwmi gmwmi.nii.gz \
      --reference-seeds seeds.txt --reference-repeat-seeds seeds_repeat.txt \
      --reference-attempts N --candidate-points fnit_seeds.npy \
      --output seed_benchmark.json --seed 0 --device cuda:0

两种随机数序列不同，指标比较空间分布，不比较逐行坐标。
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import torch

from fnit.connectome.tracking import _five_tissue_mrtrix, sample_gmwmi_seeds


def _sha(path: Path) -> str:
    """返回单个输入或输出文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """用两次官方播种与一次 FNIT 播种生成分布指标 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("five-tissue", "gmwmi", "reference-seeds", "reference-repeat-seeds", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference-attempts", type=int, default=None)
    parser.add_argument("--candidate-points", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--act-affine-mode", choices=("header-spacing", "exact"),
                        default="header-spacing")
    args = parser.parse_args()
    five_image = nib.load(args.five_tissue)
    gmwmi_image = nib.load(args.gmwmi)
    if five_image.shape != (*gmwmi_image.shape, 5) or not np.allclose(
        five_image.affine, gmwmi_image.affine
    ):
        raise ValueError("5TT and GMWMI geometry must match")
    reference = np.loadtxt(args.reference_seeds, delimiter=",", comments="#",
                           usecols=(2, 3, 4)).astype(np.float32)
    reference_repeat = np.loadtxt(args.reference_repeat_seeds, delimiter=",", comments="#",
                                  usecols=(2, 3, 4)).astype(np.float32)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    five = torch.as_tensor(np.asarray(five_image.dataobj, dtype=np.float32).copy(), device=device)
    gmwmi = torch.as_tensor(np.asarray(gmwmi_image.dataobj, dtype=np.float32).copy(), device=device)
    affine = torch.as_tensor(five_image.affine, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = perf_counter()
    candidate_tensor = sample_gmwmi_seeds(gmwmi, five, affine, len(reference), generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    candidate = candidate_tensor.cpu().numpy()
    if args.candidate_points is not None:
        args.candidate_points.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.candidate_points, candidate)
    effective_affine = affine.clone()
    spacing = torch.as_tensor(five_image.header.get_zooms()[:3], dtype=torch.float64, device=device)
    if args.act_affine_mode == "header-spacing":
        effective_affine[:3, :3] *= spacing / torch.linalg.vector_norm(
            effective_affine[:3, :3], dim=0)
    inverse = torch.linalg.inv(effective_affine)
    def fractions(points):
        """计算世界毫米种子处的 GM-WM 差值分位数。"""
        values = _five_tissue_mrtrix(five, torch.as_tensor(points, device=device), inverse)
        difference = values[:, 0] + values[:, 1] - values[:, 2]
        return torch.quantile(difference.abs(),
                              torch.tensor([0., .5, .9, .99, 1.], device=device)).cpu().tolist()
    candidate_to_ref = cKDTree(reference).query(candidate, workers=-1)[0]
    ref_to_candidate = cKDTree(candidate).query(reference, workers=-1)[0]
    bounds = np.vstack([reference, reference_repeat, candidate])
    bins = [np.arange(bounds[:, axis].min() - 8, bounds[:, axis].max() + 16, 8)
            for axis in range(3)]
    reference_hist = np.histogramdd(reference, bins=bins)[0].ravel()
    candidate_hist = np.histogramdd(candidate, bins=bins)[0].ravel()
    repeat_hist = np.histogramdd(reference_repeat, bins=bins)[0].ravel()
    support = (reference_hist + candidate_hist) > 0
    repeat_support = (reference_hist + repeat_hist) > 0
    report = {
        "input_sha256": {"five_tissue": _sha(args.five_tissue),
                         "gmwmi": _sha(args.gmwmi),
                         "reference_seeds": _sha(args.reference_seeds),
                         "reference_repeat_seeds": _sha(args.reference_repeat_seeds)},
        "software": "independent MRtrix3 Seedtest vs FNIT PyTorch GPU float32/TF32 (float64 geometry)",
        "act_affine_mode": args.act_affine_mode,
        "reference_command": "tckgen -algorithm Seedtest -seed_gmwmi GMWMI -act 5TT -seeds N -select 0 -output_seeds seeds.txt FOD seedtest.tck",
        "reference_attempts": args.reference_attempts,
        "candidate_points_sha256": (_sha(args.candidate_points)
                                    if args.candidate_points is not None else None),
        "candidate_function": "sample_gmwmi_seeds(gmwmi, five_tissue, five_tissue_affine, n_seeds, generator)",
        "seed_count": len(reference),
        "reference_repeat_seed_count": len(reference_repeat),
        "reference_abs_gm_minus_wm_quantiles_0_50_90_99_100": fractions(reference),
        "candidate_abs_gm_minus_wm_quantiles_0_50_90_99_100": fractions(candidate),
        "reference_centroid_ras_mm": reference.mean(0).tolist(),
        "candidate_centroid_ras_mm": candidate.mean(0).tolist(),
        "eight_mm_bin_pearson_on_union": float(np.corrcoef(reference_hist[support],
                                                           candidate_hist[support])[0, 1]),
        "reference_to_reference_repeat_eight_mm_bin_pearson_on_union": float(
            np.corrcoef(reference_hist[repeat_support], repeat_hist[repeat_support])[0, 1]),
        "reference_to_reference_repeat_nn_mm_quantiles_50_90_99": np.quantile(
            cKDTree(reference).query(reference_repeat, workers=-1)[0], [.5, .9, .99]).tolist(),
        "candidate_to_reference_nn_mm_quantiles_50_90_99": np.quantile(candidate_to_ref,
                                                                       [.5, .9, .99]).tolist(),
        "reference_to_candidate_nn_mm_quantiles_50_90_99": np.quantile(ref_to_candidate,
                                                                       [.5, .9, .99]).tolist(),
        "candidate_wall_seconds": elapsed,
        "candidate_cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
                                              if device.type == "cuda" else 0.,
        "comparison_limit": "different random number generators; no row-wise paired seed identity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
