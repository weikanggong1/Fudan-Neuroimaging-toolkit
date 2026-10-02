"""独立比较真实 ICA 时间序列、子空间与对齐后的 AROMA 标签。

输入是同一时间轴的两套 mixing 与噪声编号文件。先将每列居中并归一化，
再最大化绝对 Pearson r 做一对一 Hungarian 匹配；不会将相同编号视为
相同成分。可选同网格 component 和 thresholded maps 沿用时间配对与符号，
比较空间 r 和非零阈值支持域 Dice。公开报告仅保存匿名统计和 SHA。
可选逐成分表写到私有目录。
不调用 FNIT、FSL、FreeSurfer，也不重新估计 ICA。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mixing(path):
    mixing = np.loadtxt(path, ndmin=2, dtype=np.float64)
    if mixing.shape[0] < 4 or mixing.shape[1] < 1 or not np.isfinite(mixing).all():
        raise ValueError("Mixing must contain at least four finite time points")
    centered = mixing - mixing.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered, axis=0)
    if np.any(norm <= 0):
        raise ValueError("A constant mixing column cannot be matched by Pearson r")
    return centered / norm


def read_noise(path, component_count, index_base):
    text = Path(path).read_text().strip()
    tokens = re.split(r"[,\s]+", text) if text else []
    if any(not re.fullmatch(r"\d+", token) for token in tokens):
        raise ValueError("Noise file must contain comma/whitespace-separated integer IDs")
    numbers = np.asarray([int(token) - index_base for token in tokens], dtype=np.int64)
    if np.any(numbers < 0) or np.any(numbers >= component_count):
        raise ValueError("Noise ID is outside the mixing component range")
    if np.unique(numbers).size != numbers.size:
        raise ValueError("Noise file contains duplicate component IDs")
    labels = np.zeros(component_count, dtype=bool)
    labels[numbers] = True
    return labels


def distribution(values):
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def basis(matrix, rcond):
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > singular_values[0] * rcond))
    return left[:, :rank], rank


def label_comparison(candidate, reference):
    both_noise = int(np.count_nonzero(candidate & reference))
    candidate_only = int(np.count_nonzero(candidate & ~reference))
    reference_only = int(np.count_nonzero(~candidate & reference))
    both_signal = int(np.count_nonzero(~candidate & ~reference))
    union = both_noise + candidate_only + reference_only
    return {
        "matched_pairs": int(len(candidate)),
        "agreement_rate": float(np.mean(candidate == reference)),
        "noise_jaccard": both_noise / union if union else 1.0,
        "both_noise": both_noise,
        "candidate_noise_reference_signal": candidate_only,
        "candidate_signal_reference_noise": reference_only,
        "both_signal": both_signal,
    }


def spatial_alignment(args, rows, columns, signs, candidate_count, reference_count):
    """Apply the temporal pairing/sign to maps on their existing voxel grid."""
    import nibabel as nib

    supplied = {name: getattr(args, name) for name in (
        "candidate_maps", "reference_maps", "candidate_thresholded", "reference_thresholded")}
    if not any(supplied.values()):
        return None
    raw = bool(supplied["candidate_maps"])
    if raw != bool(supplied["reference_maps"]):
        raise ValueError("Spatial comparison requires both raw component maps when either is supplied")
    thresholded = bool(supplied["candidate_thresholded"])
    if thresholded != bool(supplied["reference_thresholded"]):
        raise ValueError("Spatial comparison requires both thresholded maps when either is supplied")
    supplied = {name: path for name, path in supplied.items() if path is not None}
    images = {name: nib.load(path) for name, path in supplied.items()}
    map_suffix = "maps" if raw else "thresholded"
    first = images["candidate_" + map_suffix]
    arrays = {}
    for name, image in images.items():
        count = candidate_count if name.startswith("candidate") else reference_count
        if (image.ndim != 4 or image.shape[-1] != count or image.shape[:3] != first.shape[:3]
                or not np.allclose(image.affine, first.affine, atol=1e-4, rtol=0)):
            raise ValueError("ICA maps must match their component count and share a spatial grid")
        arrays[name] = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(arrays[name]).all():
            raise ValueError("ICA component maps contain nonfinite values")
    mask = np.ones(first.shape[:3], dtype=bool)
    input_hashes = {name: sha256(path) for name, path in supplied.items()}
    if args.comparison_mask is not None:
        image = nib.load(args.comparison_mask)
        data = np.asarray(image.dataobj)
        if (image.shape != first.shape[:3] or not np.allclose(image.affine, first.affine, atol=1e-4, rtol=0)
                or not np.isfinite(data).all() or not np.isin(data, [0, 1]).all()):
            raise ValueError("ICA comparison mask must be finite, binary and on the component grid")
        mask = data > 0
        input_hashes["comparison_mask"] = sha256(args.comparison_mask)
    if not mask.any():
        raise ValueError("ICA comparison mask is empty")
    correlations, errors, dices = [], [], []
    constant_pairs = 0
    for left, right, sign in zip(rows, columns, signs):
        x = arrays["candidate_" + map_suffix][..., left][mask].astype(np.float64) * sign
        y = arrays["reference_" + map_suffix][..., right][mask].astype(np.float64)
        errors.append(float(np.sqrt(np.mean((x - y) ** 2))))
        x -= x.mean()
        y -= y.mean()
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        if denominator:
            correlations.append(float(np.clip(np.dot(x, y) / denominator, -1, 1)))
        else:
            constant_pairs += 1
        if thresholded:
            a = arrays["candidate_thresholded"][..., left][mask] != 0
            b = arrays["reference_thresholded"][..., right][mask] != 0
            denominator = int(a.sum() + b.sum())
            dices.append(float(2 * np.count_nonzero(a & b) / denominator) if denominator else 1.)
    return {
        "method": "Existing spatial maps paired by the temporal Hungarian assignment; candidate maps multiplied by temporal sign; threshold support is nonzero regardless of sign",
        "shape": list(first.shape[:3]), "same_grid": True, "all_values_finite": True,
        "map_kind": "unthresholded components" if raw else "thresholded components",
        "comparison_mask_voxels": int(mask.sum()), "matched_pairs": len(rows),
        "input_sha256": input_hashes,
        "sign_aligned_map_pearson_r": distribution(correlations) if correlations else None,
        "map_rmse": distribution(errors), "constant_map_pairs": constant_pairs,
        "nonzero_threshold_support_dice": distribution(dices) if dices else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-mixing", "reference-mixing", "candidate-noise", "reference-noise", "report-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reference-source-revision", required=True)
    parser.add_argument("--candidate-noise-index-base", choices=(0, 1), type=int, default=1)
    parser.add_argument("--reference-noise-index-base", choices=(0, 1), type=int, default=1)
    parser.add_argument("--rank-rcond", type=float, default=1e-8)
    parser.add_argument("--validated-on", help="Optional YYYY-MM-DD date for the actual control")
    for name in ("candidate-maps", "reference-maps", "candidate-thresholded", "reference-thresholded", "comparison-mask"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--private-pairs-out", type=Path,
                        help="可选：保存成分编号、符号、r 和噪声标签的私有 TSV")
    args = parser.parse_args()
    if not 0 < args.rank_rcond < 1:
        parser.error("rank-rcond must lie strictly between zero and one")
    if args.validated_on and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.validated_on):
        parser.error("validated-on must be YYYY-MM-DD")
    for revision in (args.source_revision, args.reference_source_revision):
        if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
            parser.error("source revisions must be Git identifiers")
    started = time.perf_counter()
    candidate = read_mixing(args.candidate_mixing)
    reference = read_mixing(args.reference_mixing)
    if candidate.shape[0] != reference.shape[0]:
        raise ValueError("Mixing time axes differ; no temporal shift/resampling is applied")
    candidate_noise = read_noise(args.candidate_noise, candidate.shape[1], args.candidate_noise_index_base)
    reference_noise = read_noise(args.reference_noise, reference.shape[1], args.reference_noise_index_base)
    correlation = np.clip(candidate.T @ reference, -1.0, 1.0)
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    signed_r = correlation[rows, columns]
    absolute_r = np.abs(signed_r)
    aligned_labels = label_comparison(candidate_noise[rows], reference_noise[columns])
    conditional = {}
    for threshold in (0.5, 0.8, 0.9):
        selected = absolute_r >= threshold
        conditional[str(threshold)] = (label_comparison(candidate_noise[rows[selected]], reference_noise[columns[selected]])
                                       if selected.any() else {"matched_pairs": 0})
    candidate_basis, candidate_rank = basis(candidate, args.rank_rcond)
    reference_basis, reference_rank = basis(reference, args.rank_rcond)
    principal_cosine = np.clip(np.linalg.svd(candidate_basis.T @ reference_basis, compute_uv=False), 0, 1)
    noise_candidate_basis, noise_candidate_rank = basis(candidate[:, candidate_noise], args.rank_rcond) if candidate_noise.any() else (None, 0)
    noise_reference_basis, noise_reference_rank = basis(reference[:, reference_noise], args.rank_rcond) if reference_noise.any() else (None, 0)
    noise_cosine = (np.clip(np.linalg.svd(noise_candidate_basis.T @ noise_reference_basis, compute_uv=False), 0, 1)
                    if noise_candidate_rank and noise_reference_rank else None)
    report = {
        "schema_version": 1,
        "source_revision": args.source_revision,
        "reference_source_revision": args.reference_source_revision,
        "validated_on": args.validated_on,
        "scope": "Independent alignment of existing real ICA mixing matrices and AROMA noise labels; no new ICA or full-pipeline execution.",
        "data": {"anonymous_id": "real_run_01", "time_points": int(candidate.shape[0]),
                 "candidate_components": int(candidate.shape[1]), "reference_components": int(reference.shape[1]),
                 "candidate_noise_components": int(candidate_noise.sum()), "reference_noise_components": int(reference_noise.sum())},
        "input_sha256": {name: sha256(getattr(args, name)) for name in
                         ("candidate_mixing", "reference_mixing", "candidate_noise", "reference_noise")},
        "alignment": {
            "method": "Float64 column centering/L2 normalization; Hungarian maximum sum of absolute Pearson correlations. Each component is used at most once; sign is ignored.",
            "matched_pairs": int(len(rows)), "candidate_unmatched_components": int(candidate.shape[1] - len(rows)),
            "reference_unmatched_components": int(reference.shape[1] - len(columns)),
            "matched_absolute_pearson_r": distribution(absolute_r),
            "noise_labels_on_matched_pairs": aligned_labels,
            "noise_labels_conditioned_on_minimum_absolute_r": conditional,
        },
        "subspace": {
            "method": "Float64 SVD orthonormal bases of centered/L2-normalized mixing; singular values of Q_candidate.T @ Q_reference are principal cosines.",
            "rank_rcond": args.rank_rcond, "candidate_rank": candidate_rank, "reference_rank": reference_rank,
            "all_component_principal_cosine": distribution(principal_cosine),
            "candidate_noise_rank": noise_candidate_rank, "reference_noise_rank": noise_reference_rank,
            "noise_component_principal_cosine": distribution(noise_cosine) if noise_cosine is not None else None,
        },
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "driver_sha256": sha256(__file__)},
        "wall_seconds_including_read_and_comparison": time.perf_counter() - started,
        "limits": ["Separate pre-ICA BOLD/masks/motion and separate decompositions may differ; this is not an isolated same-input ICA-equivalence test.",
                   "A matched low-correlation pair is an optimum of the global assignment, not proof that its components have the same physiological interpretation.",
                   "Label agreement is between two automatic classifications, not accuracy against ground-truth noise."],
        "privacy": "Anonymous summaries and hashes only; no subject IDs, private paths or voxel data.",
    }
    maps = spatial_alignment(args, rows, columns, np.where(signed_r < 0, -1., 1.), candidate.shape[1], reference.shape[1])
    if maps is not None:
        report["spatial_alignment"] = maps
    report["wall_seconds_including_read_and_comparison"] = time.perf_counter() - started
    if args.private_pairs_out is not None:
        args.private_pairs_out.parent.mkdir(parents=True, exist_ok=True)
        table = np.column_stack((rows + 1, columns + 1, signed_r, absolute_r,
                                 candidate_noise[rows].astype(int), reference_noise[columns].astype(int)))
        np.savetxt(args.private_pairs_out, table, fmt=("%d", "%d", "%.17g", "%.17g", "%d", "%d"), delimiter="\t",
                   header="candidate_IC_1based\treference_IC_1based\tsigned_r\tabs_r\tcandidate_noise\treference_noise", comments="")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
