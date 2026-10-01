"""固定真实 AROMA 影像，交叉比较两套末端混杂回归设计。

候选设计从其 AROMA、WM/CSF mask 和 motion 独立重建；参照设计读取
原软件 benchmark 已保存的 normalized TSV。两者都用 NumPy float64 SVD
做同一种联合投影，只比较共同脑 mask，不重新 ICA、不保存四套 4D 图。
gzip NIfTI 每张只连续解压一次，按帧保存共同区域，控制主机内存。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask(path, reference):
    image = nib.load(str(path))
    if image.shape != reference.shape[:3] or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
        raise ValueError("Mask grid differs from the AROMA image")
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("Mask contains nonfinite values")
    selected = values > 0
    if not selected.any():
        raise ValueError("Mask is empty")
    return selected


def streaming_series(path, region, tissue_masks=()):
    image = nib.load(str(path))
    if image.ndim != 4 or image.shape[:3] != region.shape or getattr(image.dataobj, "order", "F") != "F":
        raise ValueError("Expected a Fortran-order 4D NIfTI on the mask grid")
    proxy = image.dataobj
    voxel_count = int(np.prod(image.shape[:3]))
    bytes_per_frame = voxel_count * proxy.dtype.itemsize
    selected = region.ravel(order="F")
    tissues = [value.ravel(order="F") for value in tissue_masks]
    output = np.empty((int(selected.sum()), image.shape[3]), dtype=np.float32)
    means = [np.empty(image.shape[3], dtype=np.float64) for _ in tissues]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        stream.seek(proxy.offset)
        for frame in range(image.shape[3]):
            block = stream.read(bytes_per_frame)
            if len(block) != bytes_per_frame:
                raise ValueError("Truncated NIfTI payload")
            values = np.frombuffer(block, dtype=proxy.dtype, count=voxel_count).astype(np.float32)
            if proxy.slope != 1.0 or proxy.inter != 0.0:
                values = np.asarray(values.astype(np.float64) * proxy.slope + proxy.inter, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Image contains nonfinite values")
            output[:, frame] = values[selected]
            for tissue, mean in zip(tissues, means):
                mean[frame] = values[tissue].mean(dtype=np.float64)
        # Read to EOF to check gzip CRC even when the file has trailing extensions.
        while stream.read(8 * 1024 * 1024):
            pass
    return output, means


def build_design(means, motion, time_points):
    six = np.loadtxt(motion, ndmin=2, dtype=np.float64)
    if six.shape != (time_points, 6) or not np.isfinite(six).all():
        raise ValueError("Motion must have T finite rows and six columns")
    previous = np.vstack((np.zeros((1, 6)), six[:-1]))
    t = np.linspace(-1.0, 1.0, time_points)
    raw = np.column_stack((np.ones(time_points), t, (3 * t**2 - 1) / 2,
                           *means, six, previous, six**2, previous**2))
    active = np.r_[True, np.ptp(raw[:, 1:], axis=0) > 0]
    design = raw[:, active].copy()
    design[:, 1:] -= design[:, 1:].mean(axis=0)
    design /= np.linalg.norm(design, axis=0)
    return raw, design


def basis(design):
    if design.ndim != 2 or not np.isfinite(design).all():
        raise ValueError("Invalid finite design matrix")
    left, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-8))
    return left[:, :rank], rank


class Metrics:
    def __init__(self):
        self.count = 0
        self.absolute = self.squared = self.max_error = 0.0
        self.tx = self.ty = self.txy = 0.0
        self.voxel_r = []

    def add(self, candidate, reference):
        error = candidate - reference
        self.count += error.size
        self.absolute += float(np.abs(error).sum())
        self.squared += float(np.square(error).sum())
        self.max_error = max(self.max_error, float(np.abs(error).max()))
        x = candidate - candidate.mean(axis=1, keepdims=True)
        y = reference - reference.mean(axis=1, keepdims=True)
        xx, yy, xy = np.square(x).sum(axis=1), np.square(y).sum(axis=1), (x * y).sum(axis=1)
        valid = (xx > 1e-12 * x.shape[1]) & (yy > 1e-12 * y.shape[1])
        self.voxel_r.append(np.clip(xy[valid] / np.sqrt(xx[valid] * yy[valid]), -1, 1))
        self.tx += float(xx[valid].sum())
        self.ty += float(yy[valid].sum())
        self.txy += float(xy[valid].sum())

    def report(self):
        r = np.concatenate(self.voxel_r)
        return {
            "evaluated_values": int(self.count), "mae": self.absolute / self.count,
            "rmse": float(np.sqrt(self.squared / self.count)), "max_absolute_difference": self.max_error,
            "valid_voxel_temporal_r": int(r.size), "mean_voxel_temporal_r": float(r.mean()),
            "median_voxel_temporal_r": float(np.median(r)), "p05_voxel_temporal_r": float(np.percentile(r, 5)),
            "pooled_time_demeaned_r": float(np.clip(self.txy / np.sqrt(self.tx * self.ty), -1, 1)),
            "temporal_rms_threshold": 1e-6,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-aroma", "reference-aroma", "candidate-brain-mask", "reference-brain-mask",
                 "candidate-wm-mask", "candidate-csf-mask", "candidate-motion", "reference-normalized-design",
                 "candidate-clean", "reference-clean", "report-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reference-source-revision", required=True)
    parser.add_argument("--private-design-dir", type=Path, required=True)
    parser.add_argument("--voxel-chunk", type=int, default=1024)
    args = parser.parse_args()
    if args.voxel_chunk < 1:
        parser.error("voxel-chunk must be positive")
    started = time.perf_counter()
    image = nib.load(str(args.candidate_aroma))
    reference = nib.load(str(args.reference_aroma))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
        raise ValueError("AROMA grids/time axes differ; no implicit image registration is applied")
    candidate_brain = mask(args.candidate_brain_mask, image)
    reference_brain = mask(args.reference_brain_mask, image)
    common = candidate_brain & reference_brain
    if not common.any():
        raise ValueError("No common brain voxels")
    wm, csf = mask(args.candidate_wm_mask, image), mask(args.candidate_csf_mask, image)
    if np.any((wm | csf) & ~candidate_brain):
        raise ValueError("Candidate tissue masks must lie within its own brain mask")
    candidate_data, means = streaming_series(args.candidate_aroma, common, (wm, csf))
    reference_data, _ = streaming_series(args.reference_aroma, common)
    raw, candidate_design = build_design(means, args.candidate_motion, image.shape[3])
    reference_design = np.loadtxt(args.reference_normalized_design, skiprows=1, dtype=np.float64, ndmin=2)
    if candidate_design.shape != (image.shape[3], 29) or reference_design.shape != (image.shape[3], 29):
        raise ValueError("This benchmark requires both actual 29-column designs")
    for path in (args.candidate_clean, args.reference_clean):
        clean_image = nib.load(str(path))
        if clean_image.shape != image.shape or not np.allclose(clean_image.affine, image.affine, rtol=0, atol=1e-4):
            raise ValueError("Actual clean grid differs from its AROMA input")
    candidate_clean, _ = streaming_series(args.candidate_clean, common)
    reference_clean, _ = streaming_series(args.reference_clean, common)
    candidate_basis, candidate_rank = basis(candidate_design)
    reference_basis, reference_rank = basis(reference_design)
    cosine = np.clip(np.linalg.svd(candidate_basis.T @ reference_basis, compute_uv=False), 0, 1)
    args.private_design_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.private_design_dir / "candidate_raw.tsv"
    design_path = args.private_design_dir / "candidate_normalized.tsv"
    np.savetxt(raw_path, raw, fmt="%.17g", delimiter="\t")
    np.savetxt(design_path, candidate_design, fmt="%.17g", delimiter="\t")
    names = ("aroma_no_projection", "own_designs", "both_using_candidate_design", "both_using_reference_design",
             "candidate_aroma_changing_design", "reference_aroma_changing_design",
             "candidate_independent_vs_actual_clean", "reference_independent_vs_actual_clean", "actual_clean_pair")
    metrics = {name: Metrics() for name in names}

    def project(values, q):
        return values - (q @ (q.T @ values.T)).T

    for start in range(0, len(candidate_data), args.voxel_chunk):
        end = start + args.voxel_chunk
        c, n = candidate_data[start:end].astype(np.float64), reference_data[start:end].astype(np.float64)
        cc, nn = project(c, candidate_basis), project(n, reference_basis)
        cn, nc = project(c, reference_basis), project(n, candidate_basis)
        checks = {
            "aroma_no_projection": (c, n), "own_designs": (cc, nn),
            "both_using_candidate_design": (cc, nc), "both_using_reference_design": (cn, nn),
            "candidate_aroma_changing_design": (cc, cn), "reference_aroma_changing_design": (nc, nn),
            "candidate_independent_vs_actual_clean": (cc, candidate_clean[start:end].astype(np.float64)),
            "reference_independent_vs_actual_clean": (nn, reference_clean[start:end].astype(np.float64)),
            "actual_clean_pair": (candidate_clean[start:end].astype(np.float64), reference_clean[start:end].astype(np.float64)),
        }
        for name, (left, right) in checks.items():
            metrics[name].add(left, right)
    report = {
        "schema_version": 1, "source_revision": args.source_revision,
        "reference_source_revision": args.reference_source_revision,
        "scope": "Existing full real AROMA/clean data; two actual nuisance designs crossed under a single independent float64 SVD joint projection. No new ICA or pipeline run.",
        "data": {"anonymous_id": "real_run_01", "shape": list(image.shape),
                 "candidate_brain_voxels": int(candidate_brain.sum()), "reference_brain_voxels": int(reference_brain.sum()),
                 "common_brain_voxels": int(common.sum()), "candidate_wm_voxels": int(wm.sum()), "candidate_csf_voxels": int(csf.sum())},
        "design": {"candidate_shape": list(candidate_design.shape), "reference_shape": list(reference_design.shape),
                   "candidate_rank": candidate_rank, "reference_rank": reference_rank, "rank_rcond": 1e-8,
                   "method": "Intercept, linear/quadratic trends, WM/CSF means and Friston-24. Nonconstant columns centered and L2-normalized; joint projection without additional filter or mean restoration.",
                   "candidate_mean_computation": "Float64 tissue means streamed from float32 AROMA. Fortran voxel summation order may differ by float64 roundoff from production C-order; actual-clean numerical control is included.",
                   "normalized_candidate_design_sha256": sha256(design_path), "raw_candidate_design_sha256": sha256(raw_path),
                   "principal_cosine": {"mean": float(cosine.mean()), "median": float(np.median(cosine)),
                                        "p05": float(np.percentile(cosine, 5)), "minimum": float(cosine.min())}},
        "comparisons": {name: value.report() for name, value in metrics.items()},
        "input_sha256": {name: sha256(getattr(args, name)) for name in
                         ("candidate_aroma", "reference_aroma", "candidate_brain_mask", "reference_brain_mask",
                          "candidate_wm_mask", "candidate_csf_mask", "candidate_motion", "reference_normalized_design",
                          "candidate_clean", "reference_clean")},
        "software": {"numpy": np.__version__, "nibabel": nib.__version__, "driver_sha256": sha256(__file__)},
        "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "wall_seconds_including_read_hash_and_comparison": time.perf_counter() - started,
        "limits": ["Cross-projection fixes already estimated designs; tissue means are not re-estimated after swapping a design.",
                   "Own-run AROMA input, masks, motion and decompositions may differ; these controls separate measured projection effects rather than infer a unique upstream cause."],
        "privacy": "Anonymous scalars and hashes only; tissue designs, private paths and voxel data remain private.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
