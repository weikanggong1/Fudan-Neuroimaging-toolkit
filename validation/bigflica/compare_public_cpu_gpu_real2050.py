"""Compare two private real-subject BigFLICA outputs without publishing IDs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_public_cpu_gpu_real2050.py CPU_MODEL GPU_MODEL SUMMARY_JSON")
    cpu_dir, gpu_dir, summary_file = map(Path, sys.argv[1:])
    cpu = json.loads((cpu_dir / "model.json").read_text(encoding="utf-8"))
    gpu = json.loads((gpu_dir / "model.json").read_text(encoding="utf-8"))
    n_cpu, n_gpu = cpu["n_components"], gpu["n_components"]
    report = {
        "dataset": "2050 real subjects; full-mask VBM/FA/MD; task excluded",
        "same_input_signature": cpu["input_signature"] == gpu["input_signature"],
        "subjects_match": cpu["subjects"] == gpu["subjects"],
        "requested_components": {"cpu": n_cpu, "gpu": n_gpu},
        "stage_timings_s": {"cpu": cpu["timings"], "gpu": gpu["timings"]},
    }
    if not report["same_input_signature"] or not report["subjects_match"]:
        raise ValueError("CPU and GPU models did not use identical raw inputs")
    cpu_u = np.load(cpu_dir.parent / f"mmigp_{cpu['migp_dim']}" / "U.npy").astype(np.float64)
    gpu_u = np.load(gpu_dir.parent / f"mmigp_{gpu['migp_dim']}" / "U.npy").astype(np.float64)
    if cpu_u.shape != gpu_u.shape:
        raise ValueError("mMIGP U shapes differ")
    basis_sign = np.where(np.sum(cpu_u * gpu_u, axis=0) < 0, -1.0, 1.0)
    aligned_u = gpu_u * basis_sign
    with h5py.File(cpu_dir.parent / "normalized/vbm.h5", "r") as cpu_store, \
         h5py.File(gpu_dir.parent / "normalized_f32/vbm.h5", "r") as gpu_store:
        cpu_dtype = str(cpu_store["data"].dtype)
        gpu_dtype = str(gpu_store["data"].dtype)
    report["mmigp"] = {
        "cpu_normalized_dtype": cpu_dtype, "gpu_normalized_dtype": gpu_dtype,
        "u_shape": list(cpu_u.shape),
        "u_raw_relative_l2": float(np.linalg.norm(cpu_u - gpu_u) / np.linalg.norm(cpu_u)),
        "u_sign_aligned_relative_l2": float(
            np.linalg.norm(cpu_u - aligned_u) / np.linalg.norm(cpu_u)),
        "u_same_index_abs_correlations": [
            abs(_correlation(cpu_u[:, i], gpu_u[:, i])) for i in range(cpu_u.shape[1])],
        "u_same_index_signs": basis_sign.astype(int).tolist(),
        "projected": {},
    }
    cpu_projected_dir = cpu_dir.parent / f"mmigp_{cpu['migp_dim']}"
    gpu_projected_dir = gpu_dir.parent / f"mmigp_{gpu['migp_dim']}"
    for name in ("vbm", "fa", "md"):
        raw_sq = aligned_sq = reference_sq = 0.0
        with h5py.File(cpu_projected_dir / f"{name}_projected.h5", "r") as left_file, \
             h5py.File(gpu_projected_dir / f"{name}_projected.h5", "r") as right_file:
            left, right = left_file["data"], right_file["data"]
            if left.shape != right.shape:
                raise ValueError(f"{name} projected shapes differ")
            for start in range(0, left.shape[0], 2048):
                original = left[start:start + 2048].astype(np.float64)
                candidate = right[start:start + 2048].astype(np.float64)
                raw_sq += float(np.square(original - candidate).sum())
                aligned_sq += float(np.square(original - candidate * basis_sign).sum())
                reference_sq += float(np.square(original).sum())
            report["mmigp"]["projected"][name] = {
                "voxels": int(left.shape[0]), "cpu_dtype": str(left.dtype),
                "gpu_dtype": str(right.dtype),
                "raw_relative_l2": float(np.sqrt(raw_sq / reference_sq)),
                "sign_aligned_relative_l2": float(np.sqrt(aligned_sq / reference_sq)),
            }
    cpu_dictionary_dir = cpu_dir.parent / (
        f"dicl_{cpu['migp_dim']}_{cpu['dicl_dim']}_{cpu['dicl_max_iter']}_"
        f"{cpu['random_state']}_cpu_stream")
    gpu_dictionary_dir = gpu_dir.parent / (
        f"dicl_{gpu['migp_dim']}_{gpu['dicl_dim']}_{gpu['dicl_max_iter']}_"
        f"{gpu['random_state']}_{gpu['dicl_batch_size']}_"
        f"{gpu['dicl_sparse_iterations']}_rsvd1_cuda")
    report["dicl"] = {}
    for name in ("vbm", "fa", "md"):
        left = np.load(cpu_dictionary_dir / f"{name}_dictionary.npy").astype(np.float64)
        right = np.load(gpu_dictionary_dir / f"{name}_dictionary.npy").astype(np.float64)
        if left.shape != right.shape:
            raise ValueError(f"{name} dictionary shapes differ")
        aligned_basis = right * basis_sign
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(aligned_basis, axis=1)
        if np.any(left_norm == 0) or np.any(right_norm == 0):
            raise ValueError(f"{name} dictionary has a zero atom")
        atom_cosines = (left / left_norm[:, None]) @ (
            aligned_basis / right_norm[:, None]).T
        atom_rows, atom_cols = linear_sum_assignment(-np.abs(atom_cosines))
        atom_matching = np.empty(left.shape[0], dtype=int)
        atom_matching[atom_rows] = atom_cols
        matched_cosines = atom_cosines[np.arange(left.shape[0]), atom_matching]
        atom_signs = np.where(matched_cosines < 0, -1.0, 1.0)
        matched_dictionary = aligned_basis[atom_matching] * atom_signs[:, None]
        report["dicl"][name] = {
            "shape": list(left.shape),
            "raw_relative_l2": float(np.linalg.norm(left - right) / np.linalg.norm(left)),
            "basis_sign_aligned_relative_l2": float(
                np.linalg.norm(left - aligned_basis) / np.linalg.norm(left)),
            "hungarian_matched_sign_aligned_relative_l2": float(
                np.linalg.norm(left - matched_dictionary) / np.linalg.norm(left)),
            "hungarian_matched_abs_atom_cosine_min": float(
                np.abs(matched_cosines).min()),
            "hungarian_matched_abs_atom_cosine_median": float(
                np.median(np.abs(matched_cosines))),
            "hungarian_matched_abs_atom_cosine_p05": float(
                np.quantile(np.abs(matched_cosines), 0.05)),
            "hungarian_matched_abs_atom_cosine_p95": float(
                np.quantile(np.abs(matched_cosines), 0.95)),
        }
    if n_cpu == n_gpu:
        cpu_course = np.load(cpu_dir / "subj_course.npy")
        gpu_course = np.load(gpu_dir / "subj_course.npy")
        if cpu_course.shape != gpu_course.shape or cpu_course.shape[0] != 2050:
            raise ValueError("Subject course shapes differ")
        correlations = np.array([
            [_correlation(cpu_course[:, i], gpu_course[:, j]) for j in range(n_gpu)]
            for i in range(n_cpu)
        ])
        rows, cols = linear_sum_assignment(-np.abs(correlations))
        matching = np.empty(n_cpu, dtype=int)
        matching[rows] = cols
        signs = np.sign(correlations[np.arange(n_cpu), matching])
        signs[signs == 0] = 1
        report["course_abs_correlations"] = np.abs(
            correlations[np.arange(n_cpu), matching]).tolist()
        report["course_matching_method"] = "Hungarian maximum absolute correlation; maps use the same sign and permutation"
        report["component_matching_gpu_index_zero_based"] = matching.tolist()
        cpu_h = np.linalg.lstsq(cpu_u, cpu_course, rcond=None)[0]
        gpu_h = np.linalg.lstsq(gpu_u, gpu_course, rcond=None)[0]
        report["flica_h_abs_correlations_after_alignment"] = [
            abs(_correlation(cpu_h[:, i], gpu_h[:, j] * basis_sign * signs[i]))
            for i, j in enumerate(matching)
        ]
        report["spatial_zstat"] = {}
        for name in ("vbm", "fa", "md"):
            left = np.load(cpu_dir / f"{name}_zstat.npy", mmap_mode="r")
            right = np.load(gpu_dir / f"{name}_zstat.npy", mmap_mode="r")
            if left.shape != right.shape:
                raise ValueError(f"{name} z-stat shapes differ")
            correlations_by_component = []
            relative_errors = []
            degenerate = []
            for i, j in enumerate(matching):
                reference = np.asarray(left[:, i], dtype=np.float64)
                candidate = np.asarray(right[:, j], dtype=np.float64) * signs[i]
                correlations_by_component.append(_correlation(reference, candidate))
                reference_norm = np.linalg.norm(reference)
                candidate_sq = candidate @ candidate
                is_degenerate = reference_norm == 0 or candidate_sq == 0
                degenerate.append(bool(is_degenerate))
                if is_degenerate:
                    relative_errors.append(None)
                else:
                    scale = float(reference @ candidate / candidate_sq)
                    relative_errors.append(float(np.linalg.norm(reference - scale * candidate) /
                                                 reference_norm))
            report["spatial_zstat"][name] = {
                "voxels": int(left.shape[0]),
                "signed_correlations": correlations_by_component,
                "relative_l2_after_scalar_alignment": relative_errors,
                "degenerate_component": degenerate,
            }
    else:
        report["comparison_status"] = "Component counts differ; courses/maps not comparable"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
