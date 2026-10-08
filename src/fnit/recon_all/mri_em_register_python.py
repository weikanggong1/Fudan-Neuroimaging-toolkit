"""Experimental Python replacement for the fixed T1 ``mri_em_register`` stage."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .mri_em_register import (
    atlas_label_peak, estimate_image_white_matter_peak, find_all_samples,
    find_stable_samples, gca_centroid, gca_mean_volume, read_gca,
    read_masked_input, scale_input_intensity,
)
from .mri_em_register_em_candidate import EMObjective
from .mri_em_register_lta import write_voxel_lta
from .mri_em_register_optimizer import first_em_line_search
from .mri_em_register_search_source import find_optimal_linear_transform_source
from .mri_em_register_translation_source import find_optimal_translation_source


def register_t1(nu_path: str | Path, atlas_path: str | Path,
                mask_path: str | Path, output_path: str | Path, *,
                device: str = "cpu", search_backend: str = "cpu",
                candidate_chunk: int = 64, sample_chunk: int = 8192) -> dict:
    """Build a Talairach LTA without invoking a FreeSurfer executable.

    ``search_backend="cpu"`` keeps the validated source-order NumPy/Numba
    search. ``search_backend="torch"`` reuses FNIT's resident, chunked
    ``GCASearchScorer`` for translation and linear candidate scoring on an
    explicit CUDA device, while the EM refinement and LTA writing remain the
    existing Python implementation. The Torch path is opt-in until its LTA
    and downstream norm regression is complete.
    """
    if search_backend not in {"cpu", "torch"}:
        raise ValueError("search_backend must be 'cpu' or 'torch'")

    timing = {}
    overall_started = time.perf_counter()
    started = time.perf_counter()
    atlas = read_gca(atlas_path)
    masked = read_masked_input(nu_path, mask_path)
    peak, _, _ = estimate_image_white_matter_peak(atlas, masked)
    source = scale_input_intensity(masked, atlas_label_peak(atlas, 2), peak)
    stable_samples = find_stable_samples(atlas)
    center = gca_centroid(gca_mean_volume(atlas))
    timing["prepare_seconds"] = time.perf_counter() - started

    scorer = None
    if search_backend == "torch":
        from .mri_em_register_score_gpu import GCASearchScorer

        scorer = GCASearchScorer(
            stable_samples, source, device=device,
            candidate_chunk=candidate_chunk, sample_chunk=sample_chunk)

    started = time.perf_counter()
    translated, translation_history = find_optimal_translation_source(
        stable_samples, source, np.eye(4, dtype=np.float32), scorer=scorer)
    timing["translation_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    pre_em, linear_history = find_optimal_linear_transform_source(
        stable_samples, source, translated, center, translation_history[-1][0],
        scorer=scorer)
    timing["linear_search_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    objective = EMObjective(atlas, find_all_samples(atlas), source)
    matrix, cost, trials = first_em_line_search(objective, pre_em)
    timing["em_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    write_voxel_lta(output_path, matrix, nu_path, atlas_path, atlas, masked)
    timing["write_seconds"] = time.perf_counter() - started
    timing["total_seconds"] = time.perf_counter() - overall_started
    return {
        "output": str(output_path), "matrix": matrix.tolist(),
        "em_cost": cost, "em_trials": len(trials),
        "linear_iterations": len(linear_history), "timing": timing,
        "search_backend": search_backend, "device": device,
        "candidate_chunk": candidate_chunk, "sample_chunk": sample_chunk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nu", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--search-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--candidate-chunk", type=int, default=64)
    parser.add_argument("--sample-chunk", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(register_t1(
        args.nu, args.atlas, args.mask, args.output,
        device=args.device, search_backend=args.search_backend,
        candidate_chunk=args.candidate_chunk, sample_chunk=args.sample_chunk), indent=2))


if __name__ == "__main__":
    main()
