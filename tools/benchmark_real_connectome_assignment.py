"""Compare fixed real MRtrix streamlines with the PyTorch matrix stage.

The reference directory is the ds004666 FSL-5TT ACT run. This benchmarks only
endpoint assignment and aggregation: both programs receive the same full TCK,
atlas, SIFT2 weights, streamline lengths, and per-streamline FA values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.assignment import build_connectomes


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=","))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output-dir", type=Path, help="Save exact reference and PyTorch matrices")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mrtrix", default="tck2connectome")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    reference_dir = args.reference_dir.resolve()
    atlas_path = args.atlas.resolve()
    tracks_path = reference_dir / "tracks_10000.tck"
    weights_path = reference_dir / "sift2_weights.txt"
    lengths_path = reference_dir / "streamline_length.txt"
    fa_path = reference_dir / "streamline_mean_fa.txt"
    paths = (tracks_path, atlas_path, weights_path, lengths_path, fa_path)
    references = {name: reference_dir / f"{name}.csv" for name in NAMES}
    for path in (*paths, *references.values()):
        if not path.is_file():
            parser.error(f"missing benchmark input: {path}")

    started_load = time.perf_counter()
    streamlines = nib.streamlines.load(str(tracks_path)).tractogram.streamlines
    endpoints = np.stack([np.asarray(track[[0, -1]], dtype=np.float32)
                          for track in streamlines])
    atlas_image = nib.load(str(atlas_path))
    atlas = np.asarray(atlas_image.get_fdata(dtype=np.float32), dtype=np.int32)
    affine = np.asarray(atlas_image.affine, dtype=np.float32)
    weights, lengths, fa = (np.loadtxt(path, dtype=np.float32).reshape(-1)
                            for path in (weights_path, lengths_path, fa_path))
    if any(len(vector) != len(endpoints) for vector in (weights, lengths, fa)):
        raise ValueError("SIFT2, length, and FA vectors must match TCK order and count")
    reference = {name: _matrix(path) for name, path in references.items()}
    load_seconds = time.perf_counter() - started_load

    # Regenerate matrices in a temporary directory to verify that the recorded
    # CSVs follow the same command semantics; do not overwrite source outputs.
    options = {
        "count": [],
        "sift2_fbc": ["-tck_weights_in", str(weights_path)],
        "mean_length": ["-tck_weights_in", str(weights_path),
                        "-scale_file", str(lengths_path), "-stat_edge", "mean"],
        "mean_fa": ["-tck_weights_in", str(weights_path),
                    "-scale_file", str(fa_path), "-stat_edge", "mean"],
    }
    mrtrix_seconds = {name: [] for name in NAMES}
    reference_reproduction = {name: 0.0 for name in NAMES}
    with tempfile.TemporaryDirectory(prefix="fnit_real_assignment_") as temp:
        for run in range(args.repeats):
            for name in NAMES:
                output = Path(temp) / f"{name}_{run}.csv"
                command = [args.mrtrix, "-quiet", "-symmetric", "-assignment_radial_search", "4",
                           *options[name], str(tracks_path), str(atlas_path), str(output), "-nthreads", "8"]
                started = time.perf_counter()
                subprocess.run(command, check=True, capture_output=True, text=True)
                mrtrix_seconds[name].append(time.perf_counter() - started)
                reproduced = _matrix(output)
                if reproduced.shape != reference[name].shape:
                    raise ValueError(f"{name}: regenerated reference matrix shape differs")
                reference_reproduction[name] = max(
                    reference_reproduction[name], float(np.abs(reproduced - reference[name]).max())
                )

    device = torch.device(args.device)
    started_transfer = time.perf_counter()
    tensor = lambda value: torch.as_tensor(value, device=device)
    inputs = [tensor(value) for value in (endpoints, atlas, affine, weights, lengths, fa)]
    _synchronize(device)
    transfer_seconds = time.perf_counter() - started_transfer
    torch_seconds = []
    for _ in range(args.repeats):
        started_compute = time.perf_counter()
        candidate = build_connectomes(inputs[0], inputs[1], inputs[2],
                                       weights=inputs[3], lengths=inputs[4], fa=inputs[5])
        _synchronize(device)
        torch_seconds.append(time.perf_counter() - started_compute)

    metrics = {}
    for name in NAMES:
        actual = candidate[name].cpu().numpy()
        expected = reference[name]
        if actual.shape != expected.shape:
            raise ValueError(f"{name}: PyTorch matrix shape differs from reference")
        difference = actual.astype(np.float64) - expected
        metrics[name] = {
            "shape": list(actual.shape),
            "max_absolute_error_all_elements": float(np.abs(difference).max()),
            "mean_absolute_error_all_elements": float(np.abs(difference).mean()),
            "exact_elements": int(np.count_nonzero(actual == expected)),
            "total_elements": int(actual.size),
        }

    matrix_csv_sha256 = None
    if args.matrix_output_dir is not None:
        matrix_csv_sha256 = {"reference": {}, "candidate": {}}
        for side in matrix_csv_sha256:
            (args.matrix_output_dir / side).mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            filename = f"connectome_{name}.csv"
            reference_csv = args.matrix_output_dir / "reference" / filename
            candidate_csv = args.matrix_output_dir / "candidate" / filename
            reference_csv.write_bytes(references[name].read_bytes())
            np.savetxt(candidate_csv, candidate[name].cpu().numpy(), delimiter=",", fmt="%.17g")
            matrix_csv_sha256["reference"][filename] = _sha256(reference_csv)
            matrix_csv_sha256["candidate"][filename] = _sha256(candidate_csv)

    upper = np.triu_indices(reference["count"].shape[0])
    reference_assigned = int(reference["count"][upper].sum())
    candidate_assigned = int(candidate["count"].cpu().numpy()[upper].sum())
    version = subprocess.check_output([args.mrtrix, "-version"], text=True).strip().splitlines()[0]
    report = {
        "scope": "real ds004666 MRtrix FSL-5TT ACT tractogram; fixed-input matrix stage only",
        "source_dataset": "OpenNeuro ds004666",
        "reference_relative_id": "mrtrix_raw_default/fsl5tt_act_temporary",
        "atlas_relative_id": "registration_direct_affine/synthseg_gm_atlas_dwi.nii.gz",
        "track_input": "full tracks_10000.tck, matching all four original tck2connectome commands",
        "streamlines": len(endpoints),
        "assigned_reference": reference_assigned,
        "assigned_torch": candidate_assigned,
        "mrtrix_version": version,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "timing_seconds": {
            "boundary": "MRtrix subprocesses include startup and file I/O; PyTorch is GPU-resident assignment only",
            "load_tck_atlas_vectors_csv": load_seconds,
            "host_to_device": transfer_seconds,
            "torch_assignment_gpu_resident_runs": torch_seconds,
            "torch_assignment_gpu_resident_median": float(np.median(torch_seconds)),
            "mrtrix_command_wall_runs": mrtrix_seconds,
            "mrtrix_four_commands_wall_median": float(np.median([
                sum(mrtrix_seconds[name][run] for name in NAMES)
                for run in range(args.repeats)
            ])),
        },
        "recorded_reference_csv_reproduction_max_absolute_error": reference_reproduction,
        "metrics": metrics,
        "matrix_csv_sha256": matrix_csv_sha256,
        "assignment_source_sha256": _sha256(Path(build_connectomes.__code__.co_filename)),
        "benchmark_script_sha256": _sha256(Path(__file__)),
        "input_sha256": {path.name: _sha256(path) for path in (*paths, *references.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
