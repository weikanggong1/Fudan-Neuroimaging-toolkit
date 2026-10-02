"""Real full-grid TOPUP sampling: independent double tensor oracle vs CUDA.

Use a real selected b0 pair and independently generated official TOPUP field
coefficients/movement. This estimates no field, runs no TOPUP optimization,
calls no external program and writes only an anonymous JSON report. The tensor
oracle mirrors FSL Splinterpolator arithmetic; its timing is not FSL runtime.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import nibabel as nib
import numpy as np
import torch


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_oracle(path):
    # Load only this self-contained function, so pytest is not a runtime
    # dependency of the private benchmark. Keep the independent test oracle.
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "_official_reference"]
    if len(matches) != 1:
        raise ValueError("oracle file must define one _official_reference function")
    function_source = ast.get_source_segment(source, matches[0])
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=matches, type_ignores=[]), "official_tensor_oracle", "exec"), namespace)
    return namespace["_official_reference"], hashlib.sha256(function_source.encode()).hexdigest()


def _errors(candidate, reference, mask=None):
    x = candidate.detach()
    y = reference.detach()
    if mask is not None:
        x, y = x[mask], y[mask]
    delta = x.to(torch.float64) - y.to(torch.float64)
    if not delta.numel():
        return {"elements": 0, "compared": False}
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    return {"elements": delta.numel(), "compared": True,
            "exact": bool(torch.equal(x, y)),
            "different_elements": int(torch.count_nonzero(delta)),
            "finite": finite,
            "mae": float(delta.abs().mean()) if finite else None,
            "rmse": float(delta.square().mean().sqrt()) if finite else None,
            "max_absdiff": float(delta.abs().max()) if finite else None}


def _timed(function, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    outputs = function()
    end_event.record()
    torch.cuda.synchronize(device)
    row = {"wall_seconds": time.perf_counter() - started,
           "cuda_event_seconds": start_event.elapsed_time(end_event) / 1000,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    del outputs
    return row


def _prepare_real_inputs(args, device):
    from fnit.topup.core import (_TOPUPLevel, _cubic_spline_coefficients,
                                _decode_fsl_coefficient_image, _load_inputs,
                                _regrid_images)
    reference, values, acquisition, pe_axis, voxels = _load_inputs(args.imain, args.datain)
    coefficients, shape, spacing, encoded_voxels = _decode_fsl_coefficient_image(
        nib.load(str(args.fieldcoef)), device=device
    )
    if shape != values.shape[:3] or not np.allclose(voxels, encoded_voxels, atol=1e-5, rtol=0):
        raise ValueError("official coefficient geometry must match the full input grid")
    movement = np.loadtxt(args.movpar, ndmin=2)
    if movement.shape != (2, 6) or not np.isfinite(movement).all():
        raise ValueError("official movpar must have two finite six-column rows")
    flipped = bool(np.linalg.det(reference.affine[:3, :3]) > 0)
    if flipped:
        values = values[::-1].copy()
        # FSL changes image storage order; datain phase vectors stay unchanged.
    means = values.mean(axis=(0, 1, 2), dtype=np.float64)
    if np.any(np.abs(means) < np.finfo(np.float32).tiny):
        raise ValueError("each real b0 volume needs a nonzero mean")
    scaled = values * (100 / means).astype(np.float32)[None, None, None, :]
    images = torch.as_tensor(np.moveaxis(scaled, -1, 0).copy(), device=device, dtype=torch.float32)
    regridded, source_voxels = _regrid_images(images, voxels, 2, pe_axis)
    interpolation_images = _cubic_spline_coefficients(regridded)
    problem = _TOPUPLevel(interpolation_images, shape, spacing, voxels,
                          acquisition, pe_axis, 1, 0,
                          torch.as_tensor(movement, device=device, dtype=torch.float64),
                          sampling_voxel_sizes=source_voxels)
    state = problem.state(problem.parameters(coefficients, False))
    if "coordinates" not in state or len(state["coordinates"]) != 2:
        raise RuntimeError("matched TOPUP core must expose two fixed real sampling coordinates")
    coordinates = tuple(value.contiguous() for value in state["coordinates"])
    image_coefficients = tuple(value.contiguous() for value in interpolation_images)
    metadata = {"input_shape": list(values.shape), "full_grid_shape": list(shape),
                "voxel_sizes_mm": list(voxels), "phase_encode_axis": pe_axis,
                "source_grid_shape": list(regridded.shape[1:]),
                "source_voxel_sizes_mm": list(source_voxels),
                "regrid": True, "maximum_subsampling": 2,
                "canonical_x_flip": flipped, "field_knot_spacing_voxels": list(spacing),
                "image_intensity_normalization": "each real b0 volume to full-FOV mean 100; float32 scaling",
                "coordinate_rule": "official fixed fieldcoef/movpar, matched _TOPUPLevel.state, full resolution",
                "coordinate_dtype": "float32", "image_coefficient_dtype": "float32",
                "coordinates_sha256": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                                       for value in coordinates],
                "image_coefficients_sha256": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                                              for value in image_coefficients]}
    return image_coefficients, coordinates, pe_axis, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imain", type=Path, required=True, help="real selected AP/PA b0 pair NIfTI")
    parser.add_argument("--datain", type=Path, required=True, help="same two-row acqparams text")
    parser.add_argument("--fieldcoef", type=Path, required=True, help="independent official TOPUP intent-2016 coefficient NIfTI")
    parser.add_argument("--movpar", type=Path, required=True, help="independent official TOPUP 2x6 movement text")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--oracle-file", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tests/topup/test_sampling_cuda.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--abba-blocks", type=int, default=2,
                        help="each block runs tensor/cuda/cuda/tensor for each real b0 scan")
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("report must be new")
    if args.threads < 1 or args.warmup < 0 or args.abba_blocks < 1 or args.memory_limit_bytes < 1:
        raise ValueError("threads, ABBA blocks and memory limit must be positive; warmup nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this real full-grid component benchmark requires CUDA")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if args.memory_limit_bytes > properties.total_memory:
        raise ValueError("memory limit exceeds device capacity")
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_bytes / properties.total_memory, device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from fnit.topup._sampling_cuda import sample_cubic_with_derivatives_cuda
    import fnit.topup.core as topup_core
    import fnit.topup._sampling_cuda as sampling_core
    import triton
    oracle, oracle_function_sha256 = _load_oracle(args.oracle_file)
    started = time.perf_counter()
    with torch.no_grad():
        coefficients, coordinates, pe_axis, metadata = _prepare_real_inputs(args, device)
    torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - started
    torch.cuda.empty_cache()
    scans = []
    for scan in range(2):
        coeff, coord = coefficients[scan], coordinates[scan]
        functions = {
            "double_tensor_oracle": lambda: oracle(coeff, coord, pe_axis),
            "fused_official_cuda": lambda: sample_cubic_with_derivatives_cuda(
                coeff, coord, pe_axis, official_precision=True),
        }
        with torch.no_grad():
            reference = functions["double_tensor_oracle"]()
            candidate = functions["fused_official_cuda"]()
            torch.cuda.synchronize(device)
            accuracy = {"valid_exact": bool(torch.equal(candidate[2], reference[2])),
                        "reference_valid_voxels": int(reference[2].sum()),
                        "candidate_valid_voxels": int(candidate[2].sum()),
                        "values_all_voxels": _errors(candidate[0], reference[0]),
                        "values_reference_valid": _errors(candidate[0], reference[0], reference[2]),
                        "derivatives_all_voxels": [_errors(candidate[1][axis], reference[1][axis])
                                                   for axis in range(3)],
                        "derivatives_reference_valid": [_errors(candidate[1][axis], reference[1][axis], reference[2])
                                                         for axis in range(3)]}
            del candidate, reference
            for _ in range(args.warmup):
                for function in functions.values():
                    temporary = function()
                    torch.cuda.synchronize(device)
                    del temporary
            samples = {name: [] for name in functions}
            order = ("double_tensor_oracle", "fused_official_cuda",
                     "fused_official_cuda", "double_tensor_oracle")
            for block in range(args.abba_blocks):
                for position, name in enumerate(order):
                    row = _timed(functions[name], device)
                    row.update(abba_block=block, position=position)
                    samples[name].append(row)
        medians = {name: statistics.median(row["wall_seconds"] for row in rows)
                   for name, rows in samples.items()}
        scans.append({"scan_index": scan, "accuracy": accuracy, "timings": samples,
                      "wall_medians_seconds": medians,
                      "observed_tensor_to_fused_time_ratio": medians["double_tensor_oracle"] /
                          medians["fused_official_cuda"]})
        print(json.dumps({"event": "scan_complete", "scan_index": scan,
                          "wall_medians_seconds": medians}), flush=True)
    payload = {"schema_version": 1, "subjects": 1,
               "scope": "real full-resolution b0 image coefficient sampling at fixed official field/motion coordinates; no fitting",
               "timing_scope": "warm sampler calls with identical resident float32 inputs; allocations, Python launches, CUDA work and final synchronization; excludes input IO, regrid, prefilter, geometry, JIT warmup and JSON write",
               "event_scope": "CUDA stream interval including gaps between tensor-oracle kernel launches; not exclusive summed kernel time",
               "memory_scope": "per-call PyTorch allocator peak, including fixed resident inputs and allocator reserve; excludes CUDA context/other processes; Triton allocations follow its runtime",
               "oracle_scope": "independent test _official_reference double z/y/x expression; not a FSL executable performance baseline",
               "preparation_seconds": preparation_seconds,
               "preparation_scope": "input read/decode, fixed official field/motion, default regrid with maximum subsampling 2, source-grid prefilter and one target-grid state render including CUDA sampler compilation; no optimization",
               "input_sha256": {"b0_pair": _digest(args.imain), "acquisition": _digest(args.datain),
                                "official_fieldcoef": _digest(args.fieldcoef), "official_movpar": _digest(args.movpar)},
               "source_sha256": {"fnit/topup/core.py": _digest(topup_core.__file__),
                                 "fnit/topup/_sampling_cuda.py": _digest(sampling_core.__file__),
                                 "independent_oracle_file": _digest(args.oracle_file),
                                 "independent_oracle_function": oracle_function_sha256,
                                 "benchmark_script": _digest(__file__)},
               "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                            "triton": triton.__version__, "numpy": np.__version__, "nibabel": nib.__version__},
               "device": {"name": properties.name, "total_memory_bytes": properties.total_memory},
               "threads": args.threads, "warmup_per_method": args.warmup,
               "abba_blocks": args.abba_blocks, "memory_limit_bytes": args.memory_limit_bytes,
               "tf32_allowed": True, "low_precision": False, "inputs": metadata, "scans": scans}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "all_values_exact": all(
        row["accuracy"]["values_all_voxels"]["exact"] for row in scans)}), flush=True)


if __name__ == "__main__":
    main()
