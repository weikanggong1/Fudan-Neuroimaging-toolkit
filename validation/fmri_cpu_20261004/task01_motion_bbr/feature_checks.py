"""Complete real 180-frame CLI and compatibility-wrapper functional controls.

These controls complement the frozen timing adapter. They never reduce the
real input, and their repeated calls are function tests rather than a native
speed claim. Input paths and subprocess logs stay in private run directories.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_baseline import digest, parse_cpus


def _publish(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _logical_outputs(prefix, raw):
    import nibabel as nib
    import numpy as np

    image = nib.load(str(prefix) + ".nii.gz")
    values = np.asanyarray(image.dataobj)
    matrix_files = sorted(Path(str(prefix) + ".mat").glob("MAT_*"))
    parameters = np.loadtxt(str(prefix) + ".par")
    if image.shape != raw.shape or len(matrix_files) != raw.shape[3]:
        raise AssertionError("Incomplete real CLI image or matrix series")
    if parameters.shape != (raw.shape[3], 6) or not np.isfinite(parameters).all():
        raise AssertionError("Incomplete real CLI parameter series")
    if not np.array_equal(image.affine, raw.affine) or not np.isfinite(values).all():
        raise AssertionError("CLI output grid or finite-value contract failed")
    matrices = np.stack([np.loadtxt(path) for path in matrix_files])
    if matrices.shape != (raw.shape[3], 4, 4) or not np.isfinite(matrices).all():
        raise AssertionError("Invalid full CLI matrices")
    files = [Path(str(prefix) + suffix) for suffix in
             (".nii.gz", ".par", "_abs.rms", "_rel.rms", "_abs_mean.rms", "_rel_mean.rms")]
    if any(not path.is_file() for path in files):
        raise AssertionError("A requested standard CLI output is missing")
    if np.atleast_1d(np.loadtxt(files[2])).size != raw.shape[3]:
        raise AssertionError("Absolute RMS series is incomplete")
    if np.atleast_1d(np.loadtxt(files[3])).size != raw.shape[3] - 1:
        raise AssertionError("Relative RMS series is incomplete")
    return {
        "shape": list(image.shape), "dtype": str(image.get_data_dtype()),
        "affine": image.affine.tolist(), "header_sha256": hashlib.sha256(image.header.binaryblock).hexdigest(),
        "values_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        "matrices_sha256": hashlib.sha256(np.ascontiguousarray(matrices).tobytes()).hexdigest(),
        "parameters_sha256": hashlib.sha256(np.ascontiguousarray(parameters).tobytes()).hexdigest(),
        "normal_text_sha256": {path.name: digest(path) for path in files[1:] + matrix_files},
        "complete_frames": raw.shape[3], "whole_grid_values": int(values.size),
    }


def _cli_contract(args, case, raw):
    prefix = args.output_dir / "result"
    cli_program = ("import os,sys,torch; torch.set_num_threads(int(os.environ['OMP_NUM_THREADS'])); "
                   "torch.set_num_interop_threads(1); from fnit.cli import main; raise SystemExit(main(sys.argv[1:]))")
    command = [sys.executable, "-c", cli_program,
               "mcflirt", "-in", case["bold"], "-reffile", case["reference"],
               "-mats", "-plots", "-rmsrel", "-rmsabs", "-stages", "3", "-trilinear_final", "--device", "cpu"]
    environment = dict(os.environ, PYTHONPATH=str(args.source.resolve()))
    records = []
    first = None
    # The unified CLI currently propagates FileExistsError (exit 1); the
    # standalone fnit-mcflirt parser has a different exit-2 contract.
    for name, overwrite, expected_exit, suffix in (("first", False, 0, ".nii"),
                                                   ("reject_existing", False, 1, ""),
                                                   ("overwrite", True, 0, ".nii.gz")):
        log_path = args.output_dir / f"{name}.private.log"
        start = time.perf_counter()
        with log_path.open("wb") as log:
            process = subprocess.run(command + ["-out", str(prefix) + suffix] + (["--overwrite"] if overwrite else []),
                                     env=environment, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if process.returncode != expected_exit:
            raise AssertionError(f"CLI {name} exit differs from {expected_exit}; inspect private log")
        if name == "reject_existing" and "FileExistsError" not in log_path.read_text(errors="replace"):
            raise AssertionError("CLI rejection did not report an existing output")
        outputs = _logical_outputs(prefix, raw)
        if first is None:
            first = outputs
        elif outputs != first:
            raise AssertionError("Rejected/overwritten complete CLI output changed its logical result")
        records.append({"call": name, "exit_code": process.returncode, "output_prefix_suffix": suffix,
                        "process_seconds": elapsed, "logical_outputs_equal_first": outputs == first})
    return {"records": records, "complete_normal_outputs": first,
            "scope": "Complete unified CLI, existing-output rejection, then full explicit overwrite; subprocess startup/read/write included."}


def _wrapper_parity(args, case, raw):
    import nibabel as nib
    import numpy as np
    from fnit.mcflirt import TorchMCFLIRT
    from fnit.fmri.motion import estimate_motion, matrices_to_mcflirt_parameters

    reference = nib.load(case["reference"])
    # A verified positive reference tests mask-grid compatibility without
    # introducing a fabricated anatomical mask. The wrapper validates this
    # argument's grid and does not use its values in the MCFLIRT objective.
    reference_grid = reference
    records = []
    first = None
    for name, batch_size, input_bold, input_reference, mask in (
            ("batch1_path", 1, case["bold"], case["reference"], None),
            ("batch16_image_mask_grid", 16, raw, reference, reference_grid)):
        start = time.perf_counter()
        result = estimate_motion(input_bold, input_reference, device="cpu", batch_size=batch_size,
                                 iterations=(1, 1, 1), resample=True, mask=mask)
        elapsed = time.perf_counter() - start
        normalized = matrices_to_mcflirt_parameters(result.fsl_matrices, result.reference)
        arrays = {"matrices": result.fsl_matrices, "legacy_pull_parameters": result.parameters,
                  "mcflirt_parameters": normalized, "corrected": np.asanyarray(result.corrected.dataobj)}
        if arrays["matrices"].shape != (raw.shape[3], 4, 4) or arrays["corrected"].shape != raw.shape:
            raise AssertionError("Wrapper did not retain the complete real series")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise AssertionError("Wrapper returned nonfinite values")
        if first is None:
            first = arrays
        elif not all(np.array_equal(value, first[key]) for key, value in arrays.items()):
            raise AssertionError("Wrapper input/batch/mask-grid compatibility changed real outputs")
        records.append({"call": name, "api_seconds": elapsed, "complete_frames": raw.shape[3],
                        "whole_grid_values": int(arrays["corrected"].size), "equals_first": True})
    start = time.perf_counter()
    canonical = TorchMCFLIRT(device="cpu").run(raw, reference, stages=3, stage_iterations=(1, 1, 1),
                                               resample=True, interpolation="linear")
    elapsed = time.perf_counter() - start
    if not np.array_equal(canonical.matrices, first["matrices"]):
        raise AssertionError("Wrapper matrices differ from the public MCFLIRT API")
    if not np.array_equal(canonical.parameters, first["mcflirt_parameters"]):
        raise AssertionError("Conversion failed to recover the source .par convention")
    if not np.array_equal(np.asanyarray(canonical.corrected.dataobj), first["corrected"]):
        raise AssertionError("Wrapper corrected data differ from the public MCFLIRT API")
    logical_output_sha256 = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in first.items()
    }
    logical_output_sha256["canonical_header"] = hashlib.sha256(canonical.corrected.header.binaryblock).hexdigest()
    logical_output_sha256["canonical_affine"] = hashlib.sha256(np.ascontiguousarray(canonical.corrected.affine).tobytes()).hexdigest()
    return {"records": records, "canonical_api_seconds": elapsed, "canonical_all_values_equal": True,
            "logical_output_sha256": logical_output_sha256,
            "mask_scope": "Actual reference image validates grid only; no anatomical-mask or mask-dependent cost claim.",
            "parameter_conventions": "Wrapper pull parameters retained; compare canonical MCFLIRT after matrices_to_mcflirt_parameters."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--kind", choices=("cli_contract", "wrapper_parity"), required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--case", default="mc180")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--cpu-lock", type=Path, required=True)
    args = parser.parse_args()
    cpu_set = parse_cpus(args.cpu_list)
    if len(cpu_set) != args.threads:
        raise ValueError("Affinity must use exactly the declared physical CPU budget")
    os.sched_setaffinity(0, cpu_set)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with args.cpu_lock.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        import nibabel as nib
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(args.source.resolve()))
        configuration = json.loads(args.case_json.read_text())
        case = configuration[args.case]
        raw = nib.load(case["bold"])
        if raw.shape != (64, 64, 42, 180):
            raise ValueError("Use the full verified public 180-frame case")
        input_hashes = {key: digest(case[key]) for key in ("bold", "reference")}
        source_hashes = {str(path.relative_to(args.source)): digest(path)
                         for folder in ("fnit/mcflirt", "fnit/fmri", "fnit/flirt")
                         for path in (args.source / folder).glob("*.py")}
        import fnit
        if not Path(fnit.__file__).resolve().is_relative_to(args.source.resolve()):
            raise AssertionError("Imported FNIT does not belong to the declared frozen source")
        details = (_cli_contract(args, case, raw) if args.kind == "cli_contract" else
                   _wrapper_parity(args, case, raw))
        unchanged = all(digest(args.source / path) == value for path, value in source_hashes.items())
        if not unchanged:
            raise AssertionError("Frozen source changed during the functional control")
        input_unchanged = all(digest(case[key]) == value for key, value in input_hashes.items())
        if not input_unchanged:
            raise AssertionError("Actual input changed during the functional control")
        measured_seconds = sum(row.get("process_seconds", row.get("api_seconds", 0.0)) for row in details["records"])
        measured_seconds += details.get("canonical_api_seconds", 0.0)
        report = {"status": "complete", "kind": args.kind, "function": args.kind, "backend": "fnit",
                  "full_real_frames_required": True, "measured_application_seconds": measured_seconds,
                  "timing_scope": "Sum of explicitly recorded complete functional calls, excluding hashes/comparison; not a default native speed gate.",
                  "input_shape": list(raw.shape), "input_sha256": input_hashes, "input_unchanged": input_unchanged,
                  "source_revision": args.source_revision, "source_sha256": source_hashes, "source_unchanged": unchanged,
                  "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads": torch.get_num_threads(),
                  "slice_timing": False, "adapter_sha256": digest(__file__), "details": details}
        _publish(args.output_dir / "report.safe.json", report)
    print(json.dumps({"status": report["status"], "kind": args.kind, "cpu_threads": args.threads}))


if __name__ == "__main__":
    main()
