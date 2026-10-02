#!/usr/bin/env python3
"""Independent original-software T1 + tensor MMORF registration reference.

Run after the reference's own EDDY, DTIFIT and AMICO steps. This script does
not import FNIT, estimate a second FA scalar modality, or reuse FNIT matrices.
FSL applywarp consumes the original MMORF relative field directly on the
orthogonal, radiological MNI template grid validated by the frozen debug oracle.
All paths, commands and images remain in the caller's private output directory;
sanitize the JSON separately before publishing aggregate benchmark results.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import nibabel as nib
import numpy as np


MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")
SYNTHSTRIP_SHA256 = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"
SYNTHSTRIP_BYTES = 30_851_709
MMORF_COMMIT = "1c1c13b8368f05e1a79a6dafe919d6b61df36bd6"
REGISTRATION_STARTED_MARKERS = (
    "##### MAXIMUM THREADS AVAILABLE IS:", "##### THREADS BEING USED IS:",
    "Extents Old: [", "Extents New: [", "cost_init = ", "lambda_l = ",
    "cost_next = ", "Beginning iteration ", "cost = ", "lambda_lm = ",
)


def startup_retry_allowed(log_text, exit_code, output_dir):
    """Only the diagnosed pre-registration 64-byte CUDA allocation failure.

    The original 0.3.2 registration entry prints and flushes both THREADS
    markers. A failure after either marker, or after any result was saved,
    is retained without retry. Neither image geometry nor solver settings change.
    """
    if exit_code not in (-6, 134):
        return False
    if "cudaErrorMemoryAllocation" not in log_text or "thrust::" not in log_text:
        return False
    if any(marker in log_text for marker in REGISTRATION_STARTED_MARKERS):
        return False
    return not any((Path(output_dir) / (name + suffix)).exists()
                   for name in ("mmorf_warp", "mmorf_jacobian")
                   for suffix in (".nii", ".nii.gz"))


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def image_record(path, *, load_data=True):
    image = nib.load(str(path))
    record = {**file_record(path), "shape": list(image.shape),
              "affine": image.affine.tolist(),
              "stored_dtype": str(image.get_data_dtype()),
              "voxel_sizes_mm": [float(value) for value in image.header.get_zooms()[:3]],
              "intent_code": int(image.header["intent_code"])}
    if load_data:
        values = np.asarray(image.dataobj)
        record["all_finite"] = bool(np.isfinite(values).all())
        if not record["all_finite"]:
            raise ValueError("nonfinite reference image")
    return record


def require_template_contract(fa_path, t1_path, tensor_path):
    """Prove the direct MMORF-axis to FSL-relative-field contract used here.

    On an orthogonal reference grid with a negative affine determinant, FSL's
    scaled-mm coordinates and MMORF's saved reference-image-axis mm vectors
    have the same axes and scale. Oblique orthogonal grids also qualify; sheared
    or positive-determinant reference grids require an explicit conversion and
    are rejected by this benchmark driver rather than silently reinterpreted.
    """
    images = [nib.load(str(path)) for path in (fa_path, t1_path, tensor_path)]
    reference = images[1]
    linear = np.asarray(reference.affine[:3, :3], dtype=np.float64)
    sizes = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
    if not np.isfinite(reference.affine).all() or np.any(sizes <= 0):
        raise ValueError("reference geometry must be finite and nonsingular")
    if np.linalg.det(linear) >= 0:
        raise ValueError("direct original applywarp requires a radiological reference grid")
    if not np.allclose(linear.T @ linear, np.diag(sizes * sizes), rtol=1e-5, atol=1e-6):
        raise ValueError("direct original applywarp does not accept a sheared reference grid")
    if reference.ndim != 3 or images[0].ndim != 3:
        raise ValueError("T1 and FA templates must be 3D")
    if images[2].ndim != 4 or images[2].shape[-1] != 6:
        raise ValueError("tensor template must contain six FSL upper-triangular components")
    for image in images:
        if image.shape[:3] != reference.shape or not np.allclose(
                image.affine, reference.affine, rtol=0, atol=1e-5):
            raise ValueError("FA, T1 and tensor templates must share one grid")
    return {"shape": list(reference.shape), "affine": reference.affine.tolist(),
            "reference_affine_determinant": float(np.linalg.det(linear)),
            "orthogonal_reference_grid": True,
            "direct_relative_field_contract": "MMORF reference-image-axis mm equals FSL scaled-mm axes on this grid"}


def write_config(path, *, t1_brain, native_tensor, t1_reference,
                 tensor_reference, t1_matrix, tensor_matrix, identity, output_dir):
    """One T1 scalar (weight 1) + tensor (weight 1), matching the pipeline."""
    values = {
        "warp_res_init": "32", "warp_scaling": "1 1 2 2 2",
        "img_warp_space": t1_reference,
        "lambda_reg": "4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1",
        "hires": "6", "optimiser_lowres": "LM", "optimiser_hires": "MM",
        "optimiser_max_it_lowres": "5", "optimiser_max_it_hires": "5",
        "warp_out": output_dir / "mmorf_warp",
        "jac_det_out": output_dir / "mmorf_jacobian", "bias_out": "NULL",
        "img_ref_scalar": t1_reference, "img_mov_scalar": t1_brain,
        "aff_ref_scalar": identity, "aff_mov_scalar": t1_matrix,
        "use_implicit_mask": "0", "use_mask_ref_scalar": "0 0 0 0 0",
        "use_mask_mov_scalar": "0 0 0 0 0", "mask_ref_scalar": "NULL",
        "mask_mov_scalar": "NULL", "fwhm_ref_scalar": "8 8 4 2 1",
        "fwhm_mov_scalar": "8 8 4 2 1", "lambda_scalar": "1 1 1 1 1",
        "estimate_bias": "0", "bias_res_init": "32",
        "lambda_bias_reg": "1e9 1e9 1e9 1e9 1e9",
        "img_ref_tensor": tensor_reference, "img_mov_tensor": native_tensor,
        "aff_ref_tensor": identity, "aff_mov_tensor": tensor_matrix,
        "use_mask_ref_tensor": "0 0 0 0 0", "use_mask_mov_tensor": "0 0 0 0 0",
        "mask_ref_tensor": "NULL", "mask_mov_tensor": "NULL",
        "fwhm_ref_tensor": "8 8 4 2 1", "fwhm_mov_tensor": "8 8 4 2 1",
        "lambda_tensor": "1 1 1 1 1",
    }
    for value in values.values():
        if isinstance(value, Path) and any(character.isspace() for character in str(value)):
            raise ValueError("MMORF ini paths must not contain whitespace")
    path.write_text("\n".join(f"{key:<26} = {value}" for key, value in values.items()) + "\n")
    return values


class Runner:
    def __init__(self, output_dir, report, environment):
        self.output_dir, self.report, self.environment = output_dir, report, environment
        self.report_path = output_dir / "official_mmorf_report.json"
        self.commands, self.stages = [], {}

    def save(self):
        self.report["stages_seconds"] = self.stages
        self.report["subprocess_steps"] = self.commands
        temporary = self.report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.report_path)

    @contextlib.contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = time.perf_counter() - started
            self.save()

    def run(self, arguments, *, name, monitor_cuda=False):
        arguments = [str(value) for value in arguments]
        step = {"name": name, "arguments": arguments}
        self.commands.append(step)
        started = time.perf_counter()
        samples, stop = [], threading.Event()
        with (self.output_dir / f"{name}.log").open("wb") as log:
            process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT,
                                       env=self.environment, cwd=self.output_dir)

            def monitor():
                while not stop.is_set():
                    try:
                        output = subprocess.run(
                            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, timeout=5, check=True).stdout
                        for line in output.splitlines():
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) == 3 and parts[0] == str(process.pid):
                                samples.append({"seconds": time.perf_counter() - started,
                                                "gpu_uuid": parts[1],
                                                "process_memory_mib": int(parts[2])})
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                    stop.wait(5)

            thread = threading.Thread(target=monitor, daemon=True) if monitor_cuda else None
            if thread is not None:
                thread.start()
            try:
                code = process.wait()
            finally:
                stop.set()
                if thread is not None:
                    thread.join(timeout=6)
        step.update({"seconds": time.perf_counter() - started, "exit_code": code})
        if monitor_cuda:
            step["gpu_process_samples"] = samples
            step["sampled_process_gpu_memory_peak_mib"] = max(
                (sample["process_memory_mib"] for sample in samples), default=None)
            step["memory_scope"] = "nvidia-smi own native MMORF PID, sampled every 5 s; not an allocator peak or enforced cap"
        self.save()
        if code:
            raise RuntimeError(f"original-software {name} exited with code {code}")

    def run_mmorf(self, arguments):
        """At most three starts; every attempt and delay counts in the clock."""
        self.report["startup_retry_policy"] = {
            "maximum_attempts": 3, "delay_seconds": 2,
            "scope": "native MMORF constructor CUDA allocation only, before flushed registration-entry markers and before any warp/Jacobian exists",
            "all_attempts_included_in_processing_time": True,
            "numerical_parameters_changed": False,
        }
        for attempt in range(1, 4):
            name = "mmorf" if attempt == 1 else f"mmorf_startup_retry_{attempt}"
            try:
                self.run(arguments, name=name, monitor_cuda=True)
                return
            except RuntimeError:
                step = self.commands[-1]
                text = (self.output_dir / f"{name}.log").read_text(errors="replace")
                allowed = attempt < 3 and startup_retry_allowed(
                    text, step["exit_code"], self.output_dir)
                step["startup_retry"] = {
                    "attempt": attempt, "retry_allowed": allowed,
                    "log_sha256": file_record(self.output_dir / f"{name}.log")["sha256"],
                    "reason": "diagnosed_constructor_allocation_before_registration" if allowed
                              else "retry_limit_or_failure_outside_guard",
                }
                self.save()
                if not allowed:
                    raise
                time.sleep(2)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("t1", "native-dir", "t1-reference", "fa-reference", "tensor-reference",
                 "output-dir", "fsl-dir", "synthstrip-command", "synthstrip-weights"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.threads < 1:
        raise ValueError("threads must be positive")
    for name in ("t1", "native_dir", "t1_reference", "fa_reference", "tensor_reference",
                 "output_dir", "fsl_dir", "synthstrip_command", "synthstrip_weights"):
        setattr(arguments, name, getattr(arguments, name).expanduser().resolve())
    output = arguments.output_dir
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise FileExistsError("reference output directory is nonempty")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    native_tensor = arguments.native_dir / "dti_tensor.nii.gz"
    native_fa = arguments.native_dir / "dti_FA.nii.gz"
    native_maps = {
        name: arguments.native_dir / f"{'NODDI_' if name in ('ICVF', 'OD', 'ISOVF') else 'dti_'}{name}.nii.gz"
        for name in MAP_NAMES
    }
    binaries = {name: arguments.fsl_dir / "bin" / name for name in ("flirt", "mmorf", "applywarp")}
    for path in (arguments.t1, arguments.t1_reference, arguments.fa_reference,
                 arguments.tensor_reference, arguments.synthstrip_command,
                 arguments.synthstrip_weights, native_tensor, *native_maps.values(),
                 *binaries.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    grid = require_template_contract(arguments.fa_reference, arguments.t1_reference,
                                     arguments.tensor_reference)
    weight = file_record(arguments.synthstrip_weights)
    if weight != {"bytes": SYNTHSTRIP_BYTES, "sha256": SYNTHSTRIP_SHA256}:
        raise ValueError("reference SynthStrip requires the verified standard synthstrip.1.pt")
    report = {
        "schema_version": 1, "complete": False,
        "scope": "independent original CPU SynthStrip T1, two original 12-DOF FLIRT matrices, original GPU MMORF, original FSL applywarp nine maps",
        "reference": {"MMORF": "0.3.2", "MMORF_source_commit": MMORF_COMMIT},
        "configuration": {"scalar_pairs": ["T1"], "scalar_weights": [1.0],
                          "tensor_weight": 1.0, "warp_resolution_mm": [32, 32, 16, 8, 4],
                          "smoothing_mm": [8, 8, 4, 2, 1], "iterations": [5] * 5,
                          "FLIRT": {"dof": 12, "cost": "corratio"},
                          "map_interpolation": "original applywarp trilinear",
                          "SynthStrip": {"device": "cpu", "border_mm": 1,
                                         "no_csf": False, "weights": weight},
                          "threads": arguments.threads},
        "template_grid": grid,
        "adaptations": [
            "relative saved MMORF field passed to original applywarp with --rel and separate --premat; restricted to verified orthogonal radiological reference grid",
            "original MMORF does not normally save T1/tensor convenience images; original applywarp produces spline-resampled convenience outputs, without output tensor reorientation",
        ],
        "source_sha256": {Path(__file__).name: file_record(__file__)["sha256"]},
        "binaries": {name: file_record(path) for name, path in binaries.items()},
        "inputs": {"T1": image_record(arguments.t1),
                   "native_tensor": image_record(native_tensor),
                   "native_maps": {name: image_record(path) for name, path in native_maps.items()}},
        "templates": {"T1": image_record(arguments.t1_reference),
                      "FA": image_record(arguments.fa_reference),
                      "tensor": image_record(arguments.tensor_reference)},
    }
    environment = os.environ.copy()
    environment.update({"FSLDIR": str(arguments.fsl_dir), "FSLOUTPUTTYPE": "NIFTI_GZ"})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(arguments.threads)
    runner = Runner(output, report, environment)
    runner.save()
    started = time.perf_counter()
    t1_brain, t1_mask = output / "t1_brain.nii.gz", output / "t1_brain_mask.nii.gz"
    t1_matrix, tensor_matrix = output / "t1_to_MNI_affine.mat", output / "dti_FA_to_MNI_affine.mat"
    identity = output / "identity.mat"
    np.savetxt(identity, np.eye(4), fmt="%.12g")
    try:
        with runner.stage("T1_SynthStrip"):
            runner.run([arguments.synthstrip_command, "-i", arguments.t1, "-o", t1_brain,
                        "-m", t1_mask, "--model", arguments.synthstrip_weights,
                        "-b", "1", "-t", arguments.threads], name="synthstrip_T1")
        with runner.stage("T1_FLIRT"):
            runner.run([binaries["flirt"], "-in", t1_brain, "-ref", arguments.t1_reference,
                        "-omat", t1_matrix, "-dof", "12", "-cost", "corratio"], name="flirt_T1")
        with runner.stage("FA_FLIRT"):
            runner.run([binaries["flirt"], "-in", native_fa, "-ref", arguments.fa_reference,
                        "-omat", tensor_matrix, "-dof", "12", "-cost", "corratio"], name="flirt_FA")
        with runner.stage("MMORF_config_preparation"):
            config = output / "single_T1_tensor.ini"
            write_config(config, t1_brain=t1_brain, native_tensor=native_tensor,
                         t1_reference=arguments.t1_reference, tensor_reference=arguments.tensor_reference,
                         t1_matrix=t1_matrix, tensor_matrix=tensor_matrix,
                         identity=identity, output_dir=output)
            report["config_file"] = file_record(config)
        with runner.stage("MMORF_nonlinear"):
            runner.run_mmorf([binaries["mmorf"], "--config", config])
        warp = output / "mmorf_warp.nii.gz"
        with runner.stage("T1_tensor_convenience_resampling"):
            for name, image, affine in (
                ("scalar", t1_brain, t1_matrix), ("tensor", native_tensor, tensor_matrix)
            ):
                runner.run([binaries["applywarp"], f"--in={image}",
                            f"--ref={arguments.t1_reference}", f"--warp={warp}",
                            f"--premat={affine}", "--rel", "--interp=spline", "--datatype=float",
                            f"--out={output / ('mmorf_warped_' + name + '.nii.gz')}"], name="applywarp_" + name)
        with runner.stage("nine_map_propagation"):
            standard = output / "standard"
            standard.mkdir(mode=0o700, exist_ok=True)
            for name, image in native_maps.items():
                runner.run([binaries["applywarp"], f"--in={image}",
                            f"--ref={arguments.t1_reference}", f"--warp={warp}",
                            f"--premat={tensor_matrix}", "--rel", "--interp=trilinear",
                            "--datatype=float", f"--out={standard / (name + '.nii.gz')}"],
                           name="applywarp_" + name)
        report["total_processing_seconds"] = time.perf_counter() - started
        report["outputs"] = {
            "standard": {name: image_record(output / "standard" / f"{name}.nii.gz") for name in MAP_NAMES},
            "warp": image_record(warp), "jacobian": image_record(output / "mmorf_jacobian.nii.gz"),
            "T1_brain": image_record(t1_brain), "T1_mask": image_record(t1_mask),
            "warped_T1": image_record(output / "mmorf_warped_scalar.nii.gz"),
            "warped_tensor": image_record(output / "mmorf_warped_tensor.nii.gz"),
        }
        report["affines"] = {"T1": np.loadtxt(t1_matrix).tolist(),
                              "FA_tensor": np.loadtxt(tensor_matrix).tolist()}
        report["complete"] = True
        runner.save()
    except Exception as error:
        report["failure_type"] = type(error).__name__
        report["total_processing_seconds_until_failure"] = time.perf_counter() - started
        runner.save()
        raise
    print(runner.report_path)


if __name__ == "__main__":
    main()
