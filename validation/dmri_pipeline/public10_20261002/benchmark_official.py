"""Independent original-software raw AP/PA + TBSS or paired-T1 MMORF benchmark.

This validation-only driver freezes the common original FSL/AMICO reference
chain from run_official_end_to_end.py. The TBSS default preserves that chain;
MMORF runs its own official T1 SynthStrip, FLIRT, single-scalar/tensor MMORF
and native FSL applywarp via run_official_mmorf.py. No FNIT runtime is imported.
Use a shared --gpu-lock with the FNIT launcher to serialize this driver's native
GPU EDDY/MMORF. Queue wait is reported separately from subprocess execution.
Private absolute paths and command lines stay in private_logs/ and the nested
MMORF report; publish sanitized numeric reports, never those private records.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")
PE_VECTORS = {
    "i": (1, 0, 0), "i-": (-1, 0, 0),
    "j": (0, 1, 0), "j-": (0, -1, 0),
    "k": (0, 0, 1), "k-": (0, 0, -1),
}
UKB_COMMIT = "0e39a7f7eb76b55437942bfa3073512506b6c8fa"
SYNTHSTRIP_BYTES = 30_851_709
SYNTHSTRIP_SHA256 = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def image_record(path):
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    extensions = io.BytesIO()
    image.header.extensions.write_to(extensions, byteswap=False)
    record = file_record(path)
    record.update({
        "shape": list(image.shape), "dtype": str(image.get_data_dtype()),
        "decoded_dtype": str(values.dtype),
        "finite": bool(np.isfinite(values).all()),
        "affine": np.asarray(image.affine).tolist(),
        "zooms": [float(value) for value in image.header.get_zooms()],
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "header_with_extensions_sha256": hashlib.sha256(
            image.header.binaryblock + extensions.getvalue()
        ).hexdigest(),
        "sha256_decoded": hashlib.sha256(
            np.ascontiguousarray(values).tobytes()
        ).hexdigest(),
        "nonzero_voxels": int(np.count_nonzero(values)),
    })
    return record


def save_image(values, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32),
                            reference.affine, header), str(path))


def gradient_matrix(path, volume_count):
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape == (volume_count, 3) and values.shape != (3, volume_count):
        values = values.T
    if values.shape != (3, volume_count) or not np.isfinite(values).all():
        raise ValueError("gradient matrix must be finite and 3 by volume count")
    return values


def read_acquisition(raw_dir, stem):
    image = nib.load(str(raw_dir / f"{stem}.nii.gz"))
    bvals = np.loadtxt(raw_dir / f"{stem}.bval", dtype=np.float64).reshape(-1)
    if len(image.shape) != 4 or image.shape[3] != bvals.size:
        raise ValueError(f"{stem} image and bval volume counts differ")
    if not np.isfinite(bvals).all():
        raise ValueError(f"{stem} bvals contain non-finite values")
    with (raw_dir / f"{stem}.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    direction = metadata.get("PhaseEncodingDirection")
    if direction not in PE_VECTORS:
        raise ValueError(f"unsupported {stem} phase-encoding direction")
    pe = PE_VECTORS[direction]
    if "TotalReadoutTime" in metadata:
        readout = float(metadata["TotalReadoutTime"])
        readout_source = "TotalReadoutTime"
    elif "EffectiveEchoSpacing" in metadata:
        axis = next(i for i, value in enumerate(pe) if value)
        readout = float(metadata["EffectiveEchoSpacing"]) * (image.shape[axis] - 1)
        readout_source = "EffectiveEchoSpacing times phase-axis size minus one"
    else:
        raise ValueError(f"{stem} JSON needs TotalReadoutTime or EffectiveEchoSpacing")
    if not math.isfinite(readout) or readout <= 0:
        raise ValueError(f"{stem} readout time must be finite and positive")
    readout = math.floor(readout * 10000.0) / 10000.0
    indices = np.flatnonzero(bvals < 100)
    if indices.size == 0:
        raise ValueError(f"{stem} has no b<100 volume")
    return image, bvals, indices, pe, readout, readout_source


def gpu_state():
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True)
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


class ReferenceRun:
    def __init__(self, output_dir, environment, gpu_lock=None):
        self.output_dir = output_dir
        self.environment = environment
        self.logs = output_dir / "private_logs"
        self.logs.mkdir()
        self.stages = {}
        self.commands = []
        self.command_number = 0
        self.gpu_lock = gpu_lock
        self.gpu_queue_seconds = 0.0
        self.stages_gpu_queue_seconds = {}

    @contextmanager
    def gpu_slot(self, enabled):
        if not enabled or self.gpu_lock is None:
            yield 0.0
            return
        started = time.perf_counter()
        descriptor = os.open(self.gpu_lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            waited = time.perf_counter() - started
            self.gpu_queue_seconds += waited
            yield waited
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def run(self, arguments, *, name, environment=None, capture=False, gpu=False):
        self.command_number += 1
        label = f"{self.command_number:03d}_{name}"
        command = [os.fspath(value) for value in arguments]
        (self.logs / f"{label}.command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        with self.gpu_slot(gpu) as queue_seconds:
            started = time.perf_counter()
            with (self.logs / f"{label}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=self.output_dir, env=environment or self.environment,
                    stdout=log, stderr=subprocess.STDOUT, check=False, text=True,
                )
            seconds = time.perf_counter() - started
        self.commands.append({"step": name, "program": Path(command[0]).name,
                              "seconds": seconds, "gpu_queue_seconds": queue_seconds,
                              "uses_gpu": bool(gpu), "returncode": completed.returncode})
        if completed.returncode:
            raise RuntimeError(f"reference command {name} failed with exit code {completed.returncode}")
        if capture:
            return (self.logs / f"{label}.log").read_text(encoding="utf-8")
        return None

    @contextmanager
    def stage(self, name):
        print(json.dumps({"event": "stage_started", "stage": name}), flush=True)
        started = time.perf_counter()
        queue_before = self.gpu_queue_seconds
        try:
            yield
        finally:
            self.stages_gpu_queue_seconds[name] = self.gpu_queue_seconds - queue_before
            self.stages[name] = time.perf_counter() - started
            print(json.dumps({"event": "stage_finished", "stage": name,
                              "seconds": self.stages[name]}), flush=True)


def select_best_b0(run, fsl_bin, image, indices, stem):
    candidates_dir = run.output_dir / "topup" / f"{stem}_b0_candidates"
    candidates_dir.mkdir()
    # One compressed input read; no historical b0 image or selection is reused.
    all_values = np.asarray(image.dataobj, dtype=np.float32)
    paths = []
    for position, original_index in enumerate(indices):
        path = candidates_dir / f"b0_{position:04d}.nii.gz"
        save_image(all_values[..., int(original_index)], image, path)
        paths.append(path)
    del all_values
    scores = np.zeros(len(paths), dtype=np.float64)
    pairs = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            moved = candidates_dir / f"registered_{left:04d}_{right:04d}.nii.gz"
            matrix = candidates_dir / f"registered_{left:04d}_{right:04d}.mat"
            run.run([fsl_bin / "flirt", "-in", paths[left], "-ref", paths[right],
                     "-nosearch", "-dof", "6", "-o", moved, "-omat", matrix],
                    name=f"{stem}_b0_flirt_{left}_{right}")
            text = run.run([fsl_bin / "fslcc", "-t", "-1", "-p", "10", paths[right], moved],
                           name=f"{stem}_b0_fslcc_{left}_{right}", capture=True)
            fields = text.strip().split()
            if len(fields) != 3:
                raise ValueError("expected one scalar fslcc correlation for two 3D b0 images")
            correlation = float(fields[2])
            if not math.isfinite(correlation):
                raise ValueError("non-finite pairwise b0 correlation")
            scores[left] += correlation
            scores[right] += correlation
            pairs.append({"left_original_index": int(indices[left]),
                          "right_original_index": int(indices[right]),
                          "correlation": correlation})
    if len(paths) > 1:
        scores /= len(paths) - 1
    else:
        scores[:] = 1.0
    chosen = 0 if scores[0] >= 0.98 else int(np.argmax(scores))
    selected_path = run.output_dir / "topup" / f"B0_{stem}.nii.gz"
    shutil.copyfile(paths[chosen], selected_path)
    return selected_path, {
        "b0_original_indices": list(map(int, indices)),
        "pairwise_correlations": pairs, "mean_correlations": scores.tolist(),
        "selected_candidate_position": chosen,
        "selected_original_index": int(indices[chosen]),
        "best_score_below_0_95": bool(scores[chosen] < 0.95),
    }


def map_output_path(output_dir, backend, group, name):
    """Locate independently generated reference maps without sharing FNIT I/O."""
    output_dir = Path(output_dir)
    if name not in MAP_NAMES or backend not in ("tbss", "mmorf"):
        raise ValueError("unknown benchmark map or registration backend")
    if group == "native":
        prefix = "NODDI_" if name in ("ICVF", "OD", "ISOVF") else "dti_"
        return output_dir / "native" / f"{prefix}{name}.nii.gz"
    if group == "standard":
        return (output_dir / "tbss/stats" / f"all_{name}.nii.gz" if backend == "tbss"
                else output_dir / "mmorf/standard" / f"{name}.nii.gz")
    if group == "skeleton" and backend == "tbss":
        return output_dir / "tbss/stats" / f"all_{name}_skeletonised.nii.gz"
    raise ValueError("MMORF has no skeleton output; unknown map group")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case-id", required=True, help="public report alias case01 through case10")
    result.add_argument("--registration-backend", choices=("tbss", "mmorf"), default="tbss")
    result.add_argument("--gpu-lock", type=Path,
                        help="shared flock file also held by the FNIT full-process launcher")
    result.add_argument("--t1", type=Path, help="original paired T1, required for MMORF")
    result.add_argument("--t1-reference", type=Path, help="MNI brain template, required for MMORF")
    result.add_argument("--tensor-reference", type=Path, help="six-component FSL tensor template, required for MMORF")
    result.add_argument("--mmorf-runner", type=Path, help="independent run_official_mmorf.py, required for MMORF")
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True,
                        help="new root; fails if it already exists")
    result.add_argument("--fsl-dir", type=Path, required=True)
    result.add_argument("--topup-config", type=Path, required=True)
    result.add_argument("--fa-reference", type=Path, required=True)
    result.add_argument("--fa-skeleton", type=Path)
    result.add_argument("--config-prefix", type=Path,
                        help="absolute prefix for oxford_s1/s2/s3.cnf")
    result.add_argument("--amico-runner", type=Path, required=True)
    result.add_argument("--tbss-runner", type=Path)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--seed", type=int, default=12345)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--bet-fraction", type=float, default=0.2)
    result.add_argument("--brain-extractor", choices=("bet", "synthstrip"), default="synthstrip",
                        help="independent official extractor on this reference's TOPUP mean")
    result.add_argument("--synthstrip-command", type=Path,
                        help="official mri_synthstrip executable; required for synthstrip")
    result.add_argument("--synthstrip-weights", type=Path,
                        help="verified official synthstrip.1.pt; required for synthstrip")
    result.add_argument("--synthstrip-gpu", action="store_true",
                        help="pass -g to the official SynthStrip reference")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if not re.fullmatch(r"case(?:0[1-9]|10)", arguments.case_id):
        raise ValueError("case-id must be case01 through case10")
    branch_fields = (("fa_skeleton", "config_prefix", "tbss_runner")
                     if arguments.registration_backend == "tbss" else
                     ("t1", "t1_reference", "tensor_reference", "mmorf_runner"))
    if any(getattr(arguments, field) is None for field in branch_fields):
        raise ValueError("selected registration branch is missing a required argument")
    if arguments.registration_backend == "mmorf" and arguments.brain_extractor != "synthstrip":
        raise ValueError("paired-T1 MMORF reference requires official SynthStrip")
    if not 1 <= arguments.seed <= 2**32 - 1:
        raise ValueError("seed must be in [1, 2**32-1]")
    if arguments.threads < 1:
        raise ValueError("threads must be positive")
    if not 0 < arguments.bet_fraction < 1:
        raise ValueError("BET fraction must be between zero and one")
    for field in ("raw_dir", "output_dir", "fsl_dir", "topup_config", "fa_reference",
                  "amico_runner", "report", *branch_fields):
        setattr(arguments, field, getattr(arguments, field).expanduser().resolve())
    if arguments.gpu_lock is not None:
        arguments.gpu_lock = arguments.gpu_lock.expanduser().resolve()
        if not arguments.gpu_lock.parent.is_dir():
            raise FileNotFoundError("shared GPU lock parent directory is absent")
    if arguments.brain_extractor == "synthstrip":
        if arguments.synthstrip_command is None or arguments.synthstrip_weights is None:
            raise ValueError("SynthStrip reference needs its official command and verified weights")
        for field in ("synthstrip_command", "synthstrip_weights"):
            setattr(arguments, field, getattr(arguments, field).expanduser().resolve())
    if arguments.output_dir.exists() or arguments.report.exists():
        raise FileExistsError("output root and report must both be new")
    fsl_bin = arguments.fsl_dir / "bin"
    binaries = {name: fsl_bin / name for name in (
        "flirt", "fslcc", "topup", "fslmaths", "bet", "eddy_cuda10.2", "dtifit",
        "fnirt", "applywarp", "imcp",
    )}
    if arguments.registration_backend == "mmorf":
        binaries["mmorf"] = fsl_bin / "mmorf"
    # BET is a wrapper around bet2 in this FSL distribution.
    binary_records = dict(binaries)
    if (fsl_bin / "bet2").is_file():
        binary_records["bet2"] = fsl_bin / "bet2"
    required = [arguments.raw_dir / f"{stem}.{extension}"
                for stem, extensions in (("AP", ("nii.gz", "bval", "bvec", "json")),
                                         ("PA", ("nii.gz", "bval", "json")))
                for extension in extensions]
    required += [*binaries.values(), arguments.topup_config, arguments.fa_reference,
                 arguments.amico_runner, arguments.fsl_dir / "etc/fslversion"]
    if arguments.registration_backend == "tbss":
        oxford = {f"oxford_{stage}.cnf": Path(f"{arguments.config_prefix}_{stage}.cnf")
                  for stage in ("s1", "s2", "s3")}
        required += [arguments.fa_skeleton, arguments.tbss_runner, *oxford.values()]
    else:
        oxford = {}
        required += [arguments.t1, arguments.t1_reference, arguments.tensor_reference,
                     arguments.mmorf_runner]
    if arguments.brain_extractor == "synthstrip":
        required += [arguments.synthstrip_command, arguments.synthstrip_weights]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("a required raw input, binary, runner, template or configuration is absent")
    if arguments.brain_extractor == "synthstrip" and file_record(arguments.synthstrip_weights) != {
            "bytes": SYNTHSTRIP_BYTES, "sha256": SYNTHSTRIP_SHA256}:
        raise ValueError("official reference requires the verified standard SynthStrip checkpoint")
    version = (arguments.fsl_dir / "etc/fslversion").read_text().strip()
    if version != "6.0.7.4":
        raise ValueError("this benchmark requires FSL 6.0.7.4")
    for path in binaries.values():
        if not os.access(path, os.X_OK):
            raise ValueError("a required FSL program is not executable")
    environment = os.environ.copy()
    environment.update({"FSLDIR": str(arguments.fsl_dir), "FSLOUTPUTTYPE": "NIFTI_GZ"})
    for field in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
        environment[field] = str(arguments.threads)
    environment["PATH"] = str(fsl_bin) + os.pathsep + environment.get("PATH", "")
    if arguments.brain_extractor == "synthstrip":
        # The installed official wrapper invokes this distribution's fspython.
        fs_root = arguments.synthstrip_command.parent.parent
        if (fs_root / "bin/fspython").is_file():
            environment["FREESURFER_HOME"] = str(fs_root)
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("topup", "eddy", "native"):
        (arguments.output_dir / name).mkdir()
    run = ReferenceRun(arguments.output_dir, environment, arguments.gpu_lock)
    report = {
        "schema_version": 2, "case_id": arguments.case_id,
        "registration_backend": arguments.registration_backend,
        "reference": "independent FSL 6.0.7.4 and Python AMICO 2.0.3 raw AP/PA chain",
        "unmodified_UKB_v1_5": False, "official_equivalence_claim": False,
        "UKB_command_source_commit": UKB_COMMIT,
        "historical_UKB_adaptations": [
            f"T1-derived mask replaced by independent official {arguments.brain_extractor} of this reference TOPUP iout mean",
            "compiled MCR AMICO replaced by official Python AMICO 2.0.3",
            "raw downstream gradients replaced by this reference EDDY rotated gradients",
            "fixed EDDY initrand for reproducible comparison; historical wrap used time initialization",
            "all b0 candidates use audited first-if-0.98-else-best rule; not old driver -n1 behavior",
            "no GDC coefficients or GDC processing; matches the FNIT no-GDC scope",
        ],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "nibabel": nib.__version__, "fsl_version": version,
                        "threads": arguments.threads,
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
                        "FSL_thread_environment": {key: environment[key] for key in (
                            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        "AMICO_BLAS_threads": 1},
        "parameters": {"gp_seed": arguments.seed, "eddy_iterations": 8,
                       "eddy_fwhm_mm": [10, 8, 4, 2, 0, 0, 0, 0],
                       "eddy_flm": "quadratic", "eddy_resamp": "jac", "eddy_slm": "linear",
                       "eddy_ff": 10, "eddy_nvoxhp": 1000, "eddy_repol": True,
                       "eddy_sep_offs_move": True, "bet_fraction": arguments.bet_fraction,
                       "brain_extractor": arguments.brain_extractor,
                       "dti_shell": 1000, "dti_tolerance": 100,
                       "bvec_source": "own EDDY rotated", "noddi_fit_method": "AMICO",
                       "skeleton_threshold": 2000 if arguments.registration_backend == "tbss" else None,
                       "MMORF_modalities": ["single T1 scalar weight 1", "tensor weight 1"]
                       if arguments.registration_backend == "mmorf" else None,
                       "shared_gpu_lock_enabled": arguments.gpu_lock is not None},
        "input_files": {}, "FSL_binaries": {name: file_record(path) for name, path in binary_records.items()},
        "configurations": {"b02b0.cnf": file_record(arguments.topup_config),
                           **{name: file_record(path) for name, path in oxford.items()}},
        "templates": {"FA_reference": file_record(arguments.fa_reference)},
        "runner_sources": {"end_to_end": file_record(Path(__file__)),
                           "AMICO": file_record(arguments.amico_runner)},
        "gpu_before": gpu_state(),
        "timing_scope": "processing starts at raw image/JSON preparation; ends after all native/standard and TBSS skeleton saves; outer report audits and cross-pipeline comparisons are outside; nested MMORF child preflight and output audit are inside the outer process timing",
        "gpu_queue_scope": "native EDDY and whole MMORF child share flock with FNIT; queue wait is excluded from subprocess seconds and reported separately from stage/root wall times",
        "process_wall_note": "measure whole launcher separately to include Python imports, preflight, executable hashing and final report generation",
    }
    if arguments.registration_backend == "tbss":
        report["templates"]["FA_skeleton"] = file_record(arguments.fa_skeleton)
        report["runner_sources"]["TBSS"] = file_record(arguments.tbss_runner)
    else:
        report["templates"].update({"T1_reference": file_record(arguments.t1_reference),
                                    "tensor_reference": file_record(arguments.tensor_reference)})
        report["runner_sources"]["MMORF"] = file_record(arguments.mmorf_runner)
        report["input_files"]["T1w.nii.gz"] = file_record(arguments.t1)
    if arguments.brain_extractor == "synthstrip":
        report["SynthStrip"] = {
            "command": file_record(arguments.synthstrip_command),
            "weights": file_record(arguments.synthstrip_weights),
            "input": "own TOPUP iout arithmetic mean; original signal units",
            "border_mm": 1, "no_csf": False, "gpu": arguments.synthstrip_gpu,
        }
        official_script = arguments.synthstrip_command.parent.parent / "python/scripts/mri_synthstrip"
        if official_script.is_file():
            report["SynthStrip"]["python_script"] = file_record(official_script)
    for stem in ("AP", "PA"):
        for extension in ("nii.gz", "bval", "bvec", "json"):
            path = arguments.raw_dir / f"{stem}.{extension}"
            if path.is_file():
                report["input_files"][path.name] = file_record(path)
    processing_started = time.perf_counter()
    try:
        with run.stage("raw_preparation_and_b0_selection"):
            ap, bvals, ap_indices, ap_pe, ap_readout, ap_source = read_acquisition(arguments.raw_dir, "AP")
            pa, pa_bvals, pa_indices, pa_pe, pa_readout, pa_source = read_acquisition(arguments.raw_dir, "PA")
            gradient_matrix(arguments.raw_dir / "AP.bvec", bvals.size)
            if ap.shape[:3] != pa.shape[:3] or not np.allclose(ap.affine, pa.affine, atol=5e-4, rtol=0):
                raise ValueError("AP and PA must share spatial matrix and affine")
            if np.dot(ap_pe, pa_pe) != -1:
                raise ValueError("AP and PA must have opposite phase-encoding vectors")
            b0_ap, ap_choice = select_best_b0(run, fsl_bin, ap, ap_indices, "AP")
            b0_pa, pa_choice = select_best_b0(run, fsl_bin, pa, pa_indices, "PA")
            selected = np.stack((np.asarray(nib.load(str(b0_ap)).dataobj, dtype=np.float32),
                                 np.asarray(nib.load(str(b0_pa)).dataobj, dtype=np.float32)), axis=3)
            odd_z_cropped = bool(selected.shape[2] % 2)
            if odd_z_cropped:
                # TOPUP's two-frame schedule needs an even z size. The current
                # public cohort has 68 slices; fail instead of silently mismatching
                # its untrimmed EDDY input and a cropped mask.
                raise ValueError("odd-z AP/PA is outside this matched full-chain benchmark")
            topup_pair = arguments.output_dir / "topup/B0_AP_PA.nii.gz"
            save_image(selected, ap, topup_pair)
            acqp = arguments.output_dir / "topup/acqparams.txt"
            rows = np.asarray([(*ap_pe, ap_readout), (*pa_pe, pa_readout)], dtype=np.float64)
            np.savetxt(acqp, rows, fmt=("%g", "%g", "%g", "%.7g"))
            index = arguments.output_dir / "eddy/eddy_index.txt"
            np.savetxt(index, np.ones((1, bvals.size), dtype=np.int64), fmt="%d")
            reference_index = ap_choice["selected_original_index"]
            report["choices"] = {"AP": ap_choice, "PA": pa_choice,
                                 "AP_shape": list(ap.shape), "PA_shape": list(pa.shape),
                                 "AP_volume_counts": {
                                     "b0": int(np.count_nonzero(bvals < 100)),
                                     "b1000": int(np.count_nonzero(np.abs(bvals - 1000) < 100)),
                                     "b2000": int(np.count_nonzero(np.abs(bvals - 2000) < 100)),
                                     "b3000": int(np.count_nonzero(np.abs(bvals - 3000) < 100)),
                                 },
                                 "PA_b0_count": int(pa_indices.size),
                                 "acquisition_rows": rows.tolist(), "eddy_index_value": 1,
                                 "eddy_index_length": int(bvals.size),
                                 "reference_scan_no": reference_index,
                                 "readout_sources": {"AP": ap_source, "PA": pa_source},
                                 "odd_z_cropped": odd_z_cropped,
                                 "b0_selector_registration": "FLIRT -nosearch -dof 6; default corratio",
                                 "b0_selector_score": "fslcc -t -1 -p 10; mean of N-1 pairwise correlations",
                                 "b0_selector_rule": "first mean>=0.98, otherwise maximum mean"}
        with run.stage("topup"):
            field_prefix = arguments.output_dir / "topup/fieldmap_out"
            run.run([fsl_bin / "topup", f"--imain={topup_pair}", f"--datain={acqp}",
                     f"--config={arguments.topup_config}", f"--out={field_prefix}",
                     f"--fout={arguments.output_dir / 'topup/fieldmap_fout'}",
                     f"--iout={arguments.output_dir / 'topup/fieldmap_iout'}",
                     f"--jacout={arguments.output_dir / 'topup/fieldmap_jacout'}"], name="topup")
        mask_stage = ("independent_SynthStrip_mask" if arguments.brain_extractor == "synthstrip"
                      else "independent_BET_mask")
        with run.stage(mask_stage):
            mean_b0 = arguments.output_dir / "topup/fieldmap_iout_mean.nii.gz"
            brain_prefix = arguments.output_dir / "eddy/nodif_brain"
            run.run([fsl_bin / "fslmaths", arguments.output_dir / "topup/fieldmap_iout.nii.gz",
                     "-Tmean", mean_b0], name="topup_iout_mean")
            mask = arguments.output_dir / "eddy/nodif_brain_mask.nii.gz"
            if arguments.brain_extractor == "synthstrip":
                command = [arguments.synthstrip_command, "-i", mean_b0,
                           "-m", mask,
                           "--model", arguments.synthstrip_weights, "-b", "1",
                           "-t", str(arguments.threads)]
                if arguments.synthstrip_gpu:
                    command.append("-g")
                run.run(command, name="independent_official_synthstrip", gpu=arguments.synthstrip_gpu)
            else:
                run.run([fsl_bin / "bet", mean_b0, brain_prefix, "-m", "-f", str(arguments.bet_fraction)],
                        name="independent_bet")
            mask_image = nib.load(str(mask))
            if (mask_image.shape[:3] != ap.shape[:3]
                    or not np.allclose(mask_image.affine, ap.affine, atol=5e-4, rtol=0)
                    or not np.any(np.asanyarray(mask_image.dataobj) > 0)):
                raise ValueError("brain mask is empty or has different DWI geometry")
        with run.stage("eddy"):
            eddy_prefix = arguments.output_dir / "eddy/data"
            run.run([fsl_bin / "eddy_cuda10.2", f"--imain={arguments.raw_dir / 'AP.nii.gz'}",
                     f"--mask={mask}", f"--topup={field_prefix}", f"--acqp={acqp}", f"--index={index}",
                     f"--bvecs={arguments.raw_dir / 'AP.bvec'}", f"--bvals={arguments.raw_dir / 'AP.bval'}",
                     f"--out={eddy_prefix}", f"--ref_scan_no={reference_index}", f"--initrand={arguments.seed}",
                     "--flm=quadratic", "--resamp=jac", "--slm=linear", "--niter=8",
                     "--fwhm=10,8,4,2,0,0,0,0", "--ff=10", "--sep_offs_move", "--nvoxhp=1000",
                     "--repol", "--rms"], name="eddy", gpu=True)
            corrected_path = arguments.output_dir / "eddy/data.nii.gz"
            rotated_path = arguments.output_dir / "eddy/data.eddy_rotated_bvecs"
        with run.stage("shell_selection_and_dtifit"):
            corrected = nib.load(str(corrected_path))
            if corrected.shape != ap.shape:
                raise ValueError("corrected DWI matrix differs from raw AP")
            gradients = gradient_matrix(rotated_path, bvals.size)
            norms = np.linalg.norm(gradients, axis=0)
            nonzero = norms > 0
            gradients[:, nonzero] /= norms[nonzero]
            keep = (bvals < 100) | (np.abs(bvals - 1000) < 100)
            shell_path = arguments.output_dir / "native/data_1_shell.nii.gz"
            shell_bval = arguments.output_dir / "native/data_1_shell.bval"
            shell_bvec = arguments.output_dir / "native/data_1_shell.bvec"
            save_image(np.asarray(corrected.dataobj, dtype=np.float32)[..., keep], corrected, shell_path)
            np.savetxt(shell_bval, bvals[keep][None], fmt="%.10g")
            np.savetxt(shell_bvec, gradients[:, keep], fmt="%.10g")
            report["choices"]["DTI_original_indices"] = np.flatnonzero(keep).tolist()
            report["choices"]["DTI_gradients"] = "float64 unit-normalize nonzero rotated columns, select original order, save %.10g"
            report["choices"]["NODDI_gradients"] = "original complete EDDY rotated text, no additional pre-normalization"
            run.run([fsl_bin / "dtifit", "-k", shell_path, "-m", mask, "-r", shell_bvec,
                     "-b", shell_bval, "-o", arguments.output_dir / "native/dti", "--save_tensor"],
                    name="dtifit")
        with run.stage("AMICO_full_fit_and_save"):
            amico_environment = environment.copy()
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                amico_environment[name] = "1"
            run.run([sys.executable, arguments.amico_runner, "--study-dir", arguments.output_dir / "amico",
                     "--data", corrected_path, "--mask", mask,
                     "--bvals", arguments.raw_dir / "AP.bval", "--bvecs", rotated_path,
                     "--native-output", arguments.output_dir / "native",
                     "--report", arguments.output_dir / "amico_report.json", "--threads", str(arguments.threads)],
                    name="amico", environment=amico_environment)
        if arguments.registration_backend == "tbss":
            with run.stage("FA_preprocessing_and_weight"):
                fa_path = arguments.output_dir / "native/dti_FA.nii.gz"
                fa_shape = nib.load(str(fa_path)).shape[:3]
                if min(fa_shape) < 3:
                    raise ValueError("FA dimensions must permit UKB end-slice cropping")
                prepared_fa = arguments.output_dir / "native/dti_FA_preprocessed.nii.gz"
                weight = arguments.output_dir / "native/dti_FA_registration_weight.nii.gz"
                run.run([fsl_bin / "fslmaths", fa_path, "-min", "1", "-ero", "-roi",
                         "1", str(fa_shape[0] - 2), "1", str(fa_shape[1] - 2),
                         "1", str(fa_shape[2] - 2), "0", "1", prepared_fa], name="FA_preprocess")
                run.run([fsl_bin / "fslmaths", prepared_fa, "-bin", weight], name="FA_weight_mask")
                run.run([fsl_bin / "fslmaths", weight, "-dilD", "-dilD", "-sub", "1", "-abs",
                         "-add", weight, weight, "-odt", "char"], name="FA_weight")
            with run.stage("FLIRT_FNIRT_and_nine_map_propagation"):
                run.run(["/bin/bash", arguments.tbss_runner, arguments.output_dir / "tbss",
                         prepared_fa, weight, arguments.output_dir / "native", arguments.fa_reference,
                         arguments.fa_skeleton, arguments.config_prefix], name="TBSS")
        else:
            with run.stage("T1_FLIRT_MMORF_and_nine_map_propagation"):
                run.run([sys.executable, arguments.mmorf_runner,
                         "--t1", arguments.t1, "--native-dir", arguments.output_dir / "native",
                         "--t1-reference", arguments.t1_reference,
                         "--fa-reference", arguments.fa_reference,
                         "--tensor-reference", arguments.tensor_reference,
                         "--output-dir", arguments.output_dir / "mmorf",
                         "--fsl-dir", arguments.fsl_dir,
                         "--synthstrip-command", arguments.synthstrip_command,
                         "--synthstrip-weights", arguments.synthstrip_weights,
                         "--threads", str(arguments.threads)], name="MMORF", gpu=True)
        report["total_processing_seconds"] = time.perf_counter() - processing_started
        report["stages_seconds"] = run.stages
        report["gpu_queue_seconds"] = run.gpu_queue_seconds
        report["stages_gpu_queue_seconds"] = run.stages_gpu_queue_seconds
        report["stages_seconds_excluding_gpu_queue"] = {
            name: seconds - run.stages_gpu_queue_seconds.get(name, 0.0)
            for name, seconds in run.stages.items()}
        report["total_processing_seconds_excluding_gpu_queue"] = (
            report["total_processing_seconds"] - run.gpu_queue_seconds)
        report["subprocess_steps"] = run.commands
        report["gpu_after"] = gpu_state()
        amico_report = json.loads((arguments.output_dir / "amico_report.json").read_text())
        report["AMICO"] = {key: amico_report[key] for key in (
            "versions", "reference_commit", "input_shape", "mask_voxels", "scheme",
            "config", "model", "cache", "stages_seconds", "wall_seconds_including_io",
            "official_solver_seconds", "official_direction_seconds", "timing_scope",
        )}
        tbss_timing_path = arguments.output_dir / "tbss/stage_times.json"
        if tbss_timing_path.is_file():
            report["TBSS_stages_seconds"] = json.loads(tbss_timing_path.read_text())
        if arguments.registration_backend == "mmorf":
            nested_mmorf = json.loads((arguments.output_dir / "mmorf/official_mmorf_report.json").read_text())
            report["MMORF"] = {key: nested_mmorf[key] for key in (
                "stages_seconds", "total_processing_seconds", "complete", "limitations")
                if key in nested_mmorf}
            report["MMORF"]["native_gpu_observations"] = [
                {key: step[key] for key in (
                    "name", "seconds", "exit_code", "gpu_process_samples",
                    "sampled_process_gpu_memory_peak_mib", "memory_scope") if key in step}
                for step in nested_mmorf.get("subprocess_steps", []) if "gpu_process_samples" in step]
        report["outputs"] = {"native": {}, "standard": {}, "skeleton": {}}
        for name in MAP_NAMES:
            prefix = "NODDI_" if name in ("ICVF", "OD", "ISOVF") else "dti_"
            report["outputs"]["native"][name] = image_record(arguments.output_dir / "native" / f"{prefix}{name}.nii.gz")
            report["outputs"]["standard"][name] = image_record(
                map_output_path(arguments.output_dir, arguments.registration_backend, "standard", name))
            if arguments.registration_backend == "tbss":
                report["outputs"]["skeleton"][name] = image_record(
                    map_output_path(arguments.output_dir, arguments.registration_backend, "skeleton", name))
        report["preparation_outputs"] = {
            "topup_pair": image_record(topup_pair), "acqp": file_record(acqp), "index": file_record(index),
            "brain_mask": image_record(mask),
        }
        if arguments.registration_backend == "tbss":
            report["preparation_outputs"].update({
                "preprocessed_FA": image_record(prepared_fa),
                "registration_weight": image_record(weight),
                "valid_FA_mask": image_record(arguments.output_dir / "tbss/stats/mean_FA_mask.nii.gz"),
                "skeleton_mask": image_record(arguments.output_dir / "tbss/stats/mean_FA_skeleton_mask.nii.gz"),
            })
        else:
            report["preparation_outputs"].update({
                "T1_brain": image_record(arguments.output_dir / "mmorf/t1_brain.nii.gz"),
                "T1_brain_mask": image_record(arguments.output_dir / "mmorf/t1_brain_mask.nii.gz"),
                "MMORF_warp": image_record(arguments.output_dir / "mmorf/mmorf_warp.nii.gz"),
                "MMORF_jacobian": image_record(arguments.output_dir / "mmorf/mmorf_jacobian.nii.gz"),
            })
        report["intermediate_outputs"] = {
            "topup_fieldcoef": image_record(arguments.output_dir / "topup/fieldmap_out_fieldcoef.nii.gz"),
            "topup_movpar": file_record(arguments.output_dir / "topup/fieldmap_out_movpar.txt"),
            "topup_fout": image_record(arguments.output_dir / "topup/fieldmap_fout.nii.gz"),
            "topup_iout": image_record(arguments.output_dir / "topup/fieldmap_iout.nii.gz"),
            "eddy_corrected": image_record(corrected_path), "eddy_rotated_bvecs": file_record(rotated_path),
            "DTI_shell": image_record(shell_path), "DTI_shell_bval": file_record(shell_bval),
            "DTI_shell_bvec": file_record(shell_bvec),
        }
        report["parameter_map_count"] = sum(len(group) for group in report["outputs"].values())
        expected_count = 27 if arguments.registration_backend == "tbss" else 18
        report["all_maps_saved_and_finite"] = (
            report["parameter_map_count"] == expected_count and all(
                value["finite"] for group in report["outputs"].values() for value in group.values()))
        if arguments.registration_backend == "tbss":
            report["all_27_maps_saved_and_finite"] = report["all_maps_saved_and_finite"]
        report["status"] = "complete" if report["all_maps_saved_and_finite"] else "nonfinite_outputs"
    except Exception as error:
        report.update(status="failed", failure_type=type(error).__name__,
                      total_processing_seconds=time.perf_counter() - processing_started,
                      stages_seconds=run.stages, subprocess_steps=run.commands,
                      gpu_queue_seconds=run.gpu_queue_seconds,
                      stages_gpu_queue_seconds=run.stages_gpu_queue_seconds)
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "failed", "error_type": type(error).__name__}), flush=True)
        raise
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "total_processing_seconds": report["total_processing_seconds"],
                      "parameter_map_count": report["parameter_map_count"],
                      "all_maps_saved_and_finite": report["all_maps_saved_and_finite"]}), flush=True)
    if not report["all_maps_saved_and_finite"]:
        raise RuntimeError("a saved parameter map contains non-finite values")


if __name__ == "__main__":
    main()
