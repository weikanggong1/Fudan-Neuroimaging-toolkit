"""Independent FSL/AMICO raw AP/PA benchmark; never imported by FNIT runtime.

Run in the official Python AMICO environment. This reference deliberately uses
its own TOPUP field, BET mask, EDDY output, fits and registration. It adapts the
UKB v1.5 command rules to a no-T1, rotated-gradient, Python-AMICO comparison.
Actual commands and program output are stored only in private_logs/; the report
contains role names, hashes and numeric summaries, not private absolute paths.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
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
    def __init__(self, output_dir, environment):
        self.output_dir = output_dir
        self.environment = environment
        self.logs = output_dir / "private_logs"
        self.logs.mkdir()
        self.stages = {}
        self.commands = []
        self.command_number = 0

    def run(self, arguments, *, name, environment=None, capture=False):
        self.command_number += 1
        label = f"{self.command_number:03d}_{name}"
        command = [os.fspath(value) for value in arguments]
        (self.logs / f"{label}.command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        started = time.perf_counter()
        with (self.logs / f"{label}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=self.output_dir, env=environment or self.environment,
                stdout=log, stderr=subprocess.STDOUT, check=False, text=True,
            )
        seconds = time.perf_counter() - started
        self.commands.append({"step": name, "program": Path(command[0]).name,
                              "seconds": seconds, "returncode": completed.returncode})
        if completed.returncode:
            raise RuntimeError(f"reference command {name} failed with exit code {completed.returncode}")
        if capture:
            return (self.logs / f"{label}.log").read_text(encoding="utf-8")
        return None

    @contextmanager
    def stage(self, name):
        print(json.dumps({"event": "stage_started", "stage": name}), flush=True)
        started = time.perf_counter()
        try:
            yield
        finally:
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True,
                        help="new root; fails if it already exists")
    result.add_argument("--fsl-dir", type=Path, required=True)
    result.add_argument("--topup-config", type=Path, required=True)
    result.add_argument("--fa-reference", type=Path, required=True)
    result.add_argument("--fa-skeleton", type=Path, required=True)
    result.add_argument("--config-prefix", type=Path, required=True,
                        help="absolute prefix for oxford_s1/s2/s3.cnf")
    result.add_argument("--amico-runner", type=Path, required=True)
    result.add_argument("--tbss-runner", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--seed", type=int, default=12345)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--bet-fraction", type=float, default=0.2)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if not 1 <= arguments.seed <= 2**32 - 1:
        raise ValueError("seed must be in [1, 2**32-1]")
    if arguments.threads < 1:
        raise ValueError("threads must be positive")
    if not 0 < arguments.bet_fraction < 1:
        raise ValueError("BET fraction must be between zero and one")
    for field in ("raw_dir", "output_dir", "fsl_dir", "topup_config", "fa_reference",
                  "fa_skeleton", "config_prefix", "amico_runner", "tbss_runner", "report"):
        setattr(arguments, field, getattr(arguments, field).expanduser().resolve())
    if arguments.output_dir.exists() or arguments.report.exists():
        raise FileExistsError("output root and report must both be new")
    fsl_bin = arguments.fsl_dir / "bin"
    binaries = {name: fsl_bin / name for name in (
        "flirt", "fslcc", "topup", "fslmaths", "bet", "eddy_cuda10.2", "dtifit",
        "fnirt", "applywarp", "imcp",
    )}
    # BET is a wrapper around bet2 in this FSL distribution.
    binary_records = dict(binaries)
    if (fsl_bin / "bet2").is_file():
        binary_records["bet2"] = fsl_bin / "bet2"
    required = [arguments.raw_dir / f"{stem}.{extension}"
                for stem, extensions in (("AP", ("nii.gz", "bval", "bvec", "json")),
                                         ("PA", ("nii.gz", "bval", "json")))
                for extension in extensions]
    required += [*binaries.values(), arguments.topup_config, arguments.fa_reference,
                 arguments.fa_skeleton, arguments.amico_runner, arguments.tbss_runner,
                 arguments.fsl_dir / "etc/fslversion"]
    oxford = {f"oxford_{stage}.cnf": Path(f"{arguments.config_prefix}_{stage}.cnf")
              for stage in ("s1", "s2", "s3")}
    required += list(oxford.values())
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("a required raw input, binary, runner, template or configuration is absent")
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
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("topup", "eddy", "native"):
        (arguments.output_dir / name).mkdir()
    run = ReferenceRun(arguments.output_dir, environment)
    report = {
        "schema_version": 1, "reference": "independent FSL 6.0.7.4 and Python AMICO 2.0.3 raw AP/PA chain",
        "unmodified_UKB_v1_5": False, "official_equivalence_claim": False,
        "UKB_command_source_commit": UKB_COMMIT,
        "historical_UKB_adaptations": [
            "T1-derived mask replaced by independent BET of this reference TOPUP iout mean",
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
                       "dti_shell": 1000, "dti_tolerance": 100,
                       "bvec_source": "own EDDY rotated", "noddi_fit_method": "AMICO",
                       "skeleton_threshold": 2000},
        "input_files": {}, "FSL_binaries": {name: file_record(path) for name, path in binary_records.items()},
        "configurations": {"b02b0.cnf": file_record(arguments.topup_config),
                           **{name: file_record(path) for name, path in oxford.items()}},
        "templates": {"FA_reference": file_record(arguments.fa_reference),
                      "FA_skeleton": file_record(arguments.fa_skeleton)},
        "runner_sources": {"end_to_end": file_record(Path(__file__)),
                           "AMICO": file_record(arguments.amico_runner),
                           "TBSS": file_record(arguments.tbss_runner)},
        "gpu_before": gpu_state(),
        "timing_scope": "processing starts at raw image/JSON preparation; ends after all native/standard/skeleton saves; report hashes/summaries and cross-pipeline comparisons are outside",
        "process_wall_note": "measure whole launcher separately to include Python imports, preflight, executable hashing and final report generation",
    }
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
                # benchmark is 72 slices; fail instead of silently mismatching
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
        with run.stage("independent_BET_mask"):
            mean_b0 = arguments.output_dir / "topup/fieldmap_iout_mean.nii.gz"
            brain_prefix = arguments.output_dir / "eddy/nodif_brain"
            run.run([fsl_bin / "fslmaths", arguments.output_dir / "topup/fieldmap_iout.nii.gz",
                     "-Tmean", mean_b0], name="topup_iout_mean")
            run.run([fsl_bin / "bet", mean_b0, brain_prefix, "-m", "-f", str(arguments.bet_fraction)],
                    name="independent_bet")
            mask = arguments.output_dir / "eddy/nodif_brain_mask.nii.gz"
            mask_image = nib.load(str(mask))
            if (mask_image.shape[:3] != ap.shape[:3]
                    or not np.allclose(mask_image.affine, ap.affine, atol=5e-4, rtol=0)
                    or not np.any(np.asanyarray(mask_image.dataobj) > 0)):
                raise ValueError("BET mask is empty or has different DWI geometry")
        with run.stage("eddy"):
            eddy_prefix = arguments.output_dir / "eddy/data"
            run.run([fsl_bin / "eddy_cuda10.2", f"--imain={arguments.raw_dir / 'AP.nii.gz'}",
                     f"--mask={mask}", f"--topup={field_prefix}", f"--acqp={acqp}", f"--index={index}",
                     f"--bvecs={arguments.raw_dir / 'AP.bvec'}", f"--bvals={arguments.raw_dir / 'AP.bval'}",
                     f"--out={eddy_prefix}", f"--ref_scan_no={reference_index}", f"--initrand={arguments.seed}",
                     "--flm=quadratic", "--resamp=jac", "--slm=linear", "--niter=8",
                     "--fwhm=10,8,4,2,0,0,0,0", "--ff=10", "--sep_offs_move", "--nvoxhp=1000",
                     "--repol", "--rms"], name="eddy")
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
        report["total_processing_seconds"] = time.perf_counter() - processing_started
        report["stages_seconds"] = run.stages
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
        report["outputs"] = {"native": {}, "standard": {}, "skeleton": {}}
        for name in MAP_NAMES:
            prefix = "NODDI_" if name in ("ICVF", "OD", "ISOVF") else "dti_"
            report["outputs"]["native"][name] = image_record(arguments.output_dir / "native" / f"{prefix}{name}.nii.gz")
            report["outputs"]["standard"][name] = image_record(arguments.output_dir / "tbss/stats" / f"all_{name}.nii.gz")
            report["outputs"]["skeleton"][name] = image_record(arguments.output_dir / "tbss/stats" / f"all_{name}_skeletonised.nii.gz")
        report["preparation_outputs"] = {
            "topup_pair": image_record(topup_pair), "acqp": file_record(acqp), "index": file_record(index),
            "brain_mask": image_record(mask), "preprocessed_FA": image_record(prepared_fa),
            "registration_weight": image_record(weight),
            "valid_FA_mask": image_record(arguments.output_dir / "tbss/stats/mean_FA_mask.nii.gz"),
            "skeleton_mask": image_record(arguments.output_dir / "tbss/stats/mean_FA_skeleton_mask.nii.gz"),
        }
        report["intermediate_outputs"] = {
            "topup_fieldcoef": image_record(arguments.output_dir / "topup/fieldmap_out_fieldcoef.nii.gz"),
            "topup_movpar": file_record(arguments.output_dir / "topup/fieldmap_out_movpar.txt"),
            "topup_fout": image_record(arguments.output_dir / "topup/fieldmap_fout.nii.gz"),
            "topup_iout": image_record(arguments.output_dir / "topup/fieldmap_iout.nii.gz"),
            "eddy_corrected": image_record(corrected_path), "eddy_rotated_bvecs": file_record(rotated_path),
            "DTI_shell": image_record(shell_path), "DTI_shell_bval": file_record(shell_bval),
            "DTI_shell_bvec": file_record(shell_bvec),
        }
        report["all_27_maps_saved_and_finite"] = all(
            value["finite"] for group in report["outputs"].values() for value in group.values()
        )
        report["status"] = "complete" if report["all_27_maps_saved_and_finite"] else "nonfinite_outputs"
    except Exception as error:
        report.update(status="failed", failure_type=type(error).__name__,
                      total_processing_seconds=time.perf_counter() - processing_started,
                      stages_seconds=run.stages, subprocess_steps=run.commands)
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "failed", "error_type": type(error).__name__}), flush=True)
        raise
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "total_processing_seconds": report["total_processing_seconds"],
                      "all_27_maps_saved_and_finite": report["all_27_maps_saved_and_finite"]}), flush=True)
    if not report["all_27_maps_saved_and_finite"]:
        raise RuntimeError("a saved parameter map contains non-finite values")


if __name__ == "__main__":
    main()
