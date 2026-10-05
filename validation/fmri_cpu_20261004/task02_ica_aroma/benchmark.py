"""Real-input CPU benchmark adapter for PICA, ICA-AROMA, and nuisance projection.

Private manifest and fresh output directory are supplied by the coordinator.
External official programs are used only by the explicit ``official`` backend.
This adapter never crops BOLD or reduces ICA iterations/AROMA subsamples.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import nibabel as nib
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "validation" / "fmri"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def image_values(path):
    image = nib.load(str(path))
    return image, np.asarray(image.dataobj, dtype=np.float32)


def validate_image(path, reference, frames=None):
    image, values = image_values(path)
    expected = reference.shape if frames is None else (*reference.shape[:3], frames)
    if image.shape != expected or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("Benchmark output shape or affine differs from its input")
    if not np.isfinite(values).all():
        raise ValueError("Benchmark output contains nonfinite values")
    return {"shape": list(image.shape), "dtype": str(image.get_data_dtype()),
            "sha256": sha256(path)}


@contextmanager
def in_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def independent_confounds(data, affine, header, config, case, output, chunk_size):
    """Independent NumPy SVD reference; never imports FNIT projection helpers."""
    nt = data.shape[-1]
    time_axis = np.linspace(-1.0, 1.0, nt)
    columns = [np.ones(nt), time_axis, (3 * time_axis**2 - 1) / 2]
    for key in ("wm_mask", "csf_mask", "brain_mask"):
        selected = case.get(key.replace("_mask", ""), False)
        if not selected:
            continue
        mask_image = nib.load(config[key])
        if mask_image.shape != data.shape[:3] or not np.allclose(mask_image.affine, affine, atol=1e-3):
            raise ValueError("Independent tissue mask is not on the BOLD grid")
        mask = np.asarray(mask_image.dataobj) > 0
        if not mask.any():
            raise ValueError("Independent tissue mask is empty")
        columns.append(data[mask].mean(axis=0, dtype=np.float64))
    model = case.get("motion_model")
    if model is not None:
        motion = np.loadtxt(config["motion"], ndmin=2)
        if motion.shape != (nt, 6):
            raise ValueError("Independent motion input is not T x 6")
        if model == 6:
            design_motion = motion
        elif model == 12:
            design_motion = np.column_stack((motion, np.vstack((np.zeros((1, 6)), np.diff(motion, axis=0)))))
        elif model == 24:
            previous = np.vstack((np.zeros((1, 6)), motion[:-1]))
            design_motion = np.column_stack((motion, previous, motion**2, previous**2))
        else:
            raise ValueError("Unknown motion model")
        columns.extend(design_motion.T)
    design = np.column_stack(columns)
    varying = np.r_[True, np.ptp(design[:, 1:], axis=0) > 0]
    design = design[:, varying]
    design[:, 1:] -= design[:, 1:].mean(axis=0)
    norms = np.linalg.norm(design, axis=0)
    design = design[:, norms > 0] / norms[norms > 0]
    bandpass = case.get("bandpass")
    keep = None
    if bandpass is not None:
        frequencies = np.fft.rfftfreq(nt, config["tr"])
        keep = (frequencies >= bandpass[0]) & (frequencies <= bandpass[1])

    def filtered(values):
        if keep is None:
            return values
        spectrum = np.fft.rfft(values, axis=0)
        spectrum[~keep] = 0
        return np.fft.irfft(spectrum, n=nt, axis=0)

    design = filtered(design)
    norms = np.linalg.norm(design, axis=0)
    valid = norms > np.finfo(np.float64).eps * max(nt, design.shape[1])
    design = design[:, valid] / norms[valid]
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-8)) if singular.size else 0
    basis = left[:, :rank]
    flat = data.reshape((-1, nt))
    residual = np.empty_like(flat)
    for start in range(0, flat.shape[0], chunk_size):
        series = filtered(flat[start:start + chunk_size].T.astype(np.float64))
        residual[start:start + chunk_size] = (series - basis @ (basis.T @ series)).T
    copied_header = header.copy()
    copied_header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(residual.reshape(data.shape), affine, copied_header), output)
    return {"effective_rank": rank, "effective_columns": int(design.shape[1]),
            "method": "independent NumPy float64 SVD, rcond=1e-8"}


def confound_cases():
    return {
        "drift": {}, "motion6": {"motion_model": 6},
        "motion12": {"motion_model": 12}, "motion24": {"motion_model": 24},
        "tissue": {"wm": True, "csf": True}, "global": {"brain": True},
        "all": {"wm": True, "csf": True, "brain": True, "motion_model": 24},
        "all_bandpass": {"wm": True, "csf": True, "brain": True,
                         "motion_model": 24, "bandpass": [0.01, 0.1]},
    }


def official_commands(config, output, trace):
    from official_denoising import OfficialCommands
    return OfficialCommands(SimpleNamespace(
        fsl_dir=Path(config["fsl_root"]), fsl_bin=None, output_dir=output,
        # The existing FSL launcher returns 255 on this server. Acceptance
        # still requires the original executable's successful traced exit.
        allow_complete_exit255=trace, trace_original_exits=trace,
    ))


def feature_result(config, backend, output, trace):
    if backend == "fnit":
        from fnit.fmri.aroma import classify_aroma
        return classify_aroma(
            config["thresholded_maps_mni"], config["mixing"], config["ftmix"],
            config["motion"], config["aroma_csf_mask"], config["aroma_edge_mask"],
            config["aroma_outside_mask"], config["tr"], n_splits=1000, random_state=0,
        )
    if backend != "official":
        raise ValueError("Features require fnit or official backend")
    from official_denoising import checked_aroma_commands
    specification = importlib.util.spec_from_file_location("upstream_aroma", config["aroma_functions"])
    upstream = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(upstream)
    commands = official_commands(config, output, trace)
    # The official spatial helper expects masks with these original filenames.
    # This upstream revision invokes fslstats with literal relative mask
    # filenames; aromaDir does not affect those commands. Preserve the author
    # code and supply its masks in the working directory instead.
    masks = output
    for key, name in (("aroma_csf_mask", "mask_csf.nii.gz"), ("aroma_edge_mask", "mask_edge.nii.gz"),
                      ("aroma_outside_mask", "mask_out.nii.gz")):
        (masks / name).symlink_to(Path(config[key]).resolve())
    random.seed(0)
    np.random.seed(0)
    with in_directory(output), checked_aroma_commands(commands):
        motion = upstream.feature_time_series(config["mixing"], config["motion"])
        frequency = upstream.feature_frequency(config["ftmix"], config["tr"])
        edge, csf = upstream.feature_spatial(str(commands.directory) + "/", str(output),
                                           str(masks), config["thresholded_maps_mni"])
        noise = np.atleast_1d(upstream.classification(str(output), motion, edge, frequency, csf)).astype(int)
    return {"max_rp_corr": motion, "edge_fraction": edge, "high_freq_content": frequency,
            "csf_fraction": csf, "noise_indices": noise}


def run(args, config, output):
    backend = args.backend
    if args.phase == "pica":
        if backend == "fnit":
            from fnit.melodic import decompose_spatial_ica
            result = decompose_spatial_ica(
                config["bold"], config["brain_mask"], output / "ica",
                n_components=args.n_components, device=args.device, voxel_batch_size=args.chunk_size,
                max_iter=500, tolerance=1e-3, random_state=0, mm_threshold=args.mm_threshold,
            )
            return asdict(result)
        commands = official_commands(config, output, args.trace_original)
        source = nib.load(config["bold"])
        directory = output / "melodic.ica"
        argv = [f"--in={config['bold']}", f"--mask={config['brain_mask']}", f"--outdir={directory}",
                f"--tr={config['tr']}", f"--dim={args.n_components or 0}", "--dimest=lap", "--nl=pow3",
                "--eps=0.001", "--maxit=500", "--seed=0", "--nobet", "--Ostats",
                f"--mmthresh={args.mm_threshold}"]
        if args.official_report:
            argv.append("--report")
        from official_denoising import melodic_files
        wall = commands.run("melodic", argv, lambda: melodic_files(directory, source), "melodic")
        return {"n_components": melodic_files(directory, source), "command_wall_seconds": wall,
                "official_report": args.official_report}
    if args.phase == "aroma-features":
        values = feature_result(config, backend, output, args.trace_original)
        np.savez(output / "features.npz", **values)
        return {"component_count": len(values["max_rp_corr"]), "noise_count": len(values["noise_indices"]),
                "n_splits": 1000, "random_state": 0}
    if args.phase == "denoise":
        text = Path(config["noise_indices"]).read_text().strip()
        tokens = re.split(r"[,\s]+", text) if text else []
        if any(not re.fullmatch(r"\d+", token) for token in tokens):
            raise ValueError("Noise component file must contain integer indices")
        noise = np.asarray(tokens, dtype=np.int64)
        noise = noise - int(config.get("noise_index_base", 0))
        path = output / "denoised.nii.gz"
        if backend == "fnit":
            from fnit.fmri.aroma import denoise_aroma
            denoise_aroma(config["bold"], config["mixing"], noise, path, mode=args.mode,
                          device=args.device, chunk_size=args.chunk_size)
        else:
            commands = official_commands(config, output, args.trace_original)
            argv = ["-i", config["bold"], "-d", config["mixing"], "-f",
                    ",".join(str(int(index) + 1) for index in noise), "-o", str(path)]
            if args.mode == "aggr":
                argv.append("-a")
            commands.run("fsl_regfilt", argv, lambda: validate_image(path, nib.load(config["bold"])), "regfilt")
        return {"mode": args.mode, "noise_count": int(noise.size),
                "output": validate_image(path, nib.load(config["bold"]))}
    if args.phase == "confounds":
        case = confound_cases()[args.case]
        path = output / "cleaned.nii.gz"
        if backend == "fnit":
            from fnit.fmri.confounds import clean_confounds
            # The frozen baseline predates this explicit compatibility mode.
            # Its unchanged default API remains a valid old/new control.
            projection_options = ({"projection": "afni"}
                                  if args.confound_projection == "afni" else {})
            clean_confounds(
                config["confounds_bold"], path, wm_mask=config["wm_mask"] if case.get("wm") else None,
                csf_mask=config["csf_mask"] if case.get("csf") else None,
                brain_mask=config["brain_mask"], motion=config["motion"] if case.get("motion_model") else None,
                motion_model=case.get("motion_model", 24), global_signal=case.get("brain", False),
                bandpass=case.get("bandpass"), tr=config["tr"], device=args.device, chunk_size=args.chunk_size,
                **projection_options,
            )
            result = {"method": "FNIT float64 joint projection", "projection": args.confound_projection}
        elif backend == "afni":
            image, data = image_values(config["confounds_bold"])
            columns = []
            for key in ("wm_mask", "csf_mask", "brain_mask"):
                if case.get(key.replace("_mask", ""), False):
                    mask_image = nib.load(config[key])
                    if mask_image.shape != data.shape[:3] or not np.allclose(mask_image.affine, image.affine, atol=1e-3):
                        raise ValueError("AFNI nuisance mask differs from the BOLD grid")
                    mask = np.asarray(mask_image.dataobj) > 0
                    if not mask.any():
                        raise ValueError("AFNI nuisance mask is empty")
                    columns.append(data[mask].mean(axis=0, dtype=np.float64))
            model = case.get("motion_model")
            if model is not None:
                motion = np.loadtxt(config["motion"], ndmin=2)
                if motion.shape != (data.shape[-1], 6):
                    raise ValueError("AFNI motion must be T x 6")
                if model == 6:
                    design_motion = motion
                elif model == 12:
                    design_motion = np.column_stack((motion, np.vstack((np.zeros((1, 6)), np.diff(motion, axis=0)))))
                else:
                    previous = np.vstack((np.zeros((1, 6)), motion[:-1]))
                    design_motion = np.column_stack((motion, previous, motion**2, previous**2))
                columns.extend(design_motion.T)
            executable = Path(config["afni_3dtproject"])
            argv = [str(executable), "-input", config["confounds_bold"], "-prefix", str(path),
                    "-polort", "2", "-TR", str(config["tr"])]
            if columns:
                nuisance = output / "nuisance.1D"
                np.savetxt(nuisance, np.column_stack(columns), fmt="%.17g")
                argv += ["-ort", str(nuisance)]
            if case.get("bandpass"):
                argv += ["-passband", *map(str, case["bandpass"])]
            env = dict(os.environ, LD_LIBRARY_PATH=str(executable.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
            with (output / "afni.private.log").open("wb") as logfile:
                completed = subprocess.run(argv, env=env, stdout=logfile, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError("Original 3dTproject exited " + str(completed.returncode))
            result = {"method": "Original AFNI 3dTproject joint nuisance projection", "native_exit_code": completed.returncode,
                      "native_program_sha256": sha256(executable), "nuisance_columns": len(columns)}
        else:
            image, data = image_values(config["confounds_bold"])
            result = independent_confounds(data, image.affine, image.header, config, case, path, args.chunk_size)
        return {**result, "case": args.case, "output": validate_image(path, nib.load(config["confounds_bold"]))}
    if args.phase == "bids":
        if backend != "fnit":
            raise ValueError("BIDS wrapper is a FNIT API contract, not an official MELODIC mode")
        from fnit.melodic import run_melodic_bids
        result = run_melodic_bids(
            config["source_derivatives_root"], output / "derivatives", input_bold=config["bids_bold"],
            brain_mask=config["bids_brain_mask"], n_components=args.n_components,
            device=args.device, voxel_batch_size=args.chunk_size, max_iter=500,
            tolerance=1e-3, random_state=0, mm_threshold=args.mm_threshold, overwrite=False,
        )
        return asdict(result)
    raise ValueError("Unsupported phase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("pica", "aroma-features", "denoise", "confounds", "bids"), required=True)
    parser.add_argument("--backend", choices=("fnit", "official", "independent", "afni"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--mm-threshold", type=float, default=0.5)
    parser.add_argument("--mode", choices=("nonaggr", "aggr"), default="nonaggr")
    parser.add_argument("--case", choices=tuple(confound_cases()), default="all")
    parser.add_argument("--trace-original", action="store_true")
    parser.add_argument("--official-report", action="store_true")
    parser.add_argument("--gpu-memory-budget-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--confound-projection", choices=("orthogonal", "afni"), default="orthogonal")
    args = parser.parse_args()
    if args.threads < 1 or args.chunk_size < 1:
        parser.error("threads and chunk-size must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.manifest.read_text())["datasets"][args.dataset]
    source = nib.load(config["bold"])
    if source.ndim != 4 or source.shape[-1] != config["n_frames"]:
        raise ValueError("Full-frame real input differs from the private manifest")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != str(args.threads):
            raise ValueError(f"Set {key} before interpreter startup to the assigned thread budget")
    if args.backend == "fnit":
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if args.device.startswith("cuda"):
            if args.gpu_memory_budget_bytes <= 0:
                parser.error("GPU memory budget must be positive")
            torch.cuda.set_device(torch.device(args.device))
            torch.cuda.init()
            total_memory = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, args.gpu_memory_budget_bytes / total_memory), torch.device(args.device))
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    import fnit
    executed_source = Path(fnit.__file__).resolve().parent
    report = {"schema_version": 1, "dataset_alias": args.dataset, "phase": args.phase,
              "backend": args.backend, "device": args.device, "shape": list(source.shape),
              "threads": args.threads, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "host": platform.node(), "python": platform.python_version(),
              "adapter_sha256": sha256(__file__), "thread_environment": {key: os.environ.get(key) for key in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "input_sha256": {key: sha256(value) for key, value in config.items()
                  if isinstance(value, str) and Path(value).is_file()},
              "source_sha256": {name: sha256(executed_source / name) for name in
                  ("melodic/ica.py", "melodic/bids.py", "fmri/aroma.py", "fmri/confounds.py")},
              "timing_boundary": "API including load/compute/write and output checks; excluding interpreter imports and provenance hashes",
              "official_process_trace_included": args.trace_original and args.backend == "official",
              "executed": False}
    try:
        from threadpoolctl import threadpool_info
        report["blas_libraries"] = threadpool_info()
    except ImportError:
        report["blas_libraries"] = "threadpoolctl unavailable"
    started = time.perf_counter()
    report["result"] = run(args, config, output)
    if args.device.startswith("cuda") and args.backend == "fnit":
        torch.cuda.synchronize(torch.device(args.device))
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(torch.device(args.device))
        report["gpu_memory_budget_bytes"] = args.gpu_memory_budget_bytes
        report["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        report["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    report["api_wall_seconds"] = time.perf_counter() - started
    report["executed"] = True
    save_json(output / "report.private.json", report)
    public = {key: value for key, value in report.items() if key not in {"result", "blas_libraries"}}
    public["result"] = {key: value for key, value in report["result"].items()
                        if not isinstance(value, Path) and not (isinstance(value, str) and value.startswith("/"))}
    save_json(output / "report.public.json", public)


if __name__ == "__main__":
    main()
