"""Run isolated official FSL registration after native T1/FEAT preparation.

This is a benchmark, not an FNIT runtime dependency. The caller supplies the
original SynthStrip brain, FAST partial volumes, EPI reference and brain mask.
Images, paths, matrices and logs remain in the private output directory; the
public report contains anonymous checks, timings and file hashes.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def image_check(path, reference=None, vector=False):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("An official output contains nonfinite values")
    if reference is not None:
        expected = (*reference.shape[:3], 3) if vector else reference.shape[:3]
        if image.shape != expected or not np.allclose(
            image.affine, reference.affine, atol=1e-4, rtol=0
        ):
            raise ValueError("An official output has the wrong target grid")
    return {
        "shape": list(image.shape), "dtype": str(image.get_data_dtype()),
        "all_finite": True, "gzip_crc_passed": str(path).endswith(".gz"),
        "intent_code": int(image.header["intent_code"]),
    }


def matrix_check(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("An official affine is not a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("An official affine is not homogeneous")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-10:
        raise ValueError("An official affine is singular")
    return {"shape": [4, 4], "all_finite": True,
            "determinant": float(np.linalg.det(matrix[:3, :3]))}


def save_image(values, reference, path, *, vector=False):
    header = reference.header.copy()
    header.set_data_dtype(values.dtype)
    header.set_slope_inter(1, 0)
    if vector:
        header.set_intent("vector")
    nib.save(nib.Nifti1Image(values, reference.affine, header), path)


def scaled_mm(image):
    """Independently compute FSL voxel-to-scaled-mm, using stored pixdim."""
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    matrix = np.diag([*spacing, 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = spacing[0] * (image.shape[0] - 1)
    return matrix


def world_forward(matrix, moving, fixed):
    return (fixed.affine @ np.linalg.inv(scaled_mm(fixed)) @ matrix
            @ scaled_mm(moving) @ np.linalg.inv(moving.affine))


def relative_fsl_to_ras(field_path, moving, fixed, output):
    """Convert a complete fixed-grid FSL relative pull to RAS displacement."""
    field = nib.load(field_path)
    image_check(field_path, fixed, vector=True)
    displacement = np.asarray(field.dataobj, dtype=np.float64)
    voxels = np.indices(fixed.shape[:3], dtype=np.float64).reshape(3, -1)
    fixed_fsl = scaled_mm(fixed)
    source_scaled = fixed_fsl[:3, :3] @ voxels + fixed_fsl[:3, 3:4]
    source_scaled += displacement.reshape(-1, 3).T
    source_world_matrix = moving.affine @ np.linalg.inv(scaled_mm(moving))
    source_world = (source_world_matrix[:3, :3] @ source_scaled
                    + source_world_matrix[:3, 3:4])
    target_world = fixed.affine[:3, :3] @ voxels + fixed.affine[:3, 3:4]
    ras = (source_world - target_world).T.reshape(*fixed.shape[:3], 3)
    save_image(ras.astype(np.float32), fixed, output, vector=True)
    return ras


def run_stage(name, executable, arguments, outputs, args, stages):
    directory = args.private_output
    command = [str(args.fsl_root / "bin" / executable), *map(str, arguments)]
    environment = dict(os.environ, FSLDIR=str(args.fsl_root),
                       FSLOUTPUTTYPE="NIFTI_GZ", OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    environment["LD_LIBRARY_PATH"] = (str(args.fsl_root / "lib") + ":"
                                      + environment.get("LD_LIBRARY_PATH", ""))
    # The installed ELF launcher can report 255 after its real FSL child exits
    # 0. Trace only exec/exit; retain both rather than changing a return code.
    trace = directory / (name + ".exec.private.log")
    traced = shutil.which("strace")
    invoked = ([traced, "-f", "-e", "trace=execve,exit_group", "-o", str(trace)]
               + command) if traced else command
    for _, path, _, _ in outputs:
        Path(path).unlink(missing_ok=True)
    write_json(directory / "status.private.json", {"stage": name, "status": "running"})
    started = time.perf_counter()
    result = subprocess.run(invoked, env=environment, capture_output=True, text=True)
    wall = time.perf_counter() - started
    (directory / (name + ".private.log")).write_text(result.stdout + result.stderr)
    child_exits = []
    actual_child_exit = None
    if traced:
        trace_text = trace.read_text()
        child_exits = list(map(int, re.findall(r"exit_group\((-?\d+)\)", trace_text)))
        executed = re.findall(r"^(\d+) execve\(.*\) = 0$", trace_text, re.MULTILINE)
        if len(executed) > 1:
            exits = re.findall(r"^" + executed[-1] + r" exit_group\((-?\d+)\)",
                               trace_text, re.MULTILINE)
            if exits:
                actual_child_exit = int(exits[-1]) % 256
    summary = {"executable": executable, "wall_seconds_including_io": wall,
               "launcher_exit_code": result.returncode,
               "traced_exit_group_codes": child_exits,
               "actual_child_exit_code": actual_child_exit}
    write_json(directory / (name + ".command.private.json"), {"command": command, **summary})
    if result.returncode != 0 and not (result.returncode == 255 and actual_child_exit == 0):
        raise RuntimeError(f"Official {name} failed with exit {result.returncode}")
    checks = {}
    for label, path, reference, vector in outputs:
        checks[label] = matrix_check(path) if str(path).endswith(".mat") else image_check(
            path, reference, vector
        )
    stages[name] = {**summary, "outputs": checks,
                    "acceptance": "Outputs verified; original exit codes retained."}
    private = {"command": command, **stages[name]}
    write_json(directory / (name + ".command.private.json"), private)
    write_json(directory / "stages.private.json", stages)
    print(json.dumps({"completed": name, "seconds": wall,
                      "exit": result.returncode}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1-brain", "wm-pve", "csf-pve", "epi-reference", "epi-mask",
                 "mni-template", "mni-mask", "fsl-root", "private-output",
                 "report-out", "manifest-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--wait-inputs-seconds", type=float, default=0)
    parser.add_argument("--inverse-diagnostic", action="store_true",
                        help="Also run official nonlinear invwarp; excluded from primary timing")
    args = parser.parse_args()
    if args.threads < 1 or args.wait_inputs_seconds < 0:
        raise ValueError("Invalid thread count or input wait duration")
    args.private_output.mkdir(parents=True, exist_ok=True)
    needed = (args.t1_brain, args.wm_pve, args.csf_pve,
              args.epi_reference, args.epi_mask)
    deadline = time.monotonic() + args.wait_inputs_seconds
    while not all(path.exists() for path in needed):
        if time.monotonic() >= deadline:
            raise FileNotFoundError("Native anatomical and EPI inputs are not ready")
        time.sleep(5)
    t1, epi, template, mask = [nib.load(path) for path in (
        args.t1_brain, args.epi_reference, args.mni_template, args.mni_mask)]
    for image in (t1, epi, template, mask):
        if image.ndim != 3:
            raise ValueError("All registration inputs must be 3D")
    if mask.shape != template.shape or not np.allclose(mask.affine, template.affine,
                                                      atol=1e-4, rtol=0):
        raise ValueError("The template mask must match the template grid")
    template_mask = np.asarray(mask.dataobj) > 0
    if not template_mask.any():
        raise ValueError("The template mask is empty")
    output = args.private_output
    files = {name: output / name for name in (
        "MNI_brain.nii.gz", "MNI_mask.nii.gz", "T1_wmseg.nii.gz",
        "T1_to_MNI152_2mm_affine.mat", "T1_affine_in_MNI.nii.gz",
        "T1_to_MNI_coeff.nii.gz", "T1_to_MNI_fnirt_field.nii.gz",
        "T1_in_MNI.nii.gz", "T1_to_MNI_fsl_relative.nii.gz",
        "MNI152_2mm_to_T1_pull_ras.nii.gz", "example_func2highres_init.mat",
        "example_func2highres.mat", "example_func2highres.nii.gz",
        "highres2example_func.mat", "reference_to_source_world.txt",
        "EPI_to_MNI_fsl_relative.nii.gz", "MNI_to_EPI_pull_ras.nii.gz",
        "brain_MNI_raw.nii.gz", "brain_MNI152_2mm.nii.gz",
        "wm_pve_epi.nii.gz", "csf_pve_epi.nii.gz",
        "wm_epi.nii.gz", "csf_epi.nii.gz",
    )}
    if args.inverse_diagnostic:
        for name in ("MNI_to_EPI_fsl_relative.nii.gz", "EPI_to_MNI_pull_ras.nii.gz"):
            files[name] = output / name
    # Match _anatomical._produce: full template times mask, full-template header.
    brain_values = np.asarray(template.dataobj, dtype=np.float32) * template_mask
    save_image(brain_values, template, files["MNI_brain.nii.gz"])
    save_image(template_mask.astype(np.uint8), template, files["MNI_mask.nii.gz"])
    brain_template = nib.load(files["MNI_brain.nii.gz"])
    wm_pve = nib.load(args.wm_pve)
    if wm_pve.shape != t1.shape or not np.allclose(wm_pve.affine, t1.affine, atol=1e-4):
        raise ValueError("FAST WM PVE does not match T1")
    wm_seg = np.asarray(wm_pve.dataobj, dtype=np.float32) >= .5
    if not wm_seg.any():
        raise ValueError("FAST WM boundary is empty")
    save_image(wm_seg.astype(np.uint8), t1, files["T1_wmseg.nii.gz"])
    stages = {}
    run_stage("t1_affine", "flirt", [
        "-in", args.t1_brain, "-ref", files["MNI_brain.nii.gz"],
        "-dof", "12", "-cost", "corratio", "-omat", files["T1_to_MNI152_2mm_affine.mat"],
        "-out", files["T1_affine_in_MNI.nii.gz"]], [
        ("matrix", files["T1_to_MNI152_2mm_affine.mat"], None, False),
        ("image", files["T1_affine_in_MNI.nii.gz"], brain_template, False)], args, stages)
    configuration = args.fsl_root / "etc/flirtsch/T1_2_MNI152_2mm.cnf"
    run_stage("t1_nonlinear", "fnirt", [
        "--in=" + str(args.t1_brain), "--ref=" + str(files["MNI_brain.nii.gz"]),
        "--refmask=" + str(files["MNI_mask.nii.gz"]), "--config=" + str(configuration),
        "--aff=" + str(files["T1_to_MNI152_2mm_affine.mat"]),
        "--subsamp=4,4,2,2,1,1", "--miter=5,5,5,5,5,10",
        "--infwhm=8,6,5,4.5,3,2", "--reffwhm=8,6,5,4,2,0",
        "--lambda=300,150,100,50,40,30", "--estint=1,1,1,1,1,0",
        "--applyrefmask=1,1,1,1,1,1", "--applyinmask=1", "--warpres=10,10,10",
        "--imprefm=1", "--impinm=1", "--imprefval=0", "--impinval=0",
        "--ssqlambda=1", "--regmod=bending_energy", "--splineorder=3",
        "--intmod=global_non_linear_with_bias", "--intorder=5", "--biasres=50,50,50",
        "--biaslambda=10000", "--refderiv=0", "--numprec=double", "--minmet=lm",
        "--jacrange=0.01,100", "--interp=linear",
        "--cout=" + str(files["T1_to_MNI_coeff.nii.gz"]),
        "--fout=" + str(files["T1_to_MNI_fnirt_field.nii.gz"]),
        "--iout=" + str(files["T1_in_MNI.nii.gz"]),
        "--logout=" + str(output / "fnirt.private.log")], [
        ("coefficients", files["T1_to_MNI_coeff.nii.gz"], None, False),
        ("field", files["T1_to_MNI_fnirt_field.nii.gz"], brain_template, True),
        ("image", files["T1_in_MNI.nii.gz"], brain_template, False)], args, stages)
    run_stage("epi_affine", "flirt", [
        "-in", args.epi_reference, "-ref", args.t1_brain, "-dof", "6", "-cost", "normmi",
        "-omat", files["example_func2highres_init.mat"]], [
        ("matrix", files["example_func2highres_init.mat"], None, False)], args, stages)
    schedule = args.fsl_root / "etc/flirtsch/bbr.sch"
    run_stage("epi_bbr", "flirt", [
        "-in", args.epi_reference, "-ref", args.t1_brain, "-dof", "6", "-cost", "bbr",
        "-wmseg", files["T1_wmseg.nii.gz"], "-init", files["example_func2highres_init.mat"],
        "-schedule", schedule, "-omat", files["example_func2highres.mat"],
        "-out", files["example_func2highres.nii.gz"]], [
        ("matrix", files["example_func2highres.mat"], None, False),
        ("image", files["example_func2highres.nii.gz"], t1, False)], args, stages)
    run_stage("t1_dense_warp", "convertwarp", [
        "--ref=" + str(args.mni_template), "--warp1=" + str(files["T1_to_MNI_coeff.nii.gz"]),
        "--relout", "--out=" + str(files["T1_to_MNI_fsl_relative.nii.gz"])], [
        ("warp", files["T1_to_MNI_fsl_relative.nii.gz"], template, True)], args, stages)
    run_stage("compose_epi_mni", "convertwarp", [
        "--ref=" + str(args.mni_template), "--premat=" + str(files["example_func2highres.mat"]),
        "--warp1=" + str(files["T1_to_MNI_coeff.nii.gz"]), "--relout",
        "--out=" + str(files["EPI_to_MNI_fsl_relative.nii.gz"])], [
        ("warp", files["EPI_to_MNI_fsl_relative.nii.gz"], template, True)], args, stages)
    t1_pull = relative_fsl_to_ras(files["T1_to_MNI_fsl_relative.nii.gz"], t1, template,
                                  files["MNI152_2mm_to_T1_pull_ras.nii.gz"])
    epi_pull = relative_fsl_to_ras(files["EPI_to_MNI_fsl_relative.nii.gz"], epi, template,
                                   files["MNI_to_EPI_pull_ras.nii.gz"])
    bbr_matrix = np.loadtxt(files["example_func2highres.mat"])
    inverse_bbr = np.linalg.inv(world_forward(bbr_matrix, epi, t1))
    np.savetxt(files["reference_to_source_world.txt"], inverse_bbr, fmt="%.17g")
    voxels = np.indices(template.shape, dtype=np.float64).reshape(3, -1)
    world = template.affine[:3, :3] @ voxels + template.affine[:3, 3:4]
    t1_world = world + t1_pull.reshape(-1, 3).T
    composed_world = inverse_bbr[:3, :3] @ t1_world + inverse_bbr[:3, 3:4]
    delta = (composed_world - world).T.reshape(*template.shape, 3) - epi_pull
    distance = np.linalg.norm(delta[template_mask], axis=-1)
    run_stage("inverse_bbr", "convert_xfm", [
        "-omat", files["highres2example_func.mat"], "-inverse", files["example_func2highres.mat"]], [
        ("matrix", files["highres2example_func.mat"], None, False)], args, stages)
    epi_mask_image = nib.load(args.epi_mask)
    if epi_mask_image.shape != epi.shape or not np.allclose(epi_mask_image.affine, epi.affine, atol=1e-4):
        raise ValueError("The EPI mask does not match the EPI reference")
    epi_mask = np.asarray(epi_mask_image.dataobj) > 0
    for name, path in (("wm", args.wm_pve), ("csf", args.csf_pve)):
        run_stage(name + "_pve_to_epi", "flirt", [
            "-in", path, "-ref", args.epi_reference, "-applyxfm",
            "-init", files["highres2example_func.mat"], "-interp", "trilinear",
            "-out", files[name + "_pve_epi.nii.gz"]], [
            ("image", files[name + "_pve_epi.nii.gz"], epi, False)], args, stages)
        pve = np.asarray(nib.load(files[name + "_pve_epi.nii.gz"]).dataobj)
        save_image(((pve >= .8) & epi_mask).astype(np.uint8), epi, files[name + "_epi.nii.gz"])
    run_stage("epi_mask_to_mni", "applywarp", [
        "--in=" + str(args.epi_mask), "--ref=" + str(args.mni_template),
        "--warp=" + str(files["EPI_to_MNI_fsl_relative.nii.gz"]), "--rel",
        "--interp=nn", "--datatype=char", "--out=" + str(files["brain_MNI_raw.nii.gz"])], [
        ("image", files["brain_MNI_raw.nii.gz"], template, False)], args, stages)
    mni_mask = (np.asarray(nib.load(files["brain_MNI_raw.nii.gz"]).dataobj) > .5) & template_mask
    save_image(mni_mask.astype(np.uint8), template, files["brain_MNI152_2mm.nii.gz"])
    if args.inverse_diagnostic:
        run_stage("inverse_epi_mni", "invwarp", [
            "--warp=" + str(files["EPI_to_MNI_fsl_relative.nii.gz"]),
            "--ref=" + str(args.epi_reference), "--rel",
            "--out=" + str(files["MNI_to_EPI_fsl_relative.nii.gz"])], [
            ("warp", files["MNI_to_EPI_fsl_relative.nii.gz"], epi, True)], args, stages)
        relative_fsl_to_ras(files["MNI_to_EPI_fsl_relative.nii.gz"], template, epi,
                            files["EPI_to_MNI_pull_ras.nii.gz"])
    hashes = {name: sha256(path) for name, path in files.items()}
    inputs = {"t1_brain": args.t1_brain, "wm_pve": args.wm_pve, "csf_pve": args.csf_pve,
              "epi_reference": args.epi_reference, "epi_mask": args.epi_mask,
              "mni_template": args.mni_template, "mni_mask": args.mni_mask}
    report = {
        "schema_version": 1, "scope": "Official FSL registration after independent native SynthStrip/FAST/FEAT preparation; matched FNIT FNIRT branch.",
        "source_revision": args.source_revision,
        "fsl_version": (args.fsl_root / "etc/fslversion").read_text().strip(),
        "registration_backend": "fnirt", "stages": stages,
        "timing_seconds": {
            "primary_process_seconds": sum(stage["wall_seconds_including_io"] for name, stage in stages.items() if name != "inverse_epi_mni"),
            "validation_process_seconds": sum(stage["wall_seconds_including_io"] for name, stage in stages.items() if name == "inverse_epi_mni"),
        },
        "timing_scope": "Sum of individual native command walls including I/O and exec/exit tracing; NumPy conversion, file hashing, waiting and upstream preprocessing are not included. Nonlinear invwarp runs only with --inverse-diagnostic and is excluded from primary timing.",
        "inverse_diagnostic_enabled": args.inverse_diagnostic,
        "template_preparation": "Full MNI152 2mm template multiplied by the supplied mask, preserving full-template header.",
        "mask_rules": {"wm_boundary": "Native FAST WM PVE >= 0.5", "tissues_epi": "Linear-resampled native FAST WM/CSF PVE >= 0.8 intersected with native EPI brain mask", "brain_mni": "Nearest EPI brain mask through composed warp intersected with supplied template mask"},
        "composition_control_mm": {"mean": float(distance.mean()), "median": float(np.median(distance)),
                                   "p95": float(np.percentile(distance, 95)), "rms": float(np.sqrt(np.mean(distance**2))),
                                   "maximum": float(distance.max())},
        "mask_voxels": {"wm_boundary": int(wm_seg.sum()), "epi_brain": int(epi_mask.sum()),
                        "mni_brain": int(mni_mask.sum())},
        "sha256": {"inputs": {name: sha256(path) for name, path in inputs.items()},
                   "outputs": hashes, "script": sha256(__file__),
                   "fnirt_config": sha256(configuration), "bbr_schedule": sha256(schedule),
                   "executables": {name: sha256(args.fsl_root / "bin" / name) for name in (
                       "flirt", "fnirt", "convertwarp", "invwarp", "convert_xfm", "applywarp")}},
        "warp_contract": "MNI-grid RAS displacement d(p) gives T1 world p+d(p); complete affine and nonlinear are already included. EPI sample world is inverse(EPI_to_T1_world)*(p+d(p)). FSL relative fields use scaled-mm coordinates and are converted with both source/reference headers.",
        "limits": ["This run matches FNIT brain-only FNIRT inputs; it is not the UKB whole-head FNIRT protocol.", "Synthetic controls are not substituted for real registration benchmarks.", "Launcher and traced child exits are retained; geometry/header/CRC/finite checks alone do not assert registration equivalence."],
        "privacy": "Anonymous checks, scalar metrics and SHA-256 only; raw data, paths and program logs remain private.",
    }
    write_json(args.report_out, report)
    write_json(args.manifest_out, {"inputs": {k: str(v) for k, v in inputs.items()},
                                  "outputs": {k: str(v) for k, v in files.items()},
                                  "sha256": report["sha256"], "registration_backend": "fnirt",
                                  "report": str(args.report_out)})
    write_json(output / "status.private.json", {"stage": "complete", "status": "passed"})
    print(json.dumps({"completed": "registration", "composition_control_mm": report["composition_control_mm"]}), flush=True)


if __name__ == "__main__":
    main()
