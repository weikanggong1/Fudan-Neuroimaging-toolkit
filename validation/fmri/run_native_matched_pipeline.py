"""用一个私有清单连续运行单例原软件对照，记录完整进程墙钟。

主链选择 FNIRT；SynthStrip 保持原 FreeSurfer 实现，后续使用原 FSL 和
外置原 ICA-AROMA。末端混杂回归由独立 NumPy 参考实现同一投影定义。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True,
                        help="私有单例配置，包含输入、原软件和外置AROMA路径")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    case = json.loads(args.case_json.read_text())
    output = args.output_dir.resolve()
    if any((output / name).exists() for name in ("anatomy.public.json", "feat.public.json", "clean_mni_fnirt.nii.gz")):
        raise FileExistsError("Use a new native output directory")
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(__file__).resolve().parent
    fsl = Path(case["fsl_root"])
    environment = dict(os.environ, FSLDIR=str(fsl), FSLOUTPUTTYPE="NIFTI_GZ")
    environment["LD_LIBRARY_PATH"] = str(fsl / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    threads = int(case.get("threads", 8))
    environment.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    commands = []
    stages = {}

    def run(name, command):
        command = list(map(str, command))
        commands.append({"name": name, "argv": command})
        (output / "commands.private.json").write_text(json.dumps(commands, indent=2) + "\n")
        (output / "status.private.json").write_text(json.dumps({"stage": name, "status": "running"}) + "\n")
        started = time.perf_counter()
        with (output / (name + ".controller.log")).open("w") as stream:
            process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        stages[name] = {"process_wall_seconds_including_validation": time.perf_counter() - started,
                        "exit_code": process.returncode}
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect private stage log")
        print(json.dumps({"completed": name, **stages[name]}), flush=True)

    input_hashes = {name: sha256(case[name]) for name in
                    ("bold", "sbref", "t1w", "mni_template", "mni_mask", "synthstrip_weights")}
    started = time.perf_counter()
    common = ["--output-dir", output, "--fsl-root", fsl,
              "--freesurfer-root", case["freesurfer_root"], "--sbref", case["sbref"], "--threads", threads]
    if case.get("synthstrip_python"):
        common += ["--synthstrip-python", case["synthstrip_python"]]
    run("anatomy", [sys.executable, directory / "run_native_matched.py", "anatomy", *common,
                    "--t1w", case["t1w"], "--synthstrip-weights", case["synthstrip_weights"]])
    run("feat", [sys.executable, directory / "run_native_matched.py", "feat", *common,
                 "--bold", case["bold"], "--tr", case["tr"], "--highpass-seconds", 100])
    registration = output / "reg_fnirt"
    run("registration", [sys.executable, directory / "official_registration.py",
                          "--t1-brain", output / "anat/T1_brain.nii.gz",
                          "--wm-pve", output / "anat/T1_fast_pve_2.nii.gz",
                          "--csf-pve", output / "anat/T1_fast_pve_0.nii.gz",
                          "--epi-reference", output / "feat/example_func.nii.gz",
                          "--epi-mask", output / "masks/epi_mask.nii.gz",
                          "--mni-template", case["mni_template"], "--mni-mask", case["mni_mask"],
                          "--fsl-root", fsl, "--threads", threads, "--private-output", registration,
                          "--report-out", registration / "registration.public.json",
                          "--manifest-out", registration / "manifest.private.json",
                          "--source-revision", args.source_revision])
    clean_native = output / "clean_native_fnirt.nii.gz"
    run("denoising", [sys.executable, directory / "official_denoising.py", "--phase", "all",
                       "--input-bold", output / "feat/filtered_func_data.nii.gz",
                       "--brain-mask", output / "masks/epi_mask.nii.gz",
                       "--motion", output / "feat/mc/prefiltered_func_data_mcf.par",
                       "--wm-mask", registration / "wm_epi.nii.gz",
                       "--csf-mask", registration / "csf_epi.nii.gz",
                       "--output-dir", output / "aroma_fnirt", "--melodic-dir", output / "melodic.ica",
                       "--fsl-dir", fsl, "--aroma-functions", case["aroma_functions"],
                       "--classification-masks-dir", case["classification_masks_dir"],
                       "--mni-template", case["mni_template"],
                       "--epi-to-t1", registration / "example_func2highres.mat",
                       "--t1-to-mni-warp", registration / "T1_to_MNI_coeff.nii.gz",
                       "--warp-convention", "auto", "--clean-native", clean_native,
                       "--tr", case["tr"], "--source-revision", args.source_revision,
                       "--allow-complete-exit255"])
    # This original FSL call samples the original-native result, using the same
    # registration estimated above. The coefficient already includes T1 affine.
    command = [fsl / "bin/applywarp", "--in=" + str(clean_native),
               "--ref=" + case["mni_template"], "--warp=" + str(registration / "T1_to_MNI_coeff.nii.gz"),
               "--premat=" + str(registration / "example_func2highres.mat"), "--interp=spline",
               "--mask=" + str(registration / "brain_MNI152_2mm.nii.gz"),
               "--out=" + str(output / "clean_mni_fnirt.nii.gz")]
    commands.append({"name": "mni_resampling", "argv": list(map(str, command))})
    (output / "commands.private.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "status.private.json").write_text(json.dumps({"stage": "mni_resampling", "status": "running"}) + "\n")
    process_start = time.perf_counter()
    with (output / "mni_resampling.log").open("w") as stream:
        result = subprocess.run(list(map(str, command)), env=environment, stdout=stream, stderr=subprocess.STDOUT)
    native_wall = time.perf_counter() - started
    sampling_wall = time.perf_counter() - process_start
    destination = output / "clean_mni_fnirt.nii.gz"
    if result.returncode not in (0, 255) or not destination.is_file():
        raise RuntimeError("Original final applywarp failed")
    subprocess.run(["gzip", "-t", str(destination)], check=True)
    image = nib.load(destination)
    reference = nib.load(case["mni_template"])
    values = np.asarray(image.dataobj)
    mask = np.asarray(nib.load(registration / "brain_MNI152_2mm.nii.gz").dataobj) > 0
    raw = nib.load(case["bold"])
    if image.shape != (*reference.shape, raw.shape[3]) or not np.allclose(image.affine, reference.affine, atol=1e-4):
        raise ValueError("Final original MNI output has wrong grid")
    if not np.isfinite(values).all() or np.any(values[~mask]):
        raise ValueError("Invalid final original MNI values")
    stages["mni_resampling"] = {"process_wall_seconds_including_io": sampling_wall,
                                "original_exit_code": result.returncode,
                                "complete_output_checks_passed": True,
                                "shape": list(image.shape), "sha256": sha256(destination),
                                "all_finite": True, "outside_mask_max_abs": 0.0}
    report = {"schema_version": 1, "source_revision": args.source_revision,
              "registration_backend": "fnirt", "stages": stages,
              "whole_workflow_wall_seconds": native_wall, "input_sha256": input_hashes,
              "output_sha256": {"clean_native": sha256(clean_native), "clean_mni": sha256(destination)},
              "script_sha256": {name: sha256(directory / name) for name in
                                  ("run_native_matched.py", "official_registration.py", "official_denoising.py", "run_native_matched_pipeline.py")},
              "timing_boundary": "Continuous sequential workflow, including stage interpreter startup and in-stage integrity checks; input preflight hashing and final MNI integrity checks excluded. No skipped/reused computational stage.",
              "limits": ["Raw BOLD and SBRef; same archived reconstruction T1 input, not verified scanner-raw T1.",
                         "Original FreeSurfer SynthStrip, original FSL and original ICA-AROMA plus independent same-definition NumPy nuisance regression.",
                         "No GDC/B0, BET/dilation, FIX, spatial smoothing or additional bandpass.",
                         "One run on a shared host; timings are observations rather than a controlled speedup ratio."],
              "privacy": "Only anonymous scalars and hashes. Images, designs and commands with paths remain private."}
    (output / "pipeline.public.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "status.private.json").write_text(json.dumps({"stage": "complete", "status": "passed"}) + "\n")
    print(json.dumps({"completed": "native_pipeline", "whole_workflow_wall_seconds": native_wall}), flush=True)


if __name__ == "__main__":
    main()
