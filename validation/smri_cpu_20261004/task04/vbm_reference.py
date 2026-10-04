"""Independent official CPU reference for the declared SynthStrip/FAST/FSL chain.

Run through the validation reference environment, never import into FNIT.
This uses SynthStrip to align the pipeline's brain-extraction method; it is
not the standard FSL-VBM BET preprocessing pipeline. Preserve every stage's
command, return code, and time even when a later stage fails.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--synthstrip-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    report_path = output / "reference_steps.private.json"
    fsl = Path(os.environ["FSLDIR"]) / "bin"
    fs = Path(os.environ["FREESURFER_HOME"]) / "bin"
    report = {
        "status": "running",
        "scope": "official SynthStrip -> FAST -> FLIRT -> GM FNIRT -> applywarp -> Jacobian modulation",
        "standard_fsl_vbm_bet_pipeline": False,
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "input_sha256": digest(args.image),
        "template_sha256": digest(args.template),
        "reference_mask_sha256": digest(args.reference_mask),
        "synthstrip_weights_sha256": digest(args.synthstrip_weights),
        "stages": [],
    }
    started = time.perf_counter()

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command):
        stage = {"name": name, "argv": [str(value) for value in command]}
        report["stages"].append(stage)
        save()
        with (output / (name + ".log")).open("wb") as stream:
            before = time.perf_counter()
            completed = subprocess.run(stage["argv"], stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
            stage["wall_seconds"] = time.perf_counter() - before
            stage["returncode"] = completed.returncode
        save()
        if completed.returncode:
            report["status"] = "failed"
            save()
            raise SystemExit(completed.returncode)

    brain = output / "T1_brain.nii.gz"
    mask = output / "brain_mask.nii.gz"
    prefix = output / "T1_brain"
    gm = output / "T1_brain_pve_1.nii.gz"
    affine = output / "gm_affine.mat"
    coefficient = output / "gm_coeff.nii.gz"
    dense = output / "gm_dense.nii.gz"
    jacobian = output / "T1_GM_JAC_nl.nii.gz"
    warped = output / "T1_GM_to_template_GM.nii.gz"
    modulated = output / "T1_GM_to_template_GM_mod.nii.gz"
    run("synthstrip", [fs / "mri_synthstrip", "-i", args.image, "-o", brain,
                       "-m", mask, "-t", args.threads, "--model", args.synthstrip_weights])
    run("fast", [fsl / "fast", "-t", 1, "-n", 3, "-b", "-B", "-o", prefix, brain])
    run("flirt", [fsl / "flirt", "-in", gm, "-ref", args.template, "-omat", affine])
    run("fnirt", [fsl / "fnirt", "--in=" + str(gm), "--ref=" + str(args.template),
                  "--aff=" + str(affine), "--config=GM_2_MNI152GM_2mm",
                  "--refmask=" + str(args.reference_mask), "--cout=" + str(coefficient),
                  "--fout=" + str(dense), "--jout=" + str(jacobian)])
    run("applywarp", [fsl / "applywarp", "-i", gm, "-r", args.template,
                      "-w", coefficient, "-o", warped, "--interp=trilinear", "--datatype=float"])
    run("modulation", [fsl / "fslmaths", warped, "-mul", jacobian, modulated, "-odt", "float"])
    report["status"] = "complete"
    report["outputs"] = {path.name: {"sha256": digest(path), "size": path.stat().st_size}
                         for path in output.glob("*.nii.gz")}
    report["outputs"][affine.name] = {"sha256": digest(affine)}
    report["jacobian_definition"] = "Official fnirt analytic coefficient Jacobian; compare separately from FNIT dense finite differences."
    save()
    print(json.dumps({"status": "complete", "seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
