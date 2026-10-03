"""Native MRtrix nearest against full real FS labels and actual transforms."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bindings", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--controlled-only", action="store_true")
parser.add_argument("--controlled-format", choices=("mgz", "nifti", "canonical-nifti"), default="mgz")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
binary = Path("/public/software/apps/MRtrix3/3.0.3/bin/mrtransform")
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = "/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/lib"
environment["OMP_NUM_THREADS"] = "8"
bindings = json.loads(args.bindings.read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_anatomy(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modules = {name: load_anatomy(path, "anatomy_" + name)
           for name, path in (("baseline", args.baseline), ("candidate", args.candidate))}
report = {"scope": "real FS integer labels; natural transform and separately labelled controlled operator cases",
          "bindings_sha256": sha(args.bindings), "native_binary": str(binary),
          "native_binary_sha256": sha(binary), "source_sha256": {
              "baseline": sha(args.baseline), "candidate": sha(args.candidate),
              "harness": sha(__file__)}, "cases": []}


def run_case(name, source, target, transform, scope):
    directory = args.output / name
    directory.mkdir()
    matrix_path = directory / "target_to_source_world.txt"
    np.savetxt(matrix_path, transform, fmt="%.17g")
    output_path = directory / "native.nii.gz"
    command = [str(binary), str(source), str(output_path), "-linear", str(matrix_path),
               "-template", str(target), "-interp", "nearest", "-datatype", "int32",
               "-nthreads", "8"]
    started = time.perf_counter()
    with (directory / "native.log").open("w") as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    native_wall = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"native command failed: {command}")
    source_image = nib.load(source)
    target_image = nib.load(target)
    native_image = nib.load(output_path)
    # MRtrix realigns image axes; undo its output permutation/flips so the
    # comparison uses the original NiBabel target voxel order without resampling.
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(native_image.affine),
        nib.orientations.io_orientation(target_image.affine))
    native_array = nib.orientations.apply_orientation(np.asarray(native_image.dataobj), orientation)
    aligned_affine = native_image.affine @ nib.orientations.inv_ornt_aff(orientation, native_image.shape)
    affine_error = float(np.max(np.abs(aligned_affine - target_image.affine)))
    assert affine_error < 1e-4 and native_array.shape == target_image.shape[:3]
    labels = torch.as_tensor(np.asarray(source_image.dataobj, dtype=np.int32).copy())
    arguments = {"labels": labels, "source_affine": source_image.affine,
                 "target_shape": target_image.shape[:3], "target_affine": target_image.affine,
                 "target_to_source_world": transform}
    record = {"name": name, "scope": scope, "source": str(source), "target": str(target),
              "source_sha256": sha(source), "target_sha256": sha(target),
              "transform_sha256": sha(matrix_path), "transform": transform.tolist(),
              "source_affine": source_image.affine.tolist(), "target_affine": target_image.affine.tolist(),
              "native_command": command, "native_seconds": native_wall,
              "native_output_sha256": sha(output_path), "aligned_native_affine_max_error": affine_error,
              "output_voxels": int(native_array.size), "results": {}}
    arrays = {}
    for arm, module in modules.items():
        started = time.perf_counter()
        actual = module.resample_labels_nearest(**arguments)
        seconds = time.perf_counter() - started
        array = actual.numpy()
        arrays[arm] = array
        changed = array != native_array
        record["results"][arm] = {"seconds_cpu": seconds,
            "disagreements": int(np.count_nonzero(changed)),
            "nonbackground_disagreements": int(np.count_nonzero(changed & ((array != 0) | (native_array != 0)))),
            "output_array_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        np.save(directory / (arm + ".npy"), array, allow_pickle=False)
    record["baseline_candidate_disagreements"] = int(np.count_nonzero(arrays["baseline"] != arrays["candidate"]))
    report["cases"].append(record)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(record), flush=True)


for case in ("sub-CON03", "sub-CON01"):
    binding = bindings["cases"][case]
    source = Path(binding["anatomy"]["files"]["mri/aparc+aseg.mgz"]["path"])
    assert sha(source) == binding["anatomy"]["files"]["mri/aparc+aseg.mgz"]["sha256"]
    target = Path(binding["selected_inputs"]["dwi"])
    matrix = Path(binding["prior_output_dir"]) / "dwi_to_t1_world.csv"
    transform = np.loadtxt(matrix, delimiter=",")
    if not args.controlled_only:
        run_case(case + "_natural", source, target, transform,
                 "natural completed FNIT DWI-to-T1 rigid transform, unmodified full actual MRI")
    if case == "sub-CON03":
        # Full real label image and actual scanner affine; controlled half-voxel
        # shifts test operator corners and are not natural registration or SC.
        image = nib.load(source)
        if args.controlled_format == "canonical-nifti":
            image = nib.as_closest_canonical(image)
        template = args.output / "CON03_real_labels_template.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj, dtype=np.int32), image.affine), template)
        controlled_source = source if args.controlled_format == "mgz" else template
        for axis in range(3):
            shift = np.eye(4)
            shift[:3, 3] = image.affine[:3, axis] * .5
            run_case(f"sub-CON03_controlled_half_axis{axis}", controlled_source, template, shift,
                     f"controlled +half source-voxel shift of full real CON03 label MRI, {args.controlled_format}; not natural registration or E2E")
