"""Compare real FS -> 5TT/GMWMI to exact official LUT and native commands."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
for name in ("bindings", "source", "resource", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
native = Path("/public/software/apps/MRtrix3/3.0.3/bin")
colors = Path("/public/software/apps/Freesurfer/8.2.0-1/FreeSurferColorLUT.txt")
exact = Path("/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1/task_03/oracle/mrtrix3-026e850d")
reference_lut = exact / "share/mrtrix3/_5ttgen/FreeSurfer2ACT_sgm_amyg_hipp.txt"
environment = os.environ.copy()
environment.update(LD_LIBRARY_PATH="/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/lib",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


frozen = args.output / "frozen_anatomy"
frozen.mkdir()
(frozen / "__init__.py").write_text("")
shutil.copyfile(args.source, frozen / "anatomy.py")
shutil.copyfile(args.resource, frozen / args.resource.name)
spec = importlib.util.spec_from_file_location("tissue_audit", frozen / "__init__.py",
                                            submodule_search_locations=[str(frozen)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
spec = importlib.util.spec_from_file_location("tissue_audit.anatomy", frozen / "anatomy.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
bindings = json.loads(args.bindings.read_text())
report = {"scope": "full real FS labels, unchanged 5TT/GMWMI functions; CPU native reference",
          "reference_commit": "026e850d171ec2a12f09865d31b8332d23d7ecf6",
          "source_sha256": sha(args.source), "resource_sha256": sha(args.resource),
          "bindings_sha256": sha(args.bindings), "lut_sha256": sha(reference_lut),
          "FreeSurferColorLUT_sha256": sha(colors),
          "native_binary_sha256": {name: sha(native/name) for name in
              ("labelconvert", "mrcalc", "mrcat", "mrconvert", "5tt2gmwmi")}, "cases": []}


def align(image, source):
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                nib.orientations.io_orientation(source.affine))
    return nib.orientations.apply_orientation(np.asarray(image.dataobj), orientation)


for case in ("sub-CON03", "sub-CON01"):
    source_path = Path(bindings["cases"][case]["anatomy"]["files"]["mri/aparc+aseg.mgz"]["path"])
    directory = args.output / case
    directory.mkdir()
    source_image = nib.load(source_path)
    indices = directory / "indices.nii.gz"
    five_path = directory / "native_five_tissue.nii.gz"
    gmwmi_path = directory / "native_gmwmi.nii.gz"
    commands = [[str(native/"labelconvert"), str(source_path), str(colors), str(reference_lut), str(indices), "-nthreads", "8"]]
    channels = []
    for channel, name in enumerate(("cgm", "sgm", "wm", "csf", "path"), 1):
        path = directory / (name + ".mif")
        channels.append(str(path))
        commands.append([str(native/"mrcalc"), str(indices), str(channel), "-eq", str(path), "-nthreads", "8"])
    commands.extend([[str(native/"mrcat"), *channels, str(five_path), "-axis", "3", "-datatype", "float32", "-nthreads", "8"],
                     [str(native/"5tt2gmwmi"), str(five_path), str(gmwmi_path), "-nthreads", "8"]])
    command_records = []
    for index, command in enumerate(commands):
        started = time.perf_counter()
        with (directory / f"command_{index}.log").open("w") as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        command_records.append({"argv": command, "wall_seconds": time.perf_counter()-started,
                                "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"native reference failed: {command}")
    segmentation = torch.as_tensor(np.asarray(source_image.dataobj, dtype=np.int32).copy())
    started = time.perf_counter()
    five = module.freesurfer_five_tissue(segmentation)
    five_wall = time.perf_counter() - started
    started = time.perf_counter()
    gmwmi = module.gmwmi_from_five_tissue(five)
    gmwmi_wall = time.perf_counter() - started
    expected_five = align(nib.load(five_path), source_image)
    expected_gmwmi = align(nib.load(gmwmi_path), source_image)
    five_delta = five.numpy() - expected_five
    gmwmi_delta = gmwmi.numpy() - expected_gmwmi
    report["cases"].append({"case": case, "source": str(source_path), "source_sha256": sha(source_path),
        "shape": list(segmentation.shape), "native_commands": command_records,
        "five_tissue": {"cpu_seconds": five_wall, "nonzero_differences": int(np.count_nonzero(five_delta)),
                        "max_abs": float(np.max(np.abs(five_delta)))},
        "gmwmi": {"cpu_seconds": gmwmi_wall, "nonzero_differences": int(np.count_nonzero(gmwmi_delta)),
                  "max_abs": float(np.max(np.abs(gmwmi_delta)))},
        "output_sha256": {"native_five": sha(five_path), "native_gmwmi": sha(gmwmi_path)}})
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2), flush=True)
