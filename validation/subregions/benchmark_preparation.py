"""Compare preparation on saved real-subject meshes with bounded GPU memory.

Uses existing FNIT inputs and saved reference meshes; no reference software
is executed. Reports component timings separately from whole-subject runs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.context import SubregionContext
from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe
from fnit.gems.recipes.thalamus import ThalamusRecipe
import fnit.gems.smoothing as smoothing


def identity(path):
    path = Path(path)
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def load_old(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tick():
    torch.cuda.synchronize()
    return perf_counter()


@contextmanager
def phase_timings(module, names, timings):
    originals = {}
    for name in names:
        if not hasattr(module, name):
            continue
        function = getattr(module, name)
        originals[name] = function
        def wrapper(*args, _fn=function, _name=name, **kwargs):
            started = tick()
            value = _fn(*args, **kwargs)
            timings[_name] = timings.get(_name, 0.) + tick() - started
            return value
        setattr(module, name, wrapper)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(module, name, function)


def measured(function, *, module, phases):
    torch.cuda.reset_peak_memory_stats()
    timings = {}
    with torch.inference_mode(), phase_timings(module, phases, timings):
        started = tick()
        value = function()
        elapsed = tick() - started
    return value, {"seconds": elapsed, "phase_seconds": timings,
                   "remaining_seconds": elapsed - sum(timings.values()),
                   "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--old-smoothing", required=True, type=Path)
    parser.add_argument("--old-hippo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.14)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    old_smoothing = load_old(args.old_smoothing, "fnit.gems._benchmark_old_smoothing")
    old_hippo = load_old(args.old_hippo, "fnit.gems.recipes._benchmark_old_hippo")
    root = args.root
    prepared = root.parent / "fnit_subregions_plus_20260928/samseg_native_build_20260929"
    mri = root.parent / "reconall_reference_gpucw1/fs_sub01/mri"
    context = SubregionContext.prepare(mri / "norm.mgz", need_coarse=True, need_parc=True,
                                       coarse_segmentation=mri / "aseg.mgz", wmparc=mri / "wmparc.mgz")
    result = {"kind": "real_saved_mesh_preparation_comparison", "host": socket.gethostname(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "float32",
              "matmul_tf32": True, "cudnn_tf32": True, "threads": 4,
              "allocator_fraction": .14, "cases": {},
              "source_files": {"old_smoothing": identity(args.old_smoothing),
                               "new_smoothing": identity(smoothing.__file__),
                               "old_hippo": identity(args.old_hippo),
                               "new_hippo": identity(__import__(HippoAmygdalaRecipe.__module__, fromlist=["x"]).__file__)},
              "gpu_load_before": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()}
    for name, directory in (("thalamus", "port_thalamus_debug"),
                             ("hippo-amygdala-left", "port_hippo_integrated_fullfix_left_tmp")):
        stage = prepared / directory
        image_path = stage / "processedImageMasked.mgz"
        aligned_path = stage / "alignedAtlasImage.mgz"
        mesh_path = stage / "warpedOriginalMesh.txt.gz"
        lut_path = root / "atlases" / name / "compressionLookupTable.txt"
        image, aligned = nib.load(image_path), nib.load(aligned_path)
        atlas = GEMSAtlas.from_freesurfer(mesh_path, lut_path).transformed(
            np.linalg.inv(image.affine) @ aligned.affine, transform_reference=True)
        recipe = ThalamusRecipe(name, root / "atlases" / name) if name == "thalamus" else HippoAmygdalaRecipe("left", root / "atlases" / name)
        classes = recipe.intensity_groups(atlas, 0)
        row = {"inputs": [identity(path) for path in (image_path, aligned_path, mesh_path, lut_path)],
               "vertices": len(atlas.vertices), "tetrahedra": len(atlas.tetrahedra),
               "classes": int(classes.max()) + 1,
               "reference_grid_shape": (np.floor(atlas.reference_vertices.max(0)).astype(int) + 1).tolist(),
               "smoothing": {}}
        cache = {}
        for sigma in (1.5, .75):
            old, old_time = measured(lambda: old_smoothing.smooth_atlas_alphas(atlas, classes, sigma, device="cuda:0"),
                                      module=old_smoothing, phases=("rasterize_priors", "convolve1d"))
            new, new_time = measured(lambda: smoothing.smooth_atlas_alphas(atlas, classes, sigma, device="cuda:0", cache=cache),
                                      module=smoothing, phases=("rasterize_priors", "_separable_convolve"))
            row["smoothing"][str(sigma)] = {"old": old_time, "new": new_time,
                                           "observed_ratio_old_over_new": old_time["seconds"] / new_time["seconds"],
                                           "max_abs_alpha_difference": float(np.max(np.abs(old - new))),
                                           "mean_abs_alpha_difference": float(np.mean(np.abs(old - new))),
                                           "vertex_normalization_max_error": float(np.max(np.abs(new.sum(1) - 1)))}
        cache.clear()
        torch.cuda.empty_cache()
        if name == "hippo-amygdala-left":
            old_recipe = old_hippo.HippoAmygdalaRecipe("left", root / "atlases" / name)
            recipe._preparation_device = torch.device("cuda:0")
            old, old_time = measured(lambda: old_recipe.gaussian_hyperparameters(context, atlas, classes),
                                      module=old_hippo, phases=("rasterize_priors",))
            new_module = __import__(HippoAmygdalaRecipe.__module__, fromlist=["x"])
            new, new_time = measured(lambda: recipe.gaussian_hyperparameters(context, atlas, classes),
                                      module=new_module, phases=("rasterize_priors",))
            row["hyperparameters"] = {"old": old_time, "new": new_time,
                                       "old_means": old[0].tolist(), "new_means": new[0].tolist(),
                                       "max_abs_mean_difference": float(np.max(np.abs(old[0] - new[0]))),
                                       "counts_equal": bool(np.array_equal(old[1], new[1])),
                                       "observed_ratio_old_over_new": old_time["seconds"] / new_time["seconds"]}
        result["cases"][name] = row
        print(json.dumps({"case": name, "smoothing": row["smoothing"], "hyperparameters": row.get("hyperparameters")}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["gpu_load_after"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
