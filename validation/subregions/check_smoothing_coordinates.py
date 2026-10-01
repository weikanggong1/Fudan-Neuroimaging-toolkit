"""Compare real atlas alpha smoothing before and after a coordinate transform.

The C++ extension is used only as a validation reference, never by the FNIT API.
This component check is separate from whole-subject benchmarks.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.recipes import ThalamusRecipe
from fnit.gems.smoothing import smooth_atlas_alphas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--reference-extension", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("gemsbindings", args.reference_extension)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.setGlobalDefaultNumberOfThreads(4)
    atlas = GEMSAtlas.from_freesurfer(args.atlas / "AtlasMesh.gz",
                                     args.atlas / "compressionLookupTable.txt")
    classes = ThalamusRecipe("thalamus", args.atlas).intensity_groups(atlas, 0)
    grouped = np.zeros((len(atlas.vertices), int(classes.max()) + 1), np.float32)
    for channel, group in enumerate(classes):
        grouped[:, group] += atlas.alphas[:, channel]
    cached = smooth_atlas_alphas(atlas, classes, 1.5, device="cpu")
    report = {"mode": "real_atlas_coordinate_component_check", "cases": []}
    for scale in (1.0, 2.0):
        transform = np.diag([scale, scale, scale, 1.0])
        collection = native.KvlMeshCollection()
        collection.read(str(args.atlas / "AtlasMesh.gz"))
        collection.reference_mesh.alphas = grouped.astype(np.float64)
        collection.transform(native.KvlTransform(np.asfortranarray(transform)))
        collection.smooth(1.5)
        expected = np.asarray(collection.reference_mesh.alphas)
        aligned = smooth_atlas_alphas(atlas.transformed(transform), classes, 1.5, device="cpu")
        report["cases"].append({"scale": scale,
            "original_cache_mean_abs_difference": float(np.abs(cached - expected).mean()),
            "aligned_smoothing_mean_abs_difference": float(np.abs(aligned - expected).mean()),
            "aligned_smoothing_max_abs_difference": float(np.abs(aligned - expected).max())})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
