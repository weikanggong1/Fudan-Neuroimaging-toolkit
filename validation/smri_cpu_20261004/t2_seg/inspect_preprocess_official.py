"""Reference-only: inspect original preprocessing without loading the networks.

Compile the locally installed reference's pure image helper definitions;
the FNIT production runtime does not read or execute this source.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tree = ast.parse(args.source.read_text())
    by_name = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    names, pending = set(), ["preprocess"]
    while pending:
        name = pending.pop()
        if name in names:
            continue
        names.add(name)
        pending.extend(node.func.id for node in ast.walk(by_name[name])
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id in by_name and node.func.id not in names)
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "nib": nib, "os": os,
                 "gaussian_filter": gaussian_filter,
                 "RegularGridInterpolator": RegularGridInterpolator}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(args.source), "exec"), namespace)
    image, affine, header, spacing, shape, pad_index, crop_index = namespace["preprocess"](
        str(args.input), False, min_pad=128)
    np.save(args.output, image[0, ..., 0].astype(np.float32))
    source = nib.load(str(args.input))
    factors = np.sqrt(np.sum(source.affine * source.affine, axis=0))[:3]
    start = -(factors - 1) / (2 * factors)
    step = 1.0 / factors
    stop = start + step * np.ceil(np.asarray(source.shape) * factors)
    report = {"scope": "original_function_preprocessing_only",
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "input_shape": list(source.shape),
              "input_zooms": list(map(float, source.header.get_zooms())),
              "affine_factors": factors.tolist(),
              "arange_axis_counts": [len(np.arange(a, b, c)) for a, b, c in zip(start, stop, step)],
              "fixed_ceil_counts": np.ceil(np.asarray(source.shape) * factors).astype(int).tolist(),
              "padded_shape": list(image.shape), "content_shape": shape,
              "padding_index": pad_index.tolist(), "affine": affine.tolist(),
              "numpy_version": np.__version__}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
