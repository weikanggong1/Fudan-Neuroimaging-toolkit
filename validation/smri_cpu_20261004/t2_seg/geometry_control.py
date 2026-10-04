"""Check keep-geometry branches on the same real saved inference labels.

The official mode executes only the original save_volume helper. This is
an output-mode control, not another complete reconstruction benchmark.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["official", "fnit"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels, source = nib.load(args.labels), nib.load(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "official":
        import surfa as sf
        tree = ast.parse(args.source.read_text())
        by_name = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        names, pending = set(), ["save_volume"]
        while pending:
            name = pending.pop()
            if name in names:
                continue
            names.add(name)
            pending.extend(node.func.id for node in ast.walk(by_name[name])
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id in by_name and node.func.id not in names)
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"np": np, "nib": nib, "os": os, "sf": sf}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(args.source), "exec"), namespace)
        namespace["save_volume"](np.asanyarray(labels.dataobj), labels.affine, labels.header,
                                  str(args.output), dtype="int32",
                                  resample_like_image=sf.load_volume(args.input))
    else:
        from fnit.synthseg_parc.synthseg_plus import _native
        _native(labels, source).save(args.output)
    print(json.dumps({"scope": "same_real_labels_keep_geometry_output_control",
                      "mode": args.mode,
                      "label_input_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                      "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
