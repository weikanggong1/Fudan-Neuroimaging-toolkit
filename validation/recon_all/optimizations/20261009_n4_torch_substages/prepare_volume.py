"""冻结真实 conformed N4 输入；nibabel 解码，无插值和强度修改。"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output-raw", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
args = parser.parse_args()
image = nib.load(str(args.input))
array = np.asarray(image.dataobj, dtype=np.float32)
if array.ndim != 3:
    raise ValueError("N4 frozen input must be 3D")
args.output_raw.parent.mkdir(parents=True, exist_ok=True)
array.ravel(order="F").tofile(args.output_raw)
args.output_json.write_text(json.dumps({"kind": "real_frozen_conformed_n4_input_not_raw_t1_pipeline",
    "source_path": str(args.input), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    "raw_path": str(args.output_raw), "raw_sha256": hashlib.sha256(args.output_raw.read_bytes()).hexdigest(),
    "shape": list(array.shape), "spacing_mm": [float(x) for x in image.header.get_zooms()[:3]],
    "affine": image.affine.tolist(), "original_dtype": str(image.get_data_dtype()),
    "raw_dtype": "float32", "raw_order": "Fortran x-fast", "interpolation": False}, indent=2) + "\n")
