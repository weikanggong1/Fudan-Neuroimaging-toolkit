"""Capture frozen FNIT's real SynthStrip preprocessing without a network pass."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fnit._nib import load_image, new_image
import fnit._nib as image_io
import fnit.synthstrip.pipeline as pipeline


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preserve-input-geometry", action="store_true",
                        help="diagnostic only: retain original header before conform")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a fresh directory to preserve earlier controls")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    image = load_image(args.input)
    source = np.asanyarray(image.dataobj)
    frame = (pipeline._geometry_image(source, image, image.affine) if args.preserve_input_geometry
             else new_image(source, image))
    report = {"schema": "fnit.smri.cpu.strip.preprocess_control.v1",
              "input_sha256": sha256(args.input), "source_sha256": sha256(pipeline.__file__),
              "shared_nib_sha256": sha256(image_io.__file__),
              "preserve_input_geometry_prototype": args.preserve_input_geometry,
              "original_header_zooms": [float(value) for value in image.header.get_zooms()],
              "loaded_input": {"shape": list(source.shape), "dtype": str(source.dtype),
                               "array_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                               "affine": frame.affine.tolist(),
                               "zooms": [float(value) for value in frame.header.get_zooms()],
                               "minimum": float(source.min()), "maximum": float(source.max())},
              "stages": []}

    def capture(name, result):
        data = np.asanyarray(result.dataobj)
        path = args.output_dir / (name + ".private.npy")
        np.save(path, data)
        report["stages"].append({"name": name, "shape": list(data.shape), "dtype": str(data.dtype),
                                 "array_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                                 "affine": result.affine.tolist()})
        return result

    conformed = capture("conform", pipeline._conform_lia_1mm(frame))
    conformed = capture("crop", pipeline._crop_nonzero(conformed))
    shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
    conformed = capture("reshape", pipeline._reshape_center(conformed, shape))
    array = np.asanyarray(conformed.dataobj)
    array = array - array.min()
    percentile = np.percentile(array, 99)
    if percentile > 0:
        array = np.clip(array / percentile, 0, 1)
    tensor = np.ascontiguousarray(array[None, None])
    np.save(args.output_dir / "network_input.private.npy", tensor)
    report["normalized_input"] = {"array_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
                                  "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                  "percentile_99": float(percentile)}
    report["status"] = "complete"
    (args.output_dir / "control.private.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
