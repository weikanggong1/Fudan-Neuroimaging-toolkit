"""Isolated official-script observation for diagnosing actual input differences.

Execute only in the independent FreeSurfer reference environment. This driver
calls the unchanged, hash-verified official script and records its actual
conformed volume and model input/output; it is never imported by FNIT.
Observation overhead is not a production timing result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import surfa as sf
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("preserve prior control attempts and use a fresh directory")
    if sha256(args.official_script) != args.script_sha256 or sha256(args.weights) != args.weight_sha256:
        raise ValueError("official source or weight differs from the frozen provenance")
    args.output_dir.mkdir(parents=True)
    observed = {"schema": "fnit.smri.cpu.strip.original_control.v1",
                "original_script_sha256": args.script_sha256, "weight_sha256": args.weight_sha256,
                "input_sha256": sha256(args.input), "torch_version": torch.__version__,
                "worker_sha256": sha256(__file__), "conform_calls": [], "network_calls": [],
                "stages": [], "timing_scope": "Observed function bodies exclude array hashing; use independent complete CLI for speed."}
    original_load = sf.load_volume

    def capture_load(*positional, **keywords):
        started = time.perf_counter()
        result = original_load(*positional, **keywords)
        observed["stages"].append({"name": "load_input", "seconds": time.perf_counter() - started})
        data = np.asarray(result.data)
        observed["loaded_input"] = {"array_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                                    "shape": list(data.shape), "dtype": str(data.dtype),
                                    "affine": result.geom.vox2world.matrix.tolist(),
                                    "voxsize": result.geom.voxsize.tolist(),
                                    "minimum": float(data.min()), "maximum": float(data.max())}
        return result

    sf.load_volume = capture_load
    original_conform = sf.Volume.conform

    def capture_conform(self, *positional, **keywords):
        started = time.perf_counter()
        result = original_conform(self, *positional, **keywords)
        observed["stages"].append({"name": "conform", "seconds": time.perf_counter() - started})
        index = len(observed["conform_calls"])
        path = args.output_dir / ("conform_%d.private.npy" % index)
        data = np.asarray(result.data)
        np.save(path, data)
        observed["conform_calls"].append({"array_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                                           "file_sha256": sha256(path), "shape": list(data.shape),
                                           "dtype": str(data.dtype),
                                           "affine": result.geom.vox2world.matrix.tolist(),
                                           "voxsize": result.geom.voxsize.tolist()})
        return result

    sf.Volume.conform = capture_conform
    for method_name in ("crop_to_bbox", "reshape", "resample_like", "connected_component_mask", "save"):
        original_method = getattr(sf.Volume, method_name)

        def observed_method(self, *positional, _original=original_method, _name=method_name, **keywords):
            started = time.perf_counter()
            result = _original(self, *positional, **keywords)
            observed["stages"].append({"name": _name, "seconds": time.perf_counter() - started})
            return result

        setattr(sf.Volume, method_name, observed_method)
    original_load_weights = torch.load

    def observed_load_weights(*positional, **keywords):
        started = time.perf_counter()
        result = original_load_weights(*positional, **keywords)
        observed["stages"].append({"name": "load_weights", "seconds": time.perf_counter() - started})
        return result

    torch.load = observed_load_weights
    original_call = torch.nn.Module._call_impl

    def capture_network(self, *positional, **keywords):
        if self.__class__.__name__ != "StripModel":
            return original_call(self, *positional, **keywords)
        index = len(observed["network_calls"])
        array = positional[0].detach().cpu().numpy()
        path = args.output_dir / ("network_input_%d.private.npy" % index)
        np.save(path, array)
        row = {"input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
               "input_file_sha256": sha256(path), "shape": list(array.shape),
               "dtype": str(array.dtype), "threads": torch.get_num_threads()}
        started = time.perf_counter()
        output = original_call(self, *positional, **keywords)
        row["forward_seconds"] = time.perf_counter() - started
        observed["stages"].append({"name": "network_forward", "seconds": row["forward_seconds"]})
        prediction = output.detach().cpu().numpy()
        path = args.output_dir / ("network_output_%d.private.npy" % index)
        np.save(path, prediction)
        row["output_sha256"] = hashlib.sha256(prediction.tobytes()).hexdigest()
        row["output_file_sha256"] = sha256(path)
        observed["network_calls"].append(row)
        return output

    torch.nn.Module._call_impl = capture_network
    sys.argv = [str(args.official_script), "-i", str(args.input),
                "-o", str(args.output_dir / "image.nii.gz"),
                "-m", str(args.output_dir / "mask.nii.gz"),
                "-d", str(args.output_dir / "distance.nii.gz"),
                "--model", str(args.weights), "-t", "8"]
    try:
        runpy.run_path(str(args.official_script), run_name="__main__")
    except SystemExit as exit_status:
        if exit_status.code not in (None, 0):
            raise
    observed["input_unchanged"] = observed["input_sha256"] == sha256(args.input)
    observed["source_unchanged"] = args.script_sha256 == sha256(args.official_script)
    observed["weight_unchanged"] = args.weight_sha256 == sha256(args.weights)
    observed["status"] = "complete"
    (args.output_dir / "control.private.json").write_text(json.dumps(observed, indent=2) + "\n")


if __name__ == "__main__":
    main()
