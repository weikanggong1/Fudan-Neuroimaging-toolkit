"""Observe the unchanged official SynthSR in its independent reference environment.

This validation driver times original functions and saves the official floating
NPZ branch before the original byte writer mutates its array. It is never used
by production FNIT, and its observed process time is not a CLI speed result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np


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
    parser.add_argument("--ct", action="store_true")
    parser.add_argument("--disable-flipping", action="store_true")
    parser.add_argument("--disable-sharpening", action="store_true")
    parser.add_argument("--save-network-arrays", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a fresh directory and preserve prior attempts")
    if sha256(args.official_script) != args.script_sha256 or sha256(args.weights) != args.weight_sha256:
        raise ValueError("official source or model differs from frozen provenance")
    args.output_dir.mkdir(parents=True)
    observed = {"schema": "fnit.smri.cpu.sr.original_control.v1", "worker_sha256": sha256(__file__),
                "original_script_sha256": args.script_sha256, "weight_sha256": args.weight_sha256,
                "input_sha256": sha256(args.input), "stages": [], "network_calls": [],
                "timing_scope": "Original function bodies; array capture and extra NPZ write excluded. Use separate complete CLI timing."}
    namespace = runpy.run_path(str(args.official_script), run_name="_fnit_independent_reference_")
    original_globals = namespace["predict"].__globals__
    for function_name in ("preprocess", "postprocess"):
        original = original_globals[function_name]

        def measured(*positional, _function=original, _name=function_name, **keywords):
            started = time.perf_counter()
            result = _function(*positional, **keywords)
            observed["stages"].append({"name": _name, "seconds": time.perf_counter() - started})
            return result

        original_globals[function_name] = measured
    original_build = original_globals["build_model"]

    def measured_build(*positional, **keywords):
        started = time.perf_counter()
        model = original_build(*positional, **keywords)
        observed["stages"].append({"name": "model_constructor", "seconds": time.perf_counter() - started})
        observed["compute_dtype"] = str(model.compute_dtype)
        original_predict = model.predict

        def measured_predict(*prediction_args, **prediction_keywords):
            array = np.asarray(prediction_args[0])
            normalized = np.asarray(array, dtype=np.float32).transpose(0, 4, 1, 2, 3)
            row = {"input_shape": list(array.shape), "input_dtype": str(array.dtype),
                   "float32_ncdhw_input_sha256": hashlib.sha256(normalized.tobytes()).hexdigest()}
            started = time.perf_counter()
            result = original_predict(*prediction_args, **prediction_keywords)
            row["seconds"] = time.perf_counter() - started
            prediction = np.asarray(result)
            row["output_shape"] = list(prediction.shape)
            row["output_dtype"] = str(prediction.dtype)
            row["output_sha256"] = hashlib.sha256(prediction.tobytes()).hexdigest()
            if args.save_network_arrays:
                array_index = len(observed["network_calls"])
                np.save(args.output_dir / ("network_%d_input.npy" % array_index), normalized)
                np.save(args.output_dir / ("network_%d_output.npy" % array_index), prediction.transpose(0, 4, 1, 2, 3))
            observed["network_calls"].append(row)
            observed["stages"].append({"name": "network_forward", "seconds": row["seconds"]})
            return result

        model.predict = measured_predict
        return model

    original_globals["build_model"] = measured_build
    original_save = original_globals["save_volume_byte"]

    def measured_save(volume, affine, header, output):
        # The original NIfTI writer performs volume *= 2. Its NPZ branch must
        # therefore be observed first using a copy of the unmodified result.
        floating = np.array(volume, copy=True)
        observed["floating_output_dtype"] = str(floating.dtype)
        observed["floating_output_sha256"] = hashlib.sha256(floating.tobytes()).hexdigest()
        started = time.perf_counter()
        original_save(floating, affine, header, str(args.output_dir / "image.npz"))
        observed["stages"].append({"name": "write_extra_floating_npz", "seconds": time.perf_counter() - started})
        started = time.perf_counter()
        result = original_save(volume, affine, header, output)
        observed["stages"].append({"name": "write_primary_output", "seconds": time.perf_counter() - started})
        return result

    original_globals["save_volume_byte"] = measured_save
    sys.argv = [str(args.official_script), "--i", str(args.input), "--o", str(args.output_dir / "image.nii.gz"),
                "--model", str(args.weights), "--threads", "8", "--cpu"]
    for name, enabled in (("--ct", args.ct), ("--disable_flipping", args.disable_flipping),
                          ("--disable_sharpening", args.disable_sharpening)):
        if enabled:
            sys.argv.append(name)
    namespace["main"]()
    observed["input_unchanged"] = observed["input_sha256"] == sha256(args.input)
    observed["source_unchanged"] = args.script_sha256 == sha256(args.official_script)
    observed["weight_unchanged"] = args.weight_sha256 == sha256(args.weights)
    observed["tensorflow_version"] = original_globals["tf"].__version__
    observed["private_network_arrays_saved"] = args.save_network_arrays
    observed["status"] = "complete" if (args.output_dir / "image.nii.gz").is_file() else "missing_output"
    (args.output_dir / "profile.public.json").write_text(json.dumps(observed, indent=2) + "\n")
    if observed["status"] != "complete":
        raise RuntimeError("the official script caught an inference failure; inspect its original log")


if __name__ == "__main__":
    main()
