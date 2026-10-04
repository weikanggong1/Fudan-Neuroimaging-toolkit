"""Observe real SynthStrip/SynthSR stages without changing their arithmetic.

This validation worker is separate from the cold CLI timing jobs. Function
wrappers and tensor hashing add observation overhead, so its whole wall clock
must not be used as the production CLI speed result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "device": str(tensor.device),
    }


class Observer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.rows = []
        self.stack = []

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def stage(self, name):
        self.sync()
        row = {"name": name, "parent": self.stack[-1] if self.stack else None}
        self.stack.append(name)
        start = time.perf_counter()
        try:
            yield
        except Exception:
            row["status"] = "failed"
            raise
        else:
            row["status"] = "complete"
        finally:
            self.sync()
            row["seconds"] = time.perf_counter() - start
            self.stack.pop()
            self.rows.append(row)

    def wrap(self, module, name):
        original = getattr(module, name)

        def measured(*args, **kwargs):
            with self.stage(name):
                return original(*args, **kwargs)

        setattr(module, name, measured)


def image_description(path):
    if path.suffix == ".npz":
        array = np.load(path)["vol_data"]
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "all_finite": bool(np.isfinite(array).all()),
                "sha256": sha256_file(path)}
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    return {"shape": list(array.shape), "dtype": str(image.get_data_dtype()),
            "all_finite": bool(np.isfinite(array).all()),
            "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", choices=("synthstrip", "synthsr"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-csf", action="store_true")
    parser.add_argument("--border", type=float, default=1)
    parser.add_argument("--fill", type=float)
    parser.add_argument("--ct", action="store_true")
    parser.add_argument("--lowfield", action="store_true")
    parser.add_argument("--v1", action="store_true")
    parser.add_argument("--disable-flipping", action="store_true")
    parser.add_argument("--disable-sharpening", action="store_true")
    parser.add_argument("--format-checks", action="store_true")
    parser.add_argument("--input-object", action="store_true",
                        help="pass a loaded nibabel image to the public API")
    parser.add_argument("--cpu-channels-last", action="store_true",
                        help="validation prototype only; does not modify FNIT source")
    parser.add_argument("--save-network-arrays", action="store_true",
                        help="save actual inputs/outputs privately for numerical isolation")
    parser.add_argument("--cpu-bn-formula", choices=("pytorch", "subtract_first", "scale_bias"),
                        default="pytorch", help="SynthSR CPU numerical prototype only")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.report.exists() or args.output_dir.exists():
        parser.error("use new report and output directory for each attempt")
    args.output_dir.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    observed_weight = sha256_file(args.weights)
    if observed_weight != args.weight_sha256:
        raise ValueError("weight hash differs from the fixed resource manifest")
    input_hash = sha256_file(args.input)
    module = importlib.import_module("fnit." + args.feature + ".pipeline")
    source_files = sorted(Path(module.__file__).parent.glob("*.py"))
    source_hashes = {path.name: sha256_file(path) for path in source_files}
    image_io = importlib.import_module("fnit._nib")
    observer = Observer(args.device)
    if args.feature == "synthstrip":
        function_names = ("_conform_lia_1mm", "_crop_nonzero", "_reshape_center",
                          "extend_sdt", "_resample_affine", "_largest_filled_component")
        constructor = module.SynthStrip
        constructor_options = {"no_csf": args.no_csf}
        call_options = {"border": args.border, "fill": args.fill}
    else:
        function_names = ("_load_image", "resample_volume", "align_volume_to_ref",
                          "pad_volume", "crop_volume_with_idx", "gaussian_filter")
        constructor = module.SynthSR
        constructor_options = {"lowfield": args.lowfield, "v1": args.v1}
        call_options = {"ct": args.ct, "disable_flipping": args.disable_flipping,
                        "disable_sharpening": args.disable_sharpening}
    for name in function_names:
        observer.wrap(module, name)
    with observer.stage("model_constructor"):
        model = constructor(weights=args.weights, device=args.device,
                            threads=args.threads, **constructor_options)
    if args.cpu_bn_formula != "pytorch":
        if args.feature != "synthsr" or observer.device.type != "cpu":
            parser.error("BatchNorm formula experiment is only for SynthSR CPU")

        def bind_normalization(normalization):
            # These alternate FP32 evaluation expressions are diagnostic.
            # They keep the loaded parameters; production source stays intact.
            def normalize(value):
                shape = (1, -1, 1, 1, 1)
                mean = normalization.running_mean.reshape(shape)
                variance = normalization.running_var.reshape(shape)
                weight = normalization.weight.reshape(shape)
                bias = normalization.bias.reshape(shape)
                inverse_std = torch.rsqrt(variance + normalization.eps)
                scale = weight * inverse_std
                if args.cpu_bn_formula == "subtract_first":
                    return (value - mean) * scale + bias
                return value * scale + (bias - mean * scale)
            normalization.forward = normalize

        for normalization in model.model.modules():
            if isinstance(normalization, torch.nn.BatchNorm3d):
                bind_normalization(normalization)
    if args.cpu_channels_last:
        if observer.device.type != "cpu":
            parser.error("CPU layout prototype cannot change the GPU path")
        with observer.stage("cpu_layout_conversion"):
            model.model.to(memory_format=torch.channels_last_3d)
    original_forward = model.model.forward
    tensors = []

    def measured_forward(tensor):
        if args.cpu_channels_last:
            with observer.stage("cpu_input_layout_conversion"):
                tensor = tensor.contiguous(memory_format=torch.channels_last_3d)
        entry = {"input": tensor_digest(tensor)}
        with observer.stage("network_forward"):
            output = original_forward(tensor)
        entry["output"] = tensor_digest(output)
        if args.save_network_arrays:
            array_index = len(tensors)
            with observer.stage("write_private_network_arrays"):
                np.save(args.output_dir / ("network_%d_input.npy" % array_index), tensor.detach().cpu().numpy())
                np.save(args.output_dir / ("network_%d_output.npy" % array_index), output.detach().cpu().numpy())
        tensors.append(entry)
        return output

    model.model.forward = measured_forward
    if observer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(observer.device)
    with observer.stage("api_call_observed"):
        result = model(image=nib.load(str(args.input)) if args.input_object else args.input,
                       **call_options)
    paths = {"image": args.output_dir / "image.nii.gz"}
    if args.feature == "synthstrip":
        paths.update({"mask": args.output_dir / "mask.nii.gz",
                      "distance": args.output_dir / "distance.nii.gz"})
    with observer.stage("write_primary_outputs"):
        for name, path in paths.items():
            getattr(result, name).save(path) if args.feature == "synthstrip" else result.image.save(path)
    if args.format_checks:
        with observer.stage("write_format_check_outputs"):
            for suffix in (".nii", ".mgz"):
                path = args.output_dir / ("image" + suffix)
                result.image.save(path)
                paths["format_" + suffix[1:]] = path
            if args.feature == "synthsr":
                path = args.output_dir / "image.npz"
                result.image.save(path)
                paths["format_npz"] = path
    report = {
        "schema": "fnit.smri.cpu.strip_sr.profile.v1", "status": "complete",
        "feature": args.feature, "source_revision": args.source_revision,
        "source_files_sha256": source_hashes,
        "source_import_sha256": sha256_file(module.__file__),
        "shared_nib_sha256": sha256_file(image_io.__file__),
        "worker_sha256": sha256_file(__file__), "input_sha256": input_hash,
        "weight_sha256": observed_weight, "weight_size": args.weights.stat().st_size,
        "input_unchanged": input_hash == sha256_file(args.input),
        "weight_unchanged": observed_weight == sha256_file(args.weights),
        "constructor_parameters": constructor_options, "call_parameters": call_options,
        "device": str(observer.device), "threads": torch.get_num_threads(),
        "cpu_channels_last_prototype": args.cpu_channels_last,
        "production_cpu_channels_last": bool(getattr(model, "_cpu_channels_last", False)),
        "private_network_arrays_saved": args.save_network_arrays,
        "cpu_bn_formula_prototype": args.cpu_bn_formula,
        "input_object_api": args.input_object,
        "actual_precision": {"cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                             "cudnn_benchmark": torch.backends.cudnn.benchmark,
                             "cudnn_deterministic": torch.backends.cudnn.deterministic,
                             "autocast_enabled": torch.is_autocast_enabled(),
                             "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.model.parameters()})},
        "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)), "torch_version": torch.__version__,
        "stages": observer.rows, "tensor_bindings": tensors,
        "outputs": {name: image_description(path) for name, path in paths.items()},
        "timing_scope": "Observed API includes wrapper and tensor hash overhead; nested stages must not be added. Use separate uninstrumented CLI jobs for speed.",
    }
    if observer.device.type == "cuda":
        properties = torch.cuda.get_device_properties(observer.device)
        report["gpu_hardware"] = {"name": properties.name, "uuid": str(getattr(properties, "uuid", "unavailable")),
                                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  "total_memory_bytes": properties.total_memory}
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(observer.device)
        report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(observer.device)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"feature": args.feature, "status": report["status"],
                      "network_calls": len(tensors), "report_sha256": sha256_file(args.report)}))


if __name__ == "__main__":
    main()
