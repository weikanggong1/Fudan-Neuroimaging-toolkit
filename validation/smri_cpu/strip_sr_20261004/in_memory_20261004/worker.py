"""Run one default CPU API on an owned, fully decoded SpatialImage.

This benchmark worker neither changes FNIT source nor reruns references.
The copied input keeps NumPy K order. Tensor hashes and saved comparisons are
computed after the timed API call; no extra network forward is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    report = {"schema": "fnit.smri.memory_api.v1", "status": "running",
              "feature": config["feature"], "case": "case01",
              "worker_sha256": sha256(__file__), "config_sha256": sha256(args.config),
              "stages": [], "network_bindings": [], "comparisons": {},
              "overall_numerical_equivalence": "not_assessed",
              "timing_scope": "one new CPU process; preparation, loading, API, saving and posthoc work separate; reference clocks are archived",
              "private_paths": {"source_root": config["source_root"],
                                "python": sys.executable, "input": config["input"],
                                "output": str(args.output), "config": str(args.config)}}
    current = None

    @contextmanager
    def stage(name):
        nonlocal current
        current = name
        tick = time.perf_counter()
        entry = {"name": name, "status": "running"}
        try:
            yield
        except Exception:
            entry["status"] = "failed"
            raise
        else:
            entry["status"] = "complete"
        finally:
            entry["seconds"] = time.perf_counter() - tick
            report["stages"].append(entry)

    try:
        with stage("verify_frozen_source_and_resources"):
            source = Path(config["source_root"])
            manifest_path = source / "SOURCE.private.json"
            manifest = json.loads(manifest_path.read_text())
            for key in ("source_label", "head_commit", "archive_sha256"):
                if manifest[key] != config["source_identity"][key]:
                    raise ValueError("source identity differs: " + key)
            if sha256(manifest_path) != config["source_manifest_sha256"]:
                raise ValueError("SOURCE manifest bytes differ")
            for name, expected in config["source_files_sha256"].items():
                if manifest["files"][name] != expected or sha256(source / name) != expected:
                    raise ValueError("frozen source differs: " + name)
            resources = {"input": config["input_resource"], "weight": config["weight_resource"]}
            checked = {}
            for name, item in resources.items():
                path = Path(config[name])
                if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                    raise ValueError("resource differs: " + name)
                checked[name] = {"filename": path.name, **item}
            for arm, members in config["references"].items():
                for name, item in members.items():
                    if sha256(item["path"]) != item["sha256"]:
                        raise ValueError("saved reference differs: " + arm + "/" + name)
            if sha256(config["comparator"]) != config["comparator_sha256"]:
                raise ValueError("saved comparison formula differs")
            report["source"] = {**config["source_identity"],
                                "manifest_sha256": sha256(manifest_path),
                                "includes_reviewed_uncommitted_changes": manifest["includes_reviewed_uncommitted_changes"],
                                "files_verified": len(config["source_files_sha256"]),
                                "files_sha256": config["source_files_sha256"]}
            report["resources"] = checked
            report["comparator_sha256"] = config["comparator_sha256"]
            report["saved_reference_sha256"] = {
                arm: {name: item["sha256"] for name, item in members.items()}
                for arm, members in config["references"].items()}
            report["canonical_index_sha256_before"] = sha256(config["canonical_index"])
        with stage("import_runtime"):
            import nibabel as nib
            import numpy as np
            import torch
            from fnit._nib import new_image
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            module = importlib.import_module("fnit." + config["feature"] + ".pipeline")
            expected_module = (source / "src/fnit" / config["feature"] / "pipeline.py").resolve()
            if Path(module.__file__).resolve() != expected_module:
                raise ValueError("pipeline imported outside the frozen source")
            affinity = sorted(os.sched_getaffinity(0))
            if affinity != list(range(0, 29, 4)) or torch.get_num_threads() != 8:
                raise ValueError("CPU8 affinity/thread contract differs")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
                raise ValueError("this job must not initialize CUDA")
            report["runtime"] = {"hostname": socket.gethostname(), "device": "cpu",
                                 "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                                 "affinity": affinity, "torch": torch.__version__, "numpy": np.__version__,
                                 "nibabel": nib.__version__, "cuda_initialized_before": torch.cuda.is_initialized(),
                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        with stage("load_image_metadata"):
            loaded = nib.load(config["input"])
            proxy = loaded.dataobj
        with stage("decode_original_values"):
            decoded = np.asanyarray(proxy)
        with stage("copy_decoded_array_K_order"):
            copied = np.array(decoded, copy=True, order="K", subok=False)
        with stage("construct_materialized_spatial_image"):
            image = new_image(copied, loaded)
        with stage("verify_materialized_input"):
            def layout(array):
                return {"shape": list(array.shape), "dtype": str(array.dtype),
                        "strides_bytes": list(array.strides), "c_contiguous": bool(array.flags.c_contiguous),
                        "f_contiguous": bool(array.flags.f_contiguous), "owns_data": bool(array.flags.owndata)}

            same_values = np.array_equal(decoded, image.dataobj)
            same_affine = np.array_equal(loaded.affine, image.affine)
            same_zooms = np.array_equal(loaded.header.get_zooms(), image.header.get_zooms())
            forms = {}
            for name in ("qform", "sform"):
                first, code1 = getattr(loaded, "get_" + name)(coded=True)
                second, code2 = getattr(image, "get_" + name)(coded=True)
                forms[name] = {"original_code": int(code1), "materialized_code": int(code2),
                               "exact": bool(code1 == code2 and np.array_equal(first, second))}
            if not (isinstance(image, nib.spatialimages.SpatialImage) and
                    isinstance(image.dataobj, np.ndarray) and not nib.is_proxy(image.dataobj) and
                    same_values and same_affine and same_zooms and all(row["exact"] for row in forms.values()) and
                    decoded.strides == copied.strides and decoded.dtype == copied.dtype):
                raise ValueError("materialization changed decoded values, geometry or K layout")
            decoded_sha = hashlib.sha256(decoded.tobytes(order="C")).hexdigest()
            report["materialized_input"] = {
                "input_alias": "case01", "data_source": "OpenNeuro ds003138 v1.0.1 raw T1w",
                "construction": "new_image(np.array(np.asanyarray(original.dataobj), copy=True, order='K'), original)",
                "spatial_image_class": type(image).__name__, "dataobj_is_ndarray": True,
                "dataobj_is_proxy": False, "original_storage_dtype": str(loaded.get_data_dtype()),
                "proxy_slope": float(proxy.slope), "proxy_intercept": float(proxy.inter),
                "original_decoded": layout(decoded), "materialized": layout(copied),
                "values_exact": bool(same_values), "affine_exact": bool(same_affine),
                "zooms_exact": bool(same_zooms), "zooms_mm": [float(v) for v in loaded.header.get_zooms()],
                "forms": forms, "decoded_array_sha256_C_order": decoded_sha,
                "loader_policy": config["loader_policy"]}
        del decoded, loaded, proxy
        with stage("model_constructor"):
            cls = getattr(module, "SynthStrip" if config["feature"] == "synthstrip" else "SynthSR")
            model = cls(weights=config["weight"], device="cpu", threads=8)
        captures = []

        def capture(_model, inputs, output):
            captures.append((inputs[0].detach(), output.detach()))

        hook = model.model.register_forward_hook(capture)
        with stage("default_api_call"):
            result = model(image=image)
        hook.remove()
        with stage("bind_actual_network_tensors_after_api"):
            expected_calls = 1 if config["feature"] == "synthstrip" else 2
            if len(captures) != expected_calls:
                raise ValueError("unexpected extra/missing network forward")
            for first, second in captures:
                row = {}
                for name, tensor in (("input", first), ("output", second)):
                    if str(tensor.device) != "cpu":
                        raise ValueError("actual network tensor is not on CPU")
                    array = tensor.numpy()
                    row[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                 "device": str(tensor.device), "tensor_stride": list(tensor.stride()),
                                 "array_sha256_C_order": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
                report["network_bindings"].append(row)
            report["model"] = {"parameter_devices": sorted({str(p.device) for p in model.model.parameters()}),
                               "parameter_dtypes": sorted({str(p.dtype) for p in model.model.parameters()}),
                               "production_cpu_channels_last": bool(getattr(model, "_cpu_channels_last", False)),
                               "network_forwards": len(captures), "default_options_only": True,
                               "autocast_cpu_enabled": torch.is_autocast_enabled("cpu")}
            report["materialized_input"]["unchanged_after_api"] = (
                hashlib.sha256(image.dataobj.tobytes(order="C")).hexdigest() == decoded_sha)
        captures.clear()
        outputs = {"image": args.output / "image.nii.gz"}
        if config["feature"] == "synthstrip":
            outputs.update(mask=args.output / "mask.nii.gz", distance=args.output / "distance.nii.gz")
        else:
            outputs["float_image"] = args.output / "image.npz"
        with stage("save_same_call_outputs"):
            for name, path in outputs.items():
                getattr(result, name).save(path) if config["feature"] == "synthstrip" else result.image.save(path)
        report["output_sha256"] = {name: sha256(path) for name, path in outputs.items()}
        del result, model, image
        with stage("compare_saved_reference_outputs"):
            for arm, members in config["references"].items():
                report_path = args.output / ("comparison_" + arm + ".json")
                floating = "float_image" in members
                command = [sys.executable, config["comparator"], "--feature", config["feature"],
                           "--case", "case01_memory_" + arm, "--report", str(report_path),
                           "--reference-image", members["float_image" if floating else "image"]["path"],
                           "--candidate-image", str(outputs["float_image" if floating else "image"]),
                           "--gate", ("strip_" if config["feature"] == "synthstrip" else "sr_") +
                           ("official" if arm.startswith("official") else "candidate")]
                if config["feature"] == "synthstrip":
                    for name in ("mask", "distance"):
                        command += ["--reference-" + name, members[name]["path"],
                                    "--candidate-" + name, str(outputs[name])]
                tick = time.perf_counter()
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                entry = {"returncode": process.returncode, "seconds": time.perf_counter() - tick,
                         "report_sha256": sha256(report_path) if report_path.exists() else None,
                         "result": json.loads(report_path.read_text()) if report_path.exists() else None}
                geometry = {}
                for name, member in members.items():
                    if name == "float_image":
                        geometry[name] = {"status": "array_container_without_affine",
                                          "basis": "spatial geometry is separately checked on the quantized NIfTI from the same API call"}
                        continue
                    original = nib.load(member["path"])
                    candidate = nib.load(str(outputs[name]))
                    fields = ("dim", "pixdim", "datatype", "bitpix", "scl_slope", "scl_inter",
                              "qform_code", "sform_code", "xyzt_units", "intent_code", "cal_min", "cal_max")
                    row = {"shape_exact": original.shape == candidate.shape,
                           "affine_exact": bool(np.array_equal(original.affine, candidate.affine)),
                           "header_binaryblock_exact": original.header.binaryblock == candidate.header.binaryblock,
                           "header_fields_exact": {field: bool(np.array_equal(original.header[field], candidate.header[field], equal_nan=True))
                                                   for field in fields}}
                    for form in ("qform", "sform"):
                        first, code1 = getattr(original, "get_" + form)(coded=True)
                        second, code2 = getattr(candidate, "get_" + form)(coded=True)
                        row[form] = {"reference_code": int(code1), "candidate_code": int(code2),
                                     "exact": bool(code1 == code2 and np.array_equal(first, second)),
                                     "max_abs_mm": float(np.abs(first-second).max())
                                     if first is not None and second is not None else None}
                    geometry[name] = row
                entry["scientific_geometry"] = geometry
                if not report_path.exists():
                    entry["error"] = process.stderr[-2000:]
                report["comparisons"][arm] = entry
        with stage("verify_source_resources_unchanged"):
            checks = {"input": sha256(config["input"]) == config["input_resource"]["sha256"],
                      "weight": sha256(config["weight"]) == config["weight_resource"]["sha256"],
                      "source_manifest": sha256(source / "SOURCE.private.json") == config["source_manifest_sha256"],
                      "source_files": all(sha256(source / name) == value for name, value in config["source_files_sha256"].items()),
                      "saved_references": all(sha256(item["path"]) == item["sha256"]
                                              for members in config["references"].values() for item in members.values())}
            report["unchanged_after"] = checks
            if not all(checks.values()):
                raise ValueError("source/resource/reference changed")
            report["runtime"]["cuda_initialized_after"] = torch.cuda.is_initialized()
            if report["runtime"]["cuda_initialized_after"]:
                raise ValueError("CUDA was unexpectedly initialized")
        report["status"] = "complete"
        report["comparisons_complete"] = all(x["result"] is not None for x in report["comparisons"].values())
        report["all_fixed_gates_passed"] = all(x["result"] is not None and x["result"].get("fixed_gate", {}).get("passes", False)
                                              for x in report["comparisons"].values())
    except Exception as exc:
        report.update(status="failed", failed_stage=current, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["worker_seconds_including_posthoc"] = time.perf_counter() - started
        write(args.output / "report.private.json", report)


if __name__ == "__main__":
    main()
