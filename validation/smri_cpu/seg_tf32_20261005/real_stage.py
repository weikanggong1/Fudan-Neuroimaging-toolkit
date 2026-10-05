"""Real feature-prefix/weight policy diagnostic, never a complete-T1 benchmark."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    binding = json.loads(args.binding.read_text())
    for name, expected in binding["source_files"][args.arm].items():
        assert sha(args.source / "fnit/synthseg_parc" / name) == expected, name
    root = args.root
    image = root / "runs/smri_cpu_20261004/inputs/ds003138/case02_T1w.nii.gz"
    feature = root / "runs/smri_cpu_20261004/t2_seg/preprocess/official_case02.float32.npy"
    assert sha(image) == "73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21"
    assert sha(feature) == "3132a18a1524165eb3ecfa6785ef9dc209c07b68c1a601dfa83b0a29bb45ee43"
    sys.path.insert(0, str(args.source))
    import fnit
    import numpy as np
    import torch
    from fnit.weights import WEIGHT_FILES
    from fnit.synthseg_parc import segment
    from fnit.synthseg_parc import pipeline
    from fnit.synthseg_parc.labels import PARCELLATION_LABELS
    from fnit.synthseg_parc.preprocess import PreprocessedT1
    assert Path(fnit.__file__).resolve() == (args.source / "fnit/__init__.py").resolve()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    target = torch.device(args.device)
    gpu = target.type == "cuda"
    if gpu:
        assert torch.cuda.is_available()
        props = torch.cuda.get_device_properties(target)
        torch.cuda.set_per_process_memory_fraction(20_000_000_000 / props.total_memory, target)
        torch.cuda.reset_peak_memory_stats(target)
    weights = root / "workspaces/smri_cpu_20261004/assets/weights"
    weight_names = ("synthseg_2.0.h5", "synthseg_parc_2.0.h5",
                    "synthseg_segmentation_labels_2.0.npy", "synthseg_segmentation_names_2.0.npy",
                    "synthseg_topological_classes_2.0.npy")
    weight_binding = {}
    for name in weight_names:
        resource = weights / name
        assert (resource.stat().st_size, sha(resource)) == tuple(WEIGHT_FILES[name][1:])
        weight_binding[name] = {"bytes": resource.stat().st_size, "sha256": sha(resource)}
    values = np.load(feature, mmap_mode="r")
    assert values.shape == (192, 224, 256) and values.dtype == np.float32
    crop = np.array(values[32:64, 96:128, 96:128], copy=True)
    tensor = torch.from_numpy(crop).to(target)
    prepared = PreprocessedT1(tensor, np.eye(4), np.eye(4), (32, 32, 32),
        (slice(0, 32),) * 3, np.eye(4), 1.)
    # Diagnostic grid only: output geometry is deliberately not scored.
    # The original input is hash-bound, but preprocessing is replaced by its
    # saved real feature prefix. Whole-T1 gates use no such replacement.
    prepare_calls = []
    original_prepare, original_blur, original_conv = segment.preprocess_t1, segment._blur, pipeline.F.conv3d
    operation_rows = []
    failure = None

    def flags():
        return {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)}

    def entry(value, name):
        return {"operation": name, "device": str(value.device), "input_dtype": str(value.dtype),
                **flags(), "cpu_autocast": bool(torch.is_autocast_enabled("cpu")),
                "cuda_autocast": bool(torch.is_autocast_enabled("cuda"))}

    def saved_prefix(*args, **kwargs):
        prepare_calls.append(1)
        if failure == "preprocess":
            raise RuntimeError("declared preprocessing failure")
        return prepared

    def observed_blur(value):
        row = entry(value, "segmentation_gaussian_blur")
        operation_rows.append(row)
        if failure == "fast_blur":
            raise RuntimeError("declared fast blur failure")
        output = original_blur(value)
        row["output_dtype"] = str(output.dtype)
        return output

    def observed_conv(value, kernel, *args, **kwargs):
        if kernel.shape == (69, 1, 3, 3, 3) and kwargs.get("groups") == 69:
            row = entry(value, "parcellation_gaussian_blur")
            operation_rows.append(row)
            if failure == "parcel_blur":
                raise RuntimeError("declared parcel blur failure")
            output = original_conv(value, kernel, *args, **kwargs)
            row["output_dtype"] = str(output.dtype)
            return output
        return original_conv(value, kernel, *args, **kwargs)

    segment.preprocess_t1, segment._blur, pipeline.F.conv3d = saved_prefix, observed_blur, observed_conv
    report = {"schema": "fnit_parc_tf32_real_prefix/v1", "worker_sha256": sha(__file__),
        "arm": args.arm, "device": args.device, "hostname": os.uname().nodename,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "source_files": binding["source_files"][args.arm], "binding_sha256": sha(args.binding),
        "index_sha256": sha(root / "INDEX.json"), "input_sha256": sha(image),
        "feature_file_sha256": sha(feature), "feature_shape": list(values.shape),
        "crop_slices": [[32, 64], [96, 128], [96, 128]], "crop_value_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "weights": weight_binding, "cases": [], "failures": [],
        "scope": "Actual saved real FP32 T1 feature prefix and official weights; actual complete small CNNs and filters; no whole-image benchmark or geometry assertion."}
    if gpu:
        report["gpu"] = {"name": props.name, "uuid": str(props.uuid),
                         "total_bytes": props.total_memory, "allocator_budget_bytes": 20_000_000_000}
    def set_flags(cudnn):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = cudnn

    def make_models(policy, caller):
        set_flags(caller)
        before = flags()
        seg = segment.SynthSegSegmenter(weights / "synthseg_2.0.h5",
            weights / "synthseg_segmentation_labels_2.0.npy", args.device, cudnn_tf32=policy)
        options = {"cudnn_tf32": policy} if args.arm == "candidate" else {}
        parc = pipeline.SynthSegParc(weights / "synthseg_parc_2.0.h5", PARCELLATION_LABELS,
                                    args.device, **options)
        after = flags()
        if args.arm == "candidate":
            assert before == after, "constructor wrote global precision"
        for name, model in (("segmentation_network", seg.model), ("parcellation_network", parc.model)):
            forward = model.forward
            def observer(value, name=name, forward=forward, model=model):
                row = entry(value, name)
                row["model_dtypes"] = sorted({str(value.dtype) for value in model.parameters()})
                operation_rows.append(row)
                if failure == name:
                    raise RuntimeError("declared " + name + " failure")
                output = forward(value)
                row["output_dtype"] = str(output.dtype)
                return output
            model.forward = observer  # No Module hooks, no changed tensor math.
        return seg, parc, before, after

    def infer(seg, parc, policy, fast):
        options = {"cudnn_tf32": policy} if args.arm == "candidate" else {}
        return segment.run_synthseg_parc_t1(image, weights / "synthseg_2.0.h5",
            weights / "synthseg_segmentation_labels_2.0.npy", weights / "synthseg_parc_2.0.h5",
            PARCELLATION_LABELS, device=args.device,
            topology_classes=weights / "synthseg_topological_classes_2.0.npy",
            fast=fast, volumes=True, segmenter=seg, parcellator=parc, **options)

    cases = [(True, False, fast) for fast in (False, True)]
    if args.arm == "candidate":
        cases += [(policy, caller, fast) for policy, caller in ((False, True), (None, False), (None, True))
                  for fast in (False, True)]
    try:
        for number, (policy, caller, fast) in enumerate(cases):
            operation_rows.clear()
            seg, parc, before, constructed = make_models(policy, caller)
            start = time.perf_counter()
            result = infer(seg, parc, policy, fast)
            if gpu:
                torch.cuda.synchronize(target)
            after = flags()
            if args.arm == "candidate":
                assert before == after, "inference did not restore caller precision"
            effective = caller if policy is None else policy
            expected = {"matmul_tf32": True, "cudnn_tf32": effective} if gpu else before
            assert operation_rows
            assert all({key: row[key] for key in expected} == expected for row in operation_rows)
            assert all(row["device"] == str(target) and row["input_dtype"] == "torch.float32"
                       and row["output_dtype"] == "torch.float32" and not row["cpu_autocast"]
                       and not row["cuda_autocast"] for row in operation_rows)
            outputs = {"segmentation": result.segmentation, "parcellation": result.parcellation,
                       "combined": result.combined, "segmentation_posterior": result.segmentation_posterior,
                       "parcellation_soft_voxels": result.parcellation_soft_voxels}
            case = {"id": number, "policy": policy, "caller_cudnn": caller, "fast": fast,
                    "before": before, "after_construction": constructed, "after": after,
                    "api_seconds_diagnostic_only": time.perf_counter() - start,
                    "operations": list(operation_rows), "precision": getattr(result, "precision", None),
                    "outputs": {}}
            for name, value in outputs.items():
                array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                path = args.output / (str(number) + "_" + name + ".npy")
                np.save(path, array)
                case["outputs"][name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                    "value_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "saved_file_sha256": sha(path)}
            report["cases"].append(case)
            del result, outputs, seg, parc
            gc.collect()
            if gpu:
                torch.cuda.empty_cache()
        if args.arm == "candidate":
            for failure_name in ("preprocess", "segmentation_network", "fast_blur",
                                 "parcellation_network", "parcel_blur"):
                operation_rows.clear()
                seg, parc, before, _ = make_models(False, True)
                failure = failure_name
                try:
                    infer(seg, parc, False, failure_name == "fast_blur")
                    raise AssertionError("declared failure did not fire")
                except RuntimeError as error:
                    assert "declared" in str(error)
                    assert flags() == before
                    report["failures"].append({"phase": failure, "exception": str(error),
                        "before": before, "after": flags(), "operations": list(operation_rows)})
                finally:
                    failure = None
                del seg, parc
                gc.collect()
                if gpu:
                    torch.cuda.empty_cache()
            seg, parc, before, _ = make_models(True, False)
            count = len(prepare_calls)
            try:
                infer(seg, parc, False, False)
                raise AssertionError("cached mismatch not rejected")
            except ValueError as error:
                assert "cached cudnn_tf32" in str(error)
                assert len(prepare_calls) == count and flags() == before
                report["cache_rejected_before_preprocess"] = True
            del seg, parc
        if gpu:
            report["maximum_allocated_bytes"] = torch.cuda.max_memory_allocated(target)
            report["maximum_reserved_bytes"] = torch.cuda.max_memory_reserved(target)
        report["status"] = "complete"
        with (args.output / "stage.private.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                          "failures": len(report["failures"]), "arm": args.arm, "device": args.device}))
    finally:
        segment.preprocess_t1, segment._blur, pipeline.F.conv3d = original_prepare, original_blur, original_conv


if __name__ == "__main__":
    main()
