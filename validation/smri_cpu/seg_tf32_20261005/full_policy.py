"""Source-bound original-T1 ordinary parcel policy worker; no reference fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--mode", choices=("parc", "parc-fast"), required=True)
    parser.add_argument("--policy", choices=("true", "false", "none"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    binding = json.loads(args.binding.read_text())
    source_hashes = {name: sha(args.source / "fnit/synthseg_parc" / name)
                     for name in binding["source_files"][args.arm]}
    assert source_hashes == binding["source_files"][args.arm], "incorrect frozen source"
    assert args.arm != "baseline" or args.policy == "true", "old public API only supports default policy"
    raw_t1 = args.root / "runs/smri_cpu_20261004/inputs/ds003138/case02_T1w.nii.gz"
    assert sha(raw_t1) == "73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21"
    sys.path.insert(0, str(args.source))
    import fnit
    import torch
    from fnit import SynthSegPlus
    from fnit.weights import WEIGHT_FILES
    from fnit.synthseg_parc import segment, pipeline, model as parc_model
    assert Path(fnit.__file__).resolve() == (args.source / "fnit/__init__.py").resolve()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    weights = args.root / "workspaces/smri_cpu_20261004/assets/weights"
    resource_rows = {}
    for name in ("synthseg_2.0.h5", "synthseg_parc_2.0.h5",
                 "synthseg_segmentation_labels_2.0.npy", "synthseg_segmentation_names_2.0.npy",
                 "synthseg_topological_classes_2.0.npy"):
        path = weights / name
        size, digest = path.stat().st_size, sha(path)
        assert (size, digest) == tuple(WEIGHT_FILES[name][1:]), name
        resource_rows[name] = {"bytes": size, "sha256": digest}
    policy = {"true": True, "false": False, "none": None}[args.policy]
    target = torch.device(args.device)
    gpu = target.type == "cuda"
    caller_cudnn = args.policy != "none"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = caller_cudnn

    def flags():
        return {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cpu_autocast": bool(torch.is_autocast_enabled("cpu")),
                "cuda_autocast": bool(torch.is_autocast_enabled("cuda")),
                "mkldnn_enabled": bool(torch.backends.mkldnn.enabled)}

    before_construct = flags()
    operation_rows = []

    def record(value, operation, model=None):
        row = {"operation": operation, "device": str(value.device),
               "input_dtype": str(value.dtype), **flags()}
        if model is not None:
            row["model_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        operation_rows.append(row)
        return row

    # Python function observation adds no Module hooks and changes no arithmetic.
    originals = []
    for cls, name in ((segment.SegmentUNet, "segmentation_network"),
                      (parc_model.ParcUNet, "parcellation_network")):
        original = cls.forward
        originals.append((cls, original))
        def observed_forward(self, value, original=original, name=name):
            row = record(value, name, self)
            output = original(self, value)
            row["output_dtype"] = str(output.dtype)
            return output
        cls.forward = observed_forward
    original_blur, original_conv = segment._blur, pipeline.F.conv3d
    def observed_blur(value):
        row = record(value, "segmentation_gaussian_blur")
        output = original_blur(value)
        row["output_dtype"] = str(output.dtype)
        return output
    def observed_conv(value, kernel, *args, **kwargs):
        if kernel.shape == (69, 1, 3, 3, 3) and kwargs.get("groups") == 69:
            row = record(value, "parcellation_gaussian_blur")
            output = original_conv(value, kernel, *args, **kwargs)
            row["output_dtype"] = str(output.dtype)
            return output
        return original_conv(value, kernel, *args, **kwargs)
    segment._blur, pipeline.F.conv3d = observed_blur, observed_conv
    report = {"schema": "fnit_parc_tf32_original_T1/v1", "status": "running",
        "arm": args.arm, "mode": args.mode, "policy": policy, "caller_cudnn_tf32": caller_cudnn,
        "device": args.device, "hostname": os.uname().nodename,
        "source_files": source_hashes, "worker_sha256": sha(__file__), "binding_sha256": sha(args.binding),
        "index_sha256": sha(args.root / "INDEX.json"), "input_sha256": sha(raw_t1), "weights": resource_rows,
        "torch_version": torch.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "before_construction": before_construct,
        "scope": "Complete public SynthSegPlus API and all maps/CSV from original raw CC0 T1; no prefix or official output injected."}
    if gpu:
        assert torch.cuda.is_available()
        props = torch.cuda.get_device_properties(target)
        assert str(props.uuid).lower().removeprefix("gpu-") == "e25cac06-0ce8-a833-abf9-09ab18c9c9ba"
        torch.cuda.set_per_process_memory_fraction(20_000_000_000 / props.total_memory, target)
        torch.cuda.reset_peak_memory_stats(target)
        report["gpu"] = {"name": props.name, "uuid": str(props.uuid), "total_bytes": props.total_memory,
                         "allocator_budget_bytes": 20_000_000_000}
    def sync():
        if gpu:
            torch.cuda.synchronize(target)
    try:
        report["preflight_seconds"] = time.perf_counter() - process_start
        sync()
        start = time.perf_counter()
        options = {"cudnn_tf32": policy} if args.arm == "candidate" else {}
        model = SynthSegPlus(weights=weights, parc_weights=weights, device=args.device, **options)
        sync()
        report["constructor_seconds"] = time.perf_counter() - start
        report["after_construction"] = flags()
        assert report["after_construction"] == before_construct
        start = time.perf_counter()
        result = model(raw_t1, keep_geometry=False, fast=args.mode == "parc-fast", volumes=True)
        sync()
        report["api_seconds"] = time.perf_counter() - start
        report["after_inference"] = flags()
        report["caller_precision_restored"] = report["after_inference"] == before_construct
        report["actual_operations"] = operation_rows
        report["precision"] = getattr(result, "precision", None)
        report["segmentation_precision"] = model._segmenter.precision
        report["model_devices"] = {"segmentation": sorted({str(p.device) for p in model._segmenter.model.parameters()}),
                                   "parcellation": sorted({str(p.device) for p in model._parcellator.model.parameters()})}
        effective = caller_cudnn if policy is None else policy
        expected = {"matmul_tf32": True, "cudnn_tf32": effective} if gpu else {
                    name: before_construct[name] for name in ("matmul_tf32", "cudnn_tf32")}
        for row in operation_rows:
            assert all(row[name] == value for name, value in expected.items()), row
            assert row["device"] == str(target) and row["input_dtype"] == row["output_dtype"] == "torch.float32"
            assert not row["cpu_autocast"] and not row["cuda_autocast"]
            if "model_dtypes" in row:
                assert row["model_dtypes"] == ["torch.float32"]
        assert {row["operation"] for row in operation_rows} == {
            "segmentation_network", "parcellation_network", "segmentation_gaussian_blur", "parcellation_gaussian_blur"}
        # Segmentation ensembles two networks in ordinary mode; the parcel
        # head consumes the resulting image/mask once in both modes.
        expected_forwards = {"segmentation_network": 1 if args.mode == "parc-fast" else 2,
                             "parcellation_network": 1}
        for name, count in expected_forwards.items():
            assert sum(row["operation"] == name for row in operation_rows) == count
        if args.arm == "candidate":
            assert report["caller_precision_restored"]
            assert result.precision["cuda_precision_restored"]
        start = time.perf_counter()
        for name in ("segmentation", "cortical_parcellation", "combined"):
            getattr(result, name).save(args.output / (name + ".nii.gz"))
        result.write_volumes_csv(raw_t1, args.output / "volumes.csv")
        report["save_seconds"] = time.perf_counter() - start
        report["saved_outputs"] = {name: {"sha256": sha(args.output / name), "bytes": (args.output / name).stat().st_size}
            for name in ("segmentation.nii.gz", "cortical_parcellation.nii.gz", "combined.nii.gz", "volumes.csv")}
        report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if gpu:
            report["gpu"].update(max_allocated_bytes=torch.cuda.max_memory_allocated(target),
                                 max_reserved_bytes=torch.cuda.max_memory_reserved(target))
            assert report["gpu"]["max_reserved_bytes"] <= 20_000_000_000
        report["worker_seconds"] = time.perf_counter() - process_start
        report["status"] = "complete"
        with (args.output / "full.private.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({name: report[name] for name in ("status", "arm", "mode", "policy", "device", "api_seconds", "caller_precision_restored")}), flush=True)
    finally:
        for cls, original in originals:
            cls.forward = original
        segment._blur, pipeline.F.conv3d = original_blur, original_conv


if __name__ == "__main__":
    main()
