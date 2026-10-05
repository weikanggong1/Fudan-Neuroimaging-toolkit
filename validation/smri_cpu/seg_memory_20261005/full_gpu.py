"""Frozen-source complete raw-T1 SynthSeg output and resource regression worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

BASE = {
    "model.py": "292428207da96cea837919e73b3fa455dc533b5b9e7fb1bca02e3729a92cde52",
    "segment.py": "ca51439f15ea9b792f2eb4bfb7bdd6cd7f298522e03a5917dda7ed8c9438c441",
    "cpu_conv.py": "78a0fac2300a45a01ff6790d24218f825ebc21a21f2033a8e6b8c58ebb60dc47",
    "pipeline.py": "030095b124a3556f0508359317ee15108fbb2a3d45a803d80c28350d18776633",
    "postprocess.py": "5ac0a53745e972563302ebcde2f078d5fb3e6eb7607ec92e59f67d85adae8936",
    "synthseg.py": "3f0e91ff0bfa83ab8b3742a27ccecabcad83ddb650110d7347e5e4fea42ab62f",
}
CANDIDATE = {**BASE,
    "model.py": "3268749e8f36783b217a20bd6f536e1f590b462c5bfe095e01fb8981f849c26f",
    "segment.py": "1bc2c3d90a42e4cf664991ec6dd8b0499df4b1c5e788c7adb35d02ce4b2d000e",
    "cpu_join.py": "196a2c05e32f94b9e27442a4813ad9a9bb9fb2138820c653ca89cf1a77707b8f",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    start_process = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    p.add_argument("--mode", choices=("seg33", "parc", "parc-fast"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cudnn-tf32", choices=("default", "false"), default="default")
    p.add_argument("--memory-budget-bytes", type=int, default=20_000_000_000)
    args = p.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    expected = BASE if args.arm == "baseline" else CANDIDATE
    actual = {name: sha(args.source / "fnit/synthseg_parc" / name) for name in expected}
    assert actual == expected, "incorrect frozen producer"
    assert sha(args.input) == "73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21"
    sys.path.insert(0, str(args.source))
    import fnit
    import torch
    from fnit import SynthSeg, SynthSegPlus
    from fnit.weights import WEIGHT_FILES
    assert Path(fnit.__file__).resolve() == (args.source / "fnit/__init__.py").resolve()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    weight_files = ("synthseg_2.0.h5", "synthseg_parc_2.0.h5",
                    "synthseg_segmentation_labels_2.0.npy", "synthseg_segmentation_names_2.0.npy",
                    "synthseg_topological_classes_2.0.npy")
    weights = {}
    for name in weight_files:
        path = args.weights / name
        entry = WEIGHT_FILES[name]
        size, digest = path.stat().st_size, sha(path)
        assert (size, digest) == (entry[1], entry[2]), "unverified weight " + name
        weights[name] = {"bytes": size, "sha256": digest}
    report = {"schema": "fnit_seg_decoder_full/v1", "arm": args.arm, "mode": args.mode,
              "source_files": actual, "worker_sha256": sha(__file__),
              "hostname": os.uname().nodename, "input_sha256": sha(args.input),
              "weights": weights, "python_version": sys.version, "source_import": fnit.__file__,
              "torch_version": torch.__version__, "device": args.device,
              "thread_environment": {key: os.environ.get(key) for key in
                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "requested_cudnn_tf32": args.cudnn_tf32, "cpu_join_calls": 0,
              "scope": "complete model API from raw original T1; includes all requested outputs and soft volume CSV"}
    gpu = args.device != "cpu"
    target = torch.device(args.device)
    if gpu:
        assert torch.cuda.is_available()
        props = torch.cuda.get_device_properties(target)
        torch.cuda.set_per_process_memory_fraction(args.memory_budget_bytes / props.total_memory, target)
        torch.cuda.reset_peak_memory_stats(target)
        report["gpu"] = {"name": props.name, "uuid": str(getattr(props, "uuid", "unavailable")),
                         "total_bytes": props.total_memory, "memory_budget_bytes": args.memory_budget_bytes}
    if args.arm == "candidate":
        import fnit.synthseg_parc.model as parc_model
        import fnit.synthseg_parc.segment as seg_model
        original = seg_model.join_nearest_cpu
        def observed_join(skip, value):
            report["cpu_join_calls"] += 1
            return original(skip, value)
        # Python function observation is deliberately not a Module hook: real
        # caller Module hooks remain visible to and protected by qualification.
        parc_model.join_nearest_cpu = observed_join
        seg_model.join_nearest_cpu = observed_join
    report["preflight_seconds"] = time.perf_counter() - start_process
    def sync():
        if gpu:
            torch.cuda.synchronize(target)
    sync()
    start_model = time.perf_counter()
    if args.mode == "seg33":
        options = {"cudnn_tf32": False} if args.cudnn_tf32 == "false" else {}
        model = SynthSeg(weights=args.weights, device=args.device, threads=8, **options)
    else:
        model = SynthSegPlus(weights=args.weights, parc_weights=args.weights, device=args.device)
        if args.cudnn_tf32 == "false":
            from fnit.synthseg_parc.segment import SynthSegSegmenter
            model._segmenter = SynthSegSegmenter(model.segment_weights, model.segmentation_labels,
                                                 device=args.device, cudnn_tf32=False)
    sync()
    report["construct_seconds"] = time.perf_counter() - start_model
    report["before_inference_global_precision"] = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled}
    start_api = time.perf_counter()
    if args.mode == "seg33":
        result = model(args.input, keep_geometry=False)
        precision = result.precision
        outputs = {"segmentation": result.segmentation}
    else:
        result = model(args.input, keep_geometry=False, fast=args.mode == "parc-fast", volumes=True)
        precision = model._segmenter.precision
        outputs = {"segmentation": result.segmentation, "cortical_parcellation": result.cortical_parcellation,
                   "combined": result.combined}
    sync()
    report["api_seconds"] = time.perf_counter() - start_api
    report["forward_precision"] = precision
    report["after_inference_global_precision"] = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled}
    report["scope_precision_restored"] = (
        report["before_inference_global_precision"] == report["after_inference_global_precision"])
    report["model_devices"] = sorted({str(value.device) for value in
        (model.segmenter.model.parameters() if args.mode == "seg33" else model._segmenter.model.parameters())})
    if args.mode != "seg33":
        report["parc_model_devices"] = sorted({str(value.device) for value in model._parcellator.model.parameters()})
        report["parc_model_dtypes"] = sorted({str(value.dtype) for value in model._parcellator.model.parameters()})
    start_save = time.perf_counter()
    for name, image in outputs.items():
        image.save(args.output / (name + ".nii.gz"))
    result.write_volumes_csv(args.input, args.output / "volumes.csv")
    report["save_seconds"] = time.perf_counter() - start_save
    report["complete_api_save_seconds"] = time.perf_counter() - start_model
    report["saved_outputs"] = {name: {"sha256": sha(args.output / (name + ".nii.gz")),
                                      "bytes": (args.output / (name + ".nii.gz")).stat().st_size}
                               for name in outputs}
    report["saved_outputs"]["volumes.csv"] = {"sha256": sha(args.output / "volumes.csv"),
                                               "bytes": (args.output / "volumes.csv").stat().st_size}
    report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["post_global_precision"] = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                                        "mkldnn_enabled": torch.backends.mkldnn.enabled}
    if gpu:
        report["gpu"].update(max_allocated_bytes=torch.cuda.max_memory_allocated(target),
                              max_reserved_bytes=torch.cuda.max_memory_reserved(target))
        assert report["gpu"]["max_allocated_bytes"] <= args.memory_budget_bytes
        assert report["cpu_join_calls"] == 0
    elif args.arm == "candidate":
        assert report["cpu_join_calls"] == (8 if args.mode in ("seg33", "parc-fast") else 12)
    report["worker_seconds"] = time.perf_counter() - start_process
    report["status"] = "complete"
    (args.output / "full.private.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("arm", "mode", "device", "status", "api_seconds", "complete_api_save_seconds", "cpu_join_calls")}), flush=True)


if __name__ == "__main__":
    main()
