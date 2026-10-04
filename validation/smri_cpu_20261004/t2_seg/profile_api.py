"""Real-image SynthSeg/WMH stage profiler; independent benchmark only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", choices=("synthseg", "parc", "wmh"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--keep-geometry", action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--profile-convolutions", action="store_true")
    args = parser.parse_args()
    import_start = time.perf_counter()
    import fnit
    import torch
    import numpy as np
    import fnit.synthseg_parc.synthseg as synthseg_module
    import fnit.synthseg_parc.segment as segment_module
    import fnit.synthseg_parc.preprocess as preprocess_module
    import fnit.synthseg_parc.postprocess as postprocess_module
    from fnit.synthseg_parc import SynthSeg, SynthSegPlus
    from fnit.wmh_synthseg import WMHSynthSeg
    from fnit.wmh_synthseg.pipeline import _write_volumes_csv
    import fnit.wmh_synthseg.pipeline as wmh_module
    import_seconds = time.perf_counter() - import_start
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_times = {}
    convolution_times = {}
    stage_calls = {}
    handles = []

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))

    def instrument(module, name):
        function = getattr(module, name)

        def measured(*positional, **keywords):
            synchronize()
            start = time.perf_counter()
            value = function(*positional, **keywords)
            synchronize()
            elapsed = time.perf_counter() - start
            key = module.__name__ + "." + name
            stage_times[key] = stage_times.get(key, 0.0) + elapsed
            stage_calls[key] = stage_calls.get(key, 0) + 1
            return value

        setattr(module, name, measured)

    for module, name in ((synthseg_module, "preprocess_t1"),
                         (segment_module, "preprocess_t1"),
                         (synthseg_module, "postprocess_segmentation"),
                         (segment_module, "postprocess_segmentation"),
                         (postprocess_module, "largest_connected_component"),
                         (synthseg_module, "_official_soft_volumes"),
                         (wmh_module, "myzoom_torch")):
        instrument(module, name)

    synchronize()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    start = time.perf_counter()
    if args.feature == "synthseg":
        model = SynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    elif args.feature == "parc":
        model = SynthSegPlus(weights=args.weights, parc_weights=args.weights,
                             device=args.device)
    else:
        model = WMHSynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    synchronize()
    model_load_seconds = time.perf_counter() - start

    def attach(network, prefix):
        if network is None:
            return
        for name, layer in network.named_modules():
            if not isinstance(layer, torch.nn.Conv3d):
                continue
            key = prefix + "." + name

            def before(module, positional, key=key):
                synchronize()
                module._benchmark_start = time.perf_counter()

            def after(module, positional, result, key=key):
                synchronize()
                row = convolution_times.setdefault(key, {"seconds": 0.0, "calls": 0})
                row["seconds"] += time.perf_counter() - module._benchmark_start
                row["calls"] += 1
                row["input_shape"] = list(positional[0].shape)
                row["output_shape"] = list(result.shape)
                row["input_dtype"] = str(positional[0].dtype)

            handles.extend((layer.register_forward_pre_hook(before),
                            layer.register_forward_hook(after)))

    if args.profile_convolutions:
        if args.feature == "synthseg":
            attach(model.segmenter.model, "segmentation")
        elif args.feature == "wmh":
            attach(model.model, "wmh")
        else:
            # Match lazy construction and then profile the actual two networks.
            from fnit.synthseg_parc.segment import SynthSegSegmenter
            from fnit.synthseg_parc.pipeline import SynthSegParc
            from fnit.synthseg_parc.labels import PARCELLATION_LABELS
            model._segmenter = SynthSegSegmenter(model.segment_weights,
                                                model.segmentation_labels, model.device)
            model._parcellator = SynthSegParc(model.parc_weights, PARCELLATION_LABELS,
                                            model.device)
            attach(model._segmenter.model, "segmentation")
            attach(model._parcellator.model, "parcellation")
    synchronize()
    start = time.perf_counter()
    if args.feature == "synthseg":
        result = model(args.input, keep_geometry=args.keep_geometry)
    elif args.feature == "parc":
        result = model(args.input, keep_geometry=args.keep_geometry, fast=args.fast,
                       volumes=True)
    else:
        result = model(args.input, crop=args.crop, save_lesion_probabilities=True)
    synchronize()
    inference_seconds = time.perf_counter() - start
    for handle in handles:
        handle.remove()
    start = time.perf_counter()
    output = args.output_dir / "segmentation.nii.gz"
    if args.feature == "parc":
        result.combined.save(output)
        result.segmentation.save(args.output_dir / "anatomy.nii.gz")
        result.cortical_parcellation.save(args.output_dir / "parcellation.nii.gz")
    else:
        result.segmentation.save(output)
    csv_path = args.output_dir / "volumes.csv"
    if args.feature == "wmh":
        result.lesion_probability.save(args.output_dir / "segmentation.lesion_probs.nii.gz")
        _write_volumes_csv(result.volumes_mm3, output, csv_path)
    else:
        result.write_volumes_csv(args.input, csv_path)
    save_seconds = time.perf_counter() - start
    report = {
        "schema": "fnit_smri_seg_cpu_profile/v1",
        "feature": args.feature, "device": args.device,
        "options": {"fast": args.fast, "crop": args.crop,
                    "keep_geometry": args.keep_geometry,
                    "profile_convolutions": args.profile_convolutions},
        "source_import": fnit.__file__, "input_sha256": digest(args.input),
        "hostname": os.uname().nodename,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled_after_call": torch.backends.mkldnn.enabled,
        "seconds": {"import": import_seconds, "model_constructor": model_load_seconds,
                    "inference_and_result": inference_seconds, "save": save_seconds},
        "nested_stages_seconds": stage_times, "nested_stage_calls": stage_calls,
        "convolutions": convolution_times,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                    for p in args.output_dir.iterdir() if p.is_file()},
        "precision": getattr(result, "precision", {}),
        "warning": "Profiling callbacks add overhead; nested clocks overlap. Separate CLI runs determine speed.",
    }
    if args.device.startswith("cuda"):
        device = torch.device(args.device)
        report["cuda_memory"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["seconds"]), flush=True)


if __name__ == "__main__":
    main()
