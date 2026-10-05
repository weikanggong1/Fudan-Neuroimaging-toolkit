"""Private, no-Module-hooks timing observer for one frozen ordinary CPU SynthSeg.

This only delegates to the current implementation. No convolution, tensor
layout, precision, hook, input or inference parameter is changed. MRI outputs
stay in the task's private run directory. Observed wall time is diagnostic.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


class Observer:
    """Record nested wall clocks; tensors are never retained by observations."""

    def __init__(self):
        self.rows = []
        self.stack = []
        self.patches = []
        self.bookkeeping_seconds = 0.0
        self.tensor_type = ()

    def shape(self, value):
        if isinstance(value, self.tensor_type):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        return {}

    @contextmanager
    def span(self, name, metadata=None):
        setup_start = time.perf_counter()
        row = {"name": name, "parent": self.stack[-1]["name"] if self.stack else None,
               "metadata": metadata or {}, "children_seconds": 0.0}
        self.stack.append(row)
        start = time.perf_counter()
        self.bookkeeping_seconds += start - setup_start
        try:
            yield row
        finally:
            stop = time.perf_counter()
            row["inclusive_seconds"] = stop - start
            row["exclusive_seconds"] = row["inclusive_seconds"] - row.pop("children_seconds")
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children_seconds"] += row["inclusive_seconds"]
            self.rows.append(row)
            self.bookkeeping_seconds += time.perf_counter() - stop

    def patch(self, owner, name, wrapper):
        original = getattr(owner, name)
        replacement = wrapper(original)
        self.patches.append((owner, name, original))
        setattr(owner, name, replacement)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def wrap(self, name, metadata=None):
        def factory(original):
            def observed(*args, **kwargs):
                info = {} if metadata is None else metadata(*args, **kwargs)
                with self.span(name, info) as row:
                    value = original(*args, **kwargs)
                if isinstance(value, self.tensor_type):
                    row["output"] = self.shape(value)
                return value
            return observed
        return factory


def flags(torch):
    return {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "cpu_autocast": bool(torch.is_autocast_enabled("cpu")),
            "cuda_autocast": bool(torch.is_autocast_enabled("cuda")),
            "cuda_initialized": bool(torch.cuda.is_initialized())}


def install(observer, torch, nib, modules, model):
    """Wrap Python entry points, never Module hooks or arithmetic alternatives."""
    synthseg, segment, conv, post, pre = modules
    observer.tensor_type = torch.Tensor
    module_names = {id(layer): name for name, layer in model.segmenter.model.named_modules()}
    for item in model.segmenter.model.modules():
        assert not item._forward_hooks and not item._forward_pre_hooks
    observer.patch(synthseg, "preprocess_t1", observer.wrap("preprocess"))
    observer.patch(nib, "load", observer.wrap("nifti_header_load"))
    observer.patch(nib.dataobj_images.DataobjImage, "get_fdata", observer.wrap("nifti_data_decode"))
    observer.patch(pre, "_resample_1mm", observer.wrap("resample_1mm"))
    observer.patch(pre, "_align_ras", observer.wrap("align_ras"))
    observer.patch(pre.np, "percentile", observer.wrap("percentile"))
    observer.patch(segment.SynthSegSegmenter, "posterior", observer.wrap("posterior"))
    observer.patch(segment.SynthSegSegmenter, "_forward",
                   observer.wrap("CNN", lambda self, image, pass_name: {"pass": pass_name, **observer.shape(image)}))
    observer.patch(segment, "_blur", observer.wrap("gaussian33", lambda image: observer.shape(image)))
    observer.patch(segment, "join_nearest_cpu", observer.wrap(
        "decoder_join", lambda skip, value: {"skip": observer.shape(skip), "source": observer.shape(value)}))
    observer.patch(conv.CPUInferenceConv3d, "forward", observer.wrap(
        "layer_conv", lambda self, image: {"layer": module_names[id(self)], **observer.shape(image),
            "kernel": list(self.kernel_size), "padding": list(self.padding), "groups": self.groups,
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled)}))

    def slab_metadata(image, weight, bias=None, *, padding=0, groups=1, maximum_slab_bytes=256 * 1024**2):
        pad = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        plane = image.shape[0] * max(image.shape[1], weight.shape[0]) * image.shape[3] * image.shape[4] * image.element_size()
        depth = max(1, min(32, maximum_slab_bytes // plane - 2 * pad[0]))
        return {**observer.shape(image), "kernel_shape": list(weight.shape), "padding": list(pad),
                "groups": groups, "slab_depth": int(depth), "slab_calls": math.ceil(image.shape[2] / depth)}
    slab_wrapper = observer.wrap("slabs", slab_metadata)
    observer.patch(conv, "convolution_slabs", slab_wrapper)
    # _blur imported this helper as an independent alias before instrumentation.
    observer.patch(segment, "convolution_slabs", slab_wrapper)
    observer.patch(conv.F, "conv3d", observer.wrap("functional_conv", lambda image, weight, *args, **kwargs: {
        **observer.shape(image), "kernel_shape": list(weight.shape), "groups": kwargs.get("groups", 1),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled)}))
    observer.patch(conv.F, "pad", observer.wrap("pad", lambda image, *args, **kwargs: observer.shape(image)))
    for name in ("elu", "max_pool3d", "softmax"):
        observer.patch(conv.F, name, observer.wrap(name, lambda image, *args, **kwargs: observer.shape(image)))
    observer.patch(torch.nn.BatchNorm3d, "forward", observer.wrap(
        "batchnorm", lambda self, image: {"layer": module_names[id(self)], **observer.shape(image)}))
    observer.patch(synthseg, "postprocess_segmentation", observer.wrap("postprocess"))
    observer.patch(post, "largest_connected_component", observer.wrap("LCC", lambda mask: observer.shape(mask)))
    observer.patch(synthseg, "_synthseg_index_with_numerical_ties", observer.wrap("tie_argmax"))
    observer.patch(synthseg, "_official_soft_volumes", observer.wrap("soft_volumes"))
    observer.patch(synthseg, "_segmentation_image", observer.wrap("segmentation_header"))

    def observe_assignment(original):
        def observed(self, index, value):
            inside_slab = bool(observer.stack) and observer.stack[-1]["name"] == "slabs"
            if not inside_slab:
                return original(self, index, value)
            with observer.span("slab_output_copy", observer.shape(value)):
                return original(self, index, value)
        return observed
    observer.patch(torch.Tensor, "__setitem__", observe_assignment)

    # These wrappers observe only calls made by postprocess Python. They invoke
    # the original descriptors once, including the original inplace division.
    for method, name in (("sum", "posterior_sum"), ("argmax", "posterior_argmax"),
                         ("__itruediv__", "posterior_normalize_divide"), ("clone", "posterior_clone")):
        def factory(original, name=name):
            def observed(self, *args, **kwargs):
                inside_post = any(row["name"] == "postprocess" for row in observer.stack)
                if not inside_post:
                    return original(self, *args, **kwargs)
                with observer.span(name, observer.shape(self)):
                    return original(self, *args, **kwargs)
            return observed
        observer.patch(torch.Tensor, method, factory)


def main():
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    args.output.mkdir(mode=0o700)
    binding = json.loads(args.binding.read_text())
    expected = binding["source_files"]
    before_sources = {name: sha(args.source / "fnit/synthseg_parc" / name) for name in expected}
    assert before_sources == expected
    image = args.root / binding["input_fnit_relative_path"]
    assert sha(image) == binding["input_sha256"]
    weights = args.root / binding["weights_fnit_relative_path"]
    verified_weights = {name: {"bytes": (weights/name).stat().st_size, "sha256": sha(weights/name)}
                        for name in binding["weights"]}
    assert verified_weights == binding["weights"]
    sys.path.insert(0, str(args.source))
    import fnit
    import nibabel as nib
    import torch
    from fnit import SynthSeg
    from fnit.synthseg_parc import synthseg, segment, cpu_conv, postprocess, preprocess
    assert Path(fnit.__file__).resolve() == (args.source / "fnit/__init__.py").resolve()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    assert sorted(os.sched_getaffinity(0)) == binding["cpu_affinity"]
    initial_flags = flags(torch)
    assert not initial_flags["cuda_initialized"]
    preflight = time.perf_counter() - process_start
    start = time.perf_counter()
    model = SynthSeg(weights=weights, device="cpu", threads=8)
    constructor = time.perf_counter() - start
    observer = Observer()
    report = {"schema": "fnit_ordinary33_cpu_nohooks_profile/v1", "status": "started",
        "scope": "one complete original-T1 API under Python timing observers; not a formal speed benchmark",
        "worker_sha256": sha(__file__), "binding_sha256": sha(args.binding),
        "source_files": before_sources, "input_sha256": binding["input_sha256"], "weights": verified_weights,
        "hostname": os.uname().nodename, "torch_version": torch.__version__,
        "source_import": fnit.__file__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "index_sha256": sha(args.root / "INDEX.json"), "preflight_seconds": preflight,
        "constructor_seconds": constructor, "initial_flags": initial_flags,
        "after_constructor_flags": flags(torch), "no_module_hooks": True}
    try:
        install(observer, torch, nib, (synthseg, segment, cpu_conv, postprocess, preprocess), model)
        api_start = time.perf_counter()
        with observer.span("API"):
            result = model(image, keep_geometry=False)
        report["observed_api_seconds"] = time.perf_counter() - api_start
        report["precision"] = result.precision
        report["after_API_flags"] = flags(torch)
        # Save scientific outputs before diagnostic assertions or comparison.
        start = time.perf_counter()
        result.segmentation.save(args.output / "segmentation.nii.gz")
        result.write_volumes_csv(image, args.output / "volumes.csv")
        report["save_seconds"] = time.perf_counter() - start
        report["saved_outputs"] = {name: {"sha256": sha(args.output / name),
                                         "bytes": (args.output/name).stat().st_size}
            for name in ("segmentation.nii.gz", "volumes.csv")}
        report["source_after"] = {name: sha(args.source / "fnit/synthseg_parc" / name) for name in expected}
        report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["events"] = observer.rows
        report["observer_bookkeeping_seconds_lower_bound"] = observer.bookkeeping_seconds
        report["observer_limitations"] = [
            "Bookkeeping estimate excludes wrapper dispatch and metadata preparation overhead",
            "Observed API may differ due to instrumentation and shared CPU load; do not infer formal speed",
            "Functional convolution clock includes the unchanged ATen work; no internal ATen operator trace",
            "Nested clocks included in parents, use exclusive sums for accounting",
        ]
        joins = [row for row in observer.rows if row["name"] == "decoder_join"]
        cnn = [row for row in observer.rows if row["name"] == "CNN"]
        report["diagnostic_gates"] = {
            "source_unchanged": report["source_after"] == before_sources,
            "actual_cpu_join_calls_8": len(joins) == 8,
            "actual_network_forwards_2": len(cnn) == 2,
            "actual_network_grid_192_224_256": all(row["metadata"]["shape"] == [1,1,192,224,256] for row in cnn),
            "CPU_flags_unchanged": initial_flags == report["after_constructor_flags"] == report["after_API_flags"],
            "no_module_hooks_after": all(not value._forward_hooks and not value._forward_pre_hooks
                                          for value in model.segmenter.model.modules()),
            "saved_output_files_match_predeclared_existing_candidate": report["saved_outputs"] == binding["saved_outputs"],
        }
        report["status"] = "complete_gates_passed" if all(report["diagnostic_gates"].values()) else "complete_gate_failed"
        report["worker_seconds"] = time.perf_counter() - process_start
        with (args.output / "profile.private.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "observed_api_seconds", "save_seconds", "diagnostic_gates")}), flush=True)
        assert all(report["diagnostic_gates"].values())
    finally:
        observer.restore()


if __name__ == "__main__":
    main()
