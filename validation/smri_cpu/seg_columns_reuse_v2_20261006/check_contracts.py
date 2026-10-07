"""Prepared exact copy/actual-weight contracts. Explicit review flag required; no MRI."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch
import weakref

from bindings import check_runtime, check_sources, flags, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-contracts", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise RuntimeError("new contract receipt path required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(name) != "8" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("CPU-only eight-thread environment required")
    if any(name in os.environ for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "OPENBLAS_CORETYPE")):
        raise RuntimeError("loader/core overrides are outside the frozen contract environment")
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    if sorted(os.sched_getaffinity(0)) != plan["affinity"]:
        raise RuntimeError("declared physical-core affinity required")
    resource.setrlimit(resource.RLIMIT_AS, (8_000_000_000, 8_000_000_000))
    started = time.monotonic()
    sources_before = check_sources(args.root, args.workspace, plan)
    weight_path = args.root / plan["weight"]["fnit_relative_path"]
    if identity(weight_path) != {key: plan["weight"][key] for key in ("bytes", "sha256")}:
        raise RuntimeError("actual layer weight bytes changed")
    sys.path.insert(0, str(args.root / "repo/src"))
    import torch
    from torch.nn import functional as F
    from torch.nn.modules import module
    from fnit.synthseg_parc.segment import SegmentUNet
    from fnit.synthseg_parc.cpu_conv import CPUInferenceConv3d, convolution_slabs
    import prototype
    from prototype import ColumnsReuse
    if Path(prototype.__file__).resolve() != (args.workspace / "prototype.py").resolve():
        raise RuntimeError("wrong v2 prototype import")
    for name in ("segment", "cpu_conv"):
        if Path(sys.modules["fnit.synthseg_parc." + name].__file__).resolve() != (args.root / "repo/src/fnit/synthseg_parc" / (name + ".py")).resolve():
            raise RuntimeError("wrong actual FNIT import: " + name)
    check_runtime(torch, plan)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    initial = flags(torch)
    if initial["CUDA_initialized"]:
        raise RuntimeError("unexpected CUDA initialization")
    library = args.root / plan["previous_frozen_files"]["columns_reuse.so"]["fnit_relative_path"]
    helper = ColumnsReuse(library, plan["provider_sha256"], allow_compute=True, allow_bounded_contracts=True)
    model = SegmentUNet().load_h5(weight_path).eval()
    layer = model.up[3].conv0
    assert type(layer) is CPUInferenceConv3d and tuple(layer.weight.shape) == (24, 72, 3, 3, 3)
    assert tuple(layer.bias.shape) == (24,)
    real_copy, real_gemm = helper.copy, helper.gemm
    report = {"schema": "fnit_columns_v2_bounded_contracts/v1", "status": "started_not_accepted",
              "scope": "copy oracle and six synthetic FP32 cases with actual C72->24 weights; no MRI/native/full CNN/GPU",
              "PLAN": identity(args.workspace / "PLAN.json"), "sources_before": sources_before,
              "flags_before": initial, "Torch": torch.__version__, "affinity": sorted(os.sched_getaffinity(0)),
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "parallel_info": torch.__config__.parallel_info(), "weight": identity(weight_path),
              "binary": identity(library), "provider_sha256": helper.provider_sha256,
              "copy_oracle_calls": 0, "candidate_copy_calls": 0, "candidate_SGEMM_calls": 0,
              "injected_copy_error_calls": 0, "guard_rows": [], "copy_rows": [], "numeric_rows": [],
              "MRI_calls": 0, "native_calls": 0, "model_forward_calls": 0, "new_compilation_calls": 0,
              "production_changed": False, "accepted_for_MRI": False, "completed": False}

    def require(value, message):
        if not value:
            raise RuntimeError(message)

    def bits(value):
        return value.detach().contiguous().view(torch.int32)

    def metrics(reference, actual):
        require(reference.shape == actual.shape and reference.dtype == actual.dtype, "comparison geometry")
        finite = bool(torch.isfinite(reference).all()) and bool(torch.isfinite(actual).all())
        return {"different_bits": int(torch.count_nonzero(bits(reference) != bits(actual))),
                "max_abs": float((reference - actual).abs().max()) if finite else None, "finite": finite}

    def poisoned_copy(input_ptr, columns_ptr, input_depth, height, width, output_depth, capacity):
        # Poison only the compact prefix consumed by this call; no persistent data is retained.
        ctypes.memset(columns_ptr, 0xA5, capacity * 4)
        report["candidate_copy_calls"] += 1
        return real_copy(input_ptr, columns_ptr, input_depth, height, width, output_depth, capacity)

    def counted_gemm(*values):
        report["candidate_SGEMM_calls"] += 1
        return real_gemm(*values)

    helper.copy, helper.gemm = poisoned_copy, counted_gemm
    generator = torch.Generator().manual_seed(plan["seed"])
    parameter_bits = None
    hook_names = ("_global_forward_pre_hooks", "_global_forward_hooks", "_global_backward_pre_hooks", "_global_backward_hooks")
    global_hook_policy = getattr(module, "_global_is_full_backward_hook", None)
    local_hook_policy = getattr(layer, "_is_full_backward_hook", None)
    global_hook_before = {name: list(getattr(module, name, {})) for name in hook_names}
    try:
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            parameter_bits = (bits(layer.weight).clone(), bits(layer.bias).clone())
            samples = []
            for spec in plan["numeric_cases"]:
                image = torch.randn(tuple(spec["shape"]), generator=generator)
                image[:, 0].zero_()
                image[:, 1].fill_(-0.0)
                if spec["cancellation"]:
                    image[:, 2::2].mul_(1e4)
                    image[:, 3::2].mul_(-1e4)
                if spec["strided"]:
                    shape = tuple(image.shape[:-1]) + (image.shape[-1] * 2,)
                    storage = torch.empty(shape)
                    storage[..., ::2].copy_(image)
                    image = storage[..., ::2]
                samples.append((spec["name"], image))
            # Pure copy oracles include actual numeric geometries plus H/W singleton edges.
            copy_samples = samples + [("height_width_singleton", torch.tensor([0.0, -0.0]).repeat(72).reshape(1, 72, 2, 1, 1))]
            for name, image in copy_samples:
                _, _, depth, height, width = image.shape
                maximum_m = min(14, depth) * height * width
                scratch = torch.empty(1944 * maximum_m)
                for start in range(0, depth, 14):
                    stop = min(start + 14, depth)
                    lower, upper = max(0, start - 1), min(depth, stop + 1)
                    chunk = image[:, :, lower:upper]
                    before, after = max(0, 1 - start), max(0, stop + 1 - depth)
                    if before or after:
                        chunk = F.pad(chunk, (0, 0, 0, 0, before, after))
                    chunk = chunk.contiguous()
                    m = (stop - start) * height * width
                    ctypes.memset(scratch.data_ptr(), 0xA5, scratch.numel() * 4)
                    target = scratch[:1944 * m].view(1944, m)
                    code = real_copy(chunk.data_ptr(), target.data_ptr(), chunk.shape[2], height, width, stop - start, target.numel())
                    report["copy_oracle_calls"] += 1
                    require(code == 0, "copy oracle rejected declared geometry")
                    padded = F.pad(chunk, (1, 1, 1, 1, 0, 0))
                    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
                    oracle = patches.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous().reshape(1944, m)
                    row = {"case": name, "start": start, "depth": stop - start, "M": m,
                           "different_bits": int(torch.count_nonzero(bits(target) != bits(oracle))),
                           "compact_prefix_stride": list(target.stride()), "poisoned_before_copy": True}
                    if scratch.numel() > target.numel():
                        tail = scratch[target.numel():].view(torch.uint8)
                        row["unconsumed_tail_poison_unchanged"] = bool(torch.all(tail == 0xA5))
                    report["copy_rows"].append(row)
                    require(row["different_bits"] == 0 and row.get("unconsumed_tail_poison_unchanged", True), "first copy bit/overwrite gate failed")
            for name, image in samples:
                input_bits = bits(image).clone()
                plane_bytes = 72 * image.shape[-2] * image.shape[-1] * 4
                reference = convolution_slabs(image, layer.weight, layer.bias, padding=1, maximum_slab_bytes=16 * plane_bytes)
                allocation_refs = []
                original_empty = torch.Tensor.new_empty
                def observed_empty(tensor, *values, **kwargs):
                    value = original_empty(tensor, *values, **kwargs)
                    allocation_refs.append(weakref.ref(value))
                    return value
                with patch.object(torch.Tensor, "new_empty", observed_empty):
                    actual = helper.forward(layer, image)
                row = {"case": name, "shape": list(image.shape), "old_candidate_slab_depth": 14,
                       **metrics(reference, actual), "shape_stride_dtype_equal": reference.shape == actual.shape and reference.stride() == actual.stride() and reference.dtype == actual.dtype,
                       "input_unchanged": torch.equal(bits(image), input_bits),
                       "weight_bias_unchanged": torch.equal(bits(layer.weight), parameter_bits[0]) and torch.equal(bits(layer.bias), parameter_bits[1]),
                       "output_independent": actual.untyped_storage().data_ptr() not in (image.untyped_storage().data_ptr(), layer.weight.untyped_storage().data_ptr(), layer.bias.untyped_storage().data_ptr()),
                       "explicit_workspace_allocations": len(allocation_refs),
                       "workspace_released_on_return": len(allocation_refs) == 3 and allocation_refs[0]() is None and allocation_refs[1]() is None and allocation_refs[2]() is actual,
                       "poisoned_columns_each_slab": True}
                report["numeric_rows"].append(row)
                require(row["different_bits"] == 0, "first actual-weight numeric bit gate failed")
                require(all(row[k] for k in ("finite", "shape_stride_dtype_equal", "input_unchanged", "weight_bias_unchanged", "output_independent", "workspace_released_on_return")), "numeric postcondition failed")
        report["normal_scope_flags_restored"] = flags(torch) == initial
        require(report["normal_scope_flags_restored"], "normal scope flags not restored")
        # Guard tests use sentinels and do not invoke convolution or SGEMM.
        sample = torch.zeros((1, 72, 2, 3, 4))
        sentinel = torch.zeros(())
        fallback_calls = 0
        def fallback(value):
            nonlocal fallback_calls
            fallback_calls += 1
            return sentinel
        def guard(label, value=sample, owner=layer, candidate=helper):
            before_calls = (report["candidate_copy_calls"], report["candidate_SGEMM_calls"], fallback_calls)
            result = candidate.forward(owner, value)
            passed = result is sentinel and fallback_calls == before_calls[2] + 1 and before_calls[:2] == (report["candidate_copy_calls"], report["candidate_SGEMM_calls"])
            report["guard_rows"].append({"name": label, "fallback_identity": passed, "numeric_calls": 0})
            require(passed, "fallback guard failed: " + label)
        with patch.object(layer, "forward", side_effect=fallback):
            guard("reverse_grad_enabled")
            with torch.no_grad(), torch.backends.mkldnn.flags(enabled=True):
                guard("oneDNN_enabled")
            with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
                layer.train()
                try: guard("training")
                finally: layer.eval()
                guard("float64_input", sample.double())
                guard("meta_device_guard_only_no_CUDA", torch.empty(sample.shape, device="meta"))
                guard("input_requires_grad", sample.clone().requires_grad_(True))
                with torch.autocast("cpu"):
                    guard("CPU_autocast_enabled_no_low_precision_kernel")
                for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks"):
                    with patch.dict(getattr(layer, name), {987654: lambda *args: None}):
                        guard("local_" + name)
                for name in hook_names:
                    with patch.dict(getattr(module, name), {987654: lambda *args: None}):
                        guard("global_" + name)
                class TensorSubclass(torch.Tensor): pass
                guard("image_subclass", sample.as_subclass(TensorSubclass))
                saved_weight, saved_bias = layer.weight, layer.bias
                for name in ("weight", "bias"):
                    saved = layer._parameters[name]
                    try:
                        layer._parameters[name] = saved.detach().as_subclass(TensorSubclass)
                        guard(name + "_subclass")
                    finally: layer._parameters[name] = saved
                with torch.autograd.forward_ad.dual_level():
                    guard("image_forward_AD_dual", torch.autograd.forward_ad.make_dual(sample, torch.ones_like(sample)))
                    for name in ("weight", "bias"):
                        saved = layer._parameters[name]
                        try:
                            layer._parameters[name] = torch.autograd.forward_ad.make_dual(saved, torch.ones_like(saved))
                            guard(name + "_forward_AD_dual")
                        finally: layer._parameters[name] = saved
                require(layer.weight is saved_weight and layer.bias is saved_bias, "parameter restoration")
                restricted = ColumnsReuse(library, plan["provider_sha256"], allow_compute=True, allow_bounded_contracts=False)
                guard("unsupported_shape", candidate=restricted)
        class LayerSubclass(CPUInferenceConv3d): pass
        other = LayerSubclass(72, 24, 3, padding=1).eval()
        with patch.object(other, "forward", side_effect=fallback), torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            guard("layer_subclass", owner=other)
        closed = ColumnsReuse(library, plan["provider_sha256"], allow_compute=False)
        try: closed.forward(layer, sample)
        except RuntimeError as error:
            require("not authorized" in str(error), "closed compute gate exception")
            report["default_compute_rejected"] = True
        else: raise RuntimeError("closed compute gate allowed numerical execution")
        before_fallback_error = flags(torch)
        with patch.object(layer, "forward", side_effect=RuntimeError("declared old fallback error")):
            try: helper.forward(layer, sample)
            except RuntimeError as error:
                require(str(error) == "declared old fallback error", "old fallback error changed")
                report["old_fallback_error_propagated"] = True
            else: raise RuntimeError("old fallback error swallowed")
        require(flags(torch) == before_fallback_error, "fallback exception flags changed")
        with patch.object(helper, "_provider_still_matches", side_effect=RuntimeError("declared provider gate error")):
            try:
                with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
                    helper.forward(layer, sample)
            except RuntimeError as error:
                require(str(error) == "declared provider gate error", "provider error changed")
                report["provider_error_propagated_before_allocation"] = True
            else: raise RuntimeError("provider identity error swallowed")
        require(flags(torch) == before_fallback_error, "provider exception flags changed")
        # A synthetic copy error must propagate, release buffers and preserve flags.
        error_refs = []
        original_empty = torch.Tensor.new_empty
        def error_empty(tensor, *values, **kwargs):
            value = original_empty(tensor, *values, **kwargs)
            error_refs.append(weakref.ref(value))
            return value
        def injected_copy_error(*values):
            report["injected_copy_error_calls"] += 1
            return 5
        before_exception = flags(torch)
        with patch.object(helper, "copy", injected_copy_error), patch.object(torch.Tensor, "new_empty", error_empty):
            try:
                with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
                    helper.forward(layer, sample)
            except RuntimeError as error:
                require("copy-only im2col" in str(error), "copy exception did not propagate")
                report["copy_error_propagated"] = True
            else: raise RuntimeError("copy error swallowed")
        gc.collect()
        report["exception_workspace_released"] = len(error_refs) == 3 and all(ref() is None for ref in error_refs)
        report["exception_flags_restored"] = flags(torch) == before_exception
        require(report["exception_workspace_released"] and report["exception_flags_restored"], "exception postcondition")
        helper._provider_still_matches()
        report["fallback_calls"] = fallback_calls
        require(len(report["numeric_rows"]) == 6 and len(report["copy_rows"]) == 13,
                "declared six numeric and thirteen copy/slab cases required")
        require(report["candidate_copy_calls"] == report["candidate_SGEMM_calls"] == 12,
                "declared twelve synthetic slab calls required")
        require(len(report["guard_rows"]) == fallback_calls == 23,
                "declared twenty-three fallback guards required")
        report["completed"] = True
        report["status"] = "bounded_contracts_passed_no_MRI"
    except Exception as error:
        report["status"] = "bounded_contracts_failed_stopped"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        helper.copy, helper.gemm = real_copy, real_gemm
        module._global_is_full_backward_hook = global_hook_policy
        layer._is_full_backward_hook = local_hook_policy
        report["flags_after"] = flags(torch)
        report["flags_unchanged"] = initial == report["flags_after"]
        report["global_hook_tables_unchanged"] = global_hook_before == {name: list(getattr(module, name, {})) for name in hook_names}
        report["postcondition_failures"] = {}
        try:
            report["sources_after"] = check_sources(args.root, args.workspace, plan)
            report["sources_unchanged"] = sources_before == report["sources_after"]
        except Exception as error:
            report["sources_unchanged"] = False
            report["postcondition_failures"]["source"] = str(error)
        try:
            report["weight_file_after"] = identity(weight_path)
            report["weight_file_unchanged"] = report["weight"] == report["weight_file_after"]
            report["parameter_bits_unchanged"] = parameter_bits is not None and torch.equal(bits(layer.weight), parameter_bits[0]) and torch.equal(bits(layer.bias), parameter_bits[1])
        except Exception as error:
            report["weight_file_unchanged"] = report["parameter_bits_unchanged"] = False
            report["postcondition_failures"]["weight"] = str(error)
        report["RSS_maximum_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["worker_observation_seconds"] = time.monotonic() - started
        report["valid_bounded_contracts"] = bool(report["completed"] and not report["postcondition_failures"] and report["flags_unchanged"] and report["sources_unchanged"] and report["weight_file_unchanged"] and report["parameter_bits_unchanged"] and report["global_hook_tables_unchanged"] and not report["flags_after"]["CUDA_initialized"] and report["RSS_maximum_bytes"] <= 32_000_000_000)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    require(report["valid_bounded_contracts"], "bounded contract final gate")
    print(json.dumps({name: report[name] for name in ("status", "valid_bounded_contracts", "copy_oracle_calls", "candidate_copy_calls", "candidate_SGEMM_calls", "MRI_calls")}))


if __name__ == "__main__":
    main()
