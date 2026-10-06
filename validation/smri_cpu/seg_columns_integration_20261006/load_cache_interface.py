"""One approved metadata-only lazy cache build/load; no tensor/copy/SGEMM/MRI."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import sys
import time
from phase1_bindings import check_runtime, check_sources, flags, identity, loaded_artifact, require_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-phase1", action="store_true")
    args = parser.parse_args()
    if not args.approved_phase1:
        raise RuntimeError("prepared only; separate coordinator phase1 approval required")
    os.umask(0o077)
    if args.output.exists():
        raise RuntimeError("new interface receipt required")
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    require_environment(plan)
    cache = Path(os.environ["FNIT_SYNTHSEG_CPU_CACHE"])
    if cache.exists():
        raise RuntimeError("metadata phase requires a new private cache; do not delete prior cache")
    resource.setrlimit(resource.RLIMIT_AS, (8_000_000_000, 8_000_000_000))
    started = time.monotonic()
    before = check_sources(args.root, args.workspace, plan)
    source = args.workspace / "source_candidate"
    sys.path.insert(0, str(source))
    import torch
    from fnit.synthseg_parc import cpu_columns, _cpu_columns_build as build
    for name in ("cpu_columns", "_cpu_columns_build"):
        if Path(sys.modules["fnit.synthseg_parc." + name].__file__).resolve() != (source / "fnit/synthseg_parc" / (name + ".py")).resolve():
            raise RuntimeError("wrong frozen candidate import")
    check_runtime(torch, plan)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    initial = flags(torch)
    if initial["CUDA_initialized"]:
        raise RuntimeError("unexpected CUDA initialization")
    report = {"schema":"fnit_columns_lazy_interface/v1", "status":"started_not_accepted",
        "PLAN":identity(args.workspace / "PLAN.json"), "sources_before":before, "flags_before":initial,
        "affinity":sorted(os.sched_getaffinity(0)), "threads":torch.get_num_threads(),
        "interop_threads":torch.get_num_interop_threads(), "Torch":torch.__version__,
        "compile_calls":0, "compiler_probe_calls":0, "build_input_calls":0,
        "copy_calls":0, "SGEMM_calls":0, "MRI_calls":0, "model_forward_calls":0,
        "explicit_tensor_allocation_calls":0, "numerical_engine_invoked":False,
        "tensor_allocation_count_basis":"metadata-only code path constructs no tensor; no tensor allocation profiler used",
        "accepted_for_whole":False, "complete":False}
    originals = []
    for name, key in (("_compile","compile_calls"),("_compiler","compiler_probe_calls"),("_build_inputs","build_input_calls")):
        original = getattr(build,name)
        def observed(*values, original=original, key=key, **keywords):
            report[key] += 1
            return original(*values,**keywords)
        originals.append((name,original)); setattr(build,name,observed)
    error = None
    try:
        artifact = build.build_artifact(torch)
        record = loaded_artifact(artifact, plan)
        plan["_loaded_artifact"] = record
        helper = cpu_columns._Columns(record["library"], plan["provider_sha256"], allow_compute=False)
        helper._provider_still_matches()
        assert not helper.allow_compute and not helper.allow_bounded_contracts
        assert ctypes.cast(ctypes.CDLL(None).sgemm_,ctypes.c_void_p).value == helper.sgemm_address
        assert report["compile_calls"] == report["compiler_probe_calls"] == report["build_input_calls"] == 1
        report.update(artifact=record, ABI_metadata=10404, Torch_handle_global_same_address=True,
            provider_basename=helper.provider.name, status="compiled_loaded_lazy_cache_interface_only", complete=True)
    except BaseException as caught:
        error = caught
        report.update(status="interface_failed_stop_contracts", error_type=type(caught).__name__,error=str(caught))
    finally:
        for name, original in reversed(originals): setattr(build,name,original)
        report["flags_after"] = flags(torch)
        report["flags_unchanged"] = initial == report["flags_after"]
        try:
            report["sources_after"] = check_sources(args.root,args.workspace,plan)
            report["sources_unchanged"] = before == report["sources_after"]
        except Exception as caught:
            report.update(sources_unchanged=False, source_error=str(caught))
        report["RSS_maximum_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["worker_observation_seconds"] = time.monotonic() - started
        report["valid_interface"] = bool(error is None and report["complete"] and report["flags_unchanged"]
            and report["sources_unchanged"] and not report["flags_after"]["CUDA_initialized"]
            and report["RSS_maximum_bytes"] <= 32_000_000_000)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if error is not None: raise error
    if not report["valid_interface"]: raise RuntimeError("interface final gate")
    print(json.dumps({key:report[key] for key in ("status","valid_interface","compile_calls","copy_calls","SGEMM_calls","MRI_calls")}))


if __name__ == "__main__": main()
