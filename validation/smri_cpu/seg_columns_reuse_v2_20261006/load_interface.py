"""Load the SHA-bound existing v1 binary with revised guards; no numerical calls."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource

from bindings import check_runtime, check_sources, flags, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise RuntimeError("new metadata receipt path required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be invisible")
    resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    before = check_sources(args.root, args.workspace, plan)
    import torch
    import prototype
    from prototype import ColumnsReuse
    if Path(prototype.__file__).resolve() != (args.workspace / "prototype.py").resolve():
        raise RuntimeError("wrong v2 prototype import")
    check_runtime(torch, plan)
    initial = flags(torch)
    if initial["CUDA_initialized"]:
        raise RuntimeError("unexpected CUDA initialization")
    library = args.root / plan["previous_frozen_files"]["columns_reuse.so"]["fnit_relative_path"]
    helper = ColumnsReuse(library, plan["provider_sha256"], allow_compute=False)
    helper._provider_still_matches()
    assert not helper.allow_compute and not helper.allow_bounded_contracts
    assert ctypes.cast(ctypes.CDLL(None).sgemm_, ctypes.c_void_p).value == helper.sgemm_address
    after = check_sources(args.root, args.workspace, plan)
    final = flags(torch)
    assert initial == final and not final["CUDA_initialized"] and before == after
    report = {"schema": "fnit_columns_v2_metadata_load/v1", "status": "loaded_existing_binary_only",
              "PLAN": identity(args.workspace / "PLAN.json"), "sources_before": before, "sources_after": after,
              "flags_before": initial, "flags_after": final, "ABI_metadata": 10404,
              "binary": identity(library), "provider_basename": helper.provider.name,
              "provider_sha256": helper.provider_sha256, "Torch_handle_global_same_address": True,
              "compilation_calls": 0, "copy_calls": 0, "SGEMM_calls": 0, "MRI_calls": 0,
              "explicit_tensor_allocation_calls": 0, "guard_contracts_executed": False,
              "numerical_contracts_executed": False, "production_changed": False}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in ("status", "copy_calls", "SGEMM_calls", "MRI_calls")}))


if __name__ == "__main__":
    main()
