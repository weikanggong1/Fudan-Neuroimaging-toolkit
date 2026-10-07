"""Recompute compile-only scalar evidence; no Torch, compiler or MRI execution."""
import argparse
import hashlib
import json
from pathlib import Path


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build(folder):
    plan = json.loads((folder / "PLAN.json").read_text())
    compile_report = json.loads((folder / "COMPILE.json").read_text())
    interface = json.loads((folder / "INTERFACE.json").read_text())
    queue = json.loads((folder / "QUEUE.json").read_text())
    receipts = json.loads((folder / "RECEIPTS.json").read_text())
    for name, record in receipts["files"].items():
        assert identity(folder / name) == {key: record[key] for key in ("bytes", "sha256")}
    for name, digest in plan["prototype_sources"].items():
        assert identity(folder / name)["sha256"] == digest
    plan_sha = identity(folder / "PLAN.json")["sha256"]
    assert plan_sha == queue["plan_sha256"] == compile_report["plan_sha256"]
    assert queue["prototype_files"] == plan["prototype_sources"]
    assert compile_report["source_before"] == compile_report["source_after"] == plan["source14"] == interface["source14"]
    assert compile_report["flags_before"] == compile_report["flags_after"]
    assert compile_report["Torch_version"] == interface["runtime"]["Torch"] == "2.5.1"
    assert compile_report["source_unchanged"] and compile_report["flags_unchanged"]
    assert compile_report["status"] == "compiled_loaded_interface_only" and compile_report["compiler_rc"] == 0
    assert compile_report["abi_description"] == 10404
    assert not queue["numerical_calls_authorized"] and not compile_report["compute_activation"]
    for name in ("copy_calls", "SGEMM_calls", "MRI_calls", "tensor_allocation_calls"):
        assert compile_report[name] == 0
    for name in ("global_allocator_changed", "thread_setters_called"):
        assert not compile_report[name]
    assert not compile_report["flags_after"]["cuda_initialized"]
    assert compile_report["provider_sha256"] == plan["provider_sha256"] == interface["public_provider"]["sha256"]
    assert compile_report["Torch_handle_and_global_SGEMM_same_address"]
    assert interface["public_provider"]["Torch_global_LP64_same_address"]
    assert interface["public_provider"]["NumPy_mkl_rt_different_address"]
    geometry = plan["scope_unchanged"]
    _, channels, depth, height, width = geometry["input_shape"]
    kernel_items = 27
    maximum_m, last_m = 14 * height * width, (depth % 14) * height * width
    assert geometry["K"] == channels * kernel_items == 1944
    assert maximum_m == geometry["M_full_slab"] == 802816
    assert last_m == geometry["M_last_slab"] == 573440
    assert channels * kernel_items * maximum_m * 4 == geometry["columns_workspace_maximum_bytes"]
    return {
        "schema": "fnit_seg_columns_compile_only_summary/v1",
        "status": "compiled_loaded_interface_only_pending_numeric_review",
        "plan_sha256": plan_sha,
        "prototype_sources": plan["prototype_sources"],
        "compile_receipt_sha256": identity(folder / "COMPILE.json")["sha256"],
        "interface_receipt_sha256": identity(folder / "INTERFACE.json")["sha256"],
        "production_source_commit": plan["production_source_commit"],
        "canonical_main_at_preflight": plan["canonical_main_read"],
        "source14_before_after_exact": True,
        "flags_before_after_exact": True,
        "compiler": {"return_code": 0, "wall_seconds": compile_report["compile_wall_seconds"],
                     "maxRSS_KiB": compile_report["compiler_maxRSS_KiB"],
                     "existing_separate_Conda_toolchain": True, "clean_install_validated": False},
        "library": {"bytes": compile_report["library_bytes"], "sha256": compile_report["library_sha256"],
                    "ABI_metadata": 10404, "binary_committed_or_downloaded": False},
        "provider": {"basename": compile_report["provider_basename"], "sha256": compile_report["provider_sha256"],
                     "Torch_handle_and_global_same_address": True, "different_NumPy_route_used": False},
        "numerical_copy_SGEMM_MRI_calls": {key: compile_report[key] for key in ("copy_calls", "SGEMM_calls", "MRI_calls")},
        "explicit_probe_tensor_allocation_calls": 0,
        "allocation_counter_limitation": "Source-defined explicit calls, not a dynamic trace of internal Torch import allocations.",
        "CUDA_initialized": False,
        "workspace_geometry_derived": {"M_full_slab": maximum_m, "M_last_slab": last_m,
                                       "columns_maximum_bytes": channels * kernel_items * maximum_m * 4,
                                       "output_workspace_bytes": 24 * maximum_m * 4,
                                       "lifetime": "one proposed layer call; not executed"},
        "not_completed": ["copy bit oracle", "actual-weight numerical contracts", "fallback and exception contracts",
                          "subclass and forward-AD guards require a separate revised freeze",
                          "actual saved-layer ABBA", "accuracy or speed benchmark", "whole MRI/CSV/GPU acceptance", "clean Conda installation"],
        "production_integrated": False,
        "speed_or_accuracy_conclusion": "not_assessed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare the saved summary; do not rewrite it.")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    result = build(folder)
    target = folder / "SUMMARY.json"
    if args.check:
        assert json.loads(target.read_text()) == result
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "source14": 14, "numerical_calls": 0, "summary_check": args.check}))


if __name__ == "__main__":
    main()
