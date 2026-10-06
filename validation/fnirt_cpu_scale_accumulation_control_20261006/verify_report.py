"""Verify saved scalar-control reports with the standard library only.

No checkpoint member, MRI image, scientific library or remote process is read.
"""
import hashlib
import json
import re
from pathlib import Path


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    leaf = Path(__file__).resolve().parent
    manifest = json.loads((leaf / "MANIFEST.public.json").read_text())
    actual = {str(path.relative_to(leaf)) for path in leaf.rglob("*") if path.is_file() and path.name != "MANIFEST.public.json"}
    assert set(manifest["files"]) == actual
    for name, wanted in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert identity(leaf / name) == wanted, name
    result = json.loads((leaf / "RESULT.public.json").read_text())
    summary = json.loads((leaf / "run/summary.public.json").read_text())
    controller = json.loads((leaf / "run/controller.public.json").read_text())
    cleanup = json.loads((leaf / "run/cleanup.public.json").read_text())
    lock = json.loads((leaf / "run/lock_release.public.json").read_text())
    prepared = leaf.parent / "fnirt_cpu_scale_order_prepare_20261006"
    assert identity(prepared / "freeze.public.json")["sha256"] == result["freeze_sha256"]
    assert identity(prepared / "MANIFEST.public.json")["sha256"] == result["manifest_sha256"]
    assert json.loads((prepared / "freeze.public.json").read_text()) == json.loads((leaf / "SOURCE_FREEZE.public.json").read_text())
    assert result["source_commit"] == "3dc99bef147d40299e6510f026f62283819eef31"
    assert result["canonical_main_commit"] == summary["canonical_main_commit"] == "7ff215ee86c49414b2fa6156fdf6769aa54d00f8"
    assert result["controller_exit_code"] == result["worker_exit_code"] == controller["exit_code"] == controller["worker_returncode"] == summary["exit_code"] == 0
    assert (leaf / "run/controller.exitcode").read_text() == "0\n"
    assert controller["status"] == "completed" and summary["control_completed"] and summary["baseline_passed_before_candidate"] and summary["own_math_completed_before_reference"]
    assert all(item["bitexact"] for item in summary["gates"].values())
    assert summary["gates"]["baseline_scale_full_g"]["actual"]["FP64_little_endian_hex"] == "57a8afd76fe72440"
    assert summary["gates"]["baseline_SSD"]["actual"]["FP64_little_endian_hex"] == "997d2872352a4e40"
    calls = {"NPZ_member_restore": 4, "baseline_scalar_pair": 1, "scalar_FP32_product_pair": 1, "serial_reduction_pair": 1, "native_saved_g_tail8_read": 1}
    assert summary["calls"] == result["calls"] == calls and not any(summary["prohibited_calls"].values())
    assert controller["bindings_before"] == controller["bindings_after"] == summary["bindings_before"] == summary["bindings_after"]
    assert len(summary["bindings_before"]) == 30 and result["source_bindings"] == 23 and result["input_bindings"] == 7
    assert summary["harness_before"] == summary["harness_after"] == controller["harness_before"] == controller["harness_after"]
    assert summary["operands_before"] == summary["operands_after"] and len(summary["operands_before"]) == 7
    assert summary["flags_before"] == summary["flags_after"] and summary["header_before"] == summary["header_after"]
    assert summary["flags_unchanged"] and summary["source_inputs_unchanged"] and controller["source_inputs_unchanged"]
    assert not summary["postcheck_errors"] and not controller["postcheck_errors"] and not summary["FSL_DSO_before"] and not summary["FSL_DSO_after"]
    assert summary["same_input_context_as_prior_moving_substitution"] is False and summary["does_not_explain_prior_projection_scale_error"] is True
    assert summary["native_samepoint_input_caches_established"] is False and summary["native_scale_comparison_is_posthoc_only"] is True
    assert summary["own_scalar_comparison"]["final_grouping_at_same_serial_sum"]["bitexact"] is True
    assert summary["own_scalar_comparison"]["serial_source_scale_vs_baseline"]["absolute_difference"] == 3.197442310920451e-14
    assert summary["own_scalar_comparison"]["serial_SSD_vs_baseline"]["absolute_difference"] == 2.1316282072803006e-14
    assert result["serial_candidates"] == summary["serial_candidates"] and result["posthoc_native_scale_comparison"] == summary["posthoc_native_scale_comparison"]
    assert cleanup == controller["final_cleanup"] and cleanup["returncode"] == 0 and not cleanup["still_running"] and not cleanup["errors"]
    assert not cleanup["SIGTERM_sent"] and not cleanup["SIGKILL_sent"] and not cleanup["unreaped_child_retains_lock"]
    assert all(item["reaped"] and not item["identity_present"] for item in lock["processes"].values())
    assert lock["common_lock"]["inode"] == 5185616833 and lock["common_lock"]["lock_nb_acquired"] and lock["common_lock"]["fd_closed_after_check"]
    assert result["scientific_retries"] == 0 and not result["production_integration_accepted"]
    readme = (leaf / "README.md").read_text()
    for number in range(1, 8):
        assert "## " + str(number) + "." in readme
    for name in manifest["files"]:
        if name.endswith((".json", ".md", ".py")):
            text = (leaf / name).read_text()
            assert ("/" + "cwStorage" + "/") not in text, name
            assert not re.search(r"(?<![0-9])(?:10|192[.]168)[.][0-9]+[.][0-9]+[.][0-9]+", text), name
            assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:OPENSSH|RSA) PRIVATE KEY", text), name
    print(json.dumps({"status": "passed", "payload_files": len(manifest["files"]), "original_worker_summary_sha256": identity(leaf / "run/summary.public.json")["sha256"], "scientific_runs_during_report_check": 0, "controller_exit_code": 0, "worker_exit_code": 0, "lock_released": True}))


if __name__ == "__main__":
    main()
