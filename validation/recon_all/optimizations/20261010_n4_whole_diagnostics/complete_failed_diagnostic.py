"""从带SHA的已完成事后比较检查点，只补参考QC与脑图；不重跑生产或距离。

原诊断因Numba临时模块缓存名失效而停止，其failed收据/源码保持。
恢复是明确分段的事后诊断；生产recon-all仍failed，成功整例加速仍为null。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from diagnose_failed_outputs import (execute_quality_reference, inventory_outputs,
                                     load_module, require_same_resources, sha256,
                                     validate_failed_producer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preserved-diagnostic", "benchmark", "source-root", "control-benchmark", "control-source-root",
                 "complete-validator", "reference-root", "driver", "scripts-dir", "label-table", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--reference-kind", choices=("control", "official"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4 or args.output.exists():
        raise ValueError("four threads and a new output directory required")
    previous_path = args.preserved_diagnostic / "diagnosis.json"
    previous = json.loads(previous_path.read_text())
    if previous["status"] != "diagnostic_failed_producer_still_failed":
        raise ValueError("only preserved failed post-run diagnostic checkpoint is supported")
    if "ModuleNotFoundError: No module named '_failed_existing_quality'" not in (args.preserved_diagnostic / "commands.log").read_text():
        raise ValueError("checkpoint was not the diagnosed dynamic-module cache failure")
    if previous["case"] != args.case or previous["reference_kind"] != args.reference_kind:
        raise ValueError("diagnostic checkpoint case/relation mismatch")
    manifest = json.loads((args.reference_root / "TRANSFER_MANIFEST.private.json").read_text())
    candidate, runtime, failure = validate_failed_producer(benchmark_path=args.benchmark,
        source_root=args.source_root, case=args.case, reference_manifest=manifest)
    validator = load_module(path=args.complete_validator, name="_recovery_complete_validator")
    control, _ = validator.validate_candidate(benchmark_path=args.control_benchmark,
        source_root=args.control_source_root, case=args.case, reference_manifest=manifest)
    require_same_resources(control=control, candidate=candidate, threads=4)
    for path, binding in ((args.benchmark, "benchmark_sha256"),
            (args.benchmark.parent / "subject/fnit-native-free-run.json", "run_sha256"),
            (args.benchmark.parent / "subject/scripts/mni-mesh-parallel.json", "failed_mesh_receipt_sha256"),
            (args.control_benchmark, "control_benchmark_sha256")):
        if sha256(path) != previous[binding]:
            raise ValueError("producer/control receipt changed since diagnostic checkpoint")
    # Includes the original v1 diagnostic source: it remains frozen at its measured remote path.
    for path, binding in previous["comparator_sha256"].items():
        if sha256(Path(path)) != binding:
            raise ValueError("frozen comparison/checkpoint source changed")
    subject = args.benchmark.parent / "subject"
    expected = load_module(path=args.source_root / "src/fnit/recon_all/expected_outputs.py", name="_recovery_output_manifest")
    if inventory_outputs(subject=subject, expected_paths=expected.paths()) != previous["output_inventory"]:
        raise ValueError("preserved producer files changed since comparison")
    reference = (args.control_benchmark.parent / "subject" if args.reference_kind == "control"
                 else args.reference_root / "subjects" / args.case)
    label = "candidate_vs_" + args.reference_kind
    files = [f"{name}_{label}.json" for name in ("strict", "geometry", "region", "dice", "surface", "local", "no_th3")]
    files += ["no_th3_inputs.json", "distance_cache.json", "preserved_failed_mesh_receipt.json", "quality_candidate/report.json"]
    bindings = {relative: sha256(args.preserved_diagnostic / relative) for relative in files}
    candidate_quality = json.loads((args.preserved_diagnostic / "quality_candidate/report.json").read_text())
    if candidate_quality["status"] not in ("measured", "partially_measured"):
        raise ValueError("candidate quality checkpoint did not finish")
    driver = load_module(path=args.driver, name="_recovery_frozen_driver")
    driver.configure_runtime()
    args.output.mkdir(parents=True)
    for relative in files:
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.preserved_diagnostic / relative, destination)
    state = dict(previous)
    state.pop("diagnostic_error", None)
    state.update(status="recovering_post_run_diagnostics_producer_still_failed", recovery={
        "scope": "checkpoint-based post-failure diagnostics; only reference QC and figures newly executed; not continuous raw T1 or continuous diagnostic rerun",
        "previous_diagnostic_path": str(args.preserved_diagnostic), "previous_diagnostic_sha256": sha256(previous_path),
        "previous_status": previous["status"], "previous_diagnostic_wall_seconds": previous["diagnostic_wall_seconds"],
        "completed_checkpoint_sha256": bindings,
        "recovery_script_sha256": sha256(Path(__file__)),
        "cache_fix_source_sha256": sha256(Path(__file__).with_name("diagnose_failed_outputs.py")),
        "production_execution_unchanged": "failed", "successful_whole_speedup": None})
    driver.write(args.output / "diagnosis.json", state)
    commands, started = [], time.perf_counter()
    try:
        execute_quality_reference(command=[sys.executable, str(args.scripts_dir / "benchmark_surface_quality_extended.py"),
            "--subject", str(reference), "--output", str(args.output / "quality_reference"),
            "--code-version", control["code_version"] if args.reference_kind == "control" else manifest["cases"][args.case]["official_version"],
            "--source-kind", "fnit" if args.reference_kind == "control" else "official", "--threads", "4",
            "--cross-timeout-seconds", "180", "--max-bbox-pairs", "20000000"],
            log=args.output / "commands.log", commands=commands,
            cache_directory=args.output / "jit_reference", write=driver.write)
        driver.execute([sys.executable, str(args.scripts_dir / "plot_recon_all_comparison.py"),
            "--reference", str(reference), "--candidate", str(subject),
            "--region-report", str(args.output / f"region_{label}.json"),
            "--dice-report", str(args.output / f"dice_{label}.json"),
            "--output-dir", str(args.output / "figures"), "--code-commit", candidate["code_version"]], args.output / "commands.log", commands)
        if inventory_outputs(subject=subject, expected_paths=expected.paths()) != previous["output_inventory"]:
            raise RuntimeError("preserved outputs changed during diagnostic recovery")
        for relative, expected_sha in bindings.items():
            if sha256(args.preserved_diagnostic / relative) != expected_sha or sha256(args.output / relative) != expected_sha:
                raise RuntimeError("completed comparison checkpoint changed")
        if sha256(previous_path) != state["recovery"]["previous_diagnostic_sha256"]:
            raise RuntimeError("original failed diagnosis changed")
        state.update(status="diagnostic_complete_producer_still_failed", inputs_unchanged=True,
                     quality_candidate=candidate_quality["status"],
                     quality_reference=driver.read(args.output / "quality_reference/report.json")["status"])
        return 0
    except BaseException as error:
        state.update(status="diagnostic_recovery_failed_producer_still_failed", recovery_error=repr(error))
        raise
    finally:
        state["recovery"]["recovery_wall_seconds"] = time.perf_counter() - started
        state["timing_scope"] = "original failed post-run diagnosis time and recovery time separately; not production or whole speedup"
        driver.write(args.output / "diagnosis.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
