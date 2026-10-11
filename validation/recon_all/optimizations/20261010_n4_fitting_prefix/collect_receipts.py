"""完整前段四条链结束后，校验producer/source/inputs并数值保真导出。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(*, fnit_root: Path, output: Path, exporter: Path, replacements: Path) -> dict:
    run = fnit_root / "runs/n4_fitting_prefix_20261010_v2"
    group = run / "prefixes"
    if (run / "EXIT_CODE").read_text().strip() != "0":
        raise ValueError("prefix driver did not exit successfully")
    complete = json.loads((group / "COMPLETE.json").read_bytes())
    if complete["status"] != "complete_prefixes" or complete["runs"] != 4 or complete["full_workflow_complete"]:
        raise ValueError("not four completed prefixes")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n4-prefix-json-", dir=output.parent) as temporary:
        staged = Path(temporary)
        shutil.copyfile(group / "COMPLETE.json", staged / "COMPLETE.json")
        frozen_source = fnit_root / "workspaces/recon_python_profile_20261010_v4/SOURCE_MANIFEST.json"
        source_manifest = json.loads(frozen_source.read_bytes())
        source_checked = []
        for name, item in source_manifest["files"].items():
            path = frozen_source.parent / name
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("frozen source file changed: " + name)
            source_checked.append(name)
        n4_binding_path = fnit_root / "workspaces/n4_fitting_threads_20261010_v2/CANDIDATE_BINDING.json"
        n4_binding = json.loads(n4_binding_path.read_bytes())
        n4_module = n4_binding_path.parent / "n4_simpleitk.py"
        if digest(n4_module) != n4_binding["source_sha256"]["n4_simpleitk.py"]:
            raise ValueError("N4 overlay changed after ABBA")
        shutil.copyfile(n4_binding_path, staged / "N4_overlay_binding.json")
        (staged / "frozen_source_checked.json").write_text(json.dumps({
            "source_manifest_sha256": digest(frozen_source), "files_checked": len(source_checked),
            "each_file_bytes_and_sha256_match": True, "scope": "post-execution filesystem check, not OS isolation"}, indent=2)+"\n")
        shutil.copyfile(frozen_source, staged / "frozen_source_manifest.json")
        for case in ("sub07", "sub06"):
            comparison_path = run / "comparisons" / (case + ".json")
            comparison = json.loads(comparison_path.read_bytes())
            if comparison["status"] != "complete_prefix_diagnostic" or comparison["full_workflow_complete"]:
                raise ValueError("comparison not complete")
            shutil.copyfile(comparison_path, staged / (case + "-comparison.json"))
            for fit, side in ((1, "reference"), (4, "candidate")):
                name = case + "-fit" + str(fit)
                producer = group / name / "benchmark.json"
                record = json.loads(producer.read_bytes())
                runtime = group / name / "subject/fnit-native-free-run.json"
                stage = json.loads(runtime.read_bytes())
                process_path = group / (name + "-process.json")
                process = json.loads(process_path.read_bytes())
                if (record["execution"] != "complete_prefix" or record["full_workflow_complete"]
                        or stage["status"] != "complete_prefix" or stage["stages"][-1]["name"] != "filled_auto_checkpoint"
                        or process["exit_code"] != 0 or process["execution_audit"]["blocked_events"] != 0
                        or record["input_sha256"] != record["final_input_sha256"]
                        or record["source_manifest_sha256"] != digest(frozen_source)
                        or record["N4_module_sha256"] != digest(n4_module)
                        or record["N4_binding_sha256"] != digest(n4_binding_path)
                        or record["script_sha256"] != complete["script_sha256"]
                        or record["production_runtime_sha256"] != digest(runtime)
                        or digest(producer) != comparison[side + "_producer_sha256"]
                        or digest(runtime) != comparison[side + "_runtime_sha256"]):
                    raise ValueError("producer/source/audit/hash guard failed: " + name)
                snapshot = comparison_path.with_name(case + "." + side + "_snapshot.json")
                if digest(snapshot) != digest(producer):
                    raise ValueError("snapshot bytes differ from captured producer")
                shutil.copyfile(producer, staged / (name + "-benchmark.json"))
                shutil.copyfile(runtime, staged / (name + "-runtime.json"))
                shutil.copyfile(process_path, staged / (name + "-process.json"))
                calls = record["actual_N4_caller_parameters"]
                if len(calls) != 1 or calls[0]["kwargs"]["fitting_threads"] != fit:
                    raise ValueError("N4 fitting_threads did not reach exactly one complete call")
                profile = Path(calls[0]["kwargs"]["profile_path"])
                if not profile.resolve().is_relative_to((group / name).resolve()):
                    raise ValueError("N4 profile escaped this producer")
                shutil.copyfile(profile, staged / (name + "-N4-profile.json"))
        old = fnit_root / "runs/n4_fitting_prefix_20261010_v1"
        failure = {"status": "failed_prefix_import_audit", "included_in_successful_prefix_pair": False,
            "exit_code": int((old / "EXIT_CODE").read_text()),
            "reason": "diagnostic custom audit rejected legal NumPy subprocess.run('lscpu', shell=False); v2 reuses shared startup audit unchanged",
            "is_N4_or_OOM_failure": False, "old_script_sha256": digest(fnit_root / "workspaces/n4_fitting_prefix_20261010_v1/run_prefix.py"),
            "old_driver_log_sha256": digest(old / "driver.log"),
            "old_child_log_sha256": digest(old / "prefixes/sub07-fit1.log")}
        (staged / "failed-v1.json").write_text(json.dumps(failure, indent=2) + "\n")
        specification = importlib.util.spec_from_file_location("numeric_preserving_export", exporter)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module.export_receipts(input_directory=staged, output_directory=output, replacements_file=replacements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("fnit-root", "output", "exporter", "replacements"):
        parser.add_argument("--" + option, type=Path, required=True)
    result = collect(**vars(parser.parse_args()))
    print(json.dumps({"files": len(result["files"]), "preserved_numeric_leaves": result["numeric_boolean_null_values_unchanged"]}))
