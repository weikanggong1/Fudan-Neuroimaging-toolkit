"""收集已经结束的两例N4线程ABBA；复用既有数值保真导出器。"""
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


def collect_receipts(*, fnit_root: Path, output_directory: Path,
                     exporter_file: Path, replacements_file: Path) -> dict:
    """显式FNIT根/新输出目录/既有exporter/字符串替换表→公开JSON清单。

    只选v2的8次完整file API、两例比较和源码绑定；保留v1仅诊断写JSON
    失败的原始profile与退出记录。阶段未结束、输入或源码变动直接报错，
    不导出原始影像、浮点数组、权重、许可证或上游C++源码。
    """
    run = fnit_root / "runs/n4_fitting_threads_20261010_v2"
    work = fnit_root / "workspaces/n4_fitting_threads_20261010_v2"
    group = run / "ABBA"
    complete = json.loads((group / "COMPLETE.json").read_bytes())
    if complete["status"] != "complete" or complete["runs"] != 8 or (run / "EXIT_CODE").read_text().strip() != "0":
        raise ValueError("eight ABBA runs and shell exit must be complete")
    if (run / "UNIT_EXIT_CODE").read_text().strip() != "0":
        raise ValueError("adapter contracts failed")
    binding = json.loads((work / "CANDIDATE_BINDING.json").read_bytes())
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n4-fit-json-", dir=output_directory.parent) as temporary:
        staged = Path(temporary)
        shutil.copyfile(group / "COMPLETE.json", staged / "COMPLETE.json")
        shutil.copyfile(work / "CANDIDATE_BINDING.json", staged / "source_binding.json")
        for case in ("sub07", "sub06"):
            comparison = json.loads((group / f"{case}_comparisons.json").read_bytes())
            if not comparison["same_input"] or not comparison["all_geometry_exact"] or not comparison["all_thread_context_restored"]:
                raise ValueError("input/geometry/context contract failed")
            shutil.copyfile(group / f"{case}_comparisons.json", staged / f"{case}_comparisons.json")
            for index, fit in enumerate((1, 4, 4, 1), 1):
                name = f"{case}-{index}-fit{fit}"
                directory = group / name
                raw = json.loads((directory / "run.json").read_bytes())
                process = json.loads((group / (name + "_process.json")).read_bytes())
                if (raw["status"] != "complete" or raw["fitting_threads"] != fit or process["exit_code"] != 0
                        or raw["input_sha256_before"] != raw["input_sha256_after"]
                        or raw["script_sha256"] != complete["script_sha256"]
                        or raw["source_module_sha256"] != binding["source_sha256"]["n4_simpleitk.py"]
                        or raw["script_sha256"] != binding["source_sha256"]["benchmark_threaded_n4.py"]
                        or raw["source_binding_sha256"] != digest(work / "CANDIDATE_BINDING.json")
                        or raw["details"]["actual_fitting_work_units"] != fit
                        or raw["details"]["actual_fitting_filter_threads"] != fit):
                    raise ValueError("run contract failed: " + name)
                if digest(directory / "run.json") != comparison["run_json_sha256"][index - 1]:
                    raise ValueError("comparison producer SHA changed: " + name)
                for filename in ("run.json", "profile.json"):
                    shutil.copyfile(directory / filename, staged / (name + "-" + filename))
                shutil.copyfile(group / (name + "_process.json"), staged / (name + "-process.json"))
            reference = fnit_root / "runs/recon_n4_python_adapter_20261010_v1" / case
            shutil.copyfile(reference / "report.json", staged / (case + "-existing-native-reference.json"))
            for repeat in (0, 1):
                shutil.copyfile(reference / f"native-{repeat}-profile.json", staged / f"{case}-existing-native-{repeat}-profile.json")
        old = fnit_root / "runs/n4_fitting_threads_20261010_v1"
        shutil.copyfile(old / "ABBA/sub07-1-fit1/profile.json", staged / "failed-v1-completed-N4-profile.json")
        failure = {
            "status": "failed_diagnostic_json_serialization_after_N4_write",
            "exit_code": int((old / "EXIT_CODE").read_text()),
            "included_in_ABBA": False,
            "error": "TypeError: Object of type int32 is not JSON serializable",
            "cause": "diagnostic shape used NumPy int32; v2 converts only shape entries to Python int",
            "old_script_sha256": binding["v1_failed_driver_SHA"],
            "old_log_sha256": digest(old / "benchmark.log"),
            "finished_profile_sha256": digest(old / "ABBA/sub07-1-fit1/profile.json"),
            "finished_nu0_sha256": digest(old / "ABBA/sub07-1-fit1/nu0.mgz"),
            "not_N4_or_GPU_failure": True,
        }
        (staged / "failed-v1.json").write_text(json.dumps(failure, indent=2) + "\n")
        contracts = {
            "exit_code": int((run / "UNIT_EXIT_CODE").read_text()),
            "log_sha256": digest(run / "contracts.log"),
            "log": (run / "contracts.log").read_text(),
            "scope": "seven existing adapter contracts with added thread validation/restoration checks",
        }
        (staged / "contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
        spec = importlib.util.spec_from_file_location("existing_numeric_exporter", exporter_file)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        return exporter.export_receipts(input_directory=staged, output_directory=output_directory,
                                        replacements_file=replacements_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fnit-root", "output-directory", "exporter-file", "replacements-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = collect_receipts(**vars(args))
    print(json.dumps({"files": len(result["files"]), "numeric_leaves": result["numeric_boolean_null_values_unchanged"]}))


if __name__ == "__main__":
    main()
