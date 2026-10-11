"""核对完整两例ABBA、父CUDAAPI及源码/资源版本；仅导出JSON诊断收据。"""
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
    run = fnit_root / "runs/input_stage_parallel_20261010_v2"
    workspace = fnit_root / "workspaces/input_stage_parallel_20261010_v2"
    if (run / "EXIT_CODE").read_text().strip() != "0":
        raise ValueError("complete group driver did not finish successfully")
    binding_path = workspace / "SOURCE_MANIFEST.json"
    binding = json.loads(binding_path.read_bytes())
    for name, expected in binding["overlay_sha256"].items():
        if digest(workspace / name) != expected:
            raise ValueError("overlay source changed: " + name)
    frozen = fnit_root / "workspaces/recon_python_profile_20261010_v4/SOURCE_MANIFEST.json"
    if digest(frozen) != binding["frozen_base_manifest_sha256"]:
        raise ValueError("frozen base manifest changed")
    modules = json.loads(frozen.read_bytes())["files"]
    for name, row in modules.items():
        path = frozen.parent / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("frozen file changed: " + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="input-group-json-", dir=output.parent) as temporary:
        staged = Path(temporary)
        shutil.copyfile(binding_path, staged / "overlay_source_manifest.json")
        shutil.copyfile(frozen, staged / "frozen_source_manifest.json")
        checked = {"frozen_files_checked": len(modules), "each_file_bytes_and_sha256_match": True,
                   "source_manifest_sha256": digest(frozen), "overlay_source_manifest_sha256": digest(binding_path),
                   "driver_sha256": digest(workspace / "run_abba.sh"),
                   "collector_sha256": digest(Path(__file__)),
                   "scope": "post-run actual files, not OS execution/dynamic-library/file-access isolation"}
        for case in ("sub07", "sub06"):
            gate_path = fnit_root / f"runs/input_stage_parallel_20261010_v1/gate/{case}/gate.json"
            gate = json.loads(gate_path.read_bytes())
            if (gate["status"] != "complete" or not gate["safe_for_group"]
                    or gate["float_comparison"]["different_elements"] != 0
                    or gate["float_comparison"]["reference_dtype"] != gate["float_comparison"]["candidate_dtype"]
                    or gate["final_input_sha256"] != gate["input_sha256"]
                    or gate["candidate_module_sha256"] != binding["overlay_sha256"]["n4_simpleitk.py"]):
                raise ValueError("full N4 reconstruction1/4 gate failed")
            shutil.copyfile(gate_path, staged / (case + "-reconstruction-gate.json"))
            base = run / "ABBA" / case
            abba = json.loads((base / "ABBA.json").read_bytes())
            if abba["status"] != "complete" or len(abba["runs"]) != 4 or not abba["strict_same_input_passed"]:
                raise ValueError("not four complete, unchanged output group trials: " + case)
            if [item["execution"] for item in abba["runs"]] != ["serial", "parallel", "parallel", "serial"]:
                raise ValueError("not the declared ABBA order")
            for comparison in abba["comparisons"].values():
                for name in ("nu0.mgz", "synthseg.mgz"):
                    row = comparison[name]
                    if (row["different_voxels"] or row["affine_max_abs"] or row["vox2ras_tkr_max_abs"]
                            or row["reference_dtype"] != row["candidate_dtype"]
                            or row["reference_shape"] != row["candidate_shape"]):
                        raise ValueError("voxel/geometry/dtype replacement regression")
                if not all(value == 1 for value in comparison["synthseg.mgz"]["label_dice"].values()):
                    raise ValueError("segmentation Dice regressed")
                if not comparison["csv_byte_identical"]:
                    raise ValueError("complete volume CSV changed")
            shutil.copyfile(base / "ABBA.json", staged / (case + "-ABBA.json"))
            for item in abba["runs"]:
                name = f'{case}-{item["index"]}-{item["execution"]}'
                directory = base / f'{item["index"]}-{item["execution"]}'
                _check_trial(directory, item["execution"], gate["input_sha256"], binding_path, binding)
                for source in directory.glob("*.json"):
                    shutil.copyfile(source, staged / (name + "-" + source.name))
        live = run / "parent_initialized/sub07"
        parent = json.loads((live / "run.json").read_bytes())
        if not parent["parent_initialized"] or not parent["parent_buffer_preserved"]:
            raise ValueError("live parent CUDA API gate failed")
        _check_trial(live, "parallel", json.loads((staged / "sub07-reconstruction-gate.json").read_bytes())["input_sha256"], binding_path, binding)
        for source in live.glob("*.json"):
            shutil.copyfile(source, staged / ("parent-initialized-sub07-" + source.name))
        # The buffer API output is a fifth same-input candidate, not an ABBA speed sample.
        tool = workspace / "benchmark_recon_input_stage_parallel.py"
        spec = importlib.util.spec_from_file_location("input_group_benchmark", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        api_comparison = module.compare_groups(run / "ABBA/sub07/1-serial", live)
        if not api_comparison["strict_same_input_passed"]:
            raise ValueError("initialized-parent API output changed")
        (staged / "initialized-parent-comparison.json").write_text(json.dumps(api_comparison, indent=2)+"\n")
        (staged / "post_run_checked.json").write_text(json.dumps(checked, indent=2)+"\n")
        spec = importlib.util.spec_from_file_location("numeric_preserving_export", exporter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.export_receipts(input_directory=staged, output_directory=output, replacements_file=replacements)


def _check_trial(directory: Path, execution: str, input_sha: str, binding_path: Path, binding: dict) -> None:
    producer = json.loads((directory / "run.json").read_bytes())
    group_path = directory / "group.json"
    group = json.loads(group_path.read_bytes())
    if (producer["status"] != "complete" or group["status"] != "complete" or not group["parent_state_preserved"]
            or producer["group_report_sha256"] != digest(group_path)
            or producer["source_manifest_sha256"] != digest(binding_path)
            or producer["script_sha256"] != binding["overlay_sha256"]["benchmark_recon_input_stage_parallel.py"]
            or group["runtime_helper_sha256"] != binding["overlay_sha256"]["input_stage_parallel.py"]
            or group["inputs_sha256"] != group["final_inputs_sha256"]
            or producer["input_sha256"] != input_sha or group["inputs_sha256"]["orig"] != input_sha
            or group["execution"] != execution or group["thread_budget_total"] != 4
            or set(group["workers"]) != {"n4", "synthseg"}):
        raise ValueError("producer/source/input/state guard failed: " + directory.name)
    expected = {"n4": 1, "synthseg": 3} if execution == "parallel" else {"n4": 4, "synthseg": 4}
    if group["algorithm_threads"] != expected or any(value != 0 for value in group["returncodes"].values()):
        raise ValueError("thread or returncode guard failed")
    for name, worker in group["workers"].items():
        original_worker = directory / f"group.{name}.worker.json"
        if json.loads(original_worker.read_bytes()) != worker or worker["status"] != "complete":
            raise ValueError("worker did not complete identically to joined report")
        if worker["value"]["input_sha256"] != input_sha:
            raise ValueError("worker used another source image")
    n4 = group["workers"]["n4"]["value"]
    if (n4["fitting_threads"] != 1 or n4["reconstruction_threads"] != expected["n4"]
            or n4["used_native_programs"] or n4["source_sha256"]["fnit.recon_all.n4_simpleitk"] != binding["overlay_sha256"]["n4_simpleitk.py"]):
        raise ValueError("N4 algorithm/threads/source changed")
    synthseg = group["workers"]["synthseg"]["value"]
    if synthseg["actual_model_device"] != group["device"] or synthseg["actual_model_dtype"] != "torch.float32":
        raise ValueError("SynthSeg actual model device/precision wrong")
    actual_uuid = group["workers"]["synthseg"]["actual_cuda_device"]["uuid"]
    actual_uuid = actual_uuid if actual_uuid.startswith("GPU-") else "GPU-" + actual_uuid
    sampled_uuid = group["device_process_tree"]["target_gpu_uuid"]
    if sampled_uuid is not None and actual_uuid != sampled_uuid:
        raise ValueError("GPU compute and sampled target UUID differ")
    forwards = synthseg["precision"]["forwards"]
    if len(forwards) != 2 or [row["pass"] for row in forwards] != ["original", "flipped"]:
        raise ValueError("SynthSeg ensemble incomplete")
    for row in forwards:
        if (row["device"] != group["device"] or not row["matmul_tf32"] or row["cudnn_tf32"]
                or row["input_dtype"] != "torch.float32" or row["output_dtype"] != "torch.float32"
                or row["model_dtypes"] != ["torch.float32"]
                or any(value["enabled"] for value in row["autocast"].values())):
            raise ValueError("SynthSeg actual forward policy changed")
    for name, path in (("n4", "nu0.mgz"), ("segmentation", "synthseg.mgz"), ("volumes", "synthseg.vol.csv")):
        if digest(directory / path) != group["outputs_sha256"][name]:
            raise ValueError("published output changed after group completion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("fnit-root", "output", "exporter", "replacements"):
        parser.add_argument("--" + option, type=Path, required=True)
    result = collect(**vars(parser.parse_args()))
    print(json.dumps({"files": len(result["files"]), "numeric_preserved": result["numeric_boolean_null_values_unchanged"]}))
