"""Bind historical real benchmarks to the current integrated source, read-only."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check() -> dict:
    morph_path = "validation/synthmorph/cpu_fixes_20261004/gpu_route_v35.public.json"
    world_path = "validation/multimodal_cpu_20261004/gpu_world_v20_20261004.public.json"
    morph = json.loads((ROOT / morph_path).read_text())
    world = json.loads((ROOT / world_path).read_text())
    bindings = {
        path: {
            "historical_v35_sha256": values["candidate_sha256"],
            "current_sha256": digest(path),
            "same_bytes": digest(path) == values["candidate_sha256"],
        }
        for path, values in morph["six_core_files"].items()
    }
    local_cpu_sources = {
        "src/fnit/synthmorph/_cpu_raw_sampler.py": "f87ae99bf1a4fb8fc42eb196801871dbaa7f141f135bcb4a761a7c979ebfdfcd",
        "src/fnit/synthmorph/_cpu_preprocessing.py": "1c9f370c5904804b66e4ad7a51c4b558b19982c583846f210689f2766010139b",
        "src/fnit/synthmorph/_cpu_eigen.py": "ff7936cf2d89c740e09ba164a7c6a76fcebd22e889fd33f8c1a419a25c71a426",
    }
    local_bindings = {
        path: {"benchmarked_sha256": expected, "current_sha256": digest(path),
               "same_bytes": digest(path) == expected}
        for path, expected in local_cpu_sources.items()
    }
    direct_world = {
        path: {"benchmarked_sha256": expected, "current_sha256": digest(path),
               "same_bytes": digest(path) == expected}
        for path, expected in world["source_hashes"]["candidate"].items()
    }
    tree = ast.parse((ROOT / "src/fnit/synthmorph/pipeline.py").read_text())
    callers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                   and call.func.id == "resample_world_image" for call in ast.walk(node)):
                callers.append(node.name)
    gates = {
        "ordinary_registration_five_core_files_unchanged": all(
            item["same_bytes"] for path, item in bindings.items()
            if path != "src/fnit/_world_resampling.py"
        ),
        "accepted_CPU_helpers_match_real_reports": all(item["same_bytes"] for item in local_bindings.values()),
        "world_call_is_only_public_apply_transform": callers == ["apply_transform"],
        "current_world_sampler_matches_upstream_real_report": direct_world["src/fnit/_world_resampling.py"]["same_bytes"],
        "current_world_spline_matches_upstream_real_report": direct_world["src/fnit/eddy/fsl2111_strict/spline.py"]["same_bytes"],
        "upstream_490_frame_saved_files_equal": world["full_output_proof"]["all_saved_bytes_equal"],
        "upstream_490_frame_GPU_peak_unchanged": world["peak_allocation_unchanged"],
    }
    return {
        "schema": "fnit.smri.synthmorph_integration_binding.v1",
        "audited_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Current source binding and static call graph only; no new CNN, MRI benchmark or GPU timing.",
        "historical_reports_preserved_sha256": {morph_path: digest(morph_path), world_path: digest(world_path)},
        "v35_historical_core_bindings": bindings,
        "accepted_CPU_helper_bindings": local_bindings,
        "upstream_World_report_bindings": direct_world,
        "direct_world_callers_in_synthmorph_pipeline": callers,
        "gates": gates,
        "passed": all(gates.values()),
        "limitations": [
            "Historical v35 six-core byte identity describes its frozen task source. Current World differs after the independent upstream d63c2580 CPU change; do not relabel the old report.",
            "Ordinary SynthMorph registration uses the unchanged five core files and spatial sampler, not the WorldTransformChain branch. The World delta does not require repeating ordinary joint CNN fits.",
            "The upstream World report binds the current direct sampler and spline. Its ApplyWarp wrapper file has a different SHA; this audit does not transfer a whole-wrapper benchmark to the integrated wrapper.",
            "The saved upstream full 490-frame H100 proof is reused. Its single shared-GPU timing observation does not establish stable performance.",
            "The upstream 3D fmriprep time-unit header fix applies on CPU and CUDA. A 490-frame proof does not establish unchanged headers for that 3D case.",
        ],
    }


if __name__ == "__main__":
    result = check()
    destination = Path(__file__).with_name("synthmorph_integration_binding_20261005.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}))
    raise SystemExit(0 if result["passed"] else 1)
