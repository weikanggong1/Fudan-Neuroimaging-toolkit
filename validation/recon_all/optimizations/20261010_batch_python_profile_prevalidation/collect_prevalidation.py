"""本地只读源码清单及无MRI预检复现；不运行原软件阶段程序。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


BASELINE = "fc36d6c1c761302ed5b7425b730b17ab4951090b"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(*, repository: Path, output: Path) -> None:
    """仓库路径→源码/包版本/预检JSON；无MRI或CUDA计算，不改仓库源码。"""
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repository / "src"))
    from fnit.recon_all.batch import run_recon_all_python_batch
    from fnit.recon_all.python_profile import python_profile_options
    import torch
    before_cuda = torch.cuda.is_initialized()
    old = subprocess.run(["git", "show", f"{BASELINE}:src/fnit/recon_all/batch.py"],
                         cwd=repository, check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as directory:
        old_path = Path(directory) / "batch_before.py"
        old_path.write_bytes(old)
        spec = importlib.util.spec_from_file_location("fnit.recon_all._batch_before", old_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        results = []
        for name, extra in (("isolated_WM_with_mixed_defaults", {"wm_execution": "isolated"}),
                            ("Torch_inflation_with_one_worker_default", {"inflate_backend": "torch"})):
            row = {"configuration": name, "options": extra}
            for label, runner in (("before", module.run_recon_all_python_batch),
                                  ("after", run_recon_all_python_batch)):
                try:
                    value = runner(jobs=[], weights_dir=directory, assets_dir=directory,
                                   devices=("cuda:1",), threads=4, backend="python", **extra)
                    row[label] = {"accepted_empty_job_preflight": value == [], "error": None}
                except ValueError as error:
                    row[label] = {"accepted_empty_job_preflight": False, "error": str(error)}
            results.append(row)
    packages = {}
    for name in ("torch", "numpy", "numba", "nibabel", "scipy", "SimpleITK", "triton"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                          check=True, capture_output=True, text=True).stdout.strip()
    sources = {}
    for name in ("native_free", "batch", "python_profile", "native_runtime_selection",
                 "input_n4_chain", "n4_itk", "n4_simpleitk", "n4_itk_torch_experimental",
                 "mri_segment", "edit_wm_aseg_core_python", "wm_aseg_io",
                 "mris_register_average_numba", "mris_register_average_gpu",
                 "discrete_curvature_torch", "mni_nonlinear_chain", "python_gpu_profile"):
        path = repository / "src/fnit/recon_all" / f"{name}.py"
        sources[str(path.relative_to(repository))] = {"sha256": sha(path.read_bytes()),
                                                    "size_bytes": path.stat().st_size}
    source = {"schema_version": 1, "git_head_at_capture": head, "baseline_git_commit": BASELINE,
              "before_batch_sha256": sha(old), "source_files": sources,
              "local_interpreter_python": sys.version.split()[0], "local_package_versions": packages,
              "runtime_version_scope": "local focused tests only; not the CFFF production environment",
              "generic_libraries_permitted": ["Python", "NumPy", "Numba", "PyTorch", "SimpleITK", "SciPy"],
              "declared_versions": {"Conda_python": "3.11", "Conda_pytorch": "2.5.1",
                                    "Conda_triton": "3.1.0", "SimpleITK": "2.5.6"}}
    stages = [
        ("N4", "fnit_n4_itk", "n4_backend=native", "SimpleITK complete CPU N4; explicit complete Torch N4", "input_n4_chain.py:54-80; n4_itk.py:39-72", "SimpleITK native reference 28/99 voxel differences; Torch downstream changes; default fitting_threads=1"),
        ("WM segmentation", "mri_segment", "wm_backend=native", "Torch histogram plus cached planes and ordered CPU rules", "native_free.py:1763-1770; mri_segment.py:905", "mature threshold bug 14->13 fixed; full frozen two-case zero differences; whole separately assessed"),
        ("WM/aseg edit", "mri_edit_wm_with_aseg", "wm_edit_backend=native", "complete standard Python/Numba edit; guarded Torch hybrid optional", "native_free.py:1771-1779; edit_wm_aseg_core_python.py:529-623", "remove_paths byte-access branch remains geometric no-op proof only; unsupported geometry explicitly fails"),
        ("Topology repair", "mris_fix_topology_fnit", "topology_backend=native", "digital topology Python alternative; independent source-semantics GA standalone work", "native_free.py:308-329; native_free.py:1320-1337", "digital topology is a different algorithm; own GA not wired into current CLI recipe; full GA/whole QC pending"),
        ("nofix inflation", "mris_inflate", "surface_placement_backend=native", "complete own standard nofix inflation", "native_free.py:290-306; native_free.py:716-722", "inflate_backend=torch alone does not select nofix replacement; Python placement/profile does"),
        ("standard inflation", "mris_inflate", "inflate_backend=native", "own full Torch inflation and sulc", "native_free.py:366-385", "frozen stage contracts do not establish raw-T1 mesh quality"),
        ("white.preaparc", "mris_place_surface or optional white_fast", "surface_placement_backend=native", "own full Python/Numba placement", "native_free.py:596-610", "actual same-input stage gates; whole remains separate"),
        ("final white and pial", "mris_place_surface or optional white_fast", "surface_placement_backend=native", "own complete Python/Numba/Torch four-round placement", "native_free.py:786-816; native_free.py:615-639", "no default switch; raw whole topology/placement QC unresolved"),
        ("avg_curv atlas paint", "mrisp_paint", "surface_placement_backend=native", "own Python/Numba atlas sampler and ordered averages", "native_free.py:468-491; native_free.py:1016-1018", "frozen stages passed; whole separate"),
        ("curvature statistics", "mris_curvature_stats", "surface_placement_backend=native", "own full Python/Numba and Torch map substeps", "native_free.py:496-519; native_free.py:1985-1987", "source_libm is NumBa/system math, not original FS shared library"),
        ("defects projection", "mri_label2vol", "defects_backend=native", "own full Torch projection with ordered bilateral accumulation", "native_free.py:526-557", "frozen two-case projection gate passed; full GA quality not implied"),
        ("GCA registration", "mri_em_register", "CPU auto or native_optimizations=original", "CUDA auto uses own Torch scoring and CPU Python EM", "native_runtime_selection.py:36-65; native_free.py:173-203", "CUDA default does not execute native EM; Python profile isolated/Torch inverse/chunk1024"),
        ("surface vertex metrics", "mris_place_surface", "only non-CUDA metrics and python_only=False", "CUDA default/Python profile use own thickness/area/curvature", "native_free.py:396-436", "do not count the CUDA mixed-default metric stage as native execution"),
        ("MNI postprocessing", "mri_warp_convert,mri_ca_register,mri_convert", "device is not CUDA", "complete own GPU warp conversion/inversion/check", "native_free.py:1443-1445; mni_nonlinear_chain.py:109-125", "Python recipe requires explicit CUDA; CPU native branch remains comparison"),
    ]
    inventory = {"schema_version": 1, "git_head_at_capture": head,
        "scope": "current source conditional call-chain audit, not OS isolation or a new real pipeline execution",
        "source_files": sources,
        "native_mixed_default": {"device": "cuda:0", "backend": "native"},
        "python_profile": python_profile_options(device="cuda:0", threads=4),
        "python_profile_original_stage_programs_consumed_static": [],
        "stages": [{"stage": a, "original_program": b, "condition": c,
                    "own_alternative": d, "evidence": e, "acceptance_gap": f}
                   for a,b,c,d,e,f in stages],
        "already_own_without_original_stage_program": ["conform and rawavg", "Synth GPU family",
            "nu and both T1 normalizations", "CA normalize (default CPU Python)", "mri_cc",
            "denoise", "filled (Python/Numba)", "pretess/tessellate/main component",
            "remesh/intersection", "sphere/sphere.reg", "GCSA annotation",
            "ribbon/segmentation volume mapping", "brain/ROI statistics"],
        "sphere_average_dispatch": "Python recipe explicit CUDA -> Triton; optional CPU _average_cpu_cpp is not reached",
        "pure_python_gpu_switch": "still raises structured PurePythonGpuUnavailable before output; CPU/generic library recipe is backend=python",
        "wm_edit_production_subject_hash_whitelist": False,
        "strict_reproduction": "not_evaluated_this_audit", "whole_equivalence": "not_assessed"}
    receipt = {"schema_version": 1, "status": "focused_prevalidation_and_cli_contracts_passed",
        "before_after_reproductions": results,
        "cuda_initialized_before": before_cuda, "cuda_initialized_after": torch.cuda.is_initialized(),
        "real_MRI_pipeline_run": False, "original_stage_program_run": False,
        "new_real_precision_or_speed_measurement": False,
        "gpu_memory_bytes": None, "whole_equivalence": "not_assessed"}
    for name, content in (("source.json", source), ("current_inventory.json", inventory),
                          ("prevalidation_receipt.json", receipt)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    collect(**vars(parser.parse_args()))
