"""Bounded, isolated target-preparation oracle; no registration or GEMS fit.

The official arm compiles only the pinned installed recipe prefixes that
finish at atlasAlignmentTarget assignment and targetMask save. It neither
imports the native GEMS binding nor provides utils.run in its namespace.
The production arm imports only the frozen independent FNIT preparation API.
The worker is prepared separately from authorizing or dispatching it.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def method(path, class_name, function_name):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == function_name)


def prefix(function, last_statement, name):
    function = copy.deepcopy(function)
    stop = next(i for i, node in enumerate(function.body) if last_statement(node))
    function.body = function.body[:stop + 1]
    function.name = name
    function.decorator_list = []
    # No registration runner or native optimizer may occur in the retained AST.
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("run", "Popen", "system", "collect_output", "step_optimizer"):
                raise RuntimeError("non-preparation call in retained oracle prefix")
    return function


def target_assignment(node):
    return (isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
        and t.value.id == "self" and t.attr == "atlasAlignmentTarget"
        for t in node.targets))


def target_save(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "mask" and node.value.func.attr == "save"
            and len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "targetMaskFile")


def oracle_prefixes(plan):
    paths = plan["official_sources"]
    hippo = prefix(method(paths["hippocampus"]["path"], "HippoAmygdalaSubfields", "preprocess_images"),
                   target_assignment, "prepare_hippo_prefix")
    target = prefix(method(paths["core"]["path"], "MeshModel", "align_atlas_to_seg"),
                    target_save, "prepare_target_prefix")
    utils_tree = ast.parse(Path(paths["utils"]["path"]).read_text())
    sphere = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == "spherical_strel")
    return hippo, target, sphere


def check_bindings(plan):
    assert plan["scope"] == "one_real_target_preparation_only"
    assert plan["registration_calls"] == 0 and plan["whole_GEMS_calls"] == 0
    for group in ("inputs", "official_sources", "official_runtime"):
        for row in plan[group].values():
            assert sha(row["path"]) == row["sha256"], "binding mismatch: " + group
    source = Path(plan["frozen_source"])
    actual_paths = {p.relative_to(source).as_posix() for p in source.rglob("*.py")}
    assert actual_paths == set(plan["runtime_sources"]), "FNIT source file set mismatch"
    for relative, expected in plan["runtime_sources"].items():
        assert sha(source / relative) == expected, "FNIT source mismatch: " + relative
    tree_sha = hashlib.sha256(json.dumps(plan["runtime_sources"], sort_keys=True).encode()).hexdigest()
    assert tree_sha == plan["runtime_tree_sha256"], "FNIT source tree digest mismatch"
    assert os.sched_getaffinity(0) == set(plan["affinity"])
    return {"input_sha256": {k: v["sha256"] for k, v in plan["inputs"].items()},
            "runtime_tree_sha256": plan["runtime_tree_sha256"],
            "official_source_sha256": {k: v["sha256"] for k, v in plan["official_sources"].items()}}


def observed_flags():
    flags = {"native_GEMS_binding_imported": any(name == "gems" or name.startswith("gems.")
                                               for name in sys.modules),
             "samseg_imported": any(name == "samseg" or name.startswith("samseg.")
                                    for name in sys.modules),
             "registration_calls": 0, "whole_GEMS_calls": 0,
             "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
             "removed_environment_variables_absent": all(name not in os.environ for name in
                  ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OPENBLAS_CORETYPE")),
             "address_space_limit_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
             "affinity": sorted(os.sched_getaffinity(0))}
    torch = sys.modules.get("torch")
    if torch is not None:
        flags.update(cuda_initialized=torch.cuda.is_initialized(),
                     torch_threads=torch.get_num_threads(),
                     torch_interop_threads=torch.get_num_interop_threads(),
                     tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32)
    return flags


def official(plan, output):
    # This arm runs in the installed reference Python, not FNIT's environment.
    import numpy as np
    import scipy
    import scipy.ndimage
    import surfa as sf
    started = time.monotonic()
    hippo, target, sphere = oracle_prefixes(plan)
    namespace = {"np": np, "scipy": scipy, "sf": sf, "os": os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[sphere], type_ignores=[])),
                 plan["official_sources"]["utils"]["path"], "exec"), namespace)
    namespace["utils"] = SimpleNamespace(spherical_strel=namespace["spherical_strel"])
    for function, source in ((hippo, "hippocampus"), (target, "core")):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     plan["official_sources"][source]["path"], "exec"), namespace)
    state = SimpleNamespace(
        side="right", atlasTargetSmoothing="backward", tempDir=str(output),
        atlasDumpFileName=plan["inputs"]["atlas_dump"]["path"],
        inputSeg=sf.load_volume(plan["inputs"]["aseg"]["path"]),
    )
    namespace["prepare_hippo_prefix"](state)
    namespace["prepare_target_prefix"](state)
    target_image = sf.load_volume(output / "targetMask.mgz")
    reflection = sf.load_volume(output / "flippedAtlasDump.mgz")
    return {
        "scope": "pinned_installed_recipe_prefixes_only", "seconds": time.monotonic() - started,
        "prefix_ranges": {"hippocampus": [hippo.lineno, hippo.body[-1].end_lineno],
                          "core": [target.lineno, target.body[-1].end_lineno]},
        "target_reloaded_header_geometry_RAS_mm": target_image.geom.vox2world.matrix.tolist(),
        "reflected_reloaded_header_geometry_RAS_mm": reflection.geom.vox2world.matrix.tolist(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "surfa": sf.__version__},
        "native_GEMS_binding_imported": "gems" in sys.modules,
        "registration_calls": 0, "whole_GEMS_calls": 0,
    }


def fnit(plan, output):
    sys.path.insert(0, plan["frozen_source"])
    import nibabel as nib
    import torch
    import numpy as np
    from fnit.robust_register import prepare_subregion_alignment_target, reflect_atlas_header
    assert Path(sys.modules["fnit.robust_register.preparation"].__file__).resolve() == (
        Path(plan["frozen_source"]) / "fnit/robust_register/preparation.py").resolve()
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    flags_before = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    started = time.monotonic()
    prepared = prepare_subregion_alignment_target(
        coarse_segmentation=plan["inputs"]["aseg"]["path"], target_label_ids=(53, 54),
        target_voxel_mm=1, bbox_margin_voxels=6, smoothing="backward", device="cpu",
        spatial_chunk_size=262144, memory_budget_gb=20,
    )
    reflected = reflect_atlas_header(plan["inputs"]["atlas_dump"]["path"])
    api_seconds = time.monotonic() - started
    nib.save(prepared.image, output / "targetMask.mgz")
    nib.save(reflected, output / "flippedAtlasDump.mgz")
    assert [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32] == flags_before
    assert not torch.cuda.is_initialized() and "surfa" not in sys.modules
    return {
        "seconds": time.monotonic() - started, "api_without_save_seconds": api_seconds,
        "target_report": prepared.report,
        "target_computational_affine_RAS_mm": prepared.image.affine.tolist(),
        "reflected_computational_affine_RAS_mm": reflected.affine.tolist(),
        "versions": {"numpy": np.__version__, "nibabel": nib.__version__, "torch": torch.__version__},
        "tf32_flags_unchanged": True, "cuda_initialized": torch.cuda.is_initialized(),
        "surfa_imported": "surfa" in sys.modules, "torch_threads": torch.get_num_threads(),
        "registration_calls": 0, "whole_GEMS_calls": 0,
    }


def score(plan, root):
    import nibabel as nib
    import numpy as np
    rows = {}
    for filename in ("targetMask.mgz", "flippedAtlasDump.mgz"):
        a = nib.load(root / "official" / filename)
        b = nib.load(root / "fnit" / filename)
        equal_shape = a.shape == b.shape
        data_differences = int(np.count_nonzero(np.asanyarray(a.dataobj) != np.asanyarray(b.dataobj))) if equal_shape else None
        header_fields = {name: bool(np.array_equal(a.header[name], b.header[name])) for name in (
            "version", "dims", "type", "dof", "goodRASFlag", "delta", "Mdc", "Pxyz_c",
            "tr", "flip_angle", "te", "ti", "fov")}
        rows[filename] = {
            "shape_equal": equal_shape, "shape_official": list(a.shape), "shape_fnit": list(b.shape),
            "dtype_equal": a.header.get_data_dtype() == b.header.get_data_dtype(),
            "different_voxels": data_differences, "header_fields_exact": header_fields,
            "stored_affine_exact": bool(np.array_equal(a.affine, b.affine)),
            "stored_affine_max_abs": float(np.max(np.abs(a.affine - b.affine))),
            "foreground_official": int(np.count_nonzero(a.dataobj)),
            "foreground_fnit": int(np.count_nonzero(b.dataobj)),
            "official_file_sha256": sha(root / "official" / filename),
            "fnit_file_sha256": sha(root / "fnit" / filename),
        }
        rows[filename]["gate"] = bool(equal_shape and rows[filename]["dtype_equal"]
            and data_differences == 0 and all(header_fields.values())
            and rows[filename]["stored_affine_exact"] and rows[filename]["foreground_official"] > 0)
    return {"declared_gate": "0_different_voxels_shape_dtype_MGH_fields_affine_exact_nonempty",
            "rows": rows, "all_gates_pass": all(row["gate"] for row in rows.values()),
            "storage_scope": "MGH geometry and scan fields; gzip bytes and optional metadata tags not compared",
            "registration_calls": 0, "whole_GEMS_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=("official", "fnit", "score"), required=True)
    args = parser.parse_args()
    os.umask(0o077)
    start = time.monotonic()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["run_directory"])
    output = root / args.arm
    output.mkdir(mode=0o700, exist_ok=False)
    result = {"arm": args.arm, "source_sha256": sha(__file__), "plan_sha256": sha(args.plan),
              "host": os.uname().nodename, "uid": os.getuid(),
              "affinity": sorted(os.sched_getaffinity(0)), "python_executable": sys.executable}
    code = 1
    try:
        result["bindings_before"] = check_bindings(plan)
        result["flags_before"] = observed_flags()
        assert result["flags_before"]["CUDA_VISIBLE_DEVICES"] == ""
        assert result["flags_before"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert result["flags_before"]["removed_environment_variables_absent"]
        assert resource.getrlimit(resource.RLIMIT_AS) == (plan["address_space_cap_bytes"],) * 2
        result["measurements"] = score(plan, root) if args.arm == "score" else globals()[args.arm](plan, output)
        result["status"] = "completed"
        code = 2 if args.arm == "score" and not result["measurements"]["all_gates_pass"] else 0
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        result["science_returncode"] = code
        result["flags_after"] = observed_flags()
        result["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["worker_including_preflight_seconds"] = time.monotonic() - start
        try:
            result["bindings_after"] = check_bindings(plan)
            assert not result["flags_after"]["native_GEMS_binding_imported"]
            assert not result["flags_after"]["samseg_imported"]
            assert not result["flags_after"].get("cuda_initialized", False)
            result["after_binding_and_scope_gate"] = True
        except Exception as error:
            result["after_binding_and_scope_gate"] = False
            result["after_gate_error"] = type(error).__name__ + ": " + str(error)
            if code == 0:
                code = 3
        result["final_returncode"] = code
        write_json(output / ("report.private.json" if code == 0 else "failure.private.json"), result)
    return code


if __name__ == "__main__":
    sys.exit(main())
