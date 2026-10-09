"""限域ridge同输入回归或复用现有完整API测量器；不复制计时/显存实现。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import maximum_filter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_overlay(folder, names):
    namespace = importlib.import_module("fnit.recon_all.normalization")
    loaded = {}
    for name in names:
        path = folder / (name + ".py")
        full_name = "fnit.recon_all.normalization." + name
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        setattr(namespace, name, module)
        loaded[name] = module
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage", "api"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True, help="只读原依赖src")
    parser.add_argument("--ridge-overlay", type=Path, help="候选明确两ridge源文件；省略为原完整背景传播")
    parser.add_argument("--initial-overlay", type=Path, help="已验证v7初始偏场明确两文件")
    parser.add_argument("--driver", type=Path, help="api模式复用既有完整API计时/显存驱动")
    parser.add_argument("--mri-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    args, extra = parser.parse_known_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    reference = importlib.import_module("fnit.recon_all.normalization.normalize_aseg_ridge")
    source = importlib.import_module("fnit.recon_all.normalization.normalize_aseg_source")
    initial = load_overlay(args.initial_overlay, ("normalize_aseg_source", "aseg_pipeline")) if args.initial_overlay else {}
    candidate = load_overlay(args.ridge_overlay, ("normalize_aseg_ridge", "normalize_ridge_local")) if args.ridge_overlay else {}
    if args.mode == "api":
        if args.driver is None:
            raise ValueError("api mode needs the existing full API driver")
        pipeline = importlib.import_module("fnit.recon_all.normalization.aseg_pipeline")
        if candidate:
            pipeline.medial_ridge = candidate["normalize_ridge_local"].medial_ridge_local
        spec = importlib.util.spec_from_file_location("fnit_ridge_full_api_driver", args.driver)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        # 保留现有计时器、目标设备同步、采样与完整文件比较，独立固定入口。
        sys.argv = [str(args.driver), "--source-dir", str(args.source_dir), "--mri-dir", str(args.mri_dir),
                    "--output-dir", str(args.output_dir), "--code-commit", args.code_commit,
                    "--threads", str(args.threads), *extra]
        driver.main()
        report_path = args.output_dir / "report.json"
        report = json.loads(report_path.read_text())
        report.update(ridge_backend="local-consumer" if candidate else "full",
                      ridge_wrapper_sha256=sha(__file__), delegated_driver_sha256=sha(args.driver))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        return
    if extra or not candidate or args.threads < 1 or args.repetitions < 1:
        raise ValueError("stage comparison needs candidate files and positive counts, no extra options")
    import numba
    numba.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {name: args.mri_dir / name for name in ("norm.mgz", "aseg.presurf.mgz", "brainmask.mgz")}
    hashes = {name: sha(path) for name, path in paths.items()}
    images = {name: nib.load(str(path)) for name,path in paths.items()}
    if any(image.shape != images["norm.mgz"].shape or not np.array_equal(image.affine, images["norm.mgz"].affine)
           for image in images.values()):
        raise ValueError("input grids differ")
    arrays = {name: np.asarray(image.dataobj) for name,image in images.items()}
    masked, _ = source.prepare_aseg_source(arrays["norm.mgz"], arrays["brainmask.mgz"], arrays["aseg.presurf.mgz"])
    mask = (arrays["aseg.presurf.mgz"] == 2) | (arrays["aseg.presurf.mgz"] == 41)
    required = maximum_filter(mask, size=3, mode="nearest") != 0
    rows = []
    for repetition in range(args.repetitions):
        tick = time.perf_counter()
        full, counts = reference.signed_distance(mask)
        full_ridge = reference._nonmax(full)
        full_controls, full_removed, full_peak = source.filter_aseg_ridge(masked, full_ridge)
        full_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        local, details = candidate["normalize_ridge_local"].ridge_required_distance(aseg=arrays["aseg.presurf.mgz"])
        local_ridge = candidate["normalize_aseg_ridge"]._nonmax(local)
        local_controls, local_removed, local_peak = source.filter_aseg_ridge(masked, local_ridge)
        local_seconds = time.perf_counter()-tick
        difference = np.abs(full[required].astype(np.float64)-local[required].astype(np.float64))
        comparisons = {name: {"different": int(np.count_nonzero(lhs!=rhs)), "reference_sha256": array_sha(lhs),
                              "candidate_sha256": array_sha(rhs)}
                       for name,lhs,rhs in (("ridge",full_ridge,local_ridge),("controls",full_controls,local_controls),
                                            ("removed",full_removed,local_removed))}
        row = {"repetition": repetition, "full_seconds": full_seconds, "local_seconds": local_seconds,
               "reference_marching": counts, "candidate_details": details,
               "required_distance_voxels": int(np.count_nonzero(required)),
               "required_distance_different": int(np.count_nonzero(difference)),
               "required_distance_max_abs": float(difference.max(initial=0)),
               "required_distance_p99_abs": float(np.percentile(difference,99)) if difference.size else 0.,
               "reference_peak": int(full_peak), "candidate_peak": int(local_peak), "comparisons": comparisons,
               "far_background_field_scope": "not computed by local candidate, must not be used as generic distance"}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["required_distance_different"] or any(value["different"] for value in comparisons.values()) or full_peak!=local_peak:
            raise AssertionError("candidate introduced a ridge consumer difference")
    bindings = {"reference.normalize_aseg_ridge": sha(reference.__file__),
                "reference.normalize_aseg_source": sha(source.__file__)}
    bindings.update({"candidate."+name: sha(module.__file__) for name,module in candidate.items()})
    result = {"scope":"same-input signed-marching consumer regression; no full distance equivalence or raw T1 timing",
              "code_commit":args.code_commit,"source_sha256":bindings,"script_sha256":sha(__file__),
              "input_sha256":hashes,"input_dtype":{name:str(value.dtype) for name,value in arrays.items()},
              "input_unchanged":{name:sha(path)==hashes[name] for name,path in paths.items()},
              "host":platform.node(),"affinity":sorted(os.sched_getaffinity(0)),"numba_threads":numba.get_num_threads(),
              "numpy":np.__version__,"numba":numba.__version__,"calls":rows,
              "timing_scope":"each complete medial ridge and peak/filter consumer, preloaded arrays; hashes excluded; first JIT included"}
    (args.output_dir/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
