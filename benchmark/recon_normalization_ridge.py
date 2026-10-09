"""两例自产输入的 medial ridge 分段剖析；不更改传播/筛选算法。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import nibabel as nib
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="只读冻结源码src目录")
    parser.add_argument("--mri-dir", type=Path, required=True, help="自产norm/brainmask/aseg.presurf目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的新JSON目录")
    parser.add_argument("--code-commit", required=True, help="实际依赖源码提交；SHA另记")
    parser.add_argument("--threads", type=int, default=4, help="Numba/BLAS总线程预算")
    parser.add_argument("--repetitions", type=int, default=3, help="第一次含JIT；随后同进程暖调用")
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1:
        raise ValueError("positive thread/repetition count required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_dir.resolve()))
    import numba
    numba.set_num_threads(args.threads)
    ridge_module = importlib.import_module("fnit.recon_all.normalization.normalize_aseg_ridge")
    source_module = importlib.import_module("fnit.recon_all.normalization.normalize_aseg_source")
    input_paths = {name: args.mri_dir / name for name in ("norm.mgz", "aseg.presurf.mgz", "brainmask.mgz")}
    input_sha = {name: sha(path) for name, path in input_paths.items()}
    tick = time.perf_counter()
    images = {name: nib.load(str(path)) for name, path in input_paths.items()}
    if any(image.shape != images["norm.mgz"].shape or not np.array_equal(image.affine, images["norm.mgz"].affine)
           for image in images.values()):
        raise ValueError("all inputs must share the original conform grid")
    arrays = {name: np.asarray(image.dataobj) for name, image in images.items()}
    masked, _ = source_module.prepare_aseg_source(arrays["norm.mgz"], arrays["brainmask.mgz"], arrays["aseg.presurf.mgz"])
    loading_seconds = time.perf_counter() - tick
    rows = []
    for repetition in range(args.repetitions):
        calls, captured, patches = [], {}, []
        for module, name in ((ridge_module, "_march_pass"), (ridge_module, "_nonmax"), (source_module, "_filter_ridge")):
            original = getattr(module, name)
            def wrapped(*values, _original=original, _name=name, **keywords):
                tick, cpu = time.perf_counter(), time.process_time()
                result = _original(*values, **keywords)
                row = {"name": _name, "seconds": time.perf_counter()-tick, "cpu_seconds": time.process_time()-cpu}
                if _name == "_march_pass":
                    row.update(sign=int(values[2]), alive=int(result[0]), processed=int(result[1]))
                calls.append(row)
                if _name == "_nonmax":
                    captured["distance"] = values[0]
                return result
            setattr(module, name, wrapped)
            patches.append((module, name, original))
        tick = time.perf_counter()
        try:
            ridge, details = ridge_module.medial_ridge(arrays["aseg.presurf.mgz"])
            medial_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            controls, removed, peak = source_module.filter_aseg_ridge(masked, ridge)
            filtering_seconds = time.perf_counter()-tick
        finally:
            for module, name, original in reversed(patches):
                setattr(module, name, original)
        rows.append({"repetition": repetition, "medial_seconds": medial_seconds,
                     "filtering_seconds": filtering_seconds, "calls": calls, "details": details,
                     "distance_sha256": data_sha(captured["distance"]), "ridge_sha256": data_sha(ridge),
                     "control_sha256": data_sha(controls), "removed_sha256": data_sha(removed),
                     "wm_peak": int(peak), "removed_voxels": int(np.count_nonzero(removed))})
    binding = {str(Path(module.__file__).relative_to(args.source_dir)): sha(module.__file__)
               for module in (ridge_module, source_module)}
    result = {"scope": "actual same-input existing CPU signed marching/nonmax/ordered filtering profile; not full normalization or raw T1",
              "code_commit": args.code_commit, "source_sha256": binding, "script_sha256": sha(__file__),
              "input_sha256": input_sha, "input_unchanged": {name: sha(path)==input_sha[name] for name,path in input_paths.items()},
              "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
              "affinity": sorted(os.sched_getaffinity(0)), "numba_threads": numba.get_num_threads(),
              "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
              "loading_seconds": loading_seconds, "calls": rows,
              "timing_scope": "medial includes mask/distance allocation/both ordered passes/nonmax; filtering includes peak histogram and dynamic scan; first call includes cache load or JIT; hashes excluded"}
    (args.output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.output_dir), "calls": [{"medial_seconds": r["medial_seconds"], "filtering_seconds": r["filtering_seconds"], "segments": r["calls"]} for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
