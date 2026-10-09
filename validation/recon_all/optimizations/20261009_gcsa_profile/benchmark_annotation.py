"""冻结真实GCSA输入，比较完整三图谱并拆分Gibbs初始化和有序重分类。

显式source/subject/assets/output/device/threads/mode；不调用外部软件、
不修改被试输入，使用nibabel读取注释。完整API计时包含准备/计算/写出与
目标GPU同步；进程墙钟另包含导入和哈希。参数及失败行为见同目录说明。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time


def digest(path: Path) -> str:
    """返回单个明确文件的SHA-256，不搜索其他目录或读取许可证。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_benchmark(*, source_directory: Path, subject_directory: Path,
                  assets_directory: Path, output_directory: Path, hemi: str,
                  device: str, threads: int, mode: str,
                  candidate_module: Path | None, gibbs_backend: str = "python",
                  candidate_reclassify_module: Path | None = None,
                  candidate_gibbs_module: Path | None = None) -> dict:
    """执行同半球三图谱，返回含SHA、完整输出、拆分秒数和采样范围的字典。

    所有路径具名；source目录须含src/fnit，output须不存在。hemi=lh/rh，
    device显式cuda:N或cpu，threads正整数；mode=control/guard，guard必须
    提供候选gcsa_label_python文件。表面surface RAS/mm，注释int32打包RGB
    及v2颜色表，没有空间变换。失败写partial JSON并继续抛异常。
    """
    started = time.perf_counter()
    if threads < 1 or hemi not in ("lh", "rh") or mode not in ("control", "guard"):
        raise ValueError("invalid thread/hemi/mode")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if mode == "guard" and candidate_module is None:
        raise ValueError("guard mode requires candidate_module")
    output_directory.mkdir(parents=True)
    sys.path.insert(0, str(source_directory / "src"))
    import numpy as np
    import nibabel.freesurfer.io as fsio
    import torch
    from fnit.recon_all.profiling import (ProcessTreeDeviceSampler,
                                         autocast_state, configure_cuda_allocator)
    from fnit.recon_all import gcsa_label_python as module
    baseline_module = Path(module.__file__)
    overlay = {}
    def load_overlay(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[name] = candidate
        spec.loader.exec_module(candidate)
        overlay[str(path)] = digest(path)
        return candidate
    numba_module = None
    if candidate_gibbs_module:
        numba_module = load_overlay("fnit.recon_all.gcsa_gibbs_numba", candidate_gibbs_module)
    if candidate_reclassify_module:
        load_overlay("fnit.recon_all.gcsa_reclassify", candidate_reclassify_module)
    if mode == "guard":
        spec = importlib.util.spec_from_file_location(
            "fnit.recon_all.gcsa_label_python_guard", candidate_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    allocator = configure_cuda_allocator(device, "disabled")
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
    report = {"status": "running", "mode": mode, "hemi": hemi,
        "subject": str(subject_directory), "source_directory": str(source_directory),
        "gibbs_backend": gibbs_backend, "overlay_sha256": overlay,
        "device": device, "threads": threads, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host": __import__('socket').gethostname(), "torch_version": torch.__version__,
        "cuda_allocator": allocator, "precision": {
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_autocast": autocast_state("cuda")},
        "timing_scope": "full annotation API including read, compute, write and target synchronization; process includes imports/hashes",
        "ordered_reclassification": "existing seed1234, vertex permutation, strict greater-than and asynchronous updates unchanged",
        "baseline_module_sha256": digest(baseline_module),
        "candidate_module_sha256": digest(candidate_module) if candidate_module else None,
        "input_sha256": {}, "atlases": {}}
    input_paths = [subject_directory / "surf" / f"{hemi}.{name}" for name in ("smoothwm", "sphere.reg")]
    input_paths += [subject_directory / "mri/aseg.presurf.mgz",
                    subject_directory / "label" / f"{hemi}.cortex.label",
                    assets_directory / "lib/bem/ic4.tri", assets_directory / "lib/bem/ic7.tri"]
    atlases = [(name, assets_directory / "average" /
               f"{hemi}.{prefix}.atlas.acfb40.noaparc.i12.2016-08-02.gcs")
               for name, prefix in (("aparc", "DKaparc"), ("aparc.a2009s", "CDaparc"),
                                    ("aparc.DKTatlas", "DKTaparc"))]
    input_paths += [path for _, path in atlases]
    report["input_sha256"] = {str(path): digest(path) for path in input_paths}
    report["source_sha256"] = {str(path.relative_to(source_directory)): digest(path)
        for path in sorted((source_directory / "src/fnit/recon_all").glob("gcsa*.py"))}
    def save():
        (output_directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    sampler = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid(), interval=.5)
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            sampler.sample_if_due()
            stop.wait(.1)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    base_model, base_reclassify = module.GibbsModel, module.reclassify_gibbs
    profile = {}
    if numba_module is not None:
        base_pack, base_sweep = numba_module.pack_model, numba_module.ordered_sweep
        def timed_pack(*args, **kwargs):
            tick = time.perf_counter(); value = base_pack(*args, **kwargs)
            profile["numba_pack_seconds"] = time.perf_counter() - tick
            return value
        def timed_sweep(*args, **kwargs):
            tick = time.perf_counter(); value = base_sweep(*args, **kwargs)
            profile["numba_sweep_seconds_including_jit_cache_load"] = profile.get(
                "numba_sweep_seconds_including_jit_cache_load", 0) + time.perf_counter() - tick
            return value
        numba_module.pack_model, numba_module.ordered_sweep = timed_pack, timed_sweep
    class TimedModel(base_model):
        def __init__(self, *args, **kwargs):
            tick = time.perf_counter()
            super().__init__(*args, **kwargs)
            profile["model_init_seconds"] = time.perf_counter() - tick
    def timed_reclassify(*args, **kwargs):
        tick = time.perf_counter()
        value = base_reclassify(*args, **kwargs)
        profile["reclassification_seconds"] = time.perf_counter() - tick
        return value
    module.GibbsModel, module.reclassify_gibbs = TimedModel, timed_reclassify
    try:
        tick = time.perf_counter()
        cache = module.GCSAFeatureCache(subject=subject_directory, hemi=hemi, device=device)
        if selected.type == "cuda": torch.cuda.synchronize(selected)
        report["shared_geometry_seconds_including_sync"] = time.perf_counter() - tick
        report["geometry_sha256"] = {"feature": hashlib.sha256(cache.feature.tobytes()).hexdigest(),
            "principal": hashlib.sha256(cache.principal.tobytes()).hexdigest()}
        for name, atlas in atlases:
            tick = time.perf_counter()
            output = output_directory / f"{hemi}.{name}.annot"
            profile.clear()
            backend = {} if mode == "control" else {"gibbs_backend": gibbs_backend}
            result = module.label_surface(subject=subject_directory, hemi=hemi,
                atlas_file=atlas, ico4_file=assets_directory / "lib/bem/ic4.tri",
                ico7_file=assets_directory / "lib/bem/ic7.tri", output_file=output,
                device=device, prepared=cache, **backend)
            if selected.type == "cuda": torch.cuda.synchronize(selected)
            wall = time.perf_counter() - tick
            labels, colors, names = fsio.read_annot(str(output), orig_ids=True)
            report["atlases"][name] = {"result": result, **profile,
                "full_api_wall_seconds_including_sync": wall,
                "output_sha256": digest(output), "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                "color_table_sha256": hashlib.sha256(colors.tobytes()).hexdigest(),
                "color_names": [name.decode() for name in names], "vertices": len(labels)}
            save()
        report["status"] = "complete"
        report["inputs_unchanged"] = all(digest(path) == report["input_sha256"][str(path)] for path in input_paths)
        if not report["inputs_unchanged"]: raise RuntimeError("benchmark input changed during run")
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        module.GibbsModel, module.reclassify_gibbs = base_model, base_reclassify
        if numba_module is not None:
            numba_module.pack_model, numba_module.ordered_sweep = base_pack, base_sweep
        stop.set(); thread.join(timeout=10)
        report["process_memory"] = sampler.report()
        report["process_wall_seconds_including_import_hash_write"] = time.perf_counter() - started
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_directory", "subject_directory", "assets_directory", "output_directory"):
        parser.add_argument("--" + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--mode", choices=("control", "guard"), required=True)
    parser.add_argument("--candidate-module", type=Path)
    parser.add_argument("--gibbs-backend", choices=("python", "numba"), default="python")
    parser.add_argument("--candidate-reclassify-module", type=Path)
    parser.add_argument("--candidate-gibbs-module", type=Path)
    args = parser.parse_args()
    run_benchmark(**vars(args))
