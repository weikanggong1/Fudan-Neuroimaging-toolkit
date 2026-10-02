"""真实固定输入下比较 UKBConnectome_pipeline 的多 atlas 复用。

只测试 atlas 阶段，不是原始 DWI 到 connectome 的全流程 benchmark。
调用者提供真实 DWI/FOD/FA、recon-all、TCK、逐轨指标与 DWI→T1
世界坐标变换。DWI 重建、追踪、SIFT2 和逐轨 FA 阶段被这些真实中间
结果替换；皮层图谱构建、Tian SynthMorph/FNIRT、标签重采样、合并、
DWI 网格映射与四矩阵仍执行正式实现。默认不使用 Workbench。

示例变量均应指向同一真实受试者：
python tools/benchmark_connectome_atlas_reuse.py \
  --baseline-root "$BASELINE_CHECKOUT" --dwi "$CORRECTED_DWI" \
  --bvals "$BVALUES" --bvecs "$ROTATED_BVECTORS" \
  --brain-mask "$DWI_BRAIN_MASK" --freesurfer-subject-dir "$RECON_ALL_SUBJECT" \
  --fod "$NORMALIZED_WM_FOD" --fa "$FA_IMAGE" --tracks "$FIXED_TCK" \
  --track-metrics "$TRACK_METRICS_NPZ" --transform "$DWI_TO_T1_WORLD_TXT" \
  --atlas-templates-dir "$ATLAS_TEMPLATES" --fsaverage-dir "$FSAVERAGE" \
  --mni-template "$MNI_T1" --mni-to-t1-warp "$FIXED_SYNTHMORPH_WARP" \
  --repeats 2 --device cuda:0 --output "$ATLAS_REUSE_REPORT_JSON"

track-metrics NPZ 至少含同一 TCK 顺序的 weights、lengths；可含
mean_fa，缺失时在计时前用真实 FA 和 TCK 计算。报告只保留哈希、
调用次数、计时、显存和一致性指标，不保存受试者影像或矩阵内容。
--mni-to-t1-warp 固定已生成的 SynthMorph MNI→T1 变换，只计时真实
atlas apply；此时不加载网络或权重，也不包含 registration。省略该
选项时需 --synthmorph-weights，首次 Tian 调用包含真实配准。
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import NAMES, _load, _sha256, _sync
import fnit.connectome.pipeline as candidate


DEFAULT_ATLASES = (
    "aparc+tian-s1", "aparc.a2009s+tian-s1", "schaefer200+tian-s1",
)
SUPPORTED_ATLASES = (
    "fs-aparc", "fs-aparc-a2009s", *candidate.NATIVE_TIAN_ATLASES,
    *candidate.SCHAEFER_TIAN_ATLASES,
)
SHARED_COMPONENTS = (
    "atlas_builder.py", "atlas_surface.py", "atlas_tian.py", "anatomy.py",
    "assignment.py", "freesurfer_subject.py",
)


def _baseline_module(root):
    path = root / "src/fnit/connectome/pipeline.py"
    name = "fnit.connectome._atlas_reuse_baseline_pipeline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_hashes(baseline_root):
    """Require the shared atlas operators to match the baseline checkout."""
    current_root = Path(candidate.__file__).resolve().parent
    result = {
        "baseline_pipeline": _sha256(baseline_root / "src/fnit/connectome/pipeline.py"),
        "candidate_pipeline": _sha256(Path(candidate.__file__)),
        "benchmark_tool": _sha256(Path(__file__)),
        "shared_components": {},
    }
    for name in SHARED_COMPONENTS:
        old = _sha256(baseline_root / "src/fnit/connectome" / name)
        new = _sha256(current_root / name)
        if old != new:
            raise ValueError(f"shared atlas component differs from baseline: {name}")
        result["shared_components"][name] = new
    return result


def _input_hashes(args, atlases):
    paths = {
        "dwi": args.dwi, "bvals": args.bvals, "bvecs": args.bvecs,
        "brain_mask": args.brain_mask, "fod": args.fod, "fa": args.fa,
        "tracks": args.tracks, "track_metrics": args.track_metrics,
        "dwi_to_t1_world": args.transform,
    }
    relative = {"mri/brain.mgz", "mri/aparc+aseg.mgz"}
    surface_atlases = any(name in (*candidate.NATIVE_TIAN_ATLASES,
                                  *candidate.SCHAEFER_TIAN_ATLASES) for name in atlases)
    if surface_atlases:
        relative.add("mri/ribbon.mgz")
        for hemisphere in ("lh", "rh"):
            relative.update(f"surf/{hemisphere}.{kind}" for kind in ("white", "pial"))
    for name in atlases:
        if name in candidate.NATIVE_TIAN_ATLASES:
            annotation = candidate.NATIVE_TIAN_ATLASES[name]
            relative.update(f"label/{hemi}.{annotation}.annot" for hemi in ("lh", "rh"))
        if name in candidate.SCHAEFER_TIAN_ATLASES:
            parcels, _ = candidate.SCHAEFER_TIAN_ATLASES[name]
            for hemisphere in ("lh", "rh"):
                relative.add(f"surf/{hemisphere}.sphere.reg")
                annot = f"{hemisphere}.Schaefer2018_{parcels}Parcels_7Networks_order.annot"
                paths[f"template/{annot}"] = args.atlas_templates_dir / annot
                paths[f"fsaverage/surf/{hemisphere}.sphere.reg"] = args.fsaverage_dir / f"surf/{hemisphere}.sphere.reg"
        if name == "fs-aparc-a2009s":
            relative.add("mri/aparc.a2009s+aseg.mgz")
            relative.update(f"label/{hemi}.aparc.a2009s.annot" for hemi in ("lh", "rh"))
    paths.update({f"subject/{name}": args.freesurfer_subject_dir / name for name in sorted(relative)})
    scales = set()
    for name in atlases:
        if name in candidate.NATIVE_TIAN_ATLASES:
            scales.add(1)
        elif name in candidate.SCHAEFER_TIAN_ATLASES:
            scales.add(candidate.SCHAEFER_TIAN_ATLASES[name][1])
    for scale in sorted(scales):
        for suffix in (".nii.gz", "_label.txt"):
            name = f"Tian_Subcortex_S{scale}_3T{suffix}"
            paths[f"template/{name}"] = args.atlas_templates_dir / name
    if args.mni_template is not None:
        paths["mni_template"] = args.mni_template
    if args.mni_to_t1_warp is not None:
        paths["mni_to_t1_warp"] = args.mni_to_t1_warp
    if args.tian_fnirt_coeff is not None:
        paths["tian_fnirt_coeff"] = args.tian_fnirt_coeff
    if args.synthmorph_weights is not None:
        if args.synthmorph_weights.is_dir():
            weights = [path for path in sorted(args.synthmorph_weights.rglob("*"))
                       if path.is_file() and path.suffix.lower() in (".pt", ".pth", ".h5", ".hdf5", ".npz")]
            if not weights:
                raise ValueError("SynthMorph weight directory contains no recognized model files")
            for path in weights:
                paths[f"weights/{path.relative_to(args.synthmorph_weights)}"] = path
        else:
            paths["synthmorph_weights"] = args.synthmorph_weights
    return {name: _sha256(path) for name, path in paths.items()}


def _snapshot(result):
    return {name: {
        "atlas": value.atlas.cpu().numpy().copy(),
        "affine": value.atlas_affine.cpu().numpy().copy(),
        "region_labels": value.region_labels,
        "nodes": None if value.nodes is None else [asdict(node) for node in value.nodes],
        "matrices": {key: value.matrices[key].cpu().numpy().copy() for key in NAMES},
    } for name, value in result.atlas_results.items()}


def _array_comparison(left, right):
    shape_equal = left.shape == right.shape
    dtype_equal = left.dtype == right.dtype
    exact = shape_equal and dtype_equal and np.array_equal(left, right)
    return {
        "shape_equal": shape_equal, "dtype_equal": dtype_equal,
        "exact": exact,
        "bitwise_equal": shape_equal and dtype_equal and left.tobytes() == right.tobytes(),
        "different_values": (int(np.count_nonzero(left != right)) if shape_equal else None),
        "max_abs_error": (float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                          if shape_equal and left.size else None),
        "baseline_value_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "candidate_value_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
    }


def _compare(left, right):
    if tuple(left) != tuple(right):
        raise ValueError("atlas names/order changed")
    output = {}
    for name in left:
        a, b = left[name], right[name]
        record = {
            "atlas": _array_comparison(a["atlas"], b["atlas"]),
            "affine": _array_comparison(a["affine"], b["affine"]),
            "nodes_exact": a["nodes"] == b["nodes"],
            "region_labels_exact": a["region_labels"] == b["region_labels"],
            "matrices": {key: _array_comparison(a["matrices"][key], b["matrices"][key])
                         for key in NAMES},
        }
        record["all_exact"] = (record["atlas"]["exact"] and record["affine"]["exact"]
                               and record["nodes_exact"] and record["region_labels_exact"]
                               and all(value["exact"] for value in record["matrices"].values()))
        output[name] = record
    return output


def _stage_key(name, kwargs):
    if name == "native_annotation_to_t1":
        return "cortical_native/" + kwargs["annotation"]
    if name == "schaefer_to_t1":
        return "cortical_schaefer/" + Path(kwargs["left_annot"]).name
    if name in ("synthmorph_tian_to_t1", "fnirt_tian_to_t1"):
        scale = Path(kwargs["tian_mni"]).name.rsplit("_S", 1)[1].split("_", 1)[0]
        return name + "/s" + scale
    return name


def _run(module, options, fixed, device):
    durations = defaultdict(list)
    originals = {name: getattr(module, name) for name in (
        "native_annotation_to_t1", "schaefer_to_t1", "synthmorph_tian_to_t1",
        "fnirt_tian_to_t1", "combine_cortical_tian", "resample_labels_nearest",
        "build_connectomes", "fs_aparc_atlas", "fs_aparc_a2009s_atlas",
    )}
    with ExitStack() as stack:
        # Reuse caller-supplied real intermediate data; do not benchmark the
        # DWI reconstruction, registration, tractography or SIFT2 stages.
        replacements = {
            "_image": fixed["image"], "_gradients": lambda *args: fixed["gradients"],
            "_scalar_on_grid": fixed["scalar"], "mean_bzero": lambda **kwargs: fixed["mean_b0"],
            "freesurfer_five_tissue": lambda *args: fixed["five"],
            "gmwmi_from_five_tissue": lambda *args: fixed["gmwmi"],
            "estimate_mrtrix_dhollander": lambda *args: (fixed["shells"], None, None, None, None),
            "fit_mrtrix_msmt_csd": lambda *args: (fixed["fod"], None, None),
            "normalise_mrtrix_three_tissue": lambda *args: SimpleNamespace(wm=fixed["fod"]),
            "probabilistic_tractography": lambda *args, **kwargs: module.Tractogram(
                paths=fixed["paths"], endpoints=fixed["endpoints"], lengths_mm=fixed["lengths"],
                mean_fa=None, seeds_attempted=len(fixed["paths"]),
                accepted_seeds=fixed["endpoints"][:, 0],
            ),
            "estimate_sift2_weights": lambda *args, **kwargs: fixed["weights"],
            "sample_streamline_mean_precise": lambda *args: fixed["mean_fa"],
        }
        for name, function in replacements.items():
            stack.enter_context(patch.object(module, name, function))
        for name, function in originals.items():
            def measured(*args, _name=name, _function=function, **kwargs):
                if _name == "synthmorph_tian_to_t1" and fixed["mni_to_t1_warp"] is not None:
                    # Execute the original atlas application on one caller-
                    # provided real warp; exclude already completed registration.
                    kwargs["transform"] = fixed["mni_to_t1_warp"]
                _sync(device)
                started = time.perf_counter()
                result = _function(*args, **kwargs)
                _sync(device)
                durations[_stage_key(_name, kwargs)].append(time.perf_counter() - started)
                return result
            stack.enter_context(patch.object(module, name, measured))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        started = time.perf_counter()
        result = module.UKBConnectome_pipeline(device=str(device))(**options)
        _sync(device)
        seconds = time.perf_counter() - started
        timing = {
            "atlas_tail_call_seconds": seconds,
            "timed_stage_sum_seconds": sum(sum(values) for values in durations.values()),
            "stages": {name: {"calls": len(values), "seconds": values,
                              "total_seconds": sum(values)} for name, values in durations.items()},
            "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
            "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        }
        output = _snapshot(result)
        del result
        return output, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "dwi", "bvals", "bvecs", "brain-mask",
                 "freesurfer-subject-dir", "fod", "fa", "tracks", "track-metrics",
                 "atlas-templates-dir", "fsaverage-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--transform", "--dwi-to-t1-world", type=Path, required=True,
                        help="4×4 DWI→T1 RAS-mm transform, whitespace text, CSV or .npy")
    registration = parser.add_mutually_exclusive_group(required=True)
    registration.add_argument("--mni-template", type=Path)
    registration.add_argument("--tian-fnirt-coeff", type=Path)
    parser.add_argument("--mni-to-t1-warp", type=Path,
                        help="已生成的 SynthMorph RAS displacement；固定配准，只计时 atlas apply")
    parser.add_argument("--synthmorph-weights", type=Path)
    parser.add_argument("--atlas", action="append", choices=SUPPORTED_ATLASES,
                        help="可重复；默认 aparc、a2009s、Schaefer200 各配 Tian S1")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be >=2 to alternate baseline/candidate order")
    if args.tian_fnirt_coeff is not None and args.synthmorph_weights is not None:
        parser.error("--synthmorph-weights cannot accompany --tian-fnirt-coeff")
    if args.mni_to_t1_warp is not None and args.mni_template is None:
        parser.error("--mni-to-t1-warp requires --mni-template and cannot accompany FNIRT")
    if args.mni_template is not None and args.synthmorph_weights is None and args.mni_to_t1_warp is None:
        parser.error("provide --synthmorph-weights to hash the exact model assets")
    atlases = tuple(args.atlas or DEFAULT_ATLASES)
    source_hashes = _source_hashes(args.baseline_root)
    input_hashes = _input_hashes(args, atlases)
    baseline = _baseline_module(args.baseline_root)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            17e9 / torch.cuda.get_device_properties(device).total_memory, device,
        )
    started = time.perf_counter()
    dwi, affine = candidate._image(args.dwi, device)
    reference = nib.load(str(args.dwi))
    subject = candidate.FreeSurferSubject(args.freesurfer_subject_dir)
    seg, seg_affine = candidate._image(subject.aparc_aseg, device)
    fod, fod_affine = _load(args.fod, device)
    if fod.ndim != 4 or fod.shape[-1] != 45 or fod.shape[:3] != dwi.shape[:3] or not torch.allclose(
        affine, fod_affine, atol=1e-3, rtol=0,
    ):
        raise ValueError("normalized FOD must be [X,Y,Z,45] on corrected DWI voxel grid")
    mask = candidate._scalar_on_grid(args.brain_mask, reference, device, binary=True)
    fa = candidate._scalar_on_grid(args.fa, reference, device)
    gradients = candidate._gradients(args.bvals, args.bvecs, dwi.shape[-1], affine, device)
    gradient_table = torch.cat((gradients[1].double(), gradients[0].double()[:, None]), dim=1)
    _, _, shells, _ = candidate.mrtrix_shell_centres(gradient_table)
    five = candidate.freesurfer_five_tissue(seg)
    gmwmi = candidate.gmwmi_from_five_tissue(five)
    mean_b0 = candidate.mean_bzero(dwi=dwi, bvalues=gradients[0])
    tractogram = nib.streamlines.load(str(args.tracks)).tractogram
    tractogram.to_world()
    paths = tuple(torch.as_tensor(np.asarray(path).copy(), device=device, dtype=torch.float32)
                  for path in tractogram.streamlines)
    if not paths or any(len(path) < 2 for path in paths):
        raise ValueError("TCK must contain nonempty real paths with >=2 points")
    endpoints = torch.stack([path[[0, -1]] for path in paths])
    with np.load(args.track_metrics, allow_pickle=False) as data:
        weights = torch.as_tensor(data["weights"].copy(), device=device, dtype=torch.float64)
        lengths = torch.as_tensor(data["lengths"].copy(), device=device, dtype=torch.float32)
        mean_fa = (torch.as_tensor(data["mean_fa"].copy(), device=device, dtype=torch.float32)
                   if "mean_fa" in data else candidate.sample_streamline_mean_precise(paths, fa, affine))
    if any(value.shape != (len(paths),) or not bool(torch.isfinite(value).all())
           for value in (weights, lengths, mean_fa)):
        raise ValueError("track metrics must contain one finite value per TCK streamline")
    transform = (np.load(args.transform, allow_pickle=False) if args.transform.suffix == ".npy"
                 else np.loadtxt(args.transform, delimiter="," if args.transform.suffix == ".csv" else None))
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4×4 DWI→T1 world matrix")
    mni_to_t1_warp = None
    if args.mni_to_t1_warp is not None:
        from fnit._transforms import load_dense_warp, same_geometry
        mni = nib.load(str(args.mni_template))
        mni_to_t1_warp = load_dense_warp(args.mni_to_t1_warp, source=mni)
        if not same_geometry(nib.load(str(subject.brain)), mni_to_t1_warp.target):
            raise ValueError("fixed MNI→T1 warp target must match recon-all brain geometry")
    images = {str(args.dwi): (dwi, affine), str(subject.aparc_aseg): (seg, seg_affine)}
    original_image = candidate._image
    original_scalar = candidate._scalar_on_grid
    def image(path, current_device):
        key = str(path)
        return images[key] if key in images else original_image(path, current_device)
    def scalar(path, current_reference, current_device, *, binary=False):
        if str(path) == str(args.brain_mask) and binary:
            return mask
        if str(path) == str(args.fa) and not binary:
            return fa
        return original_scalar(path, current_reference, current_device, binary=binary)
    fixed = dict(image=image, scalar=scalar, gradients=gradients, mean_b0=mean_b0,
                 five=five, gmwmi=gmwmi, shells=shells, fod=fod, paths=paths,
                 endpoints=endpoints, weights=weights, lengths=lengths, mean_fa=mean_fa,
                 mni_to_t1_warp=mni_to_t1_warp)
    options = dict(
        dwi=args.dwi, bvals=args.bvals, bvecs=args.bvecs,
        freesurfer_subject_dir=args.freesurfer_subject_dir, atlas=atlases,
        atlas_templates_dir=args.atlas_templates_dir, fsaverage_dir=args.fsaverage_dir,
        mni_template=args.mni_template, synthmorph_weights=args.synthmorph_weights,
        tian_fnirt_coeff=args.tian_fnirt_coeff, dwi_to_t1_world=transform,
        brain_mask=args.brain_mask, response_mask=args.brain_mask, fod_mask=args.brain_mask,
        normalise_mask=args.brain_mask, fa_map=args.fa, n_seeds=len(paths),
        shell_bvals=shells.cpu().tolist(),
    )
    _sync(device)
    load_seconds = time.perf_counter() - started
    report = {
        "benchmark_scope": "fixed-real-input atlas-stage benchmark, not end-to-end reconstruction",
        "fixed_front_stages": ["DWI response/FOD normalization", "DWI→T1 registration",
                               "tractography", "SIFT2", "per-track FA"],
        "real_computed_stages": ["cortical atlas", ("Tian fixed-warp apply" if mni_to_t1_warp is not None else "Tian registration/apply"),
                                 "cortical/subcortical combine", "DWI labels", "four matrices"],
        "registration_included": mni_to_t1_warp is None,
        "timing_note": "Preloading, input hashes and output checks excluded; synchronized stage profiling included. Fixed-warp mode excludes registration. Atlas apply uses the existing FNIT backend, including its CPU operations.",
        "baseline_shared_components": "same source-verified atlas operators; only pipeline orchestration differs",
        "input_sha256": input_hashes, "source_sha256": source_hashes,
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "atlases": list(atlases), "streamlines": len(paths), "load_seconds": load_seconds,
        "repeats": [],
    }
    reference_snapshot = None
    for repeat in range(args.repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        snapshots, timing = {}, {}
        for name in order:
            module = baseline if name == "baseline" else candidate
            snapshots[name], timing[name] = _run(module, options, fixed, device)
            print(f"repeat {repeat + 1} {name}: {timing[name]['atlas_tail_call_seconds']:.3f}s", flush=True)
        parity = _compare(snapshots["baseline"], snapshots["candidate"])
        repeat_parity = (_compare(reference_snapshot, snapshots["baseline"])
                         if reference_snapshot is not None else None)
        if reference_snapshot is None:
            reference_snapshot = snapshots["baseline"]
        report["repeats"].append({"order": list(order), "timing": timing,
                                  "parity": parity, "baseline_repeat_parity": repeat_parity})
    report["summary"] = {}
    for name in ("baseline", "candidate"):
        seconds = [item["timing"][name]["atlas_tail_call_seconds"] for item in report["repeats"]]
        report["summary"][name] = {"seconds": seconds, "median_seconds": statistics.median(seconds)}
    report["summary"]["speedup"] = (report["summary"]["baseline"]["median_seconds"]
                                         / report["summary"]["candidate"]["median_seconds"])
    report["all_exact"] = all(
        all(item["all_exact"] for item in repeat["parity"].values())
        and (repeat["baseline_repeat_parity"] is None
             or all(item["all_exact"] for item in repeat["baseline_repeat_parity"].values()))
        for repeat in report["repeats"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not report["all_exact"]:
        raise SystemExit("Atlas/nodes/matrices changed; inspect the written report")
    print("All atlas labels, affines, nodes and four matrices are exact.", flush=True)


if __name__ == "__main__":
    main()
