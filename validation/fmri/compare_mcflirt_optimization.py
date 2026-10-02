"""比较一例真实 BIDS volume 的 MCFLIRT 优化前后完整保存结果。

输入参数：--candidate-derivatives 与 --baseline-derivatives 是候选和冻结版本
BIDS derivatives 根目录；--candidate-intermediates 与 --baseline-intermediates
是 benchmark_bids.py 捕获目录，各包含 feat/ 和 aroma/；--mni-mask 是两份
MNI 结果共同使用的三维统计掩膜；--report-out 是匿名 JSON 输出位置；
--source-revision 与 --baseline-source-revision 分别是两次运行的 Git revision。
--candidate-source-root 与 --baseline-source-root 是两次运行的源码目录；
驱动核对最终影像副文件中的全部源码哈希，避免把旧 derivatives 当作冻结结果。

完整读取运动校正、高通、AROMA、boldref clean、MNI clean 五份四维时序，
不截取帧。输出只有误差、时间相关系数、掩膜统计、来源版本和 SHA-256，
不写出原始影像、路径、被试名、矩阵或逐体素数组。原程序不被调用。
逐空间块 float64 指标复用 ../mcflirt/benchmark_optimization.py；这里的计时
仅为结果读取、核对和哈希计算耗时，不能视为配准或完整流程运行耗时。

示例：
python validation/fmri/compare_mcflirt_optimization.py \\
  --candidate-derivatives "$CANDIDATE_DERIVATIVES" \\
  --baseline-derivatives "$FROZEN_DERIVATIVES" \\
  --candidate-intermediates "$CANDIDATE_CAPTURED_INTERMEDIATES" \\
  --baseline-intermediates "$FROZEN_CAPTURED_INTERMEDIATES" \\
  --mni-mask "$MNI_BRAIN_MASK" --report-out "$ANONYMOUS_REPORT" \\
  --source-revision "$CANDIDATE_GIT_REVISION" \\
  --baseline-source-revision "$FROZEN_GIT_REVISION" \\
  --candidate-source-root "$CANDIDATE_SOURCE_ROOT" \\
  --baseline-source-root "$FROZEN_SOURCE_ROOT"

对应功能来源：MCFLIRT 2111.0，
https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc；
Jenkinson et al. NeuroImage 17:825-841 (2002), doi:10.1006/nimg.2002.1132。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import nibabel as nib
import numpy as np


def load_comparison_helpers():
    """按本文件相对位置读取统一指标实现，不调用 FNIT 配准运行时。"""
    path = Path(__file__).resolve().parents[1] / "mcflirt/benchmark_optimization.py"
    spec = importlib.util.spec_from_file_location("fnit_mcflirt_optimization_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot locate the shared MCFLIRT comparison helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def unique_file(root, pattern, *, optional=False):
    """单例结果必须唯一；报错只说明角色，不公开被试文件名。"""
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if optional and not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one matching derivative for {pattern}; found {len(matches)}")
    return matches[0]


def load_mask(path, reference):
    """在给定参考网格读取有限且非空的三维掩膜。"""
    image = nib.load(str(path))
    if image.ndim != 3 or image.shape != reference.shape[:3] or not np.allclose(
            image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("comparison mask must use its BOLD reference grid")
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("comparison mask must contain finite values")
    region = values > 0
    if not region.any():
        raise ValueError("comparison mask is empty")
    return image, region


def mask_metrics(candidate, baseline):
    intersection = candidate & baseline
    return {"candidate_voxels": int(candidate.sum()), "baseline_voxels": int(baseline.sum()),
            "intersection_voxels": int(intersection.sum()),
            "union_voxels": int(np.count_nonzero(candidate | baseline)),
            "different_voxels": int(np.count_nonzero(candidate != baseline)),
            "exact_equal": bool(np.array_equal(candidate, baseline)),
            "dice": float(2 * intersection.sum() / (candidate.sum() + baseline.sum()))}


def source_evidence(image_path, sha256):
    """只提取副文件中的版本、已记录源码哈希和依赖版本。"""
    sidecar = image_path.with_name(image_path.name[:-7] + ".json")
    if not sidecar.is_file():
        return {"captured_source_available": False}
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    fnit = metadata.get("FNIT", {})
    provenance = fnit.get("Source") or fnit.get("Report", {}).get("source", {})
    evidence = {"sidecar_sha256": sha256(sidecar), "captured_source_available": bool(provenance)}
    source_hashes = provenance.get("SourceSHA256", {})
    declared_manifest = provenance.get("SourceManifestSHA256")
    if source_hashes:
        actual_manifest = hashlib.sha256(json.dumps(
            source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        evidence["source_manifest_sha256"] = actual_manifest
        evidence["declared_manifest_matches_sources"] = actual_manifest == declared_manifest
        selected = ("mcflirt/core.py", "mcflirt/_cost_cuda.py", "mcflirt/sampling.py",
                    "fmri/pipeline.py", "fmri/end_to_end.py", "feat/temporal.py",
                    "synthstrip/pipeline.py")
        evidence["selected_runtime_module_sha256"] = {
            "fnit." + name[:-3].replace("/", "."): source_hashes[name]
            for name in selected if name in source_hashes
            and re.fullmatch(r"[0-9a-f]{64}", str(source_hashes[name]))}
    version = str(provenance.get("Version", ""))
    if re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", version):
        evidence["fnit_version"] = version
    evidence["dependencies"] = {
        name: str(value) for name, value in provenance.get("Dependencies", {}).items()
        if name in ("torch", "numpy", "nibabel", "scipy")
        and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", str(value))}
    return evidence


def verify_executed_sources(image_path, source_root, sha256):
    """核对实际副文件与冻结源码，不把调用者提供的 revision 当作执行证据。"""
    sidecar = image_path.with_name(image_path.name[:-7] + ".json")
    if not sidecar.is_file():
        raise ValueError("final derivative has no captured runtime source manifest")
    fnit = json.loads(sidecar.read_text(encoding="utf-8")).get("FNIT", {})
    provenance = fnit.get("Source") or fnit.get("Report", {}).get("source", {})
    hashes = provenance.get("SourceSHA256", {})
    if not hashes:
        raise ValueError("final derivative has no captured runtime source hashes")
    actual_manifest = hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if actual_manifest != provenance.get("SourceManifestSHA256"):
        raise ValueError("captured runtime source manifest checksum does not match")
    package_root = (source_root / "src" / "fnit").resolve()
    for relative, expected in hashes.items():
        path = (package_root / relative).resolve()
        if not path.is_relative_to(package_root) or not path.is_file() or sha256(path) != expected:
            raise ValueError("derivative runtime manifest differs from supplied frozen source tree")
    return {"files_checked": len(hashes), "all_match": True,
            "source_manifest_sha256": actual_manifest}


def check_pair(candidate_path, baseline_path, region_image, frames):
    candidate = nib.load(str(candidate_path))
    baseline = nib.load(str(baseline_path))
    if candidate.ndim != 4 or baseline.ndim != 4 or candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline must contain matching complete 4D series")
    if candidate.shape[3] != frames:
        raise ValueError("all captured and final images must retain the same complete frame count")
    if candidate.shape[:3] != region_image.shape or not np.allclose(
            candidate.affine, region_image.affine, atol=1e-4, rtol=0):
        raise ValueError("candidate image differs from its supplied comparison region grid")
    if not np.allclose(candidate.affine, baseline.affine, atol=1e-4, rtol=0):
        raise ValueError("candidate and baseline output affines differ")
    return candidate, baseline


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, help_text in (
        ("candidate-derivatives", "候选 BIDS derivatives 根目录，必须只有一例目标时序"),
        ("baseline-derivatives", "冻结 FNIT BIDS derivatives 根目录"),
        ("candidate-intermediates", "候选 benchmark_bids 捕获目录，包含 feat/ 和 aroma/"),
        ("baseline-intermediates", "冻结 benchmark_bids 捕获目录，包含 feat/ 和 aroma/"),
        ("mni-mask", "MNI 输出网格的三维共同统计掩膜"),
        ("report-out", "匿名 JSON 报告路径；不生成影像和数组"),
        ("candidate-source-root", "候选实际运行的冻结源码根目录"),
        ("baseline-source-root", "基准实际运行的冻结源码根目录"),
    ):
        parser.add_argument("--" + name, type=Path, required=True, help=help_text)
    parser.add_argument("--source-revision", required=True, help="候选运行 Git revision，7–40 位十六进制")
    parser.add_argument("--baseline-source-revision", required=True, help="冻结运行 Git revision，7–40 位十六进制")
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if not all(re.fullmatch(r"[0-9a-fA-F]{7,40}", revision)
               for revision in (args.source_revision, args.baseline_source_revision)):
        parser.error("source revisions must be 7-40 hexadecimal Git commit characters")
    started = time.perf_counter()
    helpers, helper_path = load_comparison_helpers()
    sha256 = helpers.sha256
    paths = {}
    for label, derivatives, intermediates in (
        ("candidate", args.candidate_derivatives, args.candidate_intermediates),
        ("baseline", args.baseline_derivatives, args.baseline_intermediates),
    ):
        if not derivatives.is_dir() or not intermediates.is_dir():
            parser.error("derivative and captured intermediate roots must be existing directories")
        paths[label] = {
            "motion_corrected": intermediates / "feat/prefiltered_func_data_mcf.nii.gz",
            "feat_filtered": intermediates / "feat/filtered_func_data.nii.gz",
            "aroma_native": intermediates / "aroma/filtered_func_data_aroma.nii.gz",
            "clean_native": unique_file(derivatives, "*_space-boldref_desc-clean_bold.nii.gz"),
            "clean_mni": unique_file(derivatives, "*_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz"),
            "native_mask": intermediates / "feat/mask.nii.gz",
            "mni_output_mask": unique_file(
                derivatives, "*_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz", optional=True),
        }
        if any(not path.is_file() for path in paths[label].values() if path is not None):
            raise ValueError("a required captured image or final derivative is missing")
    verified_sources = {
        label: {role: verify_executed_sources(paths[label][role], source_root, sha256)
                for role in ("clean_native", "clean_mni")}
        for label, source_root in (("candidate", args.candidate_source_root),
                                   ("baseline", args.baseline_source_root))
    }
    baseline_reference = nib.load(str(paths["baseline"]["clean_native"]))
    if baseline_reference.ndim != 4 or baseline_reference.shape[3] < 2:
        raise ValueError("baseline must contain at least two frames of the complete real series")
    frames = baseline_reference.shape[3]
    native_mask_image, native_region = load_mask(paths["baseline"]["native_mask"], baseline_reference)
    candidate_native_mask_image, candidate_native_region = load_mask(
        paths["candidate"]["native_mask"], baseline_reference)
    del candidate_native_mask_image
    mni_reference = nib.load(str(paths["baseline"]["clean_mni"]))
    mni_mask_image, mni_region = load_mask(args.mni_mask, mni_reference)
    report = {
        "schema_version": 1, "subjects": 1, "real_frames": int(frames),
        "source_revision": args.source_revision,
        "baseline_source_revision": args.baseline_source_revision,
        "baseline_kind": "frozen_fnit",
        "native_region": "Frozen FEAT brain mask; positive voxels.",
        "mni_region": "Supplied common MNI brain mask; positive voxels.",
        "native_mask_comparison": mask_metrics(candidate_native_region, native_region),
        "images": {},
        "input_sha256": {label: {name: sha256(path) for name, path in role_paths.items() if path is not None}
                         for label, role_paths in paths.items()},
        "mni_comparison_mask_sha256": sha256(args.mni_mask),
        "captured_sources": {label: source_evidence(role_paths["clean_native"], sha256)
                             for label, role_paths in paths.items()},
        "executed_source_tree_verification": verified_sources,
        "comparison_driver_sha256": {
            "compare_mcflirt_optimization": sha256(__file__),
            "benchmark_optimization_shared_metrics": sha256(helper_path),
            "benchmark_bids_current_file": sha256(Path(__file__).resolve().parent / "benchmark_bids.py"),
        },
        "software": {"python": sys.version.split()[0], "nibabel": nib.__version__, "numpy": np.__version__},
        "scope": "One complete saved real BIDS volume run per FNIT revision. Compares captured motion-corrected, highpass and AROMA outputs plus final boldref and MNI clean derivatives; never reruns registration, denoising or original software.",
        "source_identity_note": "Git revisions are caller-supplied run labels. Every captured derivative runtime source hash must match its supplied frozen source tree. Current comparison/benchmark driver hashes identify files available during comparison, not proof of the historical launcher's identity.",
        "privacy": "Anonymous scalar aggregates and hashes only; no subject identifiers, filesystem paths, image data or per-frame/per-voxel arrays.",
        "metric_definition": "Shared helper computes decoded full-series exact equality and RMSE over all voxels and separately within the comparison mask; temporal Pearson r includes only nonconstant masked time series. Float64 statistics use spatial chunks.",
    }
    for role in ("motion_corrected", "feat_filtered", "aroma_native", "clean_native", "clean_mni"):
        region_image, region = (mni_mask_image, mni_region) if role == "clean_mni" else (native_mask_image, native_region)
        candidate, baseline = check_pair(paths["candidate"][role], paths["baseline"][role], region_image, frames)
        comparison = helpers.compare_image(candidate, paths["baseline"][role], region, frames)
        comparison["shape"] = list(candidate.shape)
        comparison["candidate_tr_header"] = float(candidate.header.get_zooms()[3])
        comparison["baseline_tr_header"] = float(baseline.header.get_zooms()[3])
        comparison["same_time_unit_and_tr_header"] = bool(
            candidate.header.get_xyzt_units()[1] == baseline.header.get_xyzt_units()[1]
            and np.isclose(candidate.header.get_zooms()[3], baseline.header.get_zooms()[3], atol=1e-6, rtol=0))
        report["images"][role] = comparison
        del candidate, baseline
    output_mask_paths = [paths[label]["mni_output_mask"] for label in ("candidate", "baseline")]
    if all(path is not None for path in output_mask_paths):
        output_regions = [load_mask(path, mni_reference)[1] for path in output_mask_paths]
        report["mni_output_mask_comparison"] = mask_metrics(*output_regions)
        common = mni_region & output_regions[0] & output_regions[1]
        report["mni_common_output_voxels"] = int(common.sum())
        if not np.array_equal(*output_regions):
            if common.any():
                candidate = nib.load(str(paths["candidate"]["clean_mni"]))
                report["mni_common_output_comparison"] = helpers.compare_image(
                    candidate, paths["baseline"]["clean_mni"], common, frames)
                del candidate
            else:
                report["mni_common_output_comparison"] = {"skipped": "No common output voxels inside the supplied MNI mask."}
    else:
        report["mni_output_mask_comparison"] = {"available": False,
                                                 "note": "A saved MNI output mask is unavailable; supplied-mask statistics remain valid."}
    report["all_decoded_images_exact_equal"] = all(
        value["whole_image"]["exact_equal"] for value in report["images"].values())
    report["comparison_wall_seconds_including_reads_and_hashes"] = time.perf_counter() - started
    report["timing_scope"] = "CPU validation reads, decoded image statistics and hashes only; not MCFLIRT or full-pipeline execution time."
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
