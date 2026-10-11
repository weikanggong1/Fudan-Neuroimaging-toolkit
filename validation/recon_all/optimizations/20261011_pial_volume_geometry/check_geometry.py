"""只读冻结七输入的 pial 头几何；不执行放置/模型或读取参考输出。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_geometry(*, subject_directory: Path, installed_directory: Path,
                   baseline_source: Path, candidate_source: Path,
                   expected_baseline_sha256: str, expected_candidate_sha256: str,
                   expected_inputs: Path, hemisphere: str, output: Path) -> dict:
    """七项自产文件只读校验，保存安全数值/hash JSON，不保存影像或私有路径。

    subject_directory 是七输入目录；installed_directory 是冻结 FNIT 安装。
    baseline_source/candidate_source 分别为两份 pial 模块，expected_*sha256
    必须匹配实际源码。expected_inputs 为七个相对路径→SHA 的 JSON；
    hemisphere 为 lh/rh。output 须尚不存在；返回校验结果字典并写出 JSON。
    MRI为同一conform网格；surface RAS→voxel为float32 (4,4)，单位mm→voxel。
    输入/源码哈希、空间或变换不同抛异常，不写成功收据、不修改输入。
    只读取MRI头及表面几何；文件哈希读压缩字节，不解码影像数组。
    """
    started = time.monotonic()
    if output.exists():
        raise FileExistsError("output receipt already exists")
    if hemisphere not in {"lh", "rh"}:
        raise ValueError("hemisphere must be lh or rh")
    if _sha(baseline_source) != expected_baseline_sha256 or \
            _sha(candidate_source) != expected_candidate_sha256:
        raise ValueError("pial source SHA differs from frozen protocol")
    expected = json.loads(expected_inputs.read_text())
    required = [f"surf/{hemisphere}.white", f"surf/autodet.gw.stats.{hemisphere}.dat",
        f"label/{hemisphere}.cortex.label", f"label/{hemisphere}.cortex+hipamyg.label",
        "mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz"]
    if not isinstance(expected, dict) or set(expected) != set(required):
        raise ValueError("expected_inputs must contain exactly the seven relative input paths")
    before = {rel: _sha(subject_directory / rel) for rel in required}
    if before != expected:
        raise ValueError("input SHA differs from frozen protocol")
    sys.path.insert(0, str(installed_directory))
    import nibabel as nib
    import numpy as np
    import torch
    import numba
    modules = []
    for name, path in (("baseline", baseline_source), ("candidate", candidate_source)):
        module_name = "fnit.recon_all._geometry_" + name
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    baseline, candidate = modules
    xyz, faces, metadata = nib.freesurfer.read_geometry(
        str(subject_directory / required[0]), read_metadata=True)
    if not np.isfinite(xyz).all():
        raise ValueError("white coordinates are not finite")
    images = [nib.load(str(subject_directory / rel)) for rel in required[-3:]]
    original = baseline.surface_ras_to_voxel(images[0].header, metadata)
    checked = candidate._validate_pial_volume_geometry(*images, metadata)
    if original.dtype != checked.dtype or original.tobytes() != checked.tobytes():
        raise ValueError("new preflight changed the original coordinate transform")
    after = {rel: _sha(subject_directory / rel) for rel in required}
    if after != before:
        raise ValueError("frozen inputs changed during read-only check")
    result = {"schema_version": 1, "status": "passed",
        "scope": "real frozen FNIT seven-input header preflight, not a placement benchmark",
        "hemisphere": hemisphere, "input_sha256_before": before, "input_sha256_after": after,
        "source_sha256": {"baseline_pial": expected_baseline_sha256,
            "candidate_pial": expected_candidate_sha256,
            "driver": _sha(Path(__file__)),
            "surface_geometry": _sha(Path(candidate.surface_ras_to_voxel.__code__.co_filename))},
        "libraries": {"nibabel": nib.__version__, "numpy": np.__version__, "torch": torch.__version__,
            "numba": numba.__version__},
        "resources": {"cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(), "numba_threads": numba.get_num_threads(),
            "CUDA_visible_devices_empty": os.environ.get("CUDA_VISIBLE_DEVICES") == ""},
        "surface": {"vertices": len(xyz), "faces": len(faces), "coordinates_finite": True,
            "volume_geometry_valid": str(metadata["valid"]).split("#", 1)[0].strip() == "1"},
        "grid_identity_basis": ["3D_shape", "scanner_RAS_affine"],
        "MRI_tkr_equal_to_brain": {rel: bool(np.array_equal(
            image.header.get_vox2ras_tkr(), images[0].header.get_vox2ras_tkr()))
            for rel, image in zip(required[-3:], images)},
        "MRI": {rel: {"shape": list(map(int, image.shape)), "dtype": str(image.get_data_dtype()),
            "affine": image.affine.tolist(), "vox2ras_tkr": image.header.get_vox2ras_tkr().tolist(),
            "MGH_direction_cosines": image.header["Mdc"].tolist(),
            "MGH_voxel_spacing_mm": image.header["delta"].tolist(),
            "MGH_center_scanner_RAS_mm": image.header["Pxyz_c"].tolist(),
            "zooms_mm": list(map(float, image.header.get_zooms()[:3]))}
            for rel, image in zip(required[-3:], images)},
        "transform": {"shape": list(checked.shape), "dtype": str(checked.dtype),
            "matrix": checked.tolist(), "bytes_equal": True, "different_elements": 0,
            "max_absolute_error": 0.0, "p99_absolute_error": 0.0,
            "baseline_bytes_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
            "candidate_bytes_sha256": hashlib.sha256(checked.tobytes()).hexdigest()},
        "cuda_initialized": torch.cuda.is_initialized(),
        "image_voxel_arrays_decoded": False, "placement_optimizer_called": False,
        "seconds_including_imports_hashes_headers_and_checks_before_receipt_write": time.monotonic() - started,
        "whole_recon_all": "not_run", "new_pial_precision_or_speed": "not_measured",
        "overall_metric_equivalence": "not_assessed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject-directory", "installed-directory", "baseline-source", "candidate-source",
                 "expected-inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    result = check_geometry(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "transform_bytes_equal": True,
        "inputs_unchanged": True, "placement_optimizer_called": False,
        "cuda_initialized": result["cuda_initialized"]}))


if __name__ == "__main__":
    main()
