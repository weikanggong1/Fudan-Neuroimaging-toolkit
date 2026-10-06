"""Replay the real cached MS-HBM mask writer without any network access.

Preserves the initial installation report and its original unit observation.
This checks image metadata; it is not an MRI pipeline accuracy benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np

import fnit.mshbm.assets_setup as mshbm


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_record(path):
    image = nib.load(path)
    values = np.array(image.dataobj, copy=True)
    record = {
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "file_sha256": sha256(path),
        "decoded_voxel_sha256_C_order": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "shape": list(image.shape),
        "affine": image.affine.tolist(),
        "dtype": str(image.get_data_dtype()),
        "units": list(image.header.get_xyzt_units()),
        "nonzero_voxels": int(np.count_nonzero(values)),
    }
    return image, values, record


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--installation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    installation = json.loads(args.installation_report.read_text())
    if installation.get("post_install_replay") or args.output.exists():
        raise FileExistsError("The completed replay evidence must not be overwritten")
    if installation.get("fresh_release_GET_count") != 97 or not installation.get("passed"):
        raise ValueError("The expected successful initial installation report is required")

    source = "src/fnit/mshbm/assets_setup.py"
    initial_sources = dict(installation["source_sha256"])
    current_sources = {relative: sha256(repo / relative) for relative in initial_sources}
    changed_sources = [relative for relative in initial_sources
                       if initial_sources[relative] != current_sources[relative]]
    if changed_sources != [source]:
        raise ValueError("Replay expects only the mask-writer source to have changed")

    resources = []
    for name, (_, size, digest) in mshbm.FILES.items():
        path = args.cache_root / name
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError("An existing resource did not match its pinned size and SHA-256")
        resources.append({"file_name": name, "size": size, "sha256": digest,
                          "retrieval": "Previously verified installation cache; no download"})

    reference_image, _, reference = image_record(args.reference)
    initial_module = installation["modules"]["mshbm"]
    if reference["file_sha256"] != initial_module["reference"]["sha256"]:
        raise ValueError("Reference differs from the initial real-template installation")
    if reference["units"] != ["mm", "unknown"]:
        raise ValueError("The verified real MNI reference should use millimetres")

    mask_path = args.cache_root / "cortical_mask.nii.gz"
    before_image, before_values, before = image_record(mask_path)
    if before["file_sha256"] != initial_module["generated_mask"]["sha256"]:
        raise ValueError("The original generated mask differs from the initial report")
    if before["units"] != ["unknown", "unknown"]:
        raise ValueError("The original unit-loss observation must remain reproducible")
    preserved_path = args.cache_root / "cortical_mask.units_before_fix.nii.gz"
    if preserved_path.exists():
        raise FileExistsError("Preserved original image already exists")
    shutil.copyfile(mask_path, preserved_path)
    if sha256(preserved_path) != before["file_sha256"]:
        raise ValueError("Preserved image byte-copy differed from the original")
    before_header = before_image.header.binaryblock

    network_calls = []

    def forbidden_urlopen(url, timeout):
        network_calls.append("attempted")
        raise RuntimeError("Network access is forbidden during cached replay")

    original_urlopen = mshbm.urlopen
    mshbm.urlopen = forbidden_urlopen
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        paths = mshbm.prepare_projection_assets(args.cache_root, args.reference)
    finally:
        elapsed = time.perf_counter() - started
        mshbm.urlopen = original_urlopen
    after_image, after_values, after = image_record(paths["cortical_mask"])
    checks = {
        "voxel_arrays_identical": bool(np.array_equal(before_values, after_values)),
        "decoded_voxel_sha256_identical": before["decoded_voxel_sha256_C_order"]
        == after["decoded_voxel_sha256_C_order"],
        "affine_identical": bool(np.array_equal(before_image.affine, after_image.affine)),
        "shape_identical": before_image.shape == after_image.shape,
        "dtype_identical": before_image.get_data_dtype() == after_image.get_data_dtype(),
        "reference_grid_preserved": after_image.shape == reference_image.shape[:3]
        and bool(np.array_equal(after_image.affine, reference_image.affine)),
        "spatial_unit_inherited": after["units"] == reference["units"],
        "no_network_access": not network_calls,
    }
    changed_header_offsets = [index for index, (old, new) in enumerate(
        zip(before_header, after_image.header.binaryblock)) if old != new]
    checks["only_xyzt_units_header_byte_changed"] = changed_header_offsets == [123]
    if not all(checks.values()):
        raise ValueError("A metadata-only replay check failed")
    replay = {
        "schema": "fnit.assets_release.mshbm_units_fix.v1",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real MNI template cached resource mask writing; no MRI pipeline, inference or GPU",
        "timing_scope": "prepare_projection_assets cache validation, CPU resampling and NIfTI saving",
        "elapsed_seconds": elapsed,
        "baseline_commit": installation["baseline_commit"],
        "source_file": source,
        "initial_installer_source_sha256": initial_sources[source],
        "fixed_installer_source_sha256": current_sources[source],
        "replay_script_sha256": sha256(Path(__file__)),
        "resource_guards": resources,
        "reference": reference,
        "before": before,
        "preserved_initial_image": {"file_name": preserved_path.name,
                                    "sha256": sha256(preserved_path)},
        "after": after,
        "changed_NIfTI_header_byte_offsets": changed_header_offsets,
        "network_attempts": len(network_calls),
        "checks": checks,
        "passed": True,
    }
    save_json(args.output, replay)

    # Original download rows, mask observation, timestamps and elapsed time remain intact.
    installation["initial_installer_source_sha256"] = initial_sources
    installation["source_sha256"] = current_sources
    installation["source_sha256_scope"] = (
        "Current source after the cached MS-HBM metadata replay. The original 97 GETs "
        "and original generated-mask observation bind initial_installer_source_sha256."
    )
    installation["post_install_replay"] = {
        "report": str(args.output.resolve().relative_to(repo)),
        "module": "mshbm",
        "fixed_installer_source_sha256": current_sources[source],
        "generated_mask_sha256": after["file_sha256"],
        "generated_mask_units": after["units"],
        "preserved_initial_mask_sha256": before["file_sha256"],
        "network_attempts": len(network_calls),
        "original_installation_evidence_preserved": True,
        "MRI_benchmark_repeated": False,
    }
    save_json(args.installation_report, installation)
    print(json.dumps({"passed": True, "checks": checks,
                      "network_attempts": len(network_calls), "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()
