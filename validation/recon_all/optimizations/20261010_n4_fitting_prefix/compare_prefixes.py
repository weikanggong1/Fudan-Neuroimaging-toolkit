"""复用已有14图/LTA/Dice比较算子；两个对象必须是已完成连续前段。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


VOLUMES = ("rawavg.mgz", "orig.mgz", "tmp/nu0.mgz", "nu.mgz", "T1.mgz",
           "brainmask.mgz", "synthseg.rca.mgz", "norm.mgz", "brain.mgz",
           "aseg.presurf.mgz", "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz", "filled.mgz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(*, reference_benchmark: Path, candidate_benchmark: Path, output: Path,
            existing_stage_orig: Path | None = None) -> dict:
    import nibabel as nib
    import numpy as np
    from fnit.recon_all import compare_native_free, ca_normalize_python
    tick = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    first_bytes, second_bytes = reference_benchmark.read_bytes(), candidate_benchmark.read_bytes()
    first, second = json.loads(first_bytes), json.loads(second_bytes)
    if (first["execution"] != "complete_prefix" or second["execution"] != "complete_prefix"
            or first["full_workflow_complete"] or second["full_workflow_complete"]):
        raise ValueError("both raw-prefix producers must have completed filled, never whole workflow")
    if first["input_sha256"] != second["input_sha256"] or (first["fitting_threads"], second["fitting_threads"]) != (1, 4):
        raise ValueError("wrong raw input or thread contrast")
    for name in ("source_manifest_sha256", "N4_module_sha256", "N4_binding_sha256", "python_profile_module_sha256",
                 "fixed_profile", "resource_sha256", "device", "CPU_affinity", "threads", "execution_audit_helper_sha256"):
        if first[name] != second[name]:
            raise ValueError("fixed pair parameter differs: " + name)
    runtime = []
    for producer in (reference_benchmark, candidate_benchmark):
        path = producer.parent / "subject/fnit-native-free-run.json"
        report = json.loads(path.read_bytes())
        if report["status"] != "complete_prefix" or report["stages"][-1]["name"] != "filled_auto_checkpoint":
            raise ValueError("runtime did not complete the declared raw-prefix barrier")
        runtime.append(report)
    report = {"scope": "same raw-T1 empty-directory continuous prefix fit1 versus fit4, no full recon-all",
        "status": "complete_prefix_diagnostic", "case": first["case"],
        "original_T1_sha256": first["input_sha256"], "fixed_parameters_same": True,
        "reference_producer_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "candidate_producer_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "reference_runtime_sha256": sha(reference_benchmark.parent / "subject/fnit-native-free-run.json"),
        "candidate_runtime_sha256": sha(candidate_benchmark.parent / "subject/fnit-native-free-run.json"),
        "source_manifest_sha256": first["source_manifest_sha256"],
        "source_AST_transformations": [first["diagnostic_AST_transformation"], second["diagnostic_AST_transformation"]],
        "comparator_sha256": sha(Path(__file__)),
        "reused_volume_comparator_module_sha256": sha(Path(compare_native_free.__file__)),
        "reused_LTA_reader_module_sha256": sha(Path(ca_normalize_python.__file__)),
        "volumes": {}, "overall_metric_equivalence": "not_assessed", "full_workflow_complete": False,
        "prefix_API_seconds": [first["public_API_seconds_including_reads_writes"], second["public_API_seconds_including_reads_writes"]]}
    for name in VOLUMES:
        a, b = (producer.parent / "subject/mri" / name for producer in (reference_benchmark, candidate_benchmark))
        before = [sha(a), sha(b)]
        images = [nib.load(str(path)) for path in (a, b)]
        arrays = [np.asarray(image.dataobj) for image in images]
        value = compare_native_free._volume(a, b)
        value["file_sha256"] = {"reference": sha(a), "candidate": sha(b)}
        if before != [value["file_sha256"]["reference"], value["file_sha256"]["candidate"]]:
            raise RuntimeError("input changed during comparison: " + name)
        value["vox2ras_tkr_max_abs"] = float(np.abs(images[0].header.get_vox2ras_tkr()-images[1].header.get_vox2ras_tkr()).max())
        value["zooms_equal"] = images[0].header.get_zooms() == images[1].header.get_zooms()
        if arrays[0].shape == arrays[1].shape:
            error = np.abs(arrays[0].astype(np.float64)-arrays[1].astype(np.float64))
            value["p99_abs_error"] = float(np.quantile(error, .99))
            if name in {"filled.mgz", "aseg.presurf.mgz", "synthseg.rca.mgz"}:
                value["per_label"] = compare_native_free._label_dice(*arrays)
            if name.startswith("wm"):
                masks = [array != 0 for array in arrays]
                total = sum(int(mask.sum()) for mask in masks)
                value["foreground_dice"] = 2 * int((masks[0] & masks[1]).sum()) / total if total else 1.0
                value["foreground_definition"] = "nonzero WM; stored byte-intensity differences separately retained"
        report["volumes"][name] = value
    paths = [producer.parent / "subject/mri/transforms/talairach.lta" for producer in (reference_benchmark, candidate_benchmark)]
    matrices = [ca_normalize_python.read_voxel_lta(path) for path in paths]
    report["talairach_lta"] = {"type": "VOX_TO_VOX", "translation_unit": "atlas voxel, not mm",
        "reference": matrices[0].tolist(), "candidate": matrices[1].tolist(),
        "different_elements": int(np.count_nonzero(matrices[0] != matrices[1])),
        "max_abs": float(np.abs(matrices[0]-matrices[1]).max()), "files_sha256": [sha(path) for path in paths]}
    if existing_stage_orig is not None:
        fresh_orig = reference_benchmark.parent / "subject/mri/orig.mgz"
        value = compare_native_free._volume(existing_stage_orig, fresh_orig)
        value["file_sha256"] = [sha(existing_stage_orig), sha(fresh_orig)]
        headers = [nib.load(str(path)).header for path in (existing_stage_orig, fresh_orig)]
        value["vox2ras_tkr_max_abs"] = float(np.abs(headers[0].get_vox2ras_tkr()-headers[1].get_vox2ras_tkr()).max())
        value["scope"] = "old79 conformed stage input versus this fresh raw-prefix orig; SHA difference alone does not imply voxel difference"
        value["is_official_reference"] = False
        report["existing_stage_input_orig_comparison"] = value
    report["diagnostic_seconds_including_reads"] = time.perf_counter()-tick
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    # 本次捕获字节与SHA一致，不依赖之后可能更新的producer元数据。
    output.with_name(output.stem+".reference_snapshot.json").write_bytes(first_bytes)
    output.with_name(output.stem+".candidate_snapshot.json").write_bytes(second_bytes)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-benchmark", "candidate-benchmark", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--existing-stage-orig", type=Path)
    args = parser.parse_args()
    result = compare(**vars(args))
    print(json.dumps({"case": result["case"], "status": result["status"],
        "volumes_different_voxels": {name: value.get("different_voxels") for name,value in result["volumes"].items()}}))
