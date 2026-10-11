"""同原始T1前段回归的显式schema白名单；原收据和影像仅留服务器。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

if __package__:
    from .export_recon_gca_receipts import numeric_leaves
else:
    from export_recon_gca_receipts import numeric_leaves


VOLUMES = ("rawavg.mgz", "orig.mgz", "tmp/nu0.mgz", "nu.mgz", "T1.mgz",
           "brainmask.mgz", "synthseg.rca.mgz", "norm.mgz", "brain.mgz",
           "aseg.presurf.mgz", "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz", "filled.mgz")
NUMBER_FIELDS = ("affine_max_abs", "reference_mean", "candidate_mean", "reference_median",
    "candidate_median", "reference_sum", "candidate_sum", "mean_difference", "sum_difference",
    "mae", "p95", "max_abs", "p99_abs_error", "vox2ras_tkr_max_abs", "foreground_dice")
INTEGER_FIELDS = ("reference_count", "candidate_count", "exact", "different_voxels")
BOOLEAN_FIELDS = ("ordered_comparable", "input_files_unchanged_during_read", "zooms_equal")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA-256 required")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric metric required")
    return value


def _integer(value, *, nonnegative=True):
    if type(value) is not int or nonnegative and value < 0:
        raise ValueError("integer metric required")
    return value


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("boolean metric required")
    return value


def _version(value):
    """只接受commit及具名manifest哈希语法；其他原版本字符串仅保留SHA。"""
    if not isinstance(value, str):
        raise ValueError("producer code version must be a string")
    match = re.fullmatch(r"([0-9a-f]{8,40})(?:\+([a-z][a-z0-9-]{0,40}):([0-9a-f]{64}))?", value)
    return {"value": value if match else None,
        "version_string_sha256": hashlib.sha256(value.encode()).hexdigest(),
        "version_schema_recognized": match is not None}


def _volume(row):
    selected = {}
    for key in ("reference_shape", "candidate_shape"):
        values = row[key]
        if not isinstance(values, list) or len(values) not in (3, 4):
            raise ValueError("MRI shape must be a 3D/4D integer list")
        selected[key] = [_integer(value) for value in values]
    for key in ("reference_dtype", "candidate_dtype"):
        value = row[key]
        if not isinstance(value, str) or re.fullmatch(r"(?:[<>|=][biufc][0-9]+|u?int[0-9]+|float[0-9]+|bool)", value) is None:
            raise ValueError("MRI dtype syntax differs from declared schema")
        selected[key] = value
    for key in NUMBER_FIELDS:
        if key in row:
            selected[key] = _number(row[key])
    for key in INTEGER_FIELDS:
        if key in row:
            selected[key] = _integer(row[key])
    for key in BOOLEAN_FIELDS:
        if key in row:
            selected[key] = _boolean(row[key])
    selected["file_sha256"] = {key: _digest(row["file_sha256"][key]) for key in ("reference", "candidate")}
    if "per_label" in row:
        if not isinstance(row["per_label"], dict):
            raise ValueError("label metrics must be a numeric-ID mapping")
        labels = {}
        for code, original in row["per_label"].items():
            if not isinstance(code, str) or re.fullmatch(r"[0-9]{1,10}", code) is None:
                raise ValueError("numeric label ID required")
            labels[code] = {"reference_voxels": _integer(original["reference_voxels"]),
                "candidate_voxels": _integer(original["candidate_voxels"]),
                "difference_voxels": _integer(original["difference_voxels"], nonnegative=False),
                "dice": _number(original["dice"])}
            if not 0 <= labels[code]["dice"] <= 1:
                raise ValueError("Dice is outside its mathematical range")
        selected["per_label"] = labels
    return selected


def export_metrics(*, prefix_report: Path, execution_receipt: Path,
                   cli_timing_receipt: Path, output_directory: Path,
                   expected_input_sha256: str) -> dict:
    """已完成只读比较→显式公共数值JSON与哈希收据，不发布原报告。

    prefix_report包含14MRI+VOX_TO_VOX LTA与三份原样生产收据的绑定；
    execution_receipt及cli_timing_receipt是此次诊断的运行/冷CLI收据。
    expected_input_sha256必选64字符原始T1 SHA；output_directory须不存在。
    验证control完整exit0、候选filled完成、三份捕获字节的SHA、安装比较
    依赖与两层完成收据。逐字段校验原数字/bool/null的类型和值，只输出
    固定MRI名称、几何/误差/标签Dice、LTA16元素和允许的版本/资源字段。
    无字符串替换，不导出私有日志/地址/路径/UUID/权重/影像/完整收据。
    返回export收据字典；任何schema/哈希/完成门失败抛异常，原产物不改。
    """
    if output_directory.exists():
        raise FileExistsError(output_directory)
    _digest(expected_input_sha256)
    raw = prefix_report.read_bytes()
    report = json.loads(raw)
    execution = json.loads(execution_receipt.read_bytes())
    cli = json.loads(cli_timing_receipt.read_bytes())
    if (report["status"] != "complete_prefix_diagnostic" or report["input_sha256"] != expected_input_sha256
            or execution["status"] != "complete_readonly_prefix_regression"
            or execution["prefix_report_sha256"] != hashlib.sha256(raw).hexdigest()
            or execution["input_sha256"] != expected_input_sha256
            or execution["comparator_sha256"] != report["comparator_sha256"]
            or cli["execution_status"] != "complete" or cli["child_exit_code"] != 0):
        raise ValueError("same completed read-only comparator receipts required")
    snapshots = {}
    for key in ("reference_producer", "candidate_producer_checkpoint", "candidate_runtime_checkpoint"):
        record = report["captured_receipts"][key]
        expected_name = prefix_report.stem + "." + key + ".json"
        if record["path"] != expected_name:
            raise ValueError("captured receipt filename differs")
        path = prefix_report.with_name(expected_name)
        if _sha(path) != _digest(record["sha256"]):
            raise ValueError("captured original receipt changed")
        snapshots[key] = json.loads(path.read_bytes())
    reference, candidate, runtime = (snapshots[key] for key in (
        "reference_producer", "candidate_producer_checkpoint", "candidate_runtime_checkpoint"))
    if (reference["execution"] != "complete" or reference["exit_code"] != 0
            or reference["input_sha256"] != expected_input_sha256
            or candidate["input_sha256"] != expected_input_sha256
            or not any(row.get("name") == "filled_auto_checkpoint" and not row.get("error") for row in runtime["stages"])):
        raise ValueError("completed same-T1 FNIT control and candidate filled checkpoint required")
    if (report["reference_code_version"] != reference["code_version"]
            or report["candidate_code_version"] != candidate["code_version"]
            or report["producer_receipt_sha256"]["reference"] != report["captured_receipts"]["reference_producer"]["sha256"]
            or report["producer_receipt_sha256"]["candidate_checkpoint"] != report["captured_receipts"]["candidate_producer_checkpoint"]["sha256"]):
        raise ValueError("producer version/hash binding differs")
    if set(report["volumes"]) != set(VOLUMES):
        raise ValueError("all fourteen declared MRI outputs required")
    volumes = {name: _volume(report["volumes"][name]) for name in VOLUMES}
    lta = report["talairach_lta"]
    if lta["type"] != "VOX_TO_VOX":
        raise ValueError("VOX_TO_VOX matrix required")
    matrix = {}
    for key in ("reference", "candidate"):
        value = lta[key]
        if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
            raise ValueError("complete four-by-four LTA required")
        matrix[key] = [[_number(number) for number in row] for row in value]
    matrix.update(type="VOX_TO_VOX", different_elements=_integer(lta["different_elements"]),
        max_abs=_number(lta["max_abs"]),
        file_sha256={key: _digest(lta["file_sha256"][key]) for key in ("reference", "candidate")})
    names = ("fnit", "fnit.recon_all.ca_normalize_python", "fnit.recon_all.compare_native_free")
    modules = {name: _digest(execution["loaded_module_sha256"][name]) for name in names}
    versions = {}
    for key in ("python", "nibabel", "numpy", "numba", "SimpleITK"):
        value = execution["versions"][key]
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.+-]{1,100}", value) is None:
            raise ValueError("library version syntax differs")
        versions[key] = value
    current_execution = candidate["execution"]
    if current_execution not in ("running", "complete", "failed"):
        raise ValueError("candidate execution state differs from schema")
    selected = dict(schema_version=1, status="complete_prefix_diagnostic",
        validation_layer="single current raw-T1 continuous prefix versus completed FNIT control",
        comparison_reference="FNIT control; not an official FreeSurfer reference",
        declared_case="public ds000114 sub-06 ses-test T1w",
        input_sha256=expected_input_sha256, candidate_whole_execution_at_capture=current_execution,
        whole_candidate_claim=False, whole_pipeline_speedup=None, overall_metric_equivalence="not_assessed",
        reference_code_version=_version(reference["code_version"]),
        candidate_code_version=_version(candidate["code_version"]),
        captured_receipt_sha256={key: _digest(report["captured_receipts"][key]["sha256"]) for key in snapshots},
        volumes=volumes, talairach_lta=matrix,
        coordinates=dict(rawavg="original imported MRI grid", other_mri="producer conform grid",
            lta="voxel-to-voxel; no diagnostic resampling", intensity_unit="stored intensity, no physical unit"),
        wm_foreground_definition="nonzero stored WM; intensity bytes compared separately",
        diagnostic=dict(comparison_seconds_including_input_reads=_number(report["comparison_seconds_including_input_reads"]),
            runner_wall_seconds_before_execution_receipt=_number(execution["diagnostic_wall_seconds_before_execution_receipt"]),
            cold_cli_wall_seconds=_number(cli["cold_cli_wall_seconds"]),
            child_peak_rss_bytes=_integer(cli["max_child_cpu_rss_bytes"]),
            runner_self_peak_rss_bytes=_integer(execution["self_peak_rss_bytes"]),
            threads=_integer(execution["threads"]), cpu_affinity=[_integer(value) for value in execution["cpu_affinity"]],
            gpu_computation=_boolean(execution["gpu_computation"]), gpu_memory_bytes=None,
            simultaneous_parent_child_peak_memory_bytes=None, versions=versions),
        source_sha256=dict(comparator=_digest(report["comparator_sha256"]),
            runner=_digest(execution["runner_sha256"]), timer=_digest(cli["timer_script_sha256"]),
            loaded_comparison_modules=modules, exporter=_sha(Path(__file__))),
        original_report_sha256=hashlib.sha256(raw).hexdigest(),
        original_execution_receipt_sha256=_sha(execution_receipt), original_cli_receipt_sha256=_sha(cli_timing_receipt))
    data = (json.dumps(selected, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()
    before, after = numeric_leaves(selected), numeric_leaves(json.loads(data))
    if [(type(value), value) for value in before] != [(type(value), value) for value in after]:
        raise RuntimeError("selected numeric values or types changed during serialization")
    output_directory.mkdir(mode=0o700, parents=True)
    (output_directory / "metrics.json").write_bytes(data)
    receipt = dict(status="complete_explicit_schema_export", original_report_sha256=hashlib.sha256(raw).hexdigest(),
        exporter_sha256=_sha(Path(__file__)), numeric_boolean_null_leaves_preserved=len(before),
        numeric_values_and_types_preserved=True, original_production_receipts_exported=False,
        payload=dict(metrics=dict(sha256=_sha(output_directory / "metrics.json"), bytes=len(data))))
    (output_directory / "EXPORT_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix-report", "execution-receipt", "cli-timing-receipt", "output-directory"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    print(json.dumps(export_metrics(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
