"""只读比较两条同原始 T1 连续链的 conform→filled，不执行生产阶段。

输入两份真实 benchmark.json 及新输出 JSON 路径；候选必须已经完成
filled_auto_checkpoint，参考必须整例 exit 0，原始 T1 SHA 必须相同。
输出逐体积数据/几何/dtype误差、离散标签 Dice、WM 非零前景 Dice、
VOX_TO_VOX LTA 矩阵误差和所有输入文件 SHA。rawavg 在原始导入网格，
其余 MRI 在各自产出的 conform 网格，
LTA 是体素到体素变换，强度无物理单位；不配准/重采样、不判断等效。
另保存三份原样生产收据快照，SHA绑定实际解析字节，而非稍后更新的文件。
失败抛异常，不修改生产目录，不重跑缺失阶段。没有独立官方命令。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.ca_normalize_python import read_voxel_lta
from fnit.recon_all.compare_native_free import _label_dice, _volume


VOLUMES = ("rawavg.mgz", "orig.mgz", "tmp/nu0.mgz", "nu.mgz", "T1.mgz",
           "brainmask.mgz", "synthseg.rca.mgz", "norm.mgz", "brain.mgz",
           "aseg.presurf.mgz", "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz", "filled.mgz")


def evaluate_prefix(*, reference_benchmark: Path, candidate_benchmark: Path,
                    output: Path) -> dict:
    """比较已完成前段并写出可重算 JSON；全部参数必填，计时为诊断墙钟。

    reference/candidate_benchmark 是生产收据路径；output 必须不存在。
    不要求候选整例完成，仅认实际已完成的前段；状态明确区分前段和整例。
    输出另有reference_producer/candidate_producer_checkpoint/
    candidate_runtime_checkpoint三份JSON字节快照。拒绝已有输出/快照和
    读取期间变化的影像；生产收据后续更新不影响已捕获的版本。
    数值严格诊断无容差救济；filled/SynthSeg/aseg 按标签报告 Dice，
    wm* 仅按非零前景报告 Dice，保留原强度字节的严格比较，不对标签算 r。
    """
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    # 运行中的benchmark会周期写出；解析和SHA必须来自同一次字节捕获。
    ref_bytes, cand_bytes = reference_benchmark.read_bytes(), candidate_benchmark.read_bytes()
    ref, cand = json.loads(ref_bytes), json.loads(cand_bytes)
    if ref.get("execution") != "complete" or ref.get("exit_code") != 0:
        raise ValueError("reference producer did not finish")
    if not cand.get("input_sha256") or cand["input_sha256"] != ref.get("input_sha256"):
        raise ValueError("original T1 hashes differ or are absent")
    left, right = reference_benchmark.parent / "subject", candidate_benchmark.parent / "subject"
    runtime_bytes = (right / "fnit-native-free-run.json").read_bytes()
    runtime = json.loads(runtime_bytes)
    if not any(row.get("name") == "filled_auto_checkpoint" and not row.get("error")
               for row in runtime["stages"]):
        raise ValueError("candidate continuous prefix has not completed filled")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    captured = {"reference_producer": ref_bytes, "candidate_producer_checkpoint": cand_bytes,
                "candidate_runtime_checkpoint": runtime_bytes}
    snapshot_paths = {key: output.with_name(output.stem + "." + key + ".json") for key in captured}
    if any(path.exists() for path in snapshot_paths.values()):
        raise FileExistsError("producer snapshot already exists")
    report = {"scope": "same original T1 continuous volume prefix diagnostic; not a complete candidate end-to-end claim",
              "reference_code_version": ref["code_version"], "candidate_code_version": cand["code_version"],
              "input_sha256": cand["input_sha256"], "candidate_whole_execution": cand["execution"],
              "producer_receipt_sha256": {"reference": hashlib.sha256(ref_bytes).hexdigest(),
                                           "candidate_checkpoint": hashlib.sha256(cand_bytes).hexdigest()},
              "captured_receipts": {key: {"path": path.name,
                  "sha256": hashlib.sha256(captured[key]).hexdigest()} for key, path in snapshot_paths.items()},
              "comparator_sha256": sha(Path(__file__)), "volumes": {},
              "overall_metric_equivalence": "not_assessed"}
    for name in VOLUMES:
        a, b = left / "mri" / name, right / "mri" / name
        file_sha_before = {"reference": sha(a), "candidate": sha(b)}
        images = (nib.load(a), nib.load(b))
        arrays = tuple(np.asarray(image.dataobj) for image in images)
        result = _volume(a, b)
        result["file_sha256"] = {"reference": sha(a), "candidate": sha(b)}
        if result["file_sha256"] != file_sha_before:
            raise RuntimeError("volume changed during diagnostic: " + name)
        result["input_files_unchanged_during_read"] = True
        result["vox2ras_tkr_max_abs"] = float(np.abs(images[0].header.get_vox2ras_tkr() - images[1].header.get_vox2ras_tkr()).max())
        result["zooms_equal"] = images[0].header.get_zooms() == images[1].header.get_zooms()
        if arrays[0].shape == arrays[1].shape:
            delta = np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64))
            result["p99_abs_error"] = float(np.quantile(delta, .99))
            if name == "filled.mgz":
                result["per_label"] = _label_dice(*arrays)
            if name.startswith("wm"):
                masks = tuple(array != 0 for array in arrays)
                denominator = sum(int(mask.sum()) for mask in masks)
                result["foreground_dice"] = 2 * int((masks[0] & masks[1]).sum()) / denominator if denominator else 1.0
                result["foreground_definition"] = "nonzero WM mask; all stored intensity values separately compared exactly"
        report["volumes"][name] = result
    a, b = left / "mri/transforms/talairach.lta", right / "mri/transforms/talairach.lta"
    matrices = (read_voxel_lta(a), read_voxel_lta(b))
    report["talairach_lta"] = {"type": "VOX_TO_VOX", "reference": matrices[0].tolist(),
        "candidate": matrices[1].tolist(), "different_elements": int(np.count_nonzero(matrices[0] != matrices[1])),
        "max_abs": float(np.abs(matrices[0] - matrices[1]).max()),
        "file_sha256": {"reference": sha(a), "candidate": sha(b)}}
    report["comparison_seconds_including_input_reads"] = time.perf_counter() - started
    report["status"] = "complete_prefix_diagnostic"
    output.parent.mkdir(parents=True, exist_ok=True)
    for key, path in snapshot_paths.items():
        with path.open("xb") as stream:
            stream.write(captured[key])
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-benchmark", type=Path, required=True)
    parser.add_argument("--candidate-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = evaluate_prefix(reference_benchmark=options.reference_benchmark,
                             candidate_benchmark=options.candidate_benchmark, output=options.output)
    print(json.dumps({"status": result["status"], "different_voxels": {
        k: v.get("different_voxels") for k, v in result["volumes"].items()}}))
