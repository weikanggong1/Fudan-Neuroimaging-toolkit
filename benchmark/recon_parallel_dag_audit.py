#!/usr/bin/env python3
"""复核 recon-all 可并行阶段：绑定源码与既有整例报告，不运行影像算法。

输入 repository 为 FNIT 仓库根目录，output 为新报告目录。读取当前调度、
MNI/Synth 输入输出接口与冻结 803aec50 的四次真实整例计时；写出 JSON、CSV
及 SHA-256 清单。没有 GPU 调用、依赖下载或生产调度修改。不满足函数定义、
既有报告或阶段计时合同时抛异常；不能将本报告当作并行运行的速度/显存验收。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path


EVIDENCE = Path("validation/recon_all/optimizations/20261009_whole_pair_a100_803aec50")
EVIDENCE_COMMIT = "803aec50248385b3e4170cc8cdaa9667035f7288"
SOURCES = (
    "src/fnit/recon_all/native_free.py",
    "src/fnit/recon_all/input_chain.py",
    "src/fnit/recon_all/input_talairach_chain.py",
    "src/fnit/recon_all/hemisphere_parallel.py",
    "src/fnit/recon_all/mni_aux_chain.py",
    "src/fnit/recon_all/mni_nonlinear_chain.py",
    "src/fnit/recon_all/finalsurfs_python.py",
    "src/fnit/recon_all/thread_budget.py",
)
STAGES = (
    "input_talairach", "n4", "SynthSeg", "mri_em_register", "mri_ca_normalize",
    "mri_cc", "mni_aux", "mni_nonlinear", "brain_finalsurfs",
    "surface_hemisphere_group", "register_hemisphere_group",
    "annotation_hemisphere_group", "finish_surface_hemisphere_group", "mesh_validation",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _definitions(repository: Path) -> tuple[dict, dict]:
    bindings, definitions = {}, {}
    for name in SOURCES:
        path = repository / name
        text = path.read_text()
        head = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=repository)
        bindings[name] = {"sha256": _sha(path), "head_sha256": hashlib.sha256(head).hexdigest()}
        bindings[name]["matches_HEAD"] = bindings[name]["sha256"] == bindings[name]["head_sha256"]
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[f"{name}:{node.name}"] = {
                    "path": name, "function": node.name,
                    "start_line": node.lineno, "end_line": node.end_lineno,
                    "source_sha256": bindings[name]["sha256"],
                }
    required = (
        "src/fnit/recon_all/native_free.py:run_synthseg_and_write",
        "src/fnit/recon_all/native_free.py:_run_white_mri_chain",
        "src/fnit/recon_all/native_free.py:_validate_meshes",
        "src/fnit/recon_all/hemisphere_parallel.py:run_hemisphere_group",
        "src/fnit/recon_all/mni_aux_chain.py:register_mni152_affine",
        "src/fnit/recon_all/mni_aux_chain.py:run_mni_aux_chain",
        "src/fnit/recon_all/mni_nonlinear_chain.py:run_mni_nonlinear_chain",
        "src/fnit/recon_all/finalsurfs_python.py:run_finalsurfs",
        "src/fnit/recon_all/input_chain.py:run_input_chain",
    )
    return bindings, {name: definitions[name] for name in required}


def build_audit(*, repository: Path, output: Path) -> dict:
    """读取源码及既有计时，输出并行依赖审计；路径均以仓库相对形式记录。

    repository 必须包含当前 FNIT 源码和冻结整例 CSV/JSON；output 默认由 CLI
    指定，不读取影像、权重或原生程序。返回与 audit.json 相同的字典。当前
    Git commit 与源码 SHA 单列；时间仅归属 EVIDENCE_COMMIT，不改标为当前测试。
    """
    repository = repository.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources, definitions = _definitions(repository)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    stage_path = repository / EVIDENCE / "stage_times.csv"
    with stage_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    wanted = [row for row in rows if row["stage"] in STAGES]
    index = {(row["case"], row["kind"], row["stage"]): row for row in wanted}
    reports = {}
    for case in ("06", "07"):
        for variant in ("control", "candidate"):
            relative = EVIDENCE / "reports/runs/recon_e2e_803aec50_cfff_20261009" / (
                f"sub{case}-{variant}") / "subject/fnit-native-free-run.json"
            path = repository / relative
            report = json.loads(path.read_text())
            if report["status"] != "complete" or report["threads"] != 4:
                raise ValueError(f"unexpected report completion/thread budget: {relative}")
            real = {row["name"]: float(row["seconds"]) for row in report["stages"]}
            for stage in STAGES:
                if float(index[(case, variant, stage)]["seconds"]) != real[stage]:
                    raise ValueError(f"CSV/runtime stage mismatch: {relative}:{stage}")
            reports[str(relative)] = {
                "sha256": _sha(path), "status": report["status"],
                "threads": report["threads"], "logical_device": report["device"],
                "cuda_allocator": report["cuda_allocator"],
                "total_seconds": report["total_seconds"],
                "total_scope": report["timing"]["total_scope"],
                "mesh_validation_status": report["mesh_validation"]["status"],
                "private_copy_seconds": {
                    value["operation"]: value["private_copy_seconds"]
                    for value in report["hemisphere_scheduling"]["groups"]
                },
            }
    dag = [
        {"node": "orig_ready", "parents": ["raw_import", "conform", "XFORM_tag_write"],
         "outputs": ["mri/orig.mgz"], "read_only_after": "run_input_chain returns"},
        {"node": "SynthSeg", "parents": ["orig_ready"],
         "inputs": ["mri/orig.mgz", "declared SynthSeg weights/LUT"],
         "outputs": ["mri/synthseg.rca.mgz", "stats/synthseg.vol.csv"],
         "join_before": "mri_cc; downstream mni_aux also consumes segmentation",
         "return_payload": ["total_intracranial_mm3", "actual forward precision"]},
        {"node": "N4", "parents": ["orig_ready"],
         "inputs": ["mri/orig.mgz", "declared backend/program"],
         "outputs": ["mri/tmp/nu0.mgz", "scripts/N4 runtime reports"],
         "join_before": "make_nu also requires completed talairach.xfm"},
        {"node": "GCA_EM", "parents": ["nu", "brainmask"],
         "inputs": ["mri/nu.mgz", "mri/brainmask.mgz", "declared GCA"],
         "outputs": ["mri/transforms/talairach.lta"], "requires_SynthSeg": False},
        {"node": "CA_normalize", "parents": ["nu", "brainmask", "GCA_EM"],
         "outputs": ["mri/norm.mgz", "mri/ctrl_pts.mgz"]},
        {"node": "mri_cc", "parents": ["SynthSeg", "CA_normalize"],
         "outputs": ["mri/aseg.auto_noCCseg.mgz", "mri/aseg.auto.mgz",
                     "mri/aseg.presurf.mgz", "mri/aseg.mgz", "mri/transforms/cc_up.lta"]},
        {"node": "MNI_affine", "parents": ["orig_ready"],
         "outputs": ["mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/aff.lta",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta"]},
        {"node": "MNI_aux", "parents": ["MNI_affine", "nu", "SynthSeg"],
         "inputs_for_full_stats": ["mri/transforms/talairach.xfm.lta"],
         "outputs": ["mri/mca-dura.mgz", "mri/vsinus.mgz", "stats/vsinus.stats"]},
        {"node": "MNI_nonlinear", "parents": ["MNI_affine", "orig_ready"],
         "inputs": ["mri/orig.mgz", "MNI affine/crop", "declared deform weights/MNI targets"],
         "outputs": ["mri/transforms/synthmorph.1.0mm.1.0mm/tmp/deform.mgz",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/tmp/reg.crop-to-invol.lta",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/tmp/reg.crop-to-full.lta",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.nii.gz",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.inv.nii.gz",
                     "mri/transforms/synthmorph.1.0mm.1.0mm/test.nii.gz"],
         "join_before": "successful output inventory/status, not brain_finalsurfs"},
        {"node": "brain_finalsurfs", "parents": ["brain", "brainmask", "MNI_aux", "entowm", "aseg.presurf"],
         "outputs": ["mri/brain.finalsurfs.mgz", "mri/brain.finalsurfs.manedit.mgz"],
         "requires_MNI_nonlinear": False},
        {"node": "surface_topology_prefix", "parents": ["filled", "norm", "brain", "wm", "aseg.presurf"],
         "join_before": "white.preaparc placement requires brain.finalsurfs",
         "compute": "mixed CPU/CUDA; optional Torch inflation adds CUDA work"},
        {"node": "mesh_validation", "parents": ["all surface/metrics/statistics groups"],
         "inputs": ["surf/{lh,rh}.{orig,white,pial,sphere.reg}"],
         "outputs": ["report mesh_validation field"], "compute": "CPU, no MNI warp reads"},
    ]
    candidates = [
        {"priority": 1, "pair": ["MNI_nonlinear", "mesh_validation"],
         "implementation": "fresh exec MNI worker at final validation; join both before complete/output inventory",
         "reason": "no hemisphere MRI snapshots remain; surfaces and transform writes are disjoint",
         "cpu_budget": "MNI <=2, validation <=2; verify effective BLAS/Numba/Torch/native thread counts",
         "gpu_admission": "release idle parent cache; measure parent live context + MNI worker simultaneously",
         "precision": "preserve FP32 deform matmul/cuDNN exception and restore parent TF32"},
        {"priority": 2, "pair": ["SynthSeg", "CA_normalize"],
         "implementation": "fresh exec SynthSeg after brainmask/GCA, concurrently with CPU CA normalize; join before mri_cc",
         "reason": "works with GPU N4 profile without two large neural GPU jobs at once",
         "cpu_budget": "each <=2; immutable orig and disjoint synthseg/norm outputs",
         "precision": "apply SynthSeg FP32 cuDNN policy after construction in child; keep parent TF32"},
        {"priority": 3, "pair": ["SynthSeg", "native_N4"],
         "implementation": "fresh exec SynthSeg after input_talairach completion; CPU N4 in independent process",
         "reason": "both consume immutable orig; N4 native profile only",
         "cpu_budget": "each <=2; benchmark reduced N4 threads rather than assuming current four-thread time",
         "gpu_admission": "does not authorize overlapping GPU N4 and SynthSeg without simultaneous-memory evidence"},
        {"priority": 4, "pair": ["MNI_nonlinear", "surface_CPU_prefix"],
         "implementation": "private minimal subject with read-only copies of orig/crop/affine; publish declared outputs after join",
         "reason": "current copytree/snapshot traverses all MRI files at every hemisphere group",
         "cpu_budget": "LH1+RH1+MNI2; may reduce surface CPU throughput",
         "gpu_admission": "surface tessellation, normals/curvature and optional inflation use GPU; add admission barrier",
         "status": "higher implementation risk; pair 1 first"},
    ]
    audit = {
        "schema_version": 1, "scope": "read-only dependency audit; no concurrent benchmark executed",
        "audited_commit": commit, "audited_sources": sources, "definition_sites": definitions,
        "audited_source_changes_relative_HEAD": [name for name, binding in sources.items()
                                                 if not binding["matches_HEAD"]],
        "timing_evidence_commit": EVIDENCE_COMMIT,
        "timing_evidence": {"csv": str(EVIDENCE / "stage_times.csv"), "csv_sha256": _sha(stage_path),
                            "reports": reports, "note": "real serial stages, not a speed prediction; nesting not additive"},
        "dag": dag, "candidates": candidates,
        "shared_write_hazards": [
            "hemisphere_parallel.copytree copies all MRI/surface/label/stats before each group; concurrent source writes are unsafe",
            "finalsurfs overwrites its output across five edits; white starts only after completion",
            "surface.defects.mgz must retain LH then RH template accumulation",
            "model constructors, torch threads and precision flags are process globals; no same-process model thread pool",
            "orig must be immutable after conform and XFORM tag write before any concurrent consumers start",
        ],
        "validation_required": [
            "same-input outputs/geometry/precision plus worker startup/read-write time",
            "same host/thread budget serial-versus-concurrent ABBA; actual combined group wall",
            "CLI and preinitialized-CUDA API; effective allocator and parent precision restored",
            "simultaneous process-tree target-GPU bytes, whole-card bound, allocated/reserved validity and sample gaps",
            "budget 20000000000 bytes; unavailable process mapping is unknown, not zero",
            "raw T1 empty-directory end-to-end pair with complete required outputs, mesh checks and numerical diagnostics",
        ],
        "measured_parallel_speedup": None, "simultaneous_gpu_budget_verified": False,
        "runtime_code_modified": False,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    with (output / "selected_stage_times.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "kind", "stage", "seconds", "scope"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(wanted)
    payloads = ("audit.json", "selected_stage_times.csv")
    manifest = {"script_sha256": _sha(Path(__file__)),
                "files": {name: _sha(output / name) for name in payloads}}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_audit(repository=args.repository, output=args.output)
    print(json.dumps({"audited_commit": result["audited_commit"], "scope": result["scope"]}))


if __name__ == "__main__":
    main()
