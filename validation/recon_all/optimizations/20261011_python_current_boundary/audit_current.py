"""复用既有24组审计，只解析声明快照；不导入生产包或运行影像阶段。"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_helper(repository):
    path = repository / "validation/recon_all/optimizations/20261011_python_call_chain_audit/audit_call_chain.py"
    spec = importlib.util.spec_from_file_location("_existing_fnit_call_audit", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, {"path": str(path.relative_to(repository)), "sha256": sha(path.read_bytes())}


def profile_from_ast(source, *, n4_backend, n4_execution, topology_backend):
    """只读取固定返回字典，不执行torch.device或任何模块初始化。"""
    path = source / "src/fnit/recon_all/python_profile.py"
    node = next(x for x in ast.parse(path.read_bytes()).body
                if isinstance(x, ast.FunctionDef) and x.name == "python_profile_options")
    returned = next(x.value for x in node.body if isinstance(x, ast.Return))
    values = dict(n4_backend=n4_backend, n4_execution=n4_execution, topology_backend=topology_backend)
    result = {ast.literal_eval(k): values[v.id] if isinstance(v, ast.Name) else ast.literal_eval(v)
              for k, v in zip(returned.keys, returned.values)}
    if len(result) != 21:
        raise ValueError("profile key count changed")
    return result


def collect(*, repository, source, output):
    """Git源码快照、SHA清单及历史公开收据→本轮静态JSON/中文表。

    repository只提供旧审计函数、结构和历史收据；全部当前函数从source
    重新读字节。source/SOURCE_MANIFEST.json须列明Git祖先与显式overlay。
    output必须不存在。不运行模型、能力探测、原程序、CUDA或依赖版本探测。
    手工分支审核和AST定位不是形式化可达性证明或动态库/OS隔离验证。
    """
    if output.exists():
        raise FileExistsError(output)
    raw = (source / "SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(raw)
    archive = subprocess.check_output(["git", "archive", manifest["git_commit"], "src"], cwd=repository)
    if sha(archive) != manifest["archive_sha256"] or len(archive) != manifest["archive_bytes"]:
        raise ValueError("declared Git archive identity differs")
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        members = {item.name: bundle.extractfile(item).read() for item in bundle if item.isfile()}
    if set(members) != set(manifest["files"]) or not set(manifest["overlays"]).issubset(members):
        raise ValueError("source manifest does not cover the declared complete Git src tree")
    for relative, item in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe source manifest path")
        data = (source / relative).read_bytes()
        if len(data) != item["bytes"] or sha(data) != item["sha256"]:
            raise ValueError("source snapshot SHA/size mismatch: " + relative)
        if relative in manifest["overlays"]:
            overlay = manifest["overlays"][relative]
            if (sha(members[relative]) != overlay["git_sha256"] or sha(data) != overlay["captured_sha256"]
                    or len(data) != overlay["bytes"]):
                raise ValueError("declared overlay does not bind old/new bytes")
        elif members[relative] != data:
            raise ValueError("undeclared source change relative to Git: " + relative)
    helper, reused = load_helper(repository)
    previous = repository / "validation/recon_all/optimizations/20261010_python_profile_static/static_audit.json"
    template = json.loads(previous.read_bytes())
    ref = lambda name, function=None: helper.function_reference(
        repository=source, path="src/fnit/" + name + ".py", function=function)
    stages = deepcopy(template["stages"])
    excluded = deepcopy(template["unselected_local_CPP_routes"])
    for row in [*stages, *excluded]:
        row["evidence"] = [helper.function_reference(repository=source,
            path=e["path"], function=e.get("function")) for e in row["evidence"]]
    for row in stages:
        row["audit_status"] = "current entry and selected guard reviewed; current whole not executed"
        if row["name"] == "tessellation/topology":
            row.update(implementation="自有数字拓扑，或显式完整有序GA→SOAP/压缩→remesh3/source_numba清理",
                selected_guard="python_profile默认digital；显式python-ga调用packaged adapter；root本次overlay传face_detection_cache=True；原程序None")
            row["evidence"].extend([ref("recon_all/native_free", "_run_and_publish_genetic_topology"),
                ref("recon_all/topology_genetic_files", "run_genetic_topology_files"),
                ref("recon_all/topology_genetic_repair", "repair_genetic_topology"),
                ref("recon_all/topology_genetic_search", "prepare_genetic_defect_evaluator"),
                ref("recon_all/topology_face_detection_cache", "detect_patch_faces_cached")])
        elif row["name"] == "N4/nu":
            row["selected_guard"] = "默认SimpleITK fitting_threads=1；显式完整Torch/in-process或isolated；不覆盖用户Torch为SimpleITK"
            row["evidence"].extend([ref("recon_all/n4_itk_torch_experimental", "correct_volume"),
                                     ref("recon_all/n4_torch_worker", "run_isolated_n4")])
        elif row["name"] == "WM aseg/pretess/Ento-ACJ":
            row["selected_guard"] = "wm_edit_backend=python；完整typed/byte有序路径，4byte原dtype/safe geometry guard；无subject whitelist或native回调"
            row["evidence"].append(ref("recon_all/wm_remove_paths", "remove_mtl_paths"))
        elif row["name"] == "两轮normalize/brainmask":
            row["selected_guard"] = "两轮device=CUDA；controls=torch；第二轮initial_bias=torch；有序ridge/反馈仍CPU"
    extras = [ref("recon_all/" + name, function) for name, function in (
        ("native_free", "run_recon_all_python"), ("batch", "run_recon_all_python_batch"),
        ("expected_outputs", "paths"), ("assets", "validate_core_assets"),
        ("assets", "download_asset"), ("hemisphere_parallel", "run_hemisphere_group"),
        ("topology_candidate_cycle", "materialize_candidate_state"),
        ("topology_candidate_state_cache", "materialize_candidate_state_cached"),
        ("anatomical_stats_rows", "anatomical_stats_rows"),
        ("anatomical_stats_global", "anatomical_stats_global_lines"),
        ("mni_warp_inverse", "fill_inverse_fields"))]
    routes = [
        ("N4", ["fnit_n4_itk"], "n4_backend=native", "执行", "input_n4_chain", "run_n4_stage", "SimpleITK完整CPU/Torch完整拟合"),
        ("WM segmentation", ["mri_segment"], "wm_backend=native", "执行", "native_free", "_run_torch_wm_segment", "Numba有序核心+Torch histogram/planar"),
        ("WM aseg edit", ["mri_edit_wm_with_aseg"], "wm_edit_backend=native", "执行", "native_free", "_run_python_wm_edit", "完整Numba/CPU编辑；hybrid另可显式选择"),
        ("Topology", ["mris_fix_topology_fnit"], "topology_backend=native", "执行", "native_free", "_prepare_native_topology", "digital或显式完整自有GA；两者算法不同"),
        ("nofix/inflated", ["mris_inflate"], "surface_placement=native或inflate=native", "执行", "native_free", "_run_accurate_sphere_pair", "自有Torch inflation/CPU动态顺序"),
        ("white.preaparc", ["mris_place_surface"], "surface_placement_backend=native", "执行", "native_free", "_place_preaparc_and_smooth", "完整Python/Numba/Torch候选与清理"),
        ("white/pial", ["mris_place_surface"], "surface_placement_backend=native", "执行", "native_free", "_finish_cortical_surface", "完整四轮自有放置"),
        ("avg_curv", ["mrisp_paint"], "surface_placement_backend=native", "执行", "native_free", "_run_avg_curv", "自有采样及有序marked平均"),
        ("curv.stats", ["mris_curvature_stats"], "python_only=False", "执行", "native_free", "_run_curvature_stats", "自有Torch离散曲率+Numba/system libm"),
        ("defects volume", ["mri_label2vol"], "defects_backend=native", "执行", "native_free", "_run_defects_volume", "Torch raw原域投射，LH→RH共享卷"),
        ("GCA", ["mri_em_register"], "optimizations=original或CPUauto", "默认CUDA auto已走自有Torch；CPUauto有能力探测", "native_free", "_run_native_em_register", "自有EM CPU+Torch候选评分/求逆"),
        ("vertex metrics", ["mris_place_surface"], "device非CUDA且python_only=False", "默认CUDA自有指标；native仍发现该二进制供放置使用", "native_free", "_run_surface_metrics", "自有area/thickness/curvature"),
        ("MNI postprocess", ["mri_warp_convert", "mri_ca_register", "mri_convert"], "device非CUDA原对照分支", "默认CUDA走自有postprocess", "mni_nonlinear_chain", "run_mni_nonlinear_chain", "自有GPU转换/逆场/采样check"),
    ]
    inventory = [dict(stage=stage, original_programs=programs, original_call_guard=guard,
        native_default_cuda_behavior=behavior, python_selected_original=False,
        own_alternative=own, evidence=ref("recon_all/"+name, function))
        for stage, programs, guard, behavior, name, function, own in routes]
    files = sorted({e["path"] for row in [*stages, *excluded] for e in row["evidence"]} |
                   {e["path"] for e in extras})
    # Syntax is an index for manual branch review, never a runtime attempt count.
    sensitive = []
    for path in files:
        text = (source / path).read_text()
        for number, line in enumerate(text.splitlines(), 1):
            if any(token in line for token in ("subprocess.", "ctypes.", "_cpu_eigen", "_average_cpu_cpp", "cpu_columns", "float16", "bfloat16")):
                sensitive.append(dict(path=path, line=number, text=line.strip()))
    historical = helper.historical_execution_audit(repository=repository)
    # Execute only the standalone literal/tuple output declaration, with imports forbidden.
    declarations = ast.parse((source / "src/fnit/recon_all/expected_outputs.py").read_bytes())
    if any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(declarations)):
        raise ValueError("expected outputs ceased to be a standalone declaration")
    namespace = {}; exec(compile(declarations, "expected_outputs.py", "exec"), namespace)
    output_paths = namespace["paths"]()
    if len(output_paths) != 138 or len(set(output_paths)) != 138:
        raise ValueError("expected outputs changed")
    stage_evidence_paths = [
        "20261010_n4_simpleitk/summary.json", "20261011_wm_remove_paths/summary.json",
        "20261011_wm_remove_paths/ten_case_roi_scan.json",
        "20261010_wm_native_independent/reports/wm_segment_sub06_v2.json",
        "20261010_wm_native_independent/reports/wm_segment_sub07_v2.json",
        "20261011_topology_genetic_face_cache_wiring/metrics.json"]
    evidence = []
    for path in stage_evidence_paths:
        full = repository / "validation/recon_all/optimizations" / path
        data = full.read_bytes(); json.loads(data)
        evidence.append(dict(path=str(full.relative_to(repository)), published_sha256=sha(data), bytes=len(data),
            scope="existing bound stage result, not a new current-code MRI run"))
    n4 = json.loads((repository / "validation/recon_all/optimizations/20261010_n4_simpleitk/summary.json").read_bytes())
    wm = json.loads((repository / "validation/recon_all/optimizations/20261011_wm_remove_paths/summary.json").read_bytes())
    measured = dict(n4={case: dict(
        uint8_different_voxels=row["file"]["comparisons"]["native_quantized"]["different_voxels"],
        maximum_absolute_error=row["file"]["comparisons"]["native_quantized"]["maximum_absolute_error"],
        p99_absolute_error=row["file"]["comparisons"]["native_quantized"]["p99_absolute_error"],
        code_version=row["file"]["code_version"], module_sha256=row["file"]["module_sha256"])
        for case,row in n4["cases"].items()},
        wm_edit={case: {name: {key: comparison[key] for key in
            ("different_voxels", "maximum_absolute_error", "p99_absolute_error", "wm_foreground_dice", "raw_bytes_equal")}
            for name,comparison in row["comparisons"].items()}
            for case,row in wm["cases"].items()},
        scope="selected exact numeric leaves from bound old stage reports; not current-code benchmark")
    dependency_files = {}
    dependency_raw = {}
    for relative in ("pyproject.toml", "environment.yml"):
        data = subprocess.check_output(["git","show",manifest["git_commit"]+":"+relative],cwd=repository)
        dependency_files[relative] = dict(sha256=sha(data),bytes=len(data))
        dependency_raw[relative]=data
    project = tomllib.loads(dependency_raw["pyproject.toml"].decode())["project"]
    dependency_binding = dict(files=dependency_files, python_requires=project["requires-python"],
        required_python_distributions=project["dependencies"],
        conda_selected_versions=[line.strip()[2:] for line in dependency_raw["environment.yml"].decode().splitlines()
            if line.strip().startswith("- ") and any(token in line for token in
                ("python=3", "pytorch=", "pytorch-cuda=", "numpy", "scipy", "numba", "nibabel", "SimpleITK", "triton", "nvidia-ml-py"))],
        installer_extensions="setuptools may build project MSM extension for other functions; not an FS recon-all stage invocation",
        installed_environment_versions="not_probed_by_this_static_task",new_cache_dependency=[])
    report = dict(schema_version=1, scope="manual static selected-path audit; no current MRI execution",
        frozen_source=dict(git_commit=manifest["git_commit"], source_manifest_sha256=sha(raw),
            verified_source_files=len(manifest["files"]), archive_sha256=manifest["archive_sha256"],
            archive_bytes=manifest["archive_bytes"], overlays=manifest["overlays"],
            protected_retessellation=manifest["protected_retessellation"]),
        audit_tool_sha256=sha(Path(__file__).read_bytes()), reused_helper=reused,
        template=dict(path=str(previous.relative_to(repository)), sha256=sha(previous.read_bytes()),
            reuse="24 group structure only; all function bytes recaptured; topology/N4/WM guards refreshed"),
        profile_default=profile_from_ast(source,n4_backend="simpleitk",n4_execution="in-process",topology_backend="python"),
        profile_explicit_ga=profile_from_ast(source,n4_backend="simpleitk",n4_execution="in-process",topology_backend="python-ga"),
        profile_explicit_torch_n4=profile_from_ast(source,n4_backend="torch",n4_execution="isolated",topology_backend="python-ga"),
        stages=stages, original_program_conditional_inventory=inventory,
        unselected_local_CPP_routes=excluded, extra_current_function_references=extras,
        syntax_review_index=sensitive, original_stage_selected_static=[],
        current_runtime_attempts=None, current_shared_library_runtime_attempts=None,
        historical_execution_audit=historical, existing_stage_evidence=evidence,
        selected_existing_stage_measurements=measured, dependency_binding=dependency_binding,
        all_outputs=dict(expected_count=138,unique_count=138,relative_paths=output_paths,
            validation="presence after mesh/MNI join; numeric agreement and mesh quality are separate gates",
            current_execution_verified=False),
        cache_review=dict(face_cache="per-defect immutable base; dynamic neighbors and added faces remain per-candidate",
            state_cache="optional state_context None on this route; own NumPy materialization; no original binary discovery",
            root_overlay_face_cache=True, adapter_core_default_face_cache=False,
            added_dependency=[], new_half_precision_selected=False),
        asset_boundary=dict(runtime="explicit assets/weights directories; sizes and SHA verified; no installer called",
            installer="download_asset may copy verified distribution assets or use licensed Release/upstream; does not compile/execute a stage",
            license="no FS program or FS license read selected in the reviewed Python route; each external asset retains separate terms"),
        generic_compiled_libraries_allowed=["SimpleITK/ITK", "PyTorch/CUDA/Triton", "NumPy/SciPy", "Numba/system libm"],
        boundaries=dict(current_runtime="not_run",current_whole="not_run",clean_OS_isolation="not_verified",
            shared_library_or_file_access="not_audited",whole_equivalence="not_assessed",
            precision_time_memory_benchmark="not_run",formal_reachability_proof=False),
        priority=["完成当前python-ga+cacheTrue全部缺陷→final_orig严格门及新的完整生产链验收；不用重写已有同名算法",
            "WM remove_paths真实active输入仍未找到；安全guard与非终止失败契约不等于该分支原生成功等价",
            "N4保留完整SITK/Torch显式选择和已测下游差异；不因量化±1声称无影响或切默认"],
        new_production_leakage_bug_found=False)
    output.mkdir(parents=True)
    (output/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    bindings = {path: manifest["files"][path] for path in files}
    (output/"reviewed_source.json").write_text(json.dumps(dict(git_commit=manifest["git_commit"],
        overlays=manifest["overlays"],files=bindings),ensure_ascii=False,indent=2)+"\n")
    table = "| 阶段 | 原版程序 | native缺省CUDA实际分支 | 独立Python实现 |\n| --- | --- | --- | --- |\n"
    for row in inventory:
        e=row["evidence"]
        table += f"| {row['stage']} | {', '.join(row['original_programs'])} | {row['native_default_cuda_behavior']} | {row['own_alternative']}；`{e['path']}:{e['line']}` |\n"
    (output/"migration_table.md").write_text("# 当前条件调用清单\n\n"+table+"\n上述原程序在本轮独立Python配方中均未选中；静态判断不代表实测执行次数为零。\n")
    return dict(stages=len(stages),function_references=sum(len(x["evidence"]) for x in stages)+len(extras),
        conditional_original_stage_groups=len(inventory), historical_events={k:sum(v["events"].values()) for k,v in historical["cases"].items()},
        source_files_reviewed=len(files), current_MRI_run=False, report_sha256=sha((output/"audit.json").read_bytes()))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "source", "output"):
        parser.add_argument("--"+name,required=True,type=Path)
    print(json.dumps(collect(**vars(parser.parse_args())),ensure_ascii=False))
