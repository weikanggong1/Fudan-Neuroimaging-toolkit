"""校验实际安装收据，按固定类型与枚举导出版本、数值和哈希。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re


PACKAGE_PATHS = tuple("fnit/recon_all/" + name for name in (
    "native_free.py", "python_profile.py", "batch.py", "hemisphere_parallel.py",
    "topology_genetic_files.py", "topology_genetic_repair.py", "topology_genetic_search.py",
    "topology_retessellation_python.py", "topology_retessellation_dynamic.py",
    "topology_edge_blocker_scan.py", "input_n4_chain.py", "n4_simpleitk.py"))
SOURCE_OVERLAY = ["src/fnit/recon_all/" + name for name in (
    "native_free.py", "topology_edge_blocker_scan.py", "topology_retessellation_dynamic.py",
    "topology_genetic_search.py", "topology_genetic_repair.py", "topology_genetic_files.py")]
PROFILE = dict(native_optimizations="torch", n4_backend="simpleitk", n4_execution="in-process",
    wm_backend="torch-optimized", gca_execution="isolated", gca_inverse_backend="torch",
    gca_candidate_chunk=1024, normalization_controls_backend="torch",
    normalization_initial_bias_backend="torch", sphere_finish_backend="torch",
    mni_execution="parallel-late", wm_execution="isolated", wm_edit_backend="python",
    defects_backend="torch", inflate_backend="torch", surface_placement_backend="python",
    topology_backend="python-ga", annotation_gibbs_backend="numba", remesh_scalar_storage="python",
    fill_backend="numba", hemisphere_workers=2)
CLI_OPTIONS = {
    "fnit.recon_all.native_free": ["python-ga", "--topology-backend", "--backend"],
    "fnit.recon_all.topology_genetic_files": ["--final-orig", "--remesh-scalar-storage", "--threads", "--blocker-scan-backend"],
}
ACTUAL_KEYS = {"git_base", "source_overlay", "source_manifest_sha256", "wheel", "status",
    "package_checks", "installation", "entry", "total_seconds_including_install_and_entry",
    "verification_script_sha256", "entry_script_sha256", "python_version", "cpu_affinity",
    "runtime_gpu_precision", "whole_recon_all", "clean_conda_full_reconstruction", "gpu_memory_bytes", "scope"}
EXPECTED_KEYS = {"status", "scope", "source_manifest_sha256", "source_file_count", "git_base",
    "source_overlay", "wheel", "package_checks", "build", "install", "entry", "entry_checks",
    "verification_script_sha256", "dependency_installation", "whole_recon_all",
    "clean_conda_full_reconstruction", "runtime_gpu_precision", "gpu_memory_bytes"}
ENTRY_KEYS = {"status", "ga_library_blocker_scan_backend_default", "selected_profile",
    "n4_runtime_dependencies", "precision_and_cuda_at_entry",
    "precision_and_cuda_after_import_and_profile_resolution", "python_version", "torch_version", "cli"}
PRECISION_KEYS = ("cuda_initialized", "matmul_allow_tf32", "cudnn_allow_tf32")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_pairs)


def _keys(value, declared):
    if type(value) is not dict or set(value) != set(declared):
        raise ValueError("record keys differ from fixed schema")


def _sha(value, length=64):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is None:
        raise ValueError("full lowercase hash required")
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("integer metric required")
    return value


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative seconds required")
    return value


def _version(value):
    if type(value) is not str or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}(?:\+[A-Za-z0-9.]+)?", value) is None:
        raise ValueError("numeric package version required")
    return value


def _constant(value, expected):
    if type(value) is not type(expected) or value != expected:
        raise ValueError("declared enumeration differs")
    return value


def _process(value, *, has_checks=False):
    _keys(value, {"exit_code", "seconds_including_io", "checks"} if has_checks else {"exit_code", "seconds_including_io"})
    _constant(value["exit_code"], 0)
    return dict(exit_code=value["exit_code"], seconds_including_io=_seconds(value["seconds_including_io"]))


def _wheel(value):
    _keys(value, {"name", "bytes", "sha256"})
    name = value["name"]
    if type(name) is not str or re.fullmatch(r"fudan_neuroimaging_toolkit-[0-9]+(?:\.[0-9]+){1,3}-py3-none-any\.whl", name) is None:
        raise ValueError("FNIT wheel basename required")
    return dict(name=name, bytes=_integer(value["bytes"], 1), sha256=_sha(value["sha256"]))


def _packages(value):
    if type(value) is not list or len(value) != 12:
        raise ValueError("twelve installed module checks required on both sides")
    selected = []
    for row, path in zip(value, PACKAGE_PATHS):
        _keys(row, {"path", "source_sha256", "wheel_sha256", "installed_sha256"})
        _constant(row["path"], path)
        hashes = {key: _sha(row[key]) for key in ("source_sha256", "wheel_sha256", "installed_sha256")}
        if len(set(hashes.values())) != 1:
            raise ValueError("source/wheel/installed module hashes differ")
        selected.append(dict(path=path, **hashes))
    return selected


def _profile(value):
    _keys(value, PROFILE)
    return {key: _constant(value[key], expected) for key, expected in PROFILE.items()}


def _dependencies(value):
    _keys(value, {"backend", "required", "simpleitk_version", "itk_version", "library_sha256"})
    return dict(backend=_constant(value["backend"], "simpleitk"), required=_constant(value["required"], True),
        simpleitk_version=_version(value["simpleitk_version"]), itk_version=_version(value["itk_version"]),
        library_sha256=_sha(value["library_sha256"]))


def _precision(value):
    _keys(value, PRECISION_KEYS)
    if any(type(value[key]) is not bool for key in PRECISION_KEYS) or value["cuda_initialized"] is not False:
        raise ValueError("uninitialized CUDA and boolean import precision required")
    return {key: value[key] for key in PRECISION_KEYS}


def _cli(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError("two declared fresh CLI checks required on both sides")
    result = []
    for row, (module, options) in zip(value, CLI_OPTIONS.items()):
        _keys(row, {"module", "exit_code", "verified_options", "output_sha256", "fresh_cli_seconds_including_imports"})
        _constant(row["module"], module)
        _constant(row["exit_code"], 0)
        _constant(row["verified_options"], options)
        result.append(dict(module=module, exit_code=row["exit_code"], verified_options=options,
            output_sha256=_sha(row["output_sha256"]),
            fresh_cli_seconds_including_imports=_seconds(row["fresh_cli_seconds_including_imports"])))
    return result


def _entry(value):
    _keys(value, ENTRY_KEYS)
    result = dict(status=_constant(value["status"], "passed"),
        ga_library_blocker_scan_backend_default=_constant(value["ga_library_blocker_scan_backend_default"], "python"),
        selected_profile=_profile(value["selected_profile"]),
        n4_runtime_dependencies=_dependencies(value["n4_runtime_dependencies"]),
        precision_and_cuda_at_entry=_precision(value["precision_and_cuda_at_entry"]),
        precision_and_cuda_after_import_and_profile_resolution=_precision(value["precision_and_cuda_after_import_and_profile_resolution"]),
        python_version=_version(value["python_version"]), torch_version=_version(value["torch_version"]), cli=_cli(value["cli"]))
    if result["precision_and_cuda_at_entry"] != result["precision_and_cuda_after_import_and_profile_resolution"]:
        raise ValueError("import changed precision or CUDA initialization")
    return result


def _unmeasured(value):
    for key in ("runtime_gpu_precision", "whole_recon_all", "clean_conda_full_reconstruction"):
        _constant(value[key], "not_assessed")
    if value["gpu_memory_bytes"] is not None:
        raise ValueError("GPU memory was not measured")


def export_installation(*, run_directory: Path, expected_report: Path, output_file: Path) -> dict:
    """真实安装收据→固定schema公共JSON；不读取影像或运行模型。

    三个必填Path分别为含report/COMPLETE的实际目录、冻结本地安装报告、
    尚不存在的新输出文件。SHA绑定读取的原字节；所有秒数保持原数字
    类型和值，显存未测保持null。固定12模块/6overlay/21profile/2CLI、
    N4与精度字段及整数/bool类型，拒绝未知键、重复JSON键与退化完成门。
    原始报告/日志/地址不发布；返回并写report dict及原报告SHA。读取失败、
    来源/键/哈希/枚举/数值不符或已有输出均抛异常，不写公共结果；原数据
    不变。对应原软件无独立CLI，七节输入输出/中文具名示例见README。
    """
    if output_file.exists():
        raise FileExistsError(output_file)
    data = (run_directory / "report.json").read_bytes()
    original_sha = hashlib.sha256(data).hexdigest()
    complete = _read_json(run_directory / "COMPLETE.json")
    _keys(complete, {"status", "report_sha256"})
    _constant(complete["status"], "complete")
    if _sha(complete["report_sha256"]) != original_sha:
        raise ValueError("completed report binding required")
    actual = json.loads(data, object_pairs_hook=_unique_pairs)
    expected = _read_json(expected_report)
    _keys(actual, ACTUAL_KEYS)
    _keys(expected, EXPECTED_KEYS)
    _constant(actual["status"], "passed")
    _constant(expected["status"], "passed")
    _constant(actual["scope"], "CFFF existing dependencies; independent wheel target/import/fresh CLI only")
    _constant(expected["scope"], "offline wheel/import/CLI; existing dependencies")
    _constant(expected["dependency_installation"], "existing environment; --no-deps --no-index")
    _integer(expected["source_file_count"], 1)
    _sha(expected["verification_script_sha256"])
    for value in (actual, expected):
        _unmeasured(value)
        _constant(value["source_overlay"], SOURCE_OVERLAY)
        _sha(value["git_base"], 40)
        _sha(value["source_manifest_sha256"])
    safe = {}
    for key in ("git_base", "source_overlay", "source_manifest_sha256", "wheel"):
        left = _wheel(actual[key]) if key == "wheel" else actual[key]
        right = _wheel(expected[key]) if key == "wheel" else expected[key]
        if left != right:
            raise ValueError("frozen source/wheel differs")
        safe[key] = right
    safe["status"] = "passed"
    actual_packages, expected_packages = _packages(actual["package_checks"]), _packages(expected["package_checks"])
    if actual_packages != expected_packages:
        raise ValueError("source/wheel/installed module differs")
    safe["package_checks"] = expected_packages
    for key in ("build", "install", "entry"):
        _process(expected[key])
    safe["installation"] = _process(actual["installation"])
    safe["entry"] = _process(actual["entry"], has_checks=True)
    observed, local = _entry(actual["entry"]["checks"]), _entry(expected["entry_checks"])
    for key in ENTRY_KEYS - {"python_version", "torch_version", "cli"}:
        if observed[key] != local[key]:
            raise ValueError("runtime entry differs")
    for checked, frozen in zip(observed["cli"], local["cli"]):
        if any(checked[key] != frozen[key] for key in ("module", "exit_code", "verified_options", "output_sha256")):
            raise ValueError("actual CLI differs")
    safe["entry"]["checks"] = observed
    safe["total_seconds_including_install_and_entry"] = _seconds(actual["total_seconds_including_install_and_entry"])
    for key in ("verification_script_sha256", "entry_script_sha256"):
        safe[key] = _sha(actual[key])
    if _version(actual["python_version"]) != observed["python_version"]:
        raise ValueError("interpreter version differs")
    safe["python_version"] = observed["python_version"]
    affinity = actual["cpu_affinity"]
    if type(affinity) is not list or not affinity or len(set(affinity)) != len(affinity):
        raise ValueError("nonempty unique CPU affinity required")
    safe["cpu_affinity"] = [_integer(value) for value in affinity]
    for key in ("runtime_gpu_precision", "whole_recon_all", "clean_conda_full_reconstruction"):
        safe[key] = "not_assessed"
    safe["gpu_memory_bytes"] = None
    safe["scope"] = "independent wheel/import/fresh CLI; existing dependencies; no MRI/GPU"
    result = {"report": safe, "private_report_sha256": original_sha}
    output_file.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_directory", "expected_report", "output_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    export_installation(**vars(parser.parse_args()))
