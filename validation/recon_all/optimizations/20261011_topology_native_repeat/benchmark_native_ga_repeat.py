"""隔离原生全GA重复性：复用冻结79 wrapper；禁止进入生产调用链。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time


EXPECTED_INPUTS = {
    "surf/lh.orig.nofix": "cfa9c1d0f5681a87d4ae709d0b093e20fdc3466ea6525b7fc94ba61a4708b0f9",
    "surf/lh.inflated.nofix": "176ceabdc00801876dbed069f0d6ec88ca9159b338415541308a8797a11d027b",
    "surf/lh.qsphere.nofix": "cf3c1e339c9d36c5245cb7d75cc11a4bd174ee3267e2f41ce8867061e9eed952",
    "mri/brain.mgz": "977f6fb298d38e10588cd636f4db38ca8c3456265b24d6d80d00d54e60e98eed",
    "mri/wm.mgz": "859a3551807921395a516466c406d6816e79ba7813c016dc97e07b326ae8953c",
}
EXPECTED_SOURCE = {
    "native_free.py": "0e3c09e3a88f746b858ac2d21167b4dffef308062731e024d6f3fb91236d05a2",
    "topology_conda_ga.py": "285d781f1514c394b8ee9141ffad5775557b407e2753a1a8691ca8668e8463d0",
    "topology_preflight_python.py": "7ae02e2c7e5133a9de544361c95b9e7def7f5d92e35f3bbc27e8ca03f8d5ec00",
}
EXPECTED_PROGRAM = "c3bb6b744f3d3ebcc0bbaf6a79b4266eb712ea51a113b06157456b560654fde6"
EXPECTED_BENCHMARK = "6de395039495083787474fd91f1b107701af74dd89fe1e489c484e3ca279060c"
EXPECTED_RUNTIME = "eabdf28d09f37ed2b85bd26082f76dd4161665169de916d4565f43f0a55d8c1e"
EXPECTED_PREMESH = "4f50b9f28e8774e28a22ce5ec4d83fa9eaca071ff093bdd6da530d0f644592c0"
EXPECTED_COMPARISON = "ea8e7faf842c8f4039a50bc3097eb2bd77752b84355f1f76134a6bccdd3a31ba"
EXPECTED_LOG = "e2d677205a4cf136b1b17b239e24fd3edb7c46fb7af3a169b3ee5d9512b0d8f6"


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def search_behaviour(log: Path) -> dict:
    """保留原日志可见的有序缺陷/Euler/最终搜索计数；不编造逐候选trace。"""
    rows = log.read_text(errors="strict").splitlines()
    selected = [line for line in rows if any(text in line for text in (
        "setting seed for random", "number of patches/generation", "number of generations",
        "before topology correction", "defects found,", "defects to be corrected",
        "CORRECTING DEFECT", "After retessellation of defect", " mutations (",
        "after topology correction"))]
    defects = []
    for line in selected:
        match = re.fullmatch(r"CORRECTING DEFECT (\d+) \(vertices=(\d+), convex hull=(\d+), v0=(\d+)\)", line)
        if match:
            number, vertices, hull, first = map(int, match.groups())
            defects.append(dict(number=number, vertices=vertices, convex_hull=hull, first_vertex=first))
    return dict(log_sha256=sha(log), ordered_behaviour_lines=selected, defects=defects,
                per_candidate_fitness_trace="not_in_reference_log")


def preflight(*, source: Path, subject: Path, benchmark: Path, runtime: Path,
              reference_log: Path, binary: Path, assets: Path,
              comparison_module: Path) -> dict:
    """只读冻结源码/公开输入/独立程序SHA；许可证只检查env文件可读性。"""
    assert subject.name == "subject"
    source_description = json.loads((source / "source.json").read_bytes())
    assert source_description["git_commit"] == "79a41cdda2397525a8952dd5af303372b352916c"
    assert sha(source / "source.tar") == source_description["source_archive_sha256"]
    assert source_description["source_archive_sha256"] == "e0a9a418669035f39f979e25cfb4b65e8cc93e200485eae7d93a2d22ddcfcaec"
    assert sha(benchmark) == EXPECTED_BENCHMARK and sha(runtime) == EXPECTED_RUNTIME
    old_benchmark = json.loads(benchmark.read_bytes())
    old_runtime = json.loads(runtime.read_bytes())
    captured_sources = {}
    for name, expected in EXPECTED_SOURCE.items():
        path = source / "src/fnit/recon_all" / name
        actual = sha(path)
        assert actual == expected == old_benchmark["source_sha256"]["recon_all/" + name]
        captured_sources[name] = actual
    assert sha(binary) == EXPECTED_PROGRAM == old_runtime["topology_repair"]["sha256"]
    assert old_benchmark["native_program_sha256"]["mris_fix_topology_fnit"] == EXPECTED_PROGRAM
    assert binary.name == "mris_fix_topology_fnit" and os.access(binary, os.X_OK)
    assert assets.is_dir() and sha(comparison_module) == EXPECTED_COMPARISON
    inputs = {}
    for name, expected in EXPECTED_INPUTS.items():
        path = subject / name
        assert sha(path) == expected
        inputs[name] = dict(sha256=expected, bytes=path.stat().st_size)
    assert sha(subject / "surf/lh.orig.premesh") == EXPECTED_PREMESH
    assert sha(reference_log) == EXPECTED_LOG
    license_file = os.environ.get("FS_LICENSE")
    assert license_file and os.access(license_file, os.R_OK), "private license not readable"
    assert list(os.sched_getaffinity(0)) == [100, 101, 102, 103]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    return dict(status="passed", scope="isolated reference only; no original stage executed in preflight",
        case="ds000114/sub07/ses-test/lh", source=source_description,
        benchmark_sha256=sha(benchmark), producer_runtime_sha256=sha(runtime),
        frozen_source_sha256=captured_sources, program_sha256=sha(binary),
        comparison_module_sha256=sha(comparison_module), inputs=inputs,
        reference_premesh_sha256=EXPECTED_PREMESH, reference_search_behaviour=search_behaviour(reference_log),
        private_license_readable=True, cpu_affinity=list(os.sched_getaffinity(0)),
        total_cpu_budget=4, actual_native_threads=1, seed=1234,
        cuda_visible_devices="", new_GPU_execution=False,
        script_sha256=sha(Path(__file__)), overall_metric_equivalence="not_assessed")


def run(*, source: Path, subject: Path, benchmark: Path, runtime: Path,
        reference_log: Path, binary: Path, assets: Path,
        comparison_module: Path, output: Path, phase: str) -> dict:
    """五张原始阶段输入→单侧原生premesh及完整同输入重复诊断。

    全路径为明确具名参数；source是只读79冻结源码，subject是旧自产参考目录，
    benchmark/runtime/log仅用于绑定和事后比较，binary仅独立benchmark调用。
    assets为既有data-only根，comparison_module复用既有全表面比较器。
    output为任务独有新目录，phase为preflight/run。固定LH、4核预算/native1核、
    seed1234、原sphere centering/full-GA；不运行remesh、后续orig或整例。
    同索引表面坐标surface RAS/mm，faces整数(F,3)。输入版本或策略不符失败。
    """
    started = time.perf_counter()
    check = preflight(source=source, subject=subject, benchmark=benchmark, runtime=runtime,
        reference_log=reference_log, binary=binary, assets=assets, comparison_module=comparison_module)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    if phase == "preflight":
        assert not (output / "preflight.json").exists()
        write(output / "preflight.json", check)
        return check
    assert (output / "preflight.json").is_file()
    assert not (output / "candidate").exists()
    copy_started = time.perf_counter()
    candidate = output / "candidate/subject"
    candidate.mkdir(mode=0o700, parents=True)
    for name, expected in EXPECTED_INPUTS.items():
        destination = candidate / name
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(subject / name, destination)
        os.chmod(destination, 0o600)
        assert sha(destination) == expected
    copy_seconds = time.perf_counter() - copy_started
    load_at = time.perf_counter()
    sys.path.insert(0, str(source / "src"))
    from fnit.recon_all.topology_conda_ga import run_topology_ga_conda
    import_seconds = time.perf_counter() - load_at
    stage_at = time.perf_counter()
    wrapper_result = run_topology_ga_conda(candidate, "lh", binary, assets)
    stage_seconds = time.perf_counter() - stage_at
    original_call = dict(wrapper_result)
    # 路径只存在私有运行收据；公开结果仅保留命令参数与相对产物名。
    write(output / "wrapper_result.private.json", original_call)
    result_path = Path(wrapper_result["output"])
    spec = importlib.util.spec_from_file_location("_existing_surface_comparison", comparison_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    surface = module._surface(subject / "surf/lh.orig.premesh", result_path)
    decoded_exact = (surface["coordinates_mm"].get("exact", -1) == surface["coordinates_mm"].get("elements", -2) and
                     surface["ordered_faces"].get("exact", -1) == surface["ordered_faces"].get("elements", -2))
    new_behaviour = search_behaviour(Path(wrapper_result["log"]))
    for name, expected in EXPECTED_INPUTS.items():
        assert sha(subject / name) == expected
        assert sha(candidate / name) == expected
    assert sha(result_path).isalnum() and sha(binary) == EXPECTED_PROGRAM
    report = dict(status="complete", phase="isolated complete native GA repeat, before remesh",
        full_workflow_complete=False, preflight=check, inputs_after_unchanged=True,
        source_import_seconds=import_seconds, input_copy_and_verify_seconds=copy_seconds,
        native_seconds=wrapper_result["native_seconds"], wrapper_wall_seconds=stage_seconds,
        native_command_arguments=wrapper_result["command"][1:-2] + ["subject", "lh"],
        candidate_premesh_sha256=sha(result_path), surface_comparison=surface,
        strict_decoded_reproduction=bool(decoded_exact and surface["volume_geometry_exact"]),
        strict_file_reproduction=sha(result_path) == EXPECTED_PREMESH,
        search_behaviour=new_behaviour,
        ordered_log_behaviour_equal=(new_behaviour["ordered_behaviour_lines"] ==
                                     check["reference_search_behaviour"]["ordered_behaviour_lines"]),
        process_load_average=list(os.getloadavg()),
        child_max_rss_bytes=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024),
        comparison_source_sha256=sha(comparison_module),
        cuda_memory_bytes=None, shared_load_scope="same host, different time; not a speed pair",
        whole_equivalence="not_assessed", full_diagnostic_wall_seconds=time.perf_counter()-started)
    write(output / "report.json", report)
    write(output / "COMPLETE.json", dict(status="complete", report_sha256=sha(output / "report.json"),
        full_workflow_complete=False, strict_decoded_reproduction=report["strict_decoded_reproduction"],
        strict_file_reproduction=report["strict_file_reproduction"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "subject", "benchmark", "runtime", "reference-log", "binary", "assets", "comparison-module", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "run"), required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result.get(key) for key in ("status", "scope", "strict_decoded_reproduction", "strict_file_reproduction")}, ensure_ascii=False))
