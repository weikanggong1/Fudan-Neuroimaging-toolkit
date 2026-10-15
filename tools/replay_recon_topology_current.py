"""从冻结 FNIT 阶段输入重放完整拓扑文件接口，并私有保存清理坐标。

仅供同输入诊断。输出须为新目录；读取自产 nofix、qsphere、brain、wm，
不读取官方参考或补写失败整例。计时包含导入后的完整文件接口与诊断写出，
不等同从原始 T1 开始的整例墙钟。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from fnit.recon_all import topology_genetic_files as files
from fnit.recon_all import topology_genetic_repair as genetic
from fnit.recon_all.topology_distance_policy import validate_distance_policy
from fnit.recon_all.topology_genetic_files import _json_value, run_genetic_topology_files
from fnit.recon_all.topology_ga_backend_policy import validate_state_backend


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n")


def run(*, input_surface: Path, input_qsphere: Path, brain: Path, wm: Path,
        hemisphere: str, output_directory: Path, threads: int,
        distance_backend: str, distance_device: str | None,
        state_backend: str) -> dict:
    """按显式生产 GA 后端回放，保存 source 清理末态和失败回调。

    四个 Path 为同一 FNIT 被试的输入，surf 为 surface RAS/mm 三角表面，
    brain/wm 为 conform uint8 MGZ；hemisphere 为 lh/rh，threads 为
    当前进程总 CPU 预算。distance_backend 和 state_backend 必须由冻结
    整例作业显式给出；GPU 距离要求 distance_device='cuda:N'，CPU 距离
    则要求 None，既不默认降级也不推断整例使用的后端。
    output_directory 必须不存在，所有输出只写该私有目录。回执包含
    stage_execution、输入/诊断 SHA、请求后端及完整阶段墙钟；失败写回执
    后原样抛错。输入坐标为 surface RAS/mm；CUDA/TF32 及线程环境由调用者固定。
    """
    started = time.perf_counter()
    paths = dict(nofix=Path(input_surface), qsphere=Path(input_qsphere),
                 brain=Path(brain), wm=Path(wm))
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    distance_device = validate_distance_policy(
        distance_backend=distance_backend, distance_device=distance_device)
    state_backend = validate_state_backend(state_backend)
    input_sha256 = {name: _sha(path) for name, path in paths.items()}
    original_repair = genetic.repair_intersections
    diagnostics = {}

    def capture_repair(**kwargs):
        corrected, report = original_repair(**kwargs)
        if kwargs["marking_backend"] == "source_numba":
            path = output_directory / "source_cleaned_current.private.npz"
            np.savez_compressed(path, vertices=corrected,
                faces=np.asarray(kwargs["faces"], np.int32),
                ripped=np.asarray(kwargs["ripped"], bool),
                ordered_neighbor_offsets=np.asarray(kwargs["ordered_neighbor_offsets"]),
                ordered_neighbors=np.asarray(kwargs["ordered_neighbors"]))
            diagnostics["source_cleaned_current_sha256"] = _sha(path)
            diagnostics["source_cleanup"] = report
        return corrected, report

    def capture_failure(**state):
        path = output_directory / "failed_current.private.npz"
        arrays = {key: value for key, value in state.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(path, **arrays)
        diagnostics["failed_current_sha256"] = _sha(path)
        diagnostics["failed_quality"] = state["quality"]
        diagnostics["failed_cleanup"] = state["current_cleanup"]

    try:
        with patch.object(genetic, "repair_intersections", side_effect=capture_repair):
            result = run_genetic_topology_files(
                input_surface=paths["nofix"], input_qsphere=paths["qsphere"],
                brain=paths["brain"], wm=paths["wm"], hemisphere=hemisphere,
                output_directory=output_directory, threads=threads,
                seed=1234, population_size=10, max_unchanged=10, niters=-1,
                initial_selection=True, vertex_eliminate=True, use_overlap=False,
                normal_status="surface", soap_iterations=5, final_orig=True,
                remesh_scalar_storage="python", face_detection_cache=True,
                blocker_scan_backend="numba", distance_backend=distance_backend,
                distance_device=distance_device, state_backend=state_backend,
                failure_state_callback=capture_failure)
        stage_execution = "complete"
    except BaseException:
        stage_execution = "failed"
        raise
    finally:
        if output_directory.exists():
            receipt = dict(schema_version=2, stage_execution=stage_execution,
                input_sha256=input_sha256, actual_module_sha256={
                    "replay_recon_topology_current": _sha(Path(__file__)),
                    "topology_genetic_repair": _sha(Path(genetic.__file__)),
                    "topology_genetic_files": _sha(Path(files.__file__))},
                algorithm_policy_requested=dict(distance_backend=distance_backend,
                    distance_device=distance_device, state_backend=state_backend),
                diagnostic=diagnostics,
                stage_wall_seconds_through_before_receipt_write=time.perf_counter() - started,
                whole_raw_t1_execution="not_assessed",
                strict_reproduction="not_assessed",
                overall_metric_equivalence="not_assessed")
            _write(output_directory / "stage_replay.private.json", receipt)
    return dict(receipt, output_sha256=result.get("output_sha256"),
                quality=result.get("quality"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-surface", "input-qsphere", "brain", "wm", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--distance-backend",
                        choices=("serial", "parallel", "gpu_defect_signed_distance"),
                        required=True)
    parser.add_argument("--distance-device",
                        help="GPU 距离必填 cuda:N；CPU 距离不得提供")
    parser.add_argument("--state-backend", choices=("legacy", "packed"), required=True)
    arguments = vars(parser.parse_args())
    result = run(**{key.replace("-", "_"): value for key, value in arguments.items()})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
