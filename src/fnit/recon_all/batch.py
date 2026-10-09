"""Multi-subject scheduler for the Conda recon-all pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
import sys


def run_recon_all_python_batch(
    jobs: list[dict], weights_dir: str | Path, assets_dir: str | Path,
    *, devices: tuple[str, ...] = ("cuda:0",), threads: int = 4,
    native_bin_dir: str | Path | None = None,
    profile_stages: bool = False, cuda_allocator_cache: str = "auto",
    hemisphere_workers: int = 1,
    native_optimizations: str = "auto",
    n4_backend: str = "native",
    n4_execution: str = "in-process",
    normalization_controls_backend: str = "cpu",
    normalization_initial_bias_backend: str = "cpu",
    wm_backend: str = "native",
    wm_execution: str = "in-process",
    wm_edit_backend: str = "native",
    defects_backend: str = "native",
    sphere_normals_backend: str = "numba",
    inflate_backend: str = "native",
    mni_execution: str = "in-process",
    gca_inverse_backend: str = "cpu",
    gca_candidate_chunk: int = 64,
    gca_execution: str = "in-process",
    fill_backend: str = "python",
    backend: str = "native",
) -> list[dict]:
    """每设备独立子进程执行单 T1，按 jobs 顺序返回完整报告列表。

    jobs 每项含 t1 原始影像和 subject_dir 空目录；weights_dir/assets_dir
    为已校验资源，devices 为互不重复的逻辑 CPU/CUDA 设备，threads 为
    每个被试线程预算（默认4），native_bin_dir 默认当前 Conda bin。
    profile_stages=False 不增加阶段同步；cuda_allocator_cache=auto
    延续低显存默认，也可在子进程初始化前选择 enabled/disabled。
    输出体积为 conform 网格，表面为 surface RAS/mm；输出/错误语义同
    单例入口。输入非法抛 ValueError/FileNotFoundError，任务失败汇总为
    RuntimeError。每设备仅执行一个被试；hemisphere_workers 默认1，设2时被试内部
    独立进程并行双侧，threads 在双侧之间分配（总预算不翻倍）。
    backend=python-gpu使用严格纯Python/CUDA profile；只要仍有未完成阶段就
    在创建被试目录前失败，不回退到原生程序。native_optimizations=auto在CUDA选择已有Torch GCA，CPU查询GCA缓存能力；
    white使用专用快速程序，pial保持原生；
    original用于原生阶段配对控制，原样传递给每个被试CLI。
    n4_backend=native与n4_execution=in-process保持默认；torch复用完整N4，
    isolated只允许cuda:N设备，以缓存exec保留父策略及完整子阶段报告。
    这两个选项原样传入CLI，非法组合在调度/创建输出前拒绝。
    normalization_controls_backend 默认cpu；torch复用两轮的同规则GPU邻域，
    仅支持显式cuda:N设备；其余控制点选择与偏置流程保持，原样传给CLI。
    normalization_initial_bias_backend 默认cpu；torch仅迁移第二轮初始
    偏置传播/平滑，要求cuda:N，原除乘/输出精度保持，原样传给CLI。
    inflate_backend默认native；torch复用完整标准inflated/sulc，要求
    全部目标cuda:N与hemisphere_workers=2，仅surface子exec启用缓存，
    nofix/球面/注册及父策略保持。非法组合在任务调度前抛ValueError。
    mni_execution默认in-process；parallel-late在所有半球写出后，将完整
    MNI非线性与CPU网格检查并行，所有设备须cuda:N且threads至少2。
    总线程在这两个任务间分配，join后才检查138输出；原样传入CLI。
    defects_backend 默认 native；torch 将缺陷投射交给同设备 PyTorch，
    保持左清零/右合并顺序和标签含义，调色板采用确定性颜色。
    wm_backend 默认 native；torch 选择已有完整混合 WM segmentation。
    wm_edit_backend 默认 native；torch-hybrid 选择静态 CUDA 子步骤与有序
    Numba WM/aseg 核心。sphere_normals_backend 默认 numba；torch 只迁移
    标准球面法向。后三种 Torch 选择均要求 CUDA，失败不静默回退。
    wm_backend=torch-optimized复用Torch直方图和缓存平面几何，有序反馈为CPU。
    wm_execution默认in-process；isolated只允许torch-optimized，以新exec
    启用阶段缓存并保留父CUDA状态，记录完整阶段墙钟，原样传入每个CLI。
    gca_inverse_backend=cpu和gca_candidate_chunk=64保留旧评分；torch求逆及
    非默认分块只允许CUDA的Torch GCA。fill_backend=python保留旧完整fill；
    gca_execution=in-process保留父缓存；isolated以新exec启用局部GCA缓存，
    不改变已初始化父CUDA或精度，仅可CUDA Torch后端。
    numba使用同顺序CPU堆，torch-numba另使用CUDA初始边界。原样传入CLI。
    """
    from .native_free import (_normalization_controls_options, _normalization_initial_bias_options,
                              _validate_inflate_backend, _validate_mni_execution)
    for target in devices:
        _normalization_controls_options(normalization_controls_backend, target)
        _normalization_initial_bias_options(normalization_initial_bias_backend, target)
        _validate_inflate_backend(inflate_backend, target, hemisphere_workers)
        _validate_mni_execution(mni_execution, target, threads)
    if mni_execution not in {"in-process", "parallel-late"}:
        raise ValueError("mni_execution must be in-process or parallel-late")
    if inflate_backend not in {"native", "torch"}:
        raise ValueError("inflate_backend must be native or torch")
    if normalization_controls_backend not in {"cpu", "torch"}:
        raise ValueError("normalization_controls_backend must be cpu or torch")
    if normalization_initial_bias_backend not in {"cpu", "torch"}:
        raise ValueError("normalization_initial_bias_backend must be cpu or torch")
    from .input_n4_chain import validate_n4_execution
    for target in devices:
        validate_n4_execution(n4_backend=n4_backend, n4_execution=n4_execution, device=target)
    from .hemisphere_parallel import validate_hemisphere_workers
    validate_hemisphere_workers(hemisphere_workers, threads)
    if not devices or len(set(devices)) != len(devices) or any(
        device != "cpu" and re.fullmatch(r"cuda:\d+", device) is None for device in devices
    ):
        raise ValueError("devices must be distinct CPU/CUDA device names")
    if threads < 1:
        raise ValueError("threads must be positive")
    if cuda_allocator_cache not in {"auto", "enabled", "disabled"}:
        raise ValueError("cuda_allocator_cache must be auto, enabled, or disabled")
    if native_optimizations not in {"auto", "original", "torch"}:
        raise ValueError("native_optimizations must be auto, original, or torch")
    if wm_backend not in {"native", "torch", "torch-optimized"}:
        raise ValueError("wm_backend must be native, torch or torch-optimized")
    if wm_backend != "native" and any(device == "cpu" for device in devices):
        raise ValueError("Torch WM backends require CUDA devices")
    if wm_execution not in {"in-process", "isolated"}:
        raise ValueError("wm_execution must be in-process or isolated")
    if wm_execution == "isolated" and wm_backend != "torch-optimized":
        raise ValueError("isolated WM requires torch-optimized backend")
    if gca_inverse_backend not in {"cpu", "torch"}:
        raise ValueError("gca_inverse_backend must be cpu or torch")
    if isinstance(gca_candidate_chunk, bool) or not isinstance(gca_candidate_chunk, int) or gca_candidate_chunk < 1:
        raise ValueError("gca_candidate_chunk must be a positive integer")
    if gca_execution not in {"in-process", "isolated"}:
        raise ValueError("gca_execution must be in-process or isolated")
    if (gca_inverse_backend != "cpu" or gca_candidate_chunk != 64 or gca_execution != "in-process") and (
        native_optimizations == "original" or any(device == "cpu" for device in devices)
    ):
        raise ValueError("GCA candidate options require the CUDA Torch GCA backend")
    if fill_backend not in {"python", "numba", "torch-numba"}:
        raise ValueError("fill_backend must be python, numba or torch-numba")
    if fill_backend == "torch-numba" and any(device == "cpu" for device in devices):
        raise ValueError("fill_backend='torch-numba' requires CUDA devices")
    if wm_edit_backend not in {"native", "torch-hybrid"}:
        raise ValueError("wm_edit_backend must be native or torch-hybrid")
    if wm_edit_backend == "torch-hybrid" and any(device == "cpu" for device in devices):
        raise ValueError("wm_edit_backend='torch-hybrid' requires CUDA devices")
    if defects_backend not in {"native", "torch"}:
        raise ValueError("defects_backend must be native or torch")
    if sphere_normals_backend not in {"numba", "torch"}:
        raise ValueError("sphere_normals_backend must be numba or torch")
    if sphere_normals_backend == "torch" and any(device == "cpu" for device in devices):
        raise ValueError("sphere_normals_backend='torch' requires CUDA devices")
    if backend not in {"native", "python-gpu"}:
        raise ValueError("backend must be native or python-gpu")
    weights, assets = Path(weights_dir).resolve(), Path(assets_dir).resolve()
    if not weights.is_dir() or not assets.is_dir():
        raise FileNotFoundError("weights_dir and assets_dir must exist")
    prepared = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"t1", "subject_dir"}:
            raise ValueError(f"job {index} needs t1 and subject_dir")
        t1, subject = Path(job["t1"]).resolve(), Path(job["subject_dir"]).resolve()
        if not t1.is_file():
            raise FileNotFoundError(t1)
        if subject.exists() and (not subject.is_dir() or any(subject.iterdir())):
            raise ValueError(f"job {index} subject_dir must be empty")
        if any(subject == previous or subject.is_relative_to(previous)
               or previous.is_relative_to(subject) for _, previous in prepared):
            raise ValueError(f"job {index} subject_dir overlaps another job")
        prepared.append((t1, subject))

    def run_device(device: str, assignments: list[tuple[int, Path, Path]]):
        results = []
        for index, t1, subject in assignments:
            command = [sys.executable, "-m", "fnit.recon_all.native_free",
                       str(t1), str(subject), "--weights-dir", str(weights),
                       "--assets-dir", str(assets), "--device", device,
                       "--threads", str(threads), "--cuda-allocator-cache", cuda_allocator_cache]
            command += ["--backend", backend]
            if n4_backend != "native":
                command += ["--n4-backend", n4_backend]
            if n4_execution != "in-process":
                command += ["--n4-execution", n4_execution]
            if normalization_controls_backend != "cpu":
                command += ["--normalization-controls-backend", normalization_controls_backend]
            if normalization_initial_bias_backend != "cpu":
                command += ["--normalization-initial-bias-backend", normalization_initial_bias_backend]
            if inflate_backend != "native":
                command += ["--inflate-backend", inflate_backend]
            if mni_execution != "in-process":
                command += ["--mni-execution", mni_execution]
            if wm_backend != "native":
                command += ["--wm-backend", wm_backend]
            if wm_execution != "in-process":
                command += ["--wm-execution", wm_execution]
            if wm_edit_backend != "native":
                command += ["--wm-edit-backend", wm_edit_backend]
            if defects_backend != "native":
                command += ["--defects-backend", defects_backend]
            if sphere_normals_backend != "numba":
                command += ["--sphere-normals-backend", sphere_normals_backend]
            if gca_inverse_backend != "cpu":
                command += ["--gca-inverse-backend", gca_inverse_backend]
            if gca_candidate_chunk != 64:
                command += ["--gca-candidate-chunk", str(gca_candidate_chunk)]
            if gca_execution != "in-process":
                command += ["--gca-execution", gca_execution]
            if fill_backend != "python":
                command += ["--fill-backend", fill_backend]
            if hemisphere_workers != 1:
                command += ["--hemisphere-workers", str(hemisphere_workers)]
            if native_optimizations != "auto":
                command += ["--native-optimizations", native_optimizations]
            if profile_stages:
                command.append("--profile-stages")
            if native_bin_dir is not None:
                command += ["--native-bin-dir", str(Path(native_bin_dir).resolve())]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                results.append((index, None, completed.stderr.strip()))
            else:
                results.append((index, json.loads(
                    (subject / "fnit-native-free-run.json").read_text()), None))
        return results

    assignments = [[] for _ in devices]
    for index, (t1, subject) in enumerate(prepared):
        assignments[index % len(devices)].append((index, t1, subject))
    reports, errors = [None] * len(prepared), []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_device, device, assigned)
                   for device, assigned in zip(devices, assignments) if assigned]
        for future in futures:
            for index, report, error in future.result():
                reports[index] = report
                if error:
                    errors.append(f"job {index}: {error}")
    if errors:
        raise RuntimeError("batch reconstruction failed: " + "; ".join(errors))
    return reports
