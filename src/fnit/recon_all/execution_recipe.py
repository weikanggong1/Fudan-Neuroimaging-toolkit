"""生产候选的程序边界及已接入CPU/GPU依赖图；不执行影像计算。"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import sys

_guard = ContextVar('fnit_original_stage_guard', default=None)
_hook_installed = False
_ENVIRONMENT_FLAG = 'FNIT_RECON_FORBID_ORIGINAL_STAGES'


def _audit_original_stage(event, arguments):
    record = _guard.get()
    if record is None or event not in {'subprocess.Popen', 'os.exec', 'os.posix_spawn'}:
        return
    executable = os.fsdecode(arguments[0])
    name = Path(executable).name
    resolved_name = Path(executable).resolve().name if '/' in executable else name
    forbidden = any(value.startswith(('mri_', 'mris_')) or value in {
        'recon-all', 'fnit_n4_itk'} for value in (name, resolved_name))
    record['observed_exec_names'].append(name)
    if forbidden:
        record['attempted_forbidden'].append(name)
        raise RuntimeError('Python production forbids original stage program: ' + name)


@contextmanager
def original_stage_guard(*, enabled: bool):
    """拦截Python直启原版阶段程序并传播到自有exec worker，退出恢复环境。

    enabled为Python候选执行标志；返回不含参数/私有路径的执行名记录。
    不初始化CUDA，不禁止通用Python库/编译器/NVML遥测。审计直接
    subprocess/os.exec/posix_spawn及其文件路径别名；不是外部shell子孙
    的完整系统调用沙箱，外部访问证明须另列。native参考须独立运行。
    """
    global _hook_installed
    if not enabled:
        yield None
        return
    if not _hook_installed:
        sys.addaudithook(_audit_original_stage)
        _hook_installed = True
    record = {'enabled': True, 'observed_exec_names': [], 'attempted_forbidden': [],
              'scope': 'direct Python subprocess/exec/posix_spawn, including path aliases; not external shell descendant sandbox'}
    token = _guard.set(record)
    previous = os.environ.get(_ENVIRONMENT_FLAG)
    os.environ[_ENVIRONMENT_FLAG] = '1'
    try:
        yield record
    finally:
        _guard.reset(token)
        if previous is None:
            os.environ.pop(_ENVIRONMENT_FLAG, None)
        else:
            os.environ[_ENVIRONMENT_FLAG] = previous


def python_runtime_selection() -> dict:
    """返回自有GCA选择；不发现、探测或提供原版程序路径。

    无输入，返回与既有选择器兼容的字典。GPU候选评分与完整CPU
    stable/all-sample有序EM均由项目接口执行；这里声明已接入的算法，
    不代表当前输入或整例验收。white/pial路径为None，不批准默认切换。
    """
    return {"mode": "torch", "em_backend": "torch", "white_binary": None,
            "pial_binary": None, "gca_capabilities": None,
            "white_capabilities": None,
            "gca_reason": "FNIT Torch candidate search and complete CPU stable/all-sample EM; no upstream program selection",
            "em_algorithm_completion": "complete_stable_and_all_sample_protocol",
            "em_validation_scope": "source-bound component reports; not a current whole reconstruction result"}


def validate_program_boundary(*, backend: str, programs: dict) -> None:
    """在输出创建前拒绝Python配方的任何原版阶段程序槽位。

    backend为入口后端，programs为名称→路径或None的声明映射。仅
    python强制所有值为None；其他历史后端保持兼容。成功返回None，
    非空槽位抛ValueError，不访问文件、初始化CUDA或自动回退。
    """
    if backend == "python":
        unexpected = sorted(name for name, path in programs.items() if path is not None)
        if unexpected:
            raise ValueError("Python execution forbids original stage programs: " + ", ".join(unexpected))


def execution_dag(*, threads: int, input_execution: str, mni_execution: str,
                  hemisphere_workers: int, backend: str) -> dict:
    """返回实际调度选择的JSON依赖图与预算，不将配置当运行证明。

    threads为全被试总预算，input_execution为serial/parallel，
    mni_execution为in-process/parallel-late，hemisphere_workers为1/2；
    返回DAG节点/屏障/预算和发布门状态。无时间或显存测量；质量、
    动态访问和峰值须由独立实际报告提供。无独立原软件命令。
    """
    return {
        "scope": "declared orchestration; runtime quality and process access assessed separately",
        "total_thread_budget": threads,
        "input": {"execution": input_execution, "shared_input": "own orig.mgz",
                  "branches": ["complete N4", "own GPU SynthSeg original+flip ensemble"],
                  "branch_threads": [1, 3] if input_execution == "parallel" else [threads, threads],
                  "join_before": "nu and normalization"},
        "hemispheres": {"workers": hemisphere_workers,
                        "threads_each": threads // 2 if hemisphere_workers == 2 else threads,
                        "barriers": ["surface", "register", "annotation", "final white", "pial"],
                        "private_outputs": True if hemisphere_workers == 2 else False,
                        "serial_dependencies": ["surface.defects lh reset then rh accumulate",
                                                "final GPU metrics lh then rh"]},
        "late_mni": {"execution": mni_execution,
                     "branches": ["complete own GPU MNI", "CPU final mesh validation"],
                     "branch_threads": [threads - threads // 2, threads // 2]
                         if mni_execution == "parallel-late" else [threads, threads],
                     "after": "all surface/label/stats writes and private directory copies",
                     "join_before": "138 output completeness"},
        "program_boundary": "forbidden" if backend == "python" else "legacy native profile",
        "automatic_native_fallback": False,
        "public_default_is_python": False,
        "required_algorithm_completion": {"gca_em": "complete stable/all-sample EM connected; input-specific acceptance recorded separately",
                                          "topology_ga": "pending component quality gate",
                                          "final_pial": "pending component quality gate"},
        "default_recipe_release": "pending GA and final pial component quality gates; complete EM connected, whole evaluation pending",
        "overall_equivalence": "not_assessed",
    }
