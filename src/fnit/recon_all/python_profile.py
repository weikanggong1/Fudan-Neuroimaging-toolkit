"""不执行原版阶段程序的完整Python候选配方，允许CPU有序核与通用库。"""
from __future__ import annotations

import torch


def python_profile_options(*, device: str = "cuda:0", threads: int = 4,
                           n4_backend: str = "simpleitk",
                           n4_execution: str = "in-process",
                           topology_backend: str = "python",
                           input_execution: str = "serial") -> dict:
    """返回独立Python候选的固定调度选项，不初始化CUDA或读写数据。

    device须为显式cuda:N；当前完整MNI/WM接口仍需GPU，CPU核允许保留。
    threads为整例正整数总预算且至少2，双侧worker各使用threads//2。
    n4_backend默认simpleitk，使用通用SimpleITK Python完整N4；显式torch
    复用既有完整200轮N4，不使用blur残差近似或原版阶段程序。n4_execution
    默认in-process，沿用父分配器；isolated仅允许torch，在新Python进程
    执行既有缓存worker。SimpleITK保持CPU；Torch使用上述CUDA设备。
    topology_backend默认python保留数字拓扑；python-ga显式选择完整自有
    有序GA、SOAP及压缩，不启动原版阶段程序。GA在各半球既有线程预算
    内运行，原域标签与压缩网格分开；完整orig清理和质量检查仍须通过。
    仅改变所选N4/拓扑键，其余调度、必要阶段和输出检查保持。
    返回具名调度字典，包括后端名称、双侧数及输入DAG选择；无单位，
    字典键数不是阶段计数。input_execution默认serial；parallel只在
    SimpleITK/in-process且总4线程时并行N4=1与SynthSeg=3，join后继续。
    topology默认完整数字拓扑候选（不同于FS GA）；全部输出仍走完整
    放置/球面/注释/指标链。复用已验证GPU邻域/偏置、球面收尾、MNI
    后置并行，以及GCA新exec缓存/1024分块/同公式GPU求逆；父缓存不改。
    不承诺严格复现或整体指标等效。非法设备或
    预算、非法N4后端或执行组合抛ValueError，缺依赖/数据由后续预检报错，
    不回退原程序。Torch已有量化/下游差异单列，选择后端不判指标等效。
    输入DAG选择属于完整配方；非法组合不隐式改回串行或触发计算。
    没有独立官方CLI；组合参考和真实验证见PYTHON_SURFACE_PIPELINE.md。
    """
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Python profile requires explicit cuda:N") from error
    if target.type != "cuda" or target.index is None:
        raise ValueError("Python profile requires explicit cuda:N")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 2:
        raise ValueError("Python profile requires total threads >= 2")
    if not isinstance(n4_backend, str) or n4_backend not in {"simpleitk", "torch"}:
        raise ValueError("Python profile N4 backend must be simpleitk or torch")
    if not isinstance(n4_execution, str) or n4_execution not in {"in-process", "isolated"}:
        raise ValueError("Python profile N4 execution must be in-process or isolated")
    if n4_execution == "isolated" and n4_backend != "torch":
        raise ValueError("isolated N4 requires n4_backend='torch' in Python profile")
    if not isinstance(topology_backend, str) or topology_backend not in {"python", "python-ga"}:
        raise ValueError("Python profile topology backend must be python or python-ga")
    if input_execution not in {"serial", "parallel"}:
        raise ValueError("Python profile input_execution must be serial or parallel")
    if input_execution == "parallel" and (
            n4_backend != "simpleitk" or n4_execution != "in-process" or threads != 4):
        raise ValueError("parallel Python input requires SimpleITK in-process and total threads=4")
    return {
        "input_execution": input_execution,
        "native_optimizations": "torch", "n4_backend": n4_backend,
        "n4_execution": n4_execution, "wm_backend": "torch-optimized",
        "gca_execution": "isolated", "gca_inverse_backend": "torch",
        "gca_candidate_chunk": 1024,
        "normalization_controls_backend": "torch",
        "normalization_initial_bias_backend": "torch",
        "sphere_finish_backend": "torch", "mni_execution": "parallel-late",
        "wm_execution": "isolated", "wm_edit_backend": "python",
        "defects_backend": "torch", "inflate_backend": "torch",
        "surface_placement_backend": "python", "topology_backend": topology_backend,
        "annotation_gibbs_backend": "numba", "remesh_scalar_storage": "python",
        "fill_backend": "numba", "hemisphere_workers": 2,
    }
