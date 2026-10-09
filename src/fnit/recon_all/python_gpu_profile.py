"""Pure-Python/CUDA recon-all capability declaration.

The regular ``native_free`` entry point is a mixed implementation: it uses
FNIT Python/PyTorch code where it is validated and independently compiled
FreeSurfer-compatible programs for stages that still lack a complete Python
replacement.  This module keeps that boundary explicit.  It is deliberately
small so a new stage can only enter the pure profile after its same-input,
connected-chain and real-T1 checks have been recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class PythonGpuCapability:
    """One stage's implementation status in the strict pure profile."""

    stage: str
    implementation: str
    device: str
    complete: bool
    reason: str


# These are the stages that are already implemented without invoking a
# FreeSurfer/FSL executable.  ``device`` describes the production CUDA path,
# not merely a function accepting a ``device`` keyword.
_READY: tuple[PythonGpuCapability, ...] = (
    PythonGpuCapability("conform", "FNIT PyTorch", "cuda", True,
                        "validated conformed 1 mm MGH output"),
    PythonGpuCapability("SynthStrip", "FNIT PyTorch", "cuda", True,
                        "FP32 cuDNN exception is applied after construction"),
    PythonGpuCapability("Talairach affine", "FNIT PyTorch", "cuda", True,
                        "SynthMorph affine child process"),
    PythonGpuCapability("SynthSeg", "FNIT PyTorch", "cuda", True,
                        "FP32 cuDNN exception is recorded"),
    PythonGpuCapability("brainmask", "FNIT PyTorch", "cuda", True,
                        "FNIT mask kernel"),
    PythonGpuCapability("MNI auxiliary/deform", "FNIT PyTorch", "cuda", True,
                        "FNIT GPU warp conversion and inverse"),
    PythonGpuCapability("surface metrics", "FNIT PyTorch", "cuda", True,
                        "thickness, area, curvature and vertex volume kernels"),
    PythonGpuCapability("finalsurfs volume edits", "FNIT PyTorch", "cuda", True,
                        "mask and entorhinal/ACJ edits use explicit CUDA"),
    PythonGpuCapability("surface Jacobian", "FNIT PyTorch (opt-in)", "cuda", False,
                        "the complete 114k-vertex map is launch/I/O bound; production keeps the measured faster CPU path"),
)

_BLOCKED: tuple[PythonGpuCapability, ...] = (
    PythonGpuCapability("N4", "Conda ITK C++", "cpu", False,
                        "complete cached Torch N4 has two raw-T1 runs: sub06 completes, sub07 fails the unchanged final white self-intersection gate; cross-input diagnosis shows quantization and case-dependent GCA amplification; default stays ITK"),
    PythonGpuCapability("mri_em_register/GCA", "FNIT PyTorch scorer + Python EM", "cuda", False,
                        "CUDA translation/linear scoring and Python EM have two raw-T1 empty-directory regressions (803aec50/0cd9cbd5); EM remains CPU and the strict pure-PyTorch GPU profile is not complete"),
    PythonGpuCapability("WM segmentation", "FNIT PyTorch/CPU hybrid", "cuda", False,
                        "complete histogram passes are exact on two real inputs with opt-in Torch; ordered strand stays CPU and full-file/raw-T1 regression is pending"),
    PythonGpuCapability("WM/aseg core edit", "FNIT Numba/Torch hybrid (opt-in)", "mixed cpu/cuda", False,
                        "complete guarded hybrid is exact on two frozen real inputs; ordered feedback stays CPU; raw-T1 integration pending"),
    PythonGpuCapability("topology GA", "Conda FreeSurfer C++", "cpu", False,
                        "Python module is only a preflight, not the patch search"),
    PythonGpuCapability("inflate/remesh/intersection", "mixed", "cpu", False,
                        "complete standard Torch inflated/sulc is wired and raw-T1 tested (589e2749); nofix inflation, dynamic remesh and intersection still retain CPU/native components"),
    PythonGpuCapability("white.preaparc/final white", "Conda FreeSurfer C++", "cpu", False,
                        "complete four-pass Python final white now matches four same-input original Conda references after label-rip correction; preaparc and pial step-norm precision are corrected; current production-fast timing and raw-T1 replacement acceptance are separate, with ordered CPU kernels still present"),
    PythonGpuCapability("pial placement", "FNIT Numba/Torch (opt-in) and Conda C++ default", "mixed cpu/cuda", False,
                        "complete four-pass Python pial with GPU candidates matches all four frozen Conda geometries after the float-sqrt correction (3644f50c); ordered CPU sampling/gradients/SOAP remain, paired production speed and raw-T1 replacement are not established; native default retained"),
    PythonGpuCapability("defects projection", "FNIT PyTorch (opt-in)", "cuda", False,
                        "complete projection is exact on two real frozen inputs; raw-T1 integration pending; topology GA is a separate blocked stage"),
    PythonGpuCapability("curvature statistics", "FNIT Torch discrete/principal maps (opt-in); Conda stats default", "mixed cpu/cuda", False,
                        "two real bilateral raw eight-map tests are complete but the exploratory error gate fails; filtering/smoothing/full curv.stats remain native"),
)


class PurePythonGpuUnavailable(RuntimeError):
    """Raised when strict pure Python/CUDA recon-all is requested too early."""

    def __init__(self, capabilities: dict):
        self.capabilities = capabilities
        super().__init__(json.dumps(capabilities, ensure_ascii=False, sort_keys=True))


def capability_report(*, device: str = "cuda:0") -> dict:
    """返回严格pure profile的实现状态，不读取影像或初始化CUDA。

    device默认cuda:0，仅描述请求设备；返回profile/device/native_programs、
    ready/blocked各阶段结构(stage/implementation/device/complete/reason)。
    已有混合实现/阶段速度证据不会自动转成纯GPU完成，不判数值或指标等效。
    当前关键原生/CPU阶段未全替代，complete保持False；不是benchmark执行器。
    """
    is_cuda = str(device).startswith("cuda")
    rows = [*(_READY if is_cuda else tuple(
        PythonGpuCapability(row.stage, row.implementation, "cpu", False,
                            "strict profile requires an explicit CUDA device")
        for row in _READY)), *_BLOCKED]
    return {
        "profile": "python-gpu",
        "device": str(device),
        "native_programs": [],
        "complete": bool(is_cuda and all(row.complete for row in rows)),
        "ready": [row.__dict__ for row in rows if row.complete],
        "blocked": [row.__dict__ for row in rows if not row.complete],
    }


def require_complete(*, device: str = "cuda:0") -> dict:
    """校验严格pure模式；device同capability_report，不创建被试目录。

    完整时返回状态字典；未完整抛PurePythonGpuUnavailable并附同一机器
    可读capabilities，生产不会回退系统程序。没有独立原软件CLI或影像输出。
    """
    report = capability_report(device=device)
    if not report["complete"]:
        raise PurePythonGpuUnavailable(report)
    return report
