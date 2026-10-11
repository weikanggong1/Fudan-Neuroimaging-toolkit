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
    PythonGpuCapability("N4", "FNIT Torch N4 / generic SimpleITK Python API", "mixed cpu/cuda", False,
                        "strict CUDA profile is incomplete; independent backend=python uses generic SimpleITK without the original stage program, or explicit complete Torch N4; two raw-T1-to-filled Torch regressions retain numerical differences"),
    PythonGpuCapability("mri_em_register/GCA", "FNIT PyTorch scorer + Python EM", "cuda", False,
                        "CUDA translation/linear scoring and Python EM have two raw-T1 empty-directory regressions (803aec50/0cd9cbd5); EM remains CPU and the strict pure-PyTorch GPU profile is not complete"),
    PythonGpuCapability("WM segmentation", "FNIT PyTorch/CPU hybrid", "cuda", False,
                        "backend=python uses complete Torch/CPU segmentation without the original stage program; ordered strand stays CPU; two raw-T1-to-filled explicit Torch N4 regressions retain WM numerical differences"),
    PythonGpuCapability("WM/aseg core edit", "FNIT Numba/Torch hybrid (opt-in)", "mixed cpu/cuda", False,
                        "backend=python uses the complete independent Numba edit; guarded Torch hybrid has two exact frozen regressions but ordered feedback remains CPU; whole quality is separate"),
    PythonGpuCapability("topology GA", "FNIT digital topology and experimental NumPy/Numba GA primitives", "cpu", False,
                        "backend=python uses independent digital topology but two raw-T1 whole runs failed mesh quality; MRI, normal, curvature and volume fitness components have separate frozen regressions; full source-semantics GA controller is unfinished"),
    PythonGpuCapability("inflate/remesh/intersection", "mixed", "cpu", False,
                        "backend=python uses independent nofix, remesh and intersection with CPU ordered kernels plus standard Torch inflated/sulc; strict CUDA-only replacement is incomplete; raw-T1 whole quality remains separate"),
    PythonGpuCapability("white.preaparc/final white", "FNIT Python/Numba with Torch substeps", "mixed cpu/cuda", False,
                        "backend=python runs complete independent placement; frozen final-white comparisons cover four same-input hemispheres, but ordered CPU kernels remain and raw-T1 whole quality failed upstream topology"),
    PythonGpuCapability("pial placement", "FNIT Numba/Torch (opt-in) and Conda C++ default", "mixed cpu/cuda", False,
                        "backend=python runs complete independent pial with GPU candidates and ordered CPU sampling/gradients/SOAP; four frozen geometries matched after float-sqrt correction, but a raw-T1 whole candidate failed unchanged self-intersection quality"),
    PythonGpuCapability("defects projection", "FNIT PyTorch (opt-in)", "cuda", False,
                        "complete projection is exact on two real frozen inputs and is wired in backend=python; raw-T1 whole topology quality failed separately, so projection does not make full GA complete"),
    PythonGpuCapability("curvature statistics", "FNIT Python/Numba complete statistics and Torch map substeps", "mixed cpu/cuda", False,
                        "independent full eight-map and text statistics passed two real bilateral frozen regressions; ordered CPU work remains, so the strict CUDA profile is incomplete; raw-T1 whole validation remains separate"),
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
