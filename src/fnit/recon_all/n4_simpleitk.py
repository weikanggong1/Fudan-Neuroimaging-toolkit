"""固定单T1 N4的独立SimpleITK Python适配器，不调用FreeSurfer程序。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import threading
import time

import nibabel as nib
import numpy as np

from .mgh_compat import save_same_dtype_mgh
from .n4_itk import _to_uchar


_THREAD_LOCK = threading.RLock()
_SIMPLEITK_VERSION = "2.5.6"


def _load_simpleitk():
    """惰性检查实际计算库；不创建图像/滤波器，不改线程或CUDA状态。"""
    instruction = (
        "N4 simpleitk requires SimpleITK==2.5.6 in the active Python environment "
        "or an explicitly configured Python dependency target; no backend fallback"
    )
    try:
        sitk = importlib.import_module("SimpleITK")
    except (ImportError, OSError) as error:
        raise ImportError(instruction + "; import failed") from error
    required = ("Version_VersionString", "Version_ITKVersionString", "ProcessObject",
                "GetImageFromArray", "GetArrayFromImage", "ShrinkImageFilter",
                "N4BiasFieldCorrectionImageFilter", "ExpImageFilter", "DivideImageFilter")
    if any(not callable(getattr(sitk, name, None)) for name in required):
        raise ImportError(instruction + "; required N4 API is missing")
    try:
        version = sitk.Version_VersionString()
        native_file = Path(sitk._SimpleITK.__file__)
    except (AttributeError, TypeError, RuntimeError) as error:
        raise ImportError(instruction + "; loaded library metadata is invalid") from error
    if version != _SIMPLEITK_VERSION:
        raise ImportError(instruction + f"; found version {version!r}")
    if not native_file.is_file():
        raise ImportError(instruction + "; loaded compiled library file is missing")
    return sitk


def require_simpleitk_runtime() -> dict:
    """执行前检查2.5.6及可读库，返回实际版本/库SHA；不读取T1或改变计算策略。

    不把某一平台wheel的SHA写成跨平台要求。缺包、ABI导入失败、未验证
    版本或库不可读均抛ImportError，无输出目录、影像/滤波器或CUDA操作。
    """
    sitk = _load_simpleitk()
    try:
        digest = hashlib.sha256()
        with Path(sitk._SimpleITK.__file__).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        itk_version = sitk.Version_ITKVersionString()
    except (OSError, RuntimeError) as error:
        raise ImportError("N4 SimpleITK==2.5.6 library cannot be read or inspected; no backend fallback") from error
    return {"backend": "simpleitk", "required": True, "simpleitk_version": _SIMPLEITK_VERSION,
            "itk_version": itk_version, "library_sha256": digest.hexdigest()}


def correct_array(image: np.ndarray, *, spacing: tuple[float, float, float],
                  reconstruction_threads: int = 1,
                  fitting_threads: int = 1) -> tuple[np.ndarray, dict]:
    """XYZ FP32强度→同shape FP32完整N4输出，间距单位mm，无插值换网格。

    spacing为三个有限正数；reconstruction_threads与fitting_threads
    默认1且均为正整数，分别控制全分辨率重建和拟合。拟合默认不变，
    显式多线程需按真实同输入回归，不预先认定浮点归约保持逐位结果。
    shrink4、cubic控制点4³、4×50轮、收敛阈值0、
    200bin、FWHM0.15、Wiener0.01，全1 maskLabel1。重建采用库原生
    GetLogBiasFieldAsImage→Exp→Divide，不读取FS资源或调用命令。
    内部计算origin0/identity方向与现有N4 raw算法相同；输出仍XYZ。
    返回新数组和实际版本/参数/分段耗时/线程恢复报告。非法输入抛
    ValueError、缺包抛ImportError、库失败传播；不降级成平滑近似。
    同进程的本模块调用串行恢复ITK线程全局值；其他ITK任务应隔离exec。
    """
    if (not isinstance(reconstruction_threads, int) or isinstance(reconstruction_threads, bool)
            or reconstruction_threads < 1):
        raise ValueError("reconstruction_threads must be a positive integer")
    if (not isinstance(fitting_threads, int) or isinstance(fitting_threads, bool)
            or fitting_threads < 1):
        raise ValueError("fitting_threads must be a positive integer")
    values = np.asarray(image)
    distances = np.asarray(spacing, dtype=np.float64)
    if (values.ndim != 3 or min(values.shape) < 8 or values.dtype != np.float32
            or not np.isfinite(values).all() or (values < 0).any()
            or not (values > 0).any() or distances.shape != (3,)
            or not np.isfinite(distances).all() or (distances <= 0).any()):
        raise ValueError("expected nonnegative finite 3D float32 image with positive signal and 3 positive mm spacings")
    sitk = _load_simpleitk()
    started = time.perf_counter()
    with _THREAD_LOCK:
        previous_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(fitting_threads)
            full = sitk.GetImageFromArray(np.ascontiguousarray(values.transpose(2, 1, 0)))
            full.SetSpacing([float(x) for x in distances])
            mask = sitk.GetImageFromArray(np.ones(tuple(reversed(values.shape)), dtype=np.uint8))
            mask.CopyInformation(full)
            shrink = sitk.ShrinkImageFilter()
            shrink.SetNumberOfWorkUnits(1)
            shrink.SetShrinkFactors([4, 4, 4])
            small, small_mask = shrink.Execute(full), shrink.Execute(mask)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetNumberOfWorkUnits(fitting_threads)
            corrector.SetNumberOfThreads(fitting_threads)
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            corrector.SetConvergenceThreshold(0.0)
            corrector.SetNumberOfControlPoints([4, 4, 4])
            corrector.SetSplineOrder(3)
            corrector.SetNumberOfHistogramBins(200)
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
            corrector.SetWienerFilterNoise(0.01)
            corrector.SetUseMaskLabel(True)
            corrector.SetMaskLabel(1)
            prepared = time.perf_counter()
            corrector.Execute(small, small_mask)
            fitted = time.perf_counter()
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(reconstruction_threads)
            log_field = corrector.GetLogBiasFieldAsImage(full)
            reconstructed = time.perf_counter()
            exp_filter = sitk.ExpImageFilter()
            divide = sitk.DivideImageFilter()
            exp_filter.SetNumberOfWorkUnits(reconstruction_threads)
            divide.SetNumberOfWorkUnits(reconstruction_threads)
            corrected = divide.Execute(full, exp_filter.Execute(log_field))
            result = np.ascontiguousarray(sitk.GetArrayFromImage(corrected).transpose(2, 1, 0))
            finished = time.perf_counter()
            if result.dtype != np.float32 or not np.isfinite(result).all():
                raise RuntimeError("SimpleITK N4 returned invalid FP32 data")
            report = {"backend": "SimpleITK Python complete fixed N4 recipe", "used_native_programs": False,
                "simpleitk_version": sitk.Version_VersionString(), "itk_version": sitk.Version_ITKVersionString(),
                "library_sha256": hashlib.sha256(Path(sitk._SimpleITK.__file__).read_bytes()).hexdigest(),
                "fitting_threads": fitting_threads, "reconstruction_threads": reconstruction_threads,
                "actual_fitting_work_units": int(corrector.GetNumberOfWorkUnits()),
                "actual_fitting_filter_threads": int(corrector.GetNumberOfThreads()),
                "recipe": {"shrink": [4,4,4], "iterations": [50,50,50,50], "convergence_threshold": 0.0,
                    "spline_order": 3, "initial_control_points": [4,4,4], "histogram_bins": 200,
                    "fwhm": 0.15, "wiener_noise": 0.01, "mask": "all ones", "mask_label": 1},
                "internal_geometry": {"origin_mm": [0.0,0.0,0.0], "direction": "identity", "spacing_mm": distances.tolist()},
                "current_level": int(corrector.GetCurrentLevel()), "elapsed_last_level_iterations": int(corrector.GetElapsedIterations()),
                "timing_parts_seconds": {"array_prepare_and_shrink": prepared-started,
                    "fitting": fitted-prepared, "full_bspline_reconstruction": reconstructed-fitted,
                    "exp_divide_and_array_download": finished-reconstructed},
                "total_seconds": finished-started}
        finally:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous_threads)
        report["restored_global_threads"] = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        report["original_global_threads"] = previous_threads
    return result, report


def correct_volume(input_file: str | Path, output_file: str | Path, *,
                   reconstruction_threads: int = 1,
                   fitting_threads: int = 1,
                   profile_path: str | Path | None = None) -> dict:
    """MGH/MGZ三维T1→同网格uint8 nu0，保留毫米affine及兼容头尾。

    input_file为有限非负强度；output_file为新输出路径。重建与拟合
    线程均默认1，独立显式指定；多拟合线程不自动用于生产profile。
    profile_path默认None，可保存源码/输入/输出/librarySHA与完整墙钟。
    量化为clip[0,255]后floor(x+.5)，不是截断或banker's rounding。
    返回correct_array报告加文件路径/版本SHA/含加载写出的墙钟。
    仅调用SimpleITK Python库，无FS程序、资产或参考输入；失败传播。
    """
    started = time.perf_counter()
    require_simpleitk_runtime()
    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("expected a 3D MGH/MGZ input")
    result, report = correct_array(np.asarray(source.dataobj, dtype=np.float32),
        spacing=tuple(float(s) for s in source.header.get_zooms()[:3]),
        reconstruction_threads=reconstruction_threads, fitting_threads=fitting_threads)
    uchar = _to_uchar(result)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.get_data_dtype() == np.dtype(np.uint8):
        save_same_dtype_mgh(input_file, output_file, uchar)
    else:
        header = source.header.copy(); header.set_data_dtype(np.uint8)
        nib.save(nib.MGHImage(uchar, source.affine, header), str(output))
    report.update(array_stage_seconds=report["total_seconds"],
        total_seconds=time.perf_counter()-started, file_io_included=True, output=str(output_file),
        input_sha256=hashlib.sha256(Path(input_file).read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if profile_path is not None:
        profile = Path(profile_path); profile.parent.mkdir(parents=True,exist_ok=True)
        profile.write_text(json.dumps(report,indent=2)+"\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file", type=Path)
    parser.add_argument("--o", required=True, dest="output_file", type=Path)
    parser.add_argument("--reconstruction-threads", type=int, default=1)
    parser.add_argument("--fitting-threads", type=int, default=1)
    parser.add_argument("--profile", dest="profile_path", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(correct_volume(**vars(args)),ensure_ascii=False))


if __name__ == "__main__":
    main()
