"""真实端到端对照中的官方 AMICO 2.0.3 NODDI 参考步骤。

输入为完整的官方 EDDY 校正 DWI、同网格二值 mask、AP bval 和旋转后的
bvec；输出在新 study 的匿名 case/AMICO/NODDI 下，并将三张参数图复制到
native-output 的 NODDI_ICVF/OD/ISOVF.nii.gz。仅用于私密 benchmark，
不属于 FNIT 运行依赖。官方流程：https://github.com/daducci/AMICO/wiki/NODDI
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time


NATIVE_MAPS = {
    "fit_NDI.nii.gz": "NODDI_ICVF.nii.gz",
    "fit_ODI.nii.gz": "NODDI_OD.nii.gz",
    "fit_FWF.nii.gz": "NODDI_ISOVF.nii.gz",
}
CONFIG_KEYS = (
    "peaks_filename", "doNormalizeSignal", "doKeepb0Intact", "doComputeRMSE",
    "doComputeNRMSE", "doSaveModulatedMaps", "doSaveCorrectedDWI", "doMergeB0",
    "doDebiasSignal", "doDirectionalAverage", "DTI_fit_method", "nthreads",
    "BLAS_nthreads", "b0_thr", "b0_min_signal", "replace_bad_voxels", "lmax",
    "ndirs", "solver_params",
)


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _rotation_cache(amico):
    # 只记录固定分辨率及是否已缓存，不写用户目录或数据路径。
    cache_directory = Path(amico.lut.dipy_home)
    return {
        str(ndirs): (cache_directory / (
            f"AMICO_aux_matrices_lmax=12_ndirs={ndirs}.pickle"
        )).is_file()
        for ndirs in amico.lut.valid_dirs()
    }


def _stage(timings, name, function):
    started = time.perf_counter()
    result = function()
    timings[name] = time.perf_counter() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True,
                        help="全新的私密目录；创建匿名 case 和独立 kernels")
    parser.add_argument("--data", type=Path, required=True,
                        help="完整官方 EDDY 校正后的 4D DWI")
    parser.add_argument("--mask", type=Path, required=True,
                        help="同网格的官方二值脑 mask")
    parser.add_argument("--bvals", type=Path, required=True,
                        help="全部 AP b-value 文件，保持原卷顺序")
    parser.add_argument("--bvecs", type=Path, required=True,
                        help="官方 EDDY 的全部旋转后 b-vector")
    parser.add_argument("--native-output", type=Path, required=True,
                        help="native 目录；写 NODDI_ICVF/OD/ISOVF 三图")
    parser.add_argument("--report", type=Path, required=True,
                        help="新的匿名 JSON 报告，不记录输入路径或被试 ID")
    parser.add_argument("--threads", type=int, default=8,
                        help="官方拟合线程数，默认 8；BLAS 始终为 1")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads 必须是正整数")
    for name in ("data", "mask", "bvals", "bvecs"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} 必须是已有文件")
    args.study_dir = args.study_dir.resolve()
    args.native_output = args.native_output.resolve()
    args.report = args.report.resolve()
    if args.study_dir.exists():
        parser.error("--study-dir 必须是全新目录")
    if args.report.exists():
        parser.error("--report 已存在")
    if any((args.native_output / name).exists() for name in NATIVE_MAPS.values()):
        parser.error("--native-output 已包含 NODDI 输出")

    # 在导入数值库前限定 CPU/BLAS；AMICO 使用自己的 nthreads 拟合线程。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import amico
    import dipy
    import nibabel
    import numpy as np
    import scipy

    if metadata.version("dmri-amico") != "2.0.3":
        raise RuntimeError("此参考脚本要求官方 dmri-amico 2.0.3")
    args.study_dir.mkdir(parents=True, exist_ok=False)
    case_directory = args.study_dir / "case"
    case_directory.mkdir()
    scheme_file = case_directory / "DWI.scheme"
    cache_before = _rotation_cache(amico)
    timings = {}
    started = time.perf_counter()
    _stage(timings, "setup", lambda: amico.core.setup(lmax=12))
    _stage(timings, "scheme", lambda: amico.util.fsl2scheme(
        bvalsFilename=str(args.bvals.resolve()),
        bvecsFilename=str(args.bvecs.resolve()),
        schemeFilename=str(scheme_file), bStep=100,
    ))

    def configure_evaluation():
        evaluation = amico.Evaluation(study_path=str(args.study_dir), subject="case")
        settings = {
            "nthreads": args.threads, "BLAS_nthreads": 1,
            "doNormalizeSignal": True, "doMergeB0": False,
            "doDebiasSignal": False, "doDirectionalAverage": False,
            "DTI_fit_method": "OLS", "peaks_filename": None,
            "doComputeRMSE": True, "doComputeNRMSE": False,
            "doSaveModulatedMaps": False,
        }
        for key, value in settings.items():
            evaluation.set_config(key, value)
        return evaluation

    evaluation = _stage(timings, "evaluation_configuration", configure_evaluation)
    _stage(timings, "load_data", lambda: evaluation.load_data(
        dwi_filename=str(args.data.resolve()),
        scheme_filename=str(scheme_file),
        mask_filename=str(args.mask.resolve()),
        b0_thr=100, b0_min_signal=0, replace_bad_voxels=None,
    ))

    def configure_model():
        evaluation.set_model(model_name="NODDI")
        evaluation.model.set(
            dPar=1.7e-3, dIso=3.0e-3,
            IC_VFs=np.linspace(0.1, 0.99, 12),
            IC_ODs=np.hstack((np.array([0.03, 0.06]), np.linspace(0.09, 0.99, 10))),
            isExvivo=False,
        )
        evaluation.set_solver(lambda1=0.5, lambda2=1e-3)

    _stage(timings, "model_configuration", configure_model)
    _stage(timings, "kernel_generate", lambda: evaluation.generate_kernels(
        regenerate=True, lmax=12, ndirs=500,
    ))
    _stage(timings, "kernel_load", evaluation.load_kernels)
    _stage(timings, "fit", evaluation.fit)
    _stage(timings, "save", evaluation.save_results)
    fitting_wall_seconds = time.perf_counter() - started

    # 官方默认目录含模型名 NODDI；不能从 case/AMICO 直接读取图像。
    configured_output = evaluation.get_config("OUTPUT_path")
    official_output = (
        Path(configured_output) if configured_output is not None
        else case_directory / "AMICO" / "NODDI"
    )

    def copy_native_maps():
        for filename in (*NATIVE_MAPS, "fit_dir.nii.gz", "fit_RMSE.nii.gz"):
            if not (official_output / filename).is_file():
                raise RuntimeError(f"官方输出缺少 {filename}")
        args.native_output.mkdir(parents=True, exist_ok=True)
        for source_name, destination_name in NATIVE_MAPS.items():
            shutil.copyfile(official_output / source_name, args.native_output / destination_name)

    _stage(timings, "copy_native", copy_native_maps)
    wall_seconds = time.perf_counter() - started
    report = {
        "scope": "one real full-brain official EDDY output; official AMICO 2.0.3 NODDI",
        "reference_commit": "df540093b60240c38a6ff2ea4ceb1181c4f3e936",
        "versions": {
            "python": platform.python_version(), "amico": amico.__version__,
            "dipy": dipy.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "nibabel": nibabel.__version__,
            "spams-cython": _distribution_version("spams-cython"),
            "dmri-dicelib": _distribution_version("dmri-dicelib"),
        },
        "input_shape": list(evaluation.niiDWI_img.shape),
        "mask_voxels": int(np.count_nonzero(evaluation.niiMASK_img == 1)),
        "scheme": {"b_step": 100, "b0_volumes": int(evaluation.scheme.b0_count),
                   "dwi_volumes": int(evaluation.scheme.dwi_count)},
        "config": {key: _json_value(evaluation.get_config(key)) for key in CONFIG_KEYS},
        "model": _json_value(evaluation.model.get_params()),
        "cache": {
            "rotation_lmax": 12, "rotation_cache_before": cache_before,
            "rotation_cache_after": _rotation_cache(amico),
            "protocol_kernels_cached_before": False, "regenerate": True,
            "protocol_kernel_files": len(list((args.study_dir / "kernels" / "NODDI").glob("A_*.npy"))),
        },
        "stages_seconds": timings,
        "fitting_wall_seconds_including_io": fitting_wall_seconds,
        "wall_seconds_including_io": wall_seconds,
        "official_solver_seconds": evaluation.get_config("fit_time"),
        "official_direction_seconds": evaluation.get_config("dirs_precomputing_time"),
        "timing_scope": "setup, scheme, load/preprocess, kernels, fit, save, native copy; excludes imports and report generation",
        "outputs": {"native_maps": list(NATIVE_MAPS.values()),
                    "diagnostic_maps": ["fit_dir.nii.gz", "fit_RMSE.nii.gz"]},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
