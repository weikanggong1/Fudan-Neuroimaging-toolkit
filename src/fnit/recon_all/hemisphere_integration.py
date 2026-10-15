"""完整Python半球连续链的生产接线；共享缺陷累计和终态指标保持原顺序。"""
import json
from pathlib import Path


def run_persistent_cortical_chain(*, subject, report, stage, common_kwargs,
                                 device, registration_device, threads,
                                 profile_stages, assets, defect_binary=None,
                                 defects_backend="torch", metrics_binary=None):
    """执行完整双侧连续链并更新本次report，返回持久调度原始报告。

    subject是自产被试目录；report为父运行dict；stage为原计时/失败回调。
    common_kwargs是每个半球操作的JSON参数；device用于共享诊断和指标，
    registration_device用于fresh worker，threads为双侧总预算。assets是
    已声明图谱模板；两个binary参数仅传None，禁止原程序回退。
    体积为conformed网格，表面为surface RAS/mm。surface成功发布后严格
    LH清零、RH累计defects；其余半球依赖各自推进。全部成功后指标仍
    按LH/RH串行计算。子失败保留其完整调度报告并抛原异常，不补输出。
    缓存仅在fresh worker启用，父已初始化CUDA的策略不变。无独立官方CLI，
    对应完整recon-all皮层链；逐项参数/例子见PERSISTENT_HEMISPHERE_PIPELINE。
    """
    from .hemisphere_parallel import HemisphereGroupError
    from .hemisphere_pipeline import run_hemisphere_pipeline
    from .native_free import (_consume_hemisphere_result, _run_defects_volume,
                              _finish_cortical_metrics)
    if common_kwargs.get("surface_placement_backend") != "python":
        raise ValueError("persistent cortical chain requires complete Python placement")
    if defect_binary is not None or metrics_binary is not None:
        raise ValueError("persistent Python chain cannot use original stage programs")
    subject = Path(subject)
    operations = ("surface", "register", "annotation", "final_white", "pial_surface")

    def published(operation, hemi, value, row):
        _consume_hemisphere_result(report, operation, hemi, value)
        if operation == "surface":
            # 调度器保证surface回调LH→RH；诊断不进入后续优化器输入。
            stage(f"defects_{hemi}", _run_defects_volume, None, subject,
                  hemi, assets, backend=defects_backend, device=device)
        (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))

    try:
        pipeline = stage("persistent_hemisphere_pipeline", run_hemisphere_pipeline,
                         subject, operations, device=str(registration_device), threads=threads,
                         kwargs=common_kwargs, stage_hook=published,
                         profile_stages=profile_stages, cuda_allocator_cache="enabled")
    except HemisphereGroupError as error:
        report["hemisphere_scheduling"]["groups"].append(error.report)
        try:
            (subject / "fnit-native-free-run.json").write_text(json.dumps(report, indent=2))
        except Exception as metadata_error:
            error.add_note(f"Persistent failure metadata could not be saved: {metadata_error!r}")
        raise
    report["hemisphere_scheduling"]["groups"].append(pipeline)
    for hemi in ("lh", "rh"):
        metrics = stage(f"finish_metrics_{hemi}", _finish_cortical_metrics,
                        subject, hemi, None, assets, device=device, python_only=True)
        report["surfaces"][hemi].update(metrics)
    return pipeline
