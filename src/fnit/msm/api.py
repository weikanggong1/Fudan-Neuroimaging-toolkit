"""统一 MSMSulc 与 MSMAll 的公开入口；数值计算沿用各自成熟实现。"""


def FNIT_newmsm(
    inputs, output_dir, *, method="msmsulc", device="cuda:0", config=None,
    execution="optimized", parallel=True, cpu_threads=None, qc_policy="report",
):
    """按 ``method`` 执行双侧球面配准并返回 ``{"L": Path, "R": Path}``。

    参数
    ----
    inputs : dict
        仅包含 ``L``、``R`` 的输入映射。``msmsulc`` 的值是
        :class:`MSMSulcInputs`，包含原生/旋转球面、脑沟图、参考球面/脑沟图
        和初始仿射；``msmall`` 的值是 :class:`MSMAllInputs`，包含源/参考
        球面和多列特征，以及可选初始球面和代价权重。两种数据结构不混用。
    output_dir : str 或 Path
        输出目录；保存双侧原生顶点顺序球面 GIFTI 和 registration_report.json。
    method : {"msmsulc", "msmall"}
        默认 ``msmsulc``，只选择现有算法，不转换特征或改变求解规则。
    device : str
        默认 ``cuda:0``；也可为 ``cpu``，保持对应后端的精度设置。
    config : MSMSulcConfig、MSMAllConfig、str、Path 或 None
        对应方法的配置对象或官方配置文件路径；None 使用该方法原有默认计划。
    execution : {"optimized", "reference"}
        保留原有执行方式，默认 ``optimized``。
    parallel : bool
        是否并行左右半球，默认 True。
    cpu_threads : int 或 None
        双侧合计主机线程预算；None 沿用原实现的环境/线程预算解析。
    qc_policy : {"report", "repair", "error"}
        MSMSulc 原生球面翻折的记录、修复或报错策略，默认 ``report``。
        MSMAll 尚不通过该参数控制输出；非默认值会报错，避免忽略请求。

    本入口不启动 newMSM 程序；输入读取、计算和报告全部交给现有 FNIT 实现。
    """
    if method not in ("msmsulc", "msmall"):
        raise ValueError("method must be 'msmsulc' or 'msmall'")
    if method == "msmall" and qc_policy != "report":
        raise ValueError("qc_policy applies only to method='msmsulc'")
    from . import run_msmall, run_msmsulc

    options = dict(device=device, config=config, execution=execution,
                   parallel=parallel, cpu_threads=cpu_threads)
    if method == "msmsulc":
        return run_msmsulc(inputs, output_dir, qc_policy=qc_policy, **options)
    return run_msmall(inputs, output_dir, **options)


__all__ = ["FNIT_newmsm"]
