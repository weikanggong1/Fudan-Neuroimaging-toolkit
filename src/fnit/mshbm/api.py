"""统一已有 profile、时序和体积 MS-HBM 输入；不更改推断内核。"""


def FNIT_mshbm(
    data, assets=None, *, input_kind="profiles", w=200.0, c=50.0,
    max_outer=50, max_em=101, max_m=300, max_lambda=101,
    censor=None, left_surface=None, right_surface=None, cortical_mask=None,
    output_dir=None, device="cuda:0", frame_chunk=8, max_distance_mm=3.0,
):
    """执行单被试 fsLR32k / HCP_40 的 17 网络 MS-HBM。

    参数
    ----
    data : 数组列表、二维数组或 NIfTI 路径
        由 ``input_kind`` 明确指定解释，不自动猜测空间或重采样。
    assets : dict 或 None
        已加载的原有 MS-HBM 先验；None 交给 load_assets 加载包内固定先验。
        已传入的对象原样复用。
    input_kind : {"profiles", "timeseries", "volume"}
        ``profiles``（默认）：至少两个 [59412,1483] session profile，
        原样传给现有 parcellate；``timeseries``：[T,59412] 皮层时序，
        经现有 profiles_from_timeseries 按原规则剔除帧并拆为两个 session；
        ``volume``：四维 MNI BOLD NIfTI，调用原有 parcellate_volume，
        需要左右中厚度表面、同网格皮层 mask 和输出目录。
    w, c : float
        空间先验和 MRF 权重，默认 200.0、50.0，直接传给原实现。
    max_outer, max_em, max_m, max_lambda : int
        profile/时序推断的原有迭代上限，默认 50、101、300、101。
        体积入口原本未暴露这些参数，选择 volume 时只能保留这些默认值；
        非默认值会报错，不会被忽略。
    censor : 数组或 None
        时序/体积入口每帧一个 0/1 值，沿用原有筛帧规则；profile 不接受。
    left_surface, right_surface : str 或 Path
        volume 所需的左右 fsLR32k 中厚度表面，处于体积的 scanner-RAS 空间。
    cortical_mask : str 或 Path
        volume 所需的同网格三维皮层 mask。
    output_dir : str 或 Path
        volume 输出目录；保存标签、皮层 CIFTI、17 网络时序/相关矩阵及
        labels_mni.nii.gz。profile/时序入口与原 parcellate 一样仅返回数据，
        不接受输出目录；可另用已有输出函数保存。
    device : str
        volume 投影和标签映射的设备，默认 cuda:0；MS-HBM 推断仍是原 CPU 核。
    frame_chunk : int
        volume GPU 采样每批帧数，默认 8。
    max_distance_mm : float
        volume 体素到中厚度顶点映射距离上限，默认 3.0 mm。

    返回
    ----
    (labels, history)
        原有 64984 顶点标签数组与迭代记录。所有分支均复用既有函数，
        不训练或修改先验，不改变网络编号、阈值、推断精度和收敛规则。
    """
    if input_kind not in ("profiles", "timeseries", "volume"):
        raise ValueError("input_kind must be 'profiles', 'timeseries' or 'volume'")
    defaults = (50, 101, 300, 101)
    iterations = (max_outer, max_em, max_m, max_lambda)
    volume_values = (left_surface, right_surface, cortical_mask, output_dir)
    if input_kind == "volume":
        if any(value is None for value in volume_values):
            raise ValueError("volume input requires left_surface, right_surface, cortical_mask and output_dir")
        if iterations != defaults:
            raise ValueError("iteration controls apply only to profiles or timeseries input")
        from . import parcellate_volume
        return parcellate_volume(
            data, left_surface, right_surface, cortical_mask, output_dir,
            assets=assets, censor=censor, w=w, c=c, device=device,
            frame_chunk=frame_chunk, max_distance_mm=max_distance_mm,
        )
    if any(value is not None for value in volume_values):
        raise ValueError("surface paths, cortical_mask and output_dir apply only to volume input")
    if input_kind == "profiles" and censor is not None:
        raise ValueError("censor applies only to timeseries or volume input")
    if device != "cuda:0" or frame_chunk != 8 or max_distance_mm != 3.0:
        raise ValueError("device, frame_chunk and max_distance_mm apply only to volume input")
    from . import load_assets, parcellate, profiles_from_timeseries
    selected_assets = load_assets() if assets is None else assets
    profiles = (profiles_from_timeseries(data, selected_assets, censor=censor)
                if input_kind == "timeseries" else data)
    return parcellate(
        profiles, selected_assets, w=w, c=c, max_outer=max_outer, max_em=max_em,
        max_m=max_m, max_lambda=max_lambda,
    )


__all__ = ["FNIT_mshbm"]
