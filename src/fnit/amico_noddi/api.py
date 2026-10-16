"""统一 AMICO 和连续 Watson NODDI 的公开模型入口。"""

from .core import TorchAMICONODDI


class FNIT_noddi(TorchAMICONODDI):
    """复用 NODDI 模型对象，以 ``method`` 选择已有拟合算法。

    参数
    ----
    device : str、torch.device 或 None
        None 自动使用可用 CUDA，否则 CPU；沿用原实现精度与设备配置。
    method : {"amico", "classic"} 或 None
        ``amico`` 为既有离散字典拟合，``classic`` 为 AMICO 初始化后进行
        连续 Watson 非线性拟合。None 使用旧 ``fit_method``，两者都未设置
        时仍为 ``amico``。显式提供两个不同值会报错。
    config : AMICONODDIConfig 或 None
        原样复用已有配置对象；None 创建原有默认配置。包括扩散率、字典、
        正则项、求解容差和 ``lut_batch_size``，不改变模型或批处理默认值。
    fit_method : {"amico", "classic"} 或 None
        旧参数的兼容写法；新调用推荐使用 ``method``。

    ``model(data, mask, bvecs, bvals)`` 返回原有 AMICONODDIResult；输入是
    对齐的四维 DWI、三维二值 mask、FSL bvec/bval 文件。原有
    ``model.run(..., output_dir=..., naming="ukb", overwrite=False)`` 和
    result.save 完整保留。输出的方向、密度、离散度、自由水图与 QC 均由
    现有实现产生，不在此处转换或重算。
    """

    def __init__(self, device=None, *, method=None, config=None, fit_method=None):
        if method is not None and method not in ("amico", "classic"):
            raise ValueError("method must be 'amico' or 'classic'")
        if fit_method is not None and fit_method not in ("amico", "classic"):
            raise ValueError("fit_method must be 'amico' or 'classic'")
        if method is not None and fit_method is not None and method != fit_method:
            raise ValueError("method and fit_method must select the same algorithm")
        selected = method if method is not None else fit_method
        super().__init__(device=device, config=config,
                         fit_method="amico" if selected is None else selected)

    @property
    def method(self):
        """当前算法名称，与原有 fit_method 保持一致。"""
        return self.fit_method


__all__ = ["FNIT_noddi"]
