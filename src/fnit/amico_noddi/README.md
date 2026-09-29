# TorchAMICONODDI 源码目录

`TorchAMICONODDI` 在 EDDY 校正后的多壳 DWI 上拟合 NODDI，默认使用 AMICO 字典；`fit_method="classic"` 从 AMICO 解开始优化连续 Watson 三室模型的体积分数、离散度、主方向和 b0 幅度。两种方法都输出 ICVF/NDI、ODI、ISOVF/FWF、方向和拟合 RMSE。运行时不调用 AMICO、NODDI Toolbox 或 DIPY。

```python
from fnit import TorchAMICONODDI

model = TorchAMICONODDI(
    device="cuda:0",  # 计算设备；可改为 "cpu"
    config=None,  # None 使用 AMICO 2.0.3 对应的默认 NODDI 参数
    fit_method="classic",  # 连续 Watson 非线性拟合；默认 "amico" 保持原数值路径
)
result = model.run(
    data="eddy/data.nii.gz",  # 输入：EDDY 校正后的 4D DWI
    mask="eddy/nodif_brain_mask.nii.gz",  # 输入：同网格 3D 脑 mask
    bvecs="eddy/data.eddy_rotated_bvecs",  # 输入：EDDY 旋转后的梯度方向
    bvals="AP.bval",  # 输入：与 DWI 顺序一致的 b-value
    output_dir="noddi",  # 输出：五张 NIfTI 的目录；QC 在 result.qc
    naming="amico",  # 输出命名：使用官方 AMICO fit_* 文件名
    overwrite=False,  # 是否覆盖已有输出
)
```

官方等价流程需在 AMICO 2.0.3 中依次运行 `fsl2scheme`、`Evaluation.load_data()`、`set_model("NODDI")`、kernel 生成/加载、solver 设置、`fit()` 和 `save_results()`；完整可执行 Python 对照见[功能说明](../../../docs/amico_noddi/README.md)。

原 AMICO 路径已在 242,261 个真实 dMRI mask 体素上与 AMICO 2.0.3 重跑，NDI、ODI、FWF 最大误差均为 `5.960e-8`。经典连续拟合的实测结果单独列于[功能说明](../../../docs/amico_noddi/README.md)；它和离散字典法不具有逐值等价关系。
