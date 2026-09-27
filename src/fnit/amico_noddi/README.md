# TorchAMICONODDI 源码目录

`TorchAMICONODDI` 在 EDDY 校正后的多壳 DWI 上拟合 AMICO NODDI，输出 ICVF/NDI、ODI、ISOVF/FWF、主方向和拟合 RMSE。核生成、AMICO/DIPY OLS tensor 初始化、线性求解和影像输出由本包完成，运行时不调用 AMICO 或 DIPY。

```python
from fnit import TorchAMICONODDI

model = TorchAMICONODDI(
    device="cuda:0",  # 计算设备；可改为 "cpu"
    config=None,  # None 使用 AMICO 2.0.3 对应的默认 NODDI 参数
)
result = model.run(
    data="eddy/data.nii.gz",  # 输入：EDDY 校正后的 4D DWI
    mask="eddy/nodif_brain_mask.nii.gz",  # 输入：同网格 3D 脑 mask
    bvecs="eddy/data.eddy_rotated_bvecs",  # 输入：EDDY 旋转后的梯度方向
    bvals="AP.bval",  # 输入：与 DWI 顺序一致的 b-value
    output_dir="noddi",  # 输出：五张 NIfTI 和 QC 的目录
    naming="amico",  # 输出命名：使用官方 AMICO fit_* 文件名
    overwrite=False,  # 是否覆盖已有输出
)
```

官方等价流程需在 AMICO 2.0.3 中依次运行 `fsl2scheme`、`Evaluation.load_data()`、`set_model("NODDI")`、kernel 生成/加载、solver 设置、`fit()` 和 `save_results()`；完整可执行 Python 对照见[功能说明](../../../docs/amico_noddi/README.md)。

FNIT 0.14.0 调用链已在 242,261 个真实 dMRI mask 体素上与 AMICO 2.0.3 重跑；报告记录的数值文件与当前 0.16.0 逐文件 SHA-256 相同。NDI、ODI、FWF 最大误差均为 `5.960e-8`，RMSE 和 726,783 个方向分量逐元素相同；FNIT H100 本次完整 API wall time `66.35 s`，峰值 CUDA allocation `9.95 GB`。逐体素结论绑定 NumPy 1.26.4 官方 manylinux OpenBLAS64 wheel；其他 NumPy 构建的 QC 不声明数值等价。输入输出、环境固定、官方 Python 序列、当前源码哈希、时间和示意图见[功能说明](../../../docs/amico_noddi/README.md)。
