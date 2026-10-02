# TorchAMICONODDI 源码目录

`TorchAMICONODDI` 在 EDDY 校正后的多壳 DWI 上拟合 NODDI，默认使用 AMICO 字典；`fit_method="classic"` 从 AMICO 解开始优化连续 Watson 三室模型的体积分数、离散度和主方向，按原版使用 b0 均值及其噪声估计。两种方法都输出 ICVF/NDI、ODI、ISOVF/FWF、方向和拟合 RMSE。运行时不调用 AMICO、NODDI Toolbox 或 DIPY。

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

优化前的 AMICO 实现在一例 242,261 个真实脑 mask 体素上与 AMICO 2.0.3 比较，NDI、ODI、FWF 最大误差分别为 `0.04073`、`0.05376`、`0.06689`；该次 NumPy BLAS 构建未匹配逐值等价验证环境。输入、输出和源码哈希、经典模式的原版对照见[功能说明](../../../docs/amico_noddi/README.md)。

2026-10-02 的更新在第一与第三阶段 NNLS 仅复用完整 float64 Gram，linear 每次按原形状重新计算，组织阶段独立求解；经典拟合接受数在设备端按 int64 精确累计、全部拟合完成后读取一次。最初同时缓存 Gram/linear 的候选未获耗时收益且增加显存，最终移除了跨组织阶段的 linear 缓存。LUT 批量仍为 400，dtype、阈值、迭代和输出契约使用原设置。

最终版本与旧 FNIT 在同一 `104×104×72×105` 真实 DWI、242,261 个脑体素上完成 H100、20 GB 进程显存上限的 ABBA 比较。包含读写的 AMICO 中位耗时为 `30.76993`→`27.45137 s`，classic 为 `68.15137`→`79.80968 s`；allocated 峰值 `9.95255`→`10.02013 GB`。五张图的未舍入解码数组、保存文件、header、affine 和结果 QC 完全相同，classic 接受更新数均为 `3,710,053`。共享 GPU 耗时波动较大，本轮未建立稳定提速；保留改动的依据是该例没有数值回归且显存增量较小。固定初始化的 classic QC 单项对照也全部相同，支持消除逐轮主机读取，末次主机计时较高的原因未隔离，不据此推断大的加速。原始计时、源码和输入绑定见[本次真实验收报告](../../../validation/dmri_pipeline/lossless_20261002.md)，测试和内部 A/B hook 见[更新与验收说明](../../../docs/amico_noddi/README.md#计算复用更新与差分验收)。
