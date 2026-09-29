# TorchEDDY 源码目录

`TorchEDDY` 的公开接口位于 `fnit.eddy`；数值实现位于 `fsl2111_strict/`。该路径使用 PyTorch 计算 FSL EDDY 2111.0 对应 UKB 配置的 b0/DWI 配准、Gaussian-process 预测、离群切片替换及 Jacobian 重采样。`topup_field.py` 读取 TOPUP 的 B-spline 系数，`ukb.py` 为单个 UKB 风格病例准备输入。运行时不调用 FSL。

```python
from fnit import TorchEDDY

result = TorchEDDY(
    device="cuda:0",  # 计算设备
).run(
    imain="AP.nii.gz",  # 输入：单被试 4D DWI
    mask="nodif_brain_mask.nii.gz",  # 输入：DWI 网格的脑掩膜
    acqp="acqparams.txt",  # 输入：PE 向量和总读出时间
    index="eddy_index.txt",  # 输入：每个 volume 对应的 acqp 行号
    bvecs="AP.bvec",  # 输入：3×N 梯度方向
    bvals="AP.bval",  # 输入：N 个 b-value
    topup="fieldmap_out",  # 输入：TOPUP 结果前缀；无 TOPUP 时设 None
    ref_scan_no=0,  # 输入：参考 volume 的 0-based 编号
    gp_seed=12345,  # 输入：可复核比较的 GP 选点随机种子
    out="eddy/data",  # 输出：结果文件前缀
    overwrite=False,  # 不覆盖已有文件
)
```

这次对照的原软件命令为 `eddy_cuda10.2 --imain=AP.nii.gz --mask=nodif_brain_mask.nii.gz --topup=fieldmap_out --acqp=acqparams.txt --index=eddy_index.txt --bvecs=AP.bvec --bvals=AP.bval --out=eddy/data --ref_scan_no=0 --initrand=12345 --flm=quadratic --resamp=jac --slm=linear --niter=8 --fwhm=10,8,4,2,0,0,0,0 --ff=10 --sep_offs_move --nvoxhp=1000 --repol --rms`。每项输入的格式、各个输出文件的结构、实测精度和时间见[功能说明](../../../docs/eddy/README.md)。
