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

2026-09-29 对照使用的原软件命令为 `eddy_cuda10.2 --imain=AP.nii.gz --mask=nodif_brain_mask.nii.gz --topup=fieldmap_out --acqp=acqparams.txt --index=eddy_index.txt --bvecs=AP.bvec --bvals=AP.bval --out=eddy/data --ref_scan_no=0 --initrand=12345 --flm=quadratic --resamp=jac --slm=linear --niter=8 --fwhm=10,8,4,2,0,0,0,0 --ff=10 --sep_offs_move --nvoxhp=1000 --repol --rms`。每项输入的格式、各个输出文件的结构、实测精度和时间见[功能说明](../../../docs/eddy/README.md)。

2026-10-02 的执行优化在单次调用内复用固定 voxel grid、二次 EC 基函数、已校验的 PE 轴、TOPUP susceptibility 样条系数及其 mirror padding。每次 GP 拟合把当轮有效掩膜复制到 CPU 一次，再按相同 glibc `rand` 顺序选点。每次 GN 更新只复用该次预测的 periodic padding；work 图像和不同 GP 的预测保持即时计算。缓存随调用结束释放，不跨被试或离群替换缓存图像。算法、八轮顺序、dtype 与输出格式保持原值。数值导数说明也修正为代码实际使用的单侧差分。

2026-10-02 真实完整八轮验收使用 `104×104×72×105` AP 数据、242,316 体素脑掩膜（5 个 b0、50 个 b≈1000、50 个 b≈2000），固定 `gp_seed=12345`、`ref_scan_no=0`，共享 H100、8 个 CPU 线程及 20,000,000,000 bytes CUDA allocation 上限。冻结 FNIT 基线 `954ad19` 与候选版的 wall time 分别为 484.753275 s 和 404.307387 s，均包含保存 I/O；CUDA allocation 峰值为 4.842674 GB 和 4.891256 GB（十进制）。八个返回数组及 QC（排除耗时和显存字段）、保存的压缩 NIfTI SHA-256/header/affine、数值 sidecars 和离群 report 全部相同。一次配对观察的 wall time 降低 16.6%；稳定速度结论需要重复配对计时，不能由此推断普遍提速或与 FSL 官方逐值等价。详见[本轮验证报告](../../../validation/dmri_pipeline/lossless_20261002.md)；旧版对 FSL 的精度边界保留在[功能说明](../../../docs/eddy/README.md)。
