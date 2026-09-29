# TorchInvWarp：在指定网格上反转 FSL 位移场

`TorchInvWarp` 把定义在 MNI 网格、内容指向 diffusion 的 diffusion→MNI pull 场，反转成定义在 diffusion 网格、内容指向 MNI 的 MNI→diffusion pull 场。反场可交给 `TorchApplyWarp`，把 MNI 掩膜最近邻重采样到 diffusion 网格。PyTorch 在 GPU 上求解每个输出体素的反向坐标；默认用全局仿射估计初始化，再迭代修正。运行时不调用 FSL。

## Python 单被试调用

```python
from fnit import TorchInvWarp

result = TorchInvWarp(device="cuda:0").run(
    reference="/data/nodif_brain_mask.nii.gz",  # 反场输出网格：diffusion 空间，对应 --ref
    warp="/data/diff_to_MNI_warp.nii.gz",  # 前向组合场：MNI 网格上的 pull 位移，对应 --warp
    warp_convention="relative",  # 输入 dense 场是相对位移；auto 按 FSL intent 识别
    output_convention="relative",  # 输出相对位移，供 applywarp 使用
    iterations=30,  # 每个体素固定点迭代的最大次数
    tolerance_mm=0.01,  # 迭代修正量停止阈值，单位 mm
    output="/data/MNI_to_diff_warp.nii.gz",  # 输出 4D NIfTI，末轴长 3，float32，单位 mm
)
print(result.image.shape, result.valid_fraction, result.qc)
```

输出 shape 是 `reference.shape + (3,)`，affine、qform、sform 沿用 `reference`；相对场 intent 为 FSL displacement 2006。`valid_fraction` 记录最终求逆位置落在原场网格内的比例；`qc` 的残差只统计这些位置。输入支持 dense 位移和 FNIRT 三次样条系数。场外位置采用边界采样，不能据此解释场外逆变换的解剖意义。

## 命令行及 FSL 对应

```bash
# --ref 是反场的 diffusion 输出网格；--warp 是组合后的 diffusion→MNI 场。
# --rel 指定输入和输出均为相对位移；--out 保存反场。
fnit invwarp --ref /data/nodif_brain_mask.nii.gz \
  --warp /data/diff_to_MNI_warp.nii.gz \
  --out /data/MNI_to_diff_warp.nii.gz --rel --device cuda:0

invwarp --ref=/data/nodif_brain_mask.nii.gz \
  --warp=/data/diff_to_MNI_warp.nii.gz \
  --out=/data/fsl_MNI_to_diff_warp.nii.gz --rel
```

独立入口 `fnit-invwarp` 接受相同选项。`--niter` 设置最大迭代数，`--abs` 选择绝对输入和输出，`--overwrite` 允许覆盖文件。Python 接口还可分别指定输入、输出约定。

## 真实数据对照与差异

同一例真实 DWI 的 FNIT dMRI pipeline TBSS 与 MMORF 分支分别求逆：前向 dense 场均为 `182×218×182×3`，反场均为 `104×104×72×3`；FNIT GPU 单次 Python 调用耗时分别 0.79、0.81 秒。两条分支的第 9 号 JHU MNI 掩膜在 diffusion 网格分别为 120、114 个体素；各自自动转换后的追踪与直接使用所保存 diffusion 掩膜的追踪完全一致。两条配准分支的场不同，掩膜体素数不用于比较算法精度。[分支验证](../../validation/probtrackx/README.md)记录追踪结果。

在各分支固定同一前向场后与 FSL 6.0.7.4 `invwarp` 比较，TBSS 反场脑内平均向量差 0.0058 mm，MNI 掩膜 Dice 0.9917；MMORF 转成 FSL dense 场后，脑内差 0.0670 mm，掩膜 Dice 0.9912。反场**未达到与 FSL 逐体素数值等价**。FSL 完整 `invwarp` 命令分别为 110.82、137.49 秒；完整命令与 FNIT Python 调用计时边界不同。两张图上排是完整 FA 切面，下排放大该小 ROI，右列显示两套二值掩膜不同的体素。比较边界与源码哈希见[同输入 FSL 报告](../../validation/invwarp/README.md)。

![TBSS：真实 MNI 掩膜反变换](figures/real_tbss_inverse_mask.png)

![MMORF：真实 MNI 掩膜反变换](figures/real_mmorf_inverse_mask.png)

## Reference

- 参考文献：Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。 `invwarp` 没有单独的方法论文。
- 原实现代码库：[FSL `fnirt`（含 `invwarp`）](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
