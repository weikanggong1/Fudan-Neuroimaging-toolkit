# N4 的纯 PyTorch 实验后端

## 功能简介

`fnit.recon_all.n4_gpu` 使用 PyTorch 在目标 GPU 上估计平滑的对数偏置场，
并输出 float32 的校正体积。`run_input_n4_chain(..., n4_backend="torch")`
提供显式实验入口；生产默认仍使用已经验证的 ITK N4 C++ 程序。

该实现目前是 FNIT 的 GPU 近似算法，不是 ITK N4 的逐步翻译。它不能在
没有真实同输入验证时替换默认 recon-all，也不能把相关性或运行时间直接解释
为整体等效。

## Python 调用

```python
from fnit.recon_all.input_n4_chain import run_input_n4_chain

result = run_input_n4_chain(
    t1="/data/sub-07_T1w.nii.gz",  # 原始 T1w，NIfTI，三维
    subject_dir="/work/sub-07",  # 空的 FNIT/FreeSurfer 风格输出目录
    weights_dir="/models",  # SynthStrip/Talairach 所需权重目录
    assets_dir="/assets",  # 模板和固定资产目录
    device="cuda:0",  # GPU 设备；不启用 FP16/BF16
    threads=8,  # 前置 CPU 阶段线程数
    n4_backend="torch",  # 显式选择实验后端
)
```

输入是原始 T1w 和前置链生成的 `mri/orig.mgz`；输出包括 `tmp/nu0.mgz`、
`mri/nu.mgz`、N4 运行时间、后处理时间和 `n4_backend`。`nu0.mgz` 保持
float32，`nu.mgz` 继续经过现有强度缩放、Talairach-ball 直方图和 uchar
写出流程。坐标和体素网格沿用 `orig.mgz`，单位是图像强度和毫米仿射。

## 命令行与原软件

现有 recon-all 命令仍默认调用：

```bash
fnit-recon-all --t1 /data/sub-07_T1w.nii.gz --subject-dir /work/sub-07
```

对应的 FreeSurfer 参考步骤是 `mri_nu_correct.mni`/`N4BiasFieldCorrection`
及其后续 `mri_segstats`、`mri_make_uchar`。纯 PyTorch 后端目前只提供
Python API 的显式实验选项，尚未接入默认 CLI。

## 真实数据验证

在 gpucw1 的 H100 上，对公开 ds000114 sub-07 的 `orig.mgz` 运行当前实验
实现，输入形状为 `256×256×256`，同步耗时 **1.242 s**，峰值 allocated
**285,213,184 bytes**，峰值 reserved **297,795,584 bytes**。与现有 ITK
`nu0.mgz` 同输入比较：Pearson `r=0.996332`、RMSE `19.2208`、P99 绝对误差
`70.8477`、最大绝对误差 `110.836`。校正后有效体素均值为 `29.1619`，ITK
参考为 `16.7821`。

这些误差仍然过大，故本次没有把 `n4_backend="torch"` 设为生产默认，也
没有声称 recon-all 已经完成纯 PyTorch 迁移。下一步应先实现 ITK N4 的多层
B-spline、直方图锐化和收敛策略，再做两例以上真实 T1 的同输入和下游
`norm/brain/wm/filled` 回归。

## 参考

- Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE TMI (2010)。
- FNIT 的 ITK 封装：`fnit.recon_all.n4_itk`。
- FNIT 的 GPU 实验实现：`fnit.recon_all.n4_gpu`。
