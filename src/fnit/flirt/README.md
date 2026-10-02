# TorchFLIRT 源码目录

[功能、输入输出与用法](../../../docs/flirt/README.md) · [真实数据验证](../../../validation/flirt/README.md) · [MNI152 1→2 mm 矩阵](assets/FSL_MNI152_T1_1mm_to_2mm.mat)

本目录实现 12-DOF/correlation-ratio、6-DOF/NMI 配准和已知线性变换的重采样，运行时不调用 FSL。

| 文件 | 职责 |
|---|---|
| `core.py` | header `pixdim` 驱动的图像金字塔、FSL cost、默认 `search 12` 的 8/4/2/1 mm 搜索，以及输出类型、背景、qform/sform 与 `applyxfm`；保留串行 cost 和 Brent 优化器。 |
| `affine_batch.py` | 在 CPU 批量构造 affine，保留参考实现的 double 运算与 float32 角度舍入。 |
| `batched.py` | 按显存分块执行候选采样、CorrRatio/NMI cost 与缓存，保持候选顺序。 |
| `search.py` | 协同调度独立 Brent 搜索；CPU 保留各搜索的状态和分支，GPU 批量计算下一轮 cost。 |
| `_batched_cuda.py` | Triton 融合候选坐标、三线性采样、边缘权重和归约；保留逐步 float32 舍入及参考 CUDA 分组，关闭 FMA。 |
| `coordinates.py` | FSL scaled-mm 与 world-RAS 矩阵转换。 |
| `types.py` | nibabel 输入检查及 `FLIRTResult`。 |
| `standalone.py` | 路径接口与原子写盘。 |
| `cli.py`、`__main__.py` | `fnit-flirt` 命令行入口。 |

CPU 默认使用串行参考路径，CUDA 默认使用批量路径；有权重的采样保留张量实现。两条路径共享修正后的 cost、schedule 和输出规则，融合执行的真实数据 gate 检查矩阵、影像、cost 与搜索结果保持不变。完整执行参数、Triton 回退规则、FSL 误差和计时边界见功能页及验证页。随包提供的 MNI152 矩阵仅对功能页列出的原始模板哈希完成验证。
