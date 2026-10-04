# TorchFLIRT 源码目录

[功能、输入输出与用法](../../../docs/flirt/README.md) · [本轮 CPU/GPU 对照](../../../validation/multimodal_cpu_20261004/README.md) · [历史真实数据验证](../../../validation/flirt/README.md) · [MNI152 1→2 mm 矩阵](assets/FSL_MNI152_T1_1mm_to_2mm.mat)

本目录实现 12-DOF/correlation-ratio、6-DOF/NMI 配准和已知线性变换的重采样，运行时不调用 FSL。

| 文件 | 职责 |
|---|---|
| `core.py` | header `pixdim` 驱动的图像金字塔、FSL cost、默认 `search 12` 的 8/4/2/1 mm 搜索，以及输出类型、背景、qform/sform 与 `applyxfm`；保留串行 cost 和 Brent 优化器。 |
| `affine_batch.py` | 在 CPU 批量构造 affine，保留参考实现的 double 运算与 float32 角度舍入。 |
| `batched.py` | 按内存或显存分块执行 affine 候选采样、CorrRatio/NMI cost 与缓存，保持候选顺序；CPU 复用融合采样和保序箱统计，保留原箱级 pack、归约及 cost。 |
| `_cpu.py` | 仅 CPU 的 Numba 融合 float32 坐标、三线性采样、权重和保序分箱累加；关闭 fastmath，保留原 PyTorch 箱级 cost、熵与搜索。CUDA 不导入此模块。 |
| `_cpu_simd.py` | 仅 CPU 的八体素 LLVM float32 坐标、三线性采样、边缘渐变和连续影像权重指令；关闭 FMA，支持任意数组 stride，非有限值、溢出及大轴端点交回安全标量路径。 |
| `search.py` | 协同调度独立 Brent 搜索；CPU 保留各搜索的状态和分支，GPU 批量计算下一轮 cost。 |
| `_batched_cuda.py` | Triton 融合候选坐标、三线性采样、边缘权重和归约；保留逐步 float32 舍入及参考 CUDA 分组，关闭 FMA。 |
| `coordinates.py` | FSL scaled-mm 与 world-RAS 矩阵转换。 |
| `types.py` | nibabel 输入检查及 `FLIRTResult`。 |
| `standalone.py` | 路径接口与原子写盘。 |
| `cli.py`、`__main__.py` | `fnit-flirt` 命令行入口。 |

CPU 默认使用串行参考搜索；单线程 correlation-ratio 使用八体素向量采样，支持连续影像权重，多线程并行独立采样行，各路径都按原体素顺序累计分箱。CPU 的有限 float32 金字塔图像复用融合输出采样器；梯度、异常输入和 CUDA 保留原分支。CPU 显式批量路径复用融合采样和保序箱统计，保留原分块、箱级 pack、compact_sum、cost 和搜索；它批量处理同一病例的 affine 候选。NMI 非有限或超范围统计回退到原张量方法。CUDA 默认使用批量路径，有权重的采样保留张量实现。两条路径共享修正后的 cost、schedule 和输出规则，融合执行的真实数据 gate 检查矩阵、影像、cost 与搜索结果保持不变。完整执行参数、Triton 回退规则、FSL 误差和计时边界见功能页及验证页。随包提供的 MNI152 矩阵仅对功能页列出的原始模板哈希完成验证。

当前 `core.py`、`_cpu.py`、`_cpu_simd.py`、`batched.py` 与 v25/v28 冻结版相同；[功能页](../../../docs/flirt/README.md#真实数据测量)列出源码哈希范围、CPU 默认/连续权重/显式批量的各版完整计时及最新 H100 两组 API 中位数。[H100 回归](../../../validation/multimodal_cpu_20261004/gpu_flirt_v25_20261004.public.json)的两个完整例各有 16 次保存调用，全部体素、矩阵和空间元数据与优化前相同，allocation 峰值相同。精度与共享负载按对应报告解读。
