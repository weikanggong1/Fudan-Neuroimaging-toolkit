# ConvertWarp CPU 与 GPU 对照：2026-10-04

## 1. 版本与计时范围

本次冻结 `all_v7` 的 ConvertWarp 官方配对已完成 **28/28** 条。全部 14 种输入/输出组合在 CPU 1/8 线程下通过精度检查。

函数源码 SHA-256 为 `5c25a942…0518c`，adapter 为 `0e71b19b…e045f`；协调任务随后冻结的 all_v11 使用相同的这两个文件。完整 SHA、每轮时间、分区误差和空间头信息见[公开机器报告](benchmark_20261004.public.json)。

nodecw10 使用独立物理核：1 线程为 `65`，8 线程为 `65,69,73,77,81,85,89,93`。组内持 ConvertWarp 专属锁，原版 FSL 6.0.7.4、起点 main 与本次 FNIT 串行交替运行。其他功能组使用另外的物理核。真实影像及其注册输出完整参与读写，没有用模拟场或小 ROI 代替。

完整进程时间包含启动、Python 导入或原版初始化、全部输入读取、转换及未压缩 `.nii` 保存；排除一轮 warmup 后取三轮配对中位数。API 单独在已导入的工作进程中 warmup 后重复三次，仍包含每次的读取与保存。原版进程与 FNIT API 的计时范围不同，速度比只用完整进程时间。

## 2. 原版与本次版本：完整进程

| 输入与输出 | FSL 1 CPU / s | FNIT 1 CPU / s | FSL/FNIT | FSL 8 CPU / s | FNIT 8 CPU / s | FSL/FNIT | 最大分量差 / mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 三次样条→相对 | 3.2809 | 2.1275 | 1.54× | 3.3974 | 2.2781 | 1.49× | 9.53674e-06 |
| 三次样条→绝对 | 3.0768 | 2.0702 | 1.49× | 3.2954 | 2.2388 | 1.47× | 1.52588e-05 |
| dense relative→相对 | 3.3592 | 2.2029 | 1.52× | 3.2081 | 2.1951 | 1.46× | 9.53674e-06 |
| dense relative→绝对 | 3.2507 | 2.2332 | 1.46× | 2.9962 | 2.3427 | 1.28× | 1.52588e-05 |
| dense absolute→相对 | 3.0960 | 2.4510 | 1.26× | 3.2450 | 2.1294 | 1.52× | 3.05176e-05 |
| dense absolute→绝对 | 3.4166 | 2.0743 | 1.65× | 3.0040 | 2.1978 | 1.37× | 3.05176e-05 |
| dense auto→相对 | 4.9479 | 2.1192 | 2.33× | 5.0399 | 2.3170 | 2.18× | 1.62125e-05 |
| dense auto→绝对 | 5.2419 | 1.9203 | 2.73× | 5.1500 | 2.3196 | 2.22× | 1.52588e-05 |
| 恒等前后矩阵→相对 | 9.5381 | 2.1139 | 4.51× | 9.5142 | 2.1888 | 4.35× | 9.53674e-06 |
| 恒等前后矩阵→绝对 | 9.0692 | 2.0440 | 4.44× | 8.7484 | 2.2215 | 3.94× | 1.52588e-05 |
| 非恒等前后矩阵→相对 | 8.7904 | 2.2453 | 3.92× | 8.7488 | 2.4787 | 3.53× | 2.47955e-05 |
| 非恒等前后矩阵→绝对 | 8.7127 | 2.5576 | 3.41× | 8.7737 | 2.1659 | 4.05× | 3.05176e-05 |
| 极小 postmat→相对 | 6.4235 | 2.1197 | 3.03× | 6.5906 | 2.0821 | 3.17× | 1.14441e-05 |
| 极小 postmat→绝对 | 6.7186 | 2.2723 | 2.96× | 6.2237 | 2.0751 | 3.00× | 1.52588e-05 |

所有已完成输出为 `91×109×91×3`、float32、有限值；affine、qform/sform 与参考一致。qform code 为 0，sform code 为 2，sform 数值差为 0。这里的 1/8 表示共同线程预算和亲和性，原版程序的实际并行利用率不由预算数推断。

## 3. 已导入 API：起点 main 与本次版本

| 输入与输出 | main 1 CPU / s | 本次 1 CPU / s | main 8 CPU / s | 本次 8 CPU / s |
|---|---:|---:|---:|---:|---:|
| 三次样条→相对 | 0.3564 | 0.2307 | 0.1847 | 0.1409 |
| 三次样条→绝对 | 0.3544 | 0.2319 | 0.3183 | 0.1154 |
| dense relative→相对 | 0.3226 | 0.3735 | 0.2258 | 0.1745 |
| dense relative→绝对 | 0.3152 | 0.3148 | 0.5351 | 0.2422 |
| dense absolute→相对 | 0.3221 | 0.2516 | 0.4419 | 0.2276 |
| dense absolute→绝对 | 0.4089 | 0.3618 | 0.3263 | 0.2070 |
| dense auto→相对 | 0.3009 | 0.2955 | 0.3848 | 0.1956 |
| dense auto→绝对 | 0.3063 | 0.3830 | 0.3314 | 0.2840 |
| 恒等前后矩阵→相对 | 0.2883 | 0.2297 | 0.2961 | 0.3189 |
| 恒等前后矩阵→绝对 | 0.3361 | 0.2617 | 0.2580 | 0.1023 |
| 非恒等前后矩阵→相对 | 0.2829 | 0.4057 | 0.1679 | 0.2641 |
| 非恒等前后矩阵→绝对 | 0.2631 | 0.3577 | 0.1618 | 0.2169 |
| 极小 postmat→相对 | 0.2928 | 0.3567 | 0.1822 | 0.1356 |
| 极小 postmat→绝对 | 0.3418 | 0.4763 | 0.1700 | 0.3561 |

main 的两矩阵和极小 postmat 有下节列出的外场错误；这些行保留历史性能观察。

## 4. 成熟子函数修复与分区精度

旧实现把 warp 网格外的查询作为边界残差延伸；FSL `concat_warps` 使用完整 float32 绝对 prewarp 的最小二乘 affine 外推。两矩阵真实案例有 27,829 个越界体素，其中 30 个在参考影像正值区域；旧版本最高差 `5.0381 mm`。极小 postmat（`2×10⁻⁷ mm`，即 `10⁻⁷ voxel`）仍会产生 9,919 个严格越界点，旧版本最高差 `2.4466 mm`。

本次只在 ConvertWarp 需要外推时拟合 affine，并按 FSL 的两次 float32 存储及严格 voxel 边界判断。先把 premat 组合结果存为 float32，再用 float64 轴边缘归约拟合，不创建整卷设计矩阵。InvWarp 默认取样规则保持。

| 案例（H100 同输入精度检查） | 全场最大差 / mm | 场外最大差 / mm | 场外 p95 / mm | 场外 p99 / mm | 正值参考场外最大差 / mm |
|---|---:|---:|---:|---:|---:|
| both_matrices_relative | 2.47955e-05 | 7.62939e-06 | 6.19888e-06 | 7.62939e-06 | 3.8147e-06 |
| both_matrices_absolute | 3.05176e-05 | 1.52588e-05 | 0 | 0 | 0 |
| tiny_post_relative | 1.14441e-05 | 7.62939e-06 | 5.72205e-06 | 7.62939e-06 | 无该区域体素 |
| tiny_post_absolute | 1.52588e-05 | 7.62939e-06 | 0 | 0 | 无该区域体素 |

CPU 1/8 的全场、场内、场外、正值参考区及其交集指标也逐条记录在机器报告；完整视野没有因脑区筛选而被排除。

## 5. MMORF 扩展：真实完整场

原生 FA、MMORF warp、reference 和 affine 已核对来自同一保存 run。转换输出为 `182×218×182×3`。此输入约定没有直接匹配的原版 ConvertWarp CLI，以下比较起点 main 与本次版本。全部 CPU 与 H100 相对/绝对输出逐位一致，最大差为 0。

| 设备 | 输出 | main 完整进程 / s | 本次完整进程 / s | main 已导入 API / s | 本次已导入 API / s |
|---|---|---:|---:|---:|---:|
| CPU 1 线程 | 相对 | 4.9828 | 4.5246 | 2.8913 | 2.6046 |
| CPU 1 线程 | 绝对 | 4.6486 | 3.8713 | 2.4003 | 2.0116 |
| CPU 8 线程 | 相对 | 4.2844 | 4.1531 | 1.9959 | 1.8160 |
| CPU 8 线程 | 绝对 | 4.1135 | 3.5661 | 2.0346 | 1.8231 |
| H100 PCIe | 相对 | 4.4081 | 4.2323 | 1.6006 | 1.6082 |
| H100 PCIe | 绝对 | 4.1878 | 4.2113 | 1.5489 | 1.5885 |

MMORF CPU 绝对输出省去未使用的参考 scaled-mm 网格；有限值检查修复了含 NaN/Inf 的场仍返回 `valid_fraction=1` 的错误。正常场仍按 reference-axis mm、world 和 FSL scaled-mm 坐标顺序转换。

## 6. GPU 性能门与分步骤诊断

H100 使用同一 GPU、公共计时锁及 20,000,000,000 byte PyTorch 分配上限；每个版本分别独立导入，两轮 AB/BA 工作进程各记录三次已导入 API，合计六个样本。

每个独立进程另记录最后一次调用的 peak allocation；下面两版峰值列均在两个进程中一致，不将其称作每次调用都已测量。

| 案例 | 修复前 v5 API / s | 本次 v7 API / s | v5 / v7 最后调用 peak allocation（B） | 原版最大差 / mm |
|---|---:|---:|---:|---:|
| coef_relative | 0.0575 | 0.0661 | 186,100,736 / 186,100,736 | 9.53674e-06 |
| both_matrices_relative | 0.0685 | 0.0652 | 186,100,736 / 225,821,696 | 2.47955e-05 |
| both_matrices_absolute | 0.0613 | 0.0859 | 186,100,736 / 225,821,696 | 3.05176e-05 |
| tiny_post_relative | 0.0807 | 0.0694 | 186,100,736 / 225,821,696 | 1.14441e-05 |
| tiny_post_absolute | 0.0646 | 0.0675 | 186,100,736 / 225,821,696 | 1.52588e-05 |

默认三次样条转换另做 main/v5/v7 的 ABA 顺序诊断：六次完整加载、转换和保存；再独立测已解码真实图像的计算与 D2H、以及保存；最后独立 profiler 一次，不把 profiler 开销计入墙钟时间。

| 版本 | 完整读写 API / ms | 读取解码 / ms | 已解码计算+D2H+头信息 / ms | 保存 / ms | CUDA 核累计 / ms（两次） |
|---|---:|---:|---:|---:|---|
| main | 61.156 | 18.377 | 21.499 | 48.305 | 0.884 / 0.878 |
| v5 | 70.851 | 19.671 | 22.611 | 51.314 | 0.876 / 0.887 |
| v7 | 74.159 | 17.665 | 16.823 | 50.300 | 0.878 / 0.878 |

这三版默认路径均为 108 个 CUDA 核、9 次 copy/memset；本次峰值 allocation 为 186,100,736 byte，main 为 186,101,248 byte。默认 CUDA 数值与 main/v5 逐位一致，核数量、累计计算与分配峰值未增加。全量读写的本次中位时间仍偏高，GPU 背景利用率在不同工作进程间为 0–100%，保存也有波动；所有原始样本和背景观测保留。上述阶段分别测量，不把其独立中位数相加作为完整调用时间。

网格外四个案例的最后调用 allocation 各增加 39,720,960 B；为正确 affine 外推新增了必要计算，表中两矩阵绝对输出的 GPU API 墙钟观察高于旧版本；该成本与 CPU 取样优化分开报告。第一次 GPU 工作进程在设置分配上限时发生 CUDA OOM，尚未进入函数；保留失败日志后在独立目录重试，成功数据才进入此表。

## 7. 公开脑图与复现

示例来自 [OpenNeuro ds000114 v1.0.2](https://openneuro.org/datasets/ds000114/versions/1.0.2) 的完整公开去面部 T1。该数据的 [dataset_description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds000114/master/dataset_description.json) 标明 CC0；共享 FLIRT 初始化和官方四级 FNIRT 已完整成功执行。图中为包含嵌入 affine 的 FSL scaled-mm pull 位移幅值及 FNIT−FSL 向量差，叠加 FSL MNI152 标准模板的灰度解剖和脑 mask。

![完整公共样本的FSL/FNIT场幅值和向量差](assets/public_convertwarp_comparison.png)

公共全场最大分量差 `1.5259×10⁻⁵ mm`；图的切面为 MNI x=0、y=−24、z=12 mm。此图的转换在 headcw 单核完成，只用于精度示例。CPU 速度表使用上面的 nodecw10 配对。

[可编辑 SVG](assets/public_convertwarp_comparison.svg)、[公共来源 SHA 与完整误差](assets/public_figure_report.public.json)、[图指标](assets/public_convertwarp_comparison.json)随图保存。绘图脚本为 [`plot_public_comparison.py`](plot_public_comparison.py)，使用项目主环境已声明的 Matplotlib；实际旧服务器环境缺此包时，图在独立 overlay 环境生成，计时环境没有修改。

标准 FSL 案例使用 [`benchmark_multimodal_cpu_convertwarp.py`](../../tools/benchmark_multimodal_cpu_convertwarp.py)，由[统一 runner](../../tools/benchmark_multimodal_cpu.py)运行隔离官方程序；生产接口仍只调用本包计算。该 adapter 的 `run_case` 全量读写，`reference_command` 输出同输入命令，`compare_case` 在计时外计算所有区域和空间头检查。

## 8. 最近版本与验证记录

| 版本 | 结果与变更 |
|---|---|
| v7，2026-10-04 | 严格 float32 voxel 边界与保守静态范围；两矩阵和极小 postmat 场外误差修复；44项 CPU/实际CUDA功能回归；本页列最新冻结函数的真实全量对照。 |
| v6，2026-10-04 | 实现全场 float32 prewarp 的 best-fit affine。普通两矩阵 CPU1/8 官方 max 分别2.4796e-5/3.0518e-5 mm；后续补上严格极小位移。GPU记录见[此前报告](gpu_benchmark_v5_v6_20261004.public.json)。 |
| v3，2026-10-04 | CPU复用成熟取样器、跳过恒等矩阵与无用网格；首次真实官方配对发现5.0381mm外场错误。[历史聚合报告](cpu_benchmark_v3_20261004.public.json)保留失败边界。 |
| 起点main | 既有TBSS/MMORF真实转换与重采样对照，见[早期验证](../../validation/convertwarp/README.md)；其原版CLI与GPU API计时边界不同。 |

## 9. 上游与参考

- [FSL FNIRT / ConvertWarp 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#convertwarp)。
- 原算法核对项目已附带的 `fugue-2201.3/convertwarp.cc`、`warpfns-2203.0/warpfns.cc` 及 `newimage-2203.11/newimage.h`。新增实现使用 PyTorch 归约，没有复制无关上游程序。
- 数据：Gorgolewski 等，*A test-retest fMRI dataset for motor, language and spatial attention functions*；[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3641991/)，数据 DOI `10.18112/openneuro.ds000114.v1.0.2`。
