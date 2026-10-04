# FNIRT CPU 融合采样诊断

本页记录完整真实网格上的采样函数检查。输入为一张 FP32 移动影像和完整的体素坐标，输出为插值值、有效掩膜及可选的三个体素方向梯度。完整求解器的精度、停止条件和官方配对计时见 [CPU 验证](CPU_BENCHMARK.md)。

## 实现与精度范围

CPU helper 保留 `registration._trilinear_sample` 的八邻点零填充、`1e-8` 有效范围、每步 FP32 乘加和梯度计算顺序。每个查询独立计算，1/8 线程使用相同算式；Numba 的临时线程 mask 按 PyTorch 预算设置，并在异常后恢复。

CUDA 沿用原张量采样。非有限输入、中间溢出、不支持的 dtype 或布局、极端坐标和自动求导输入沿用原张量路径。影像与坐标没有值缓存，每次读取当前输入；输出独立分配。

生产命名的 helper 和已测实验 kernel 除文档字符串外具有相同 AST。局部 83 项检查通过，覆盖值、掩膜和三个梯度的全部 FP32 位、正负零、subnormal、1/4096/4097 轴边界、回退、输入原位修改、输出 alias、线程恢复与 CPU/CUDA 路由守护。详见 [局部记录](assets/cpu-sampling-local-regression-20261004.public.json)。

独立 scalar LLVM 检查的 70 条浮点乘加减指令没有 fast/contract 标志或 FMA。定向 subnormal 检查在 1/8 线程下切换 `flush_denormal=False → True → False`，六组值、掩膜及梯度均逐位一致；范围是该本地环境的联动设置。见 [浮点环境记录](assets/cpu-sampling-fp-environment-unit-20261004.public.json)。

## 完整真实网格的单次观测

查询来自真实 TBSS 第三进程的首次正常 evaluator，起点为官方第二进程保存的系数和强度参数。移动影像为 `104×104×72`，完整目标查询为 `3×182×218×182`，共 7,221,032 点。直接读取保存的原 Torch tensor，并核验 shape、stride、源码和逻辑数据 SHA-256；未改变输入布局。

每个格子为一次加载后采样调用。新 helper 的有限值扫描、输出分配、采样和输出检查均计入；小查询 JIT 准备单列，首次为 14.074 s。

| CPU 线程 | 输出 | 原张量采样 | 融合采样 | 值、掩膜和梯度位差 |
|---|---|---:|---:|---:|
| 1 | 值、有效掩膜 | 60.333 s | 4.291 s | 0 |
| 1 | 值、有效掩膜、三个梯度 | 60.593 s | 5.271 s | 0 |
| 8 | 值、有效掩膜 | 13.435 s | 3.960 s | 0 |
| 8 | 值、有效掩膜、三个梯度 | 10.168 s | 4.286 s | 0 |

两种线程预算各核对全部 7,221,032 个插值值和有效标记，以及 21,663,096 个梯度值。这里的时间范围限于采样函数，端到端速度由完整同线程配对另行记录。影像和坐标保留在私密服务器；公开记录包含几何、聚合精度、时钟及源码哈希。见 [完整网格诊断](assets/cpu-sampling-real-grid-probe-20261004.public.json)。

## 源码记录

| 源码 | SHA-256 |
|---|---|
| 本次原张量参照 `registration.py` | `74fc259e2096e64153a591488303b8ccdfdf19d4ba0b2b91891222b905740e89` |
| 实测实验 kernel | `c8e62eca397463ed185e845250288c3bb6437c112fa10c6bbfa388c397f9a96e` |
| 生产命名 `_sampling_cpu.py` | `a7e10ff4cc33edecc17b6ca378a5cfabeeb8f1fd8c41213efa06a1d16ae62368` |

本轮记录不包含新完整求解轨迹。采样接入后的完整 TBSS 输出、solver QC、停止条件及 CUDA 验收由 [统一验证页](../../validation/multimodal_cpu_20261004/README.md)维护。
