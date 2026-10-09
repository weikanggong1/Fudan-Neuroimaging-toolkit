# 2026-10-09 PyTorch placement 正则梯度

基线本地提交 `937263e04eb1be1a5053952ff2b673213cdf966f`。本轮候选通过实际
模块 SHA-256 绑定；它不是一个已发布 main 的整例结果。CPU 参考函数来自
gpucw1 已有源码，每个被调用模块的哈希保存在 JSON，不用服务器旧 repo
提交 `4edcd4b1b408fa4ceb08fa646b4f619b3cc9d365` 冒充候选提交。

## 已完成

- 完整固定网格 regularizer 转写：signed averaging、normal/tangent spring、
  两跳 quadratic curvature，以及原定义的逐列 float32 5×5 QR。
- 完整 Python pial 及 white 首轮前缀显式接入
  `regularization_backend="torch"`；默认 CPU 和原动态碰撞/目标/清理保持。
- 三项 gpucw1 CPU 源次序回归通过；最终本地代码 15 项针对性回归通过，
  10.74 s（包含原 pial 拒绝停止、索引、奇异QR失败及统计缓存测试）。
- 真实公开 ds000114 sub-07 自产冻结 MRI/white/labels 双侧 ABBA；两份报告
  均保留初次慢的候选，没有删除不利性能结果。

| 分块 | 半球 | CPU中位数(s) | Torch中位数(s) | CPU/Torch | 不同元素/max/P99 |
|---:|---|---:|---:|---:|---:|
| 4096 | LH | 0.221993 | 0.452718 | 0.490× | 0/0/0 |
| 4096 | RH | 0.213553 | 0.446270 | 0.479× | 0/0/0 |
| 32768 | LH | 0.217123 | 0.097106 | 2.236× | 0/0/0 |
| 32768 | RH | 0.208376 | 0.090175 | 2.311× | 0/0/0 |

瓶颈是小块 5×5 QR 的 CUDA kernel 调度；增大块保持原运算次序、候选范围
与全部输出。32768 版本 allocated 峰值 LH/RH 为 140668416/127419904
bytes；reserved 均为 197132288 bytes。这是内核分配器观测，不是整例或
父子进程同时总显存。固定上下文建立 0.164/0.151 s 单列。

实际硬件：gpucw1、Intel Xeon Gold 6430（128逻辑CPU）；H100 PCIe GPU0
UUID `GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e`。显式 PyTorch/Numba/
OpenMP/BLAS 四线程；CPU亲和性未固定，当前允许 0–127。TF32 默认开启，
本内核显式 float32/float64，不启用半精度。共享主机单次 ABBA 不能宣布
稳定吞吐。输入 SHA-256、候选及 CPU 模块 SHA-256 均在原 JSON。

32768 GPU 表实际 regularizer 源码 SHA-256 为
`3db2cca686387f3c1a38cf7cee656a7b92e779f162050b50ca7e9903fed7482b`。
最终本地源码另删除 SNRM2 比值的最小正数钳制，恢复原定义的 subnormal
处理，并在非有限 QR 梯度进入优化器前明确报错；15 项 CPU 回归包含该版本。
未将旧 GPU 报告哈希改成最终文件哈希。

## 正在运行及未验证

完整 LH pial CPU→Torch 同输入配对保存在服务器固定目录：
`runs/recon_torch_surface_20261009/full_pial_sub07_lh_v1/`。
持久会话为 headcw 的 `fnit-surface-full-pial-20261009`，每步保存坐标哈希、
SSE/RMS、dt、接受/拒绝以及四轮边界。完整报告将检查输出文件、有序面、
坐标与全部试步轨迹；结束前不发布整步加速。此次冻结 v1 不在运行时改写。

原始 T1 整例、最终 white 完整实现、整体指标等效及干净环境隔离未验证；
完整 pial 相对官方/当前 Conda 的同输入比较需使用新增的独立参考脚本。
官方重复性须针对本次七项输入哈希重测或找到匹配记录，当前为
`not_assessed`。无法用历史 sub01 的一致性为当前 sub07 背书。

## 复现入口

- [内核脚本](../../python_gpu_port/benchmark_placement_regularization.py)
- [完整 pial 配对](../../python_gpu_port/benchmark_placement_full_pial.py)
- [隔离原生参考及重复性](../../python_gpu_port/benchmark_placement_reference.py)
- [中文接口说明](../../../../docs/recon_all/PYTORCH_PLACEMENT_REGULARIZATION.md)

首次脚本因旧服务器法向模块缺少 `CoordinateNormalCache` 在 CUDA 之前
退出；原失败日志保留服务器 `regularizer_sub07_startup_import_failure.log`。
固定梯度 benchmark 改用旧 CPU 兼容法向入口；完整 pial 仅上传本轮模块和
必要的当前法向/斥力依赖到独立 workspace，未替换服务器正式 repo 或环境。
