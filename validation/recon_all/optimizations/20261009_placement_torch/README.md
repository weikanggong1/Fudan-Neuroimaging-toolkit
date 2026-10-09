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

## 完整 pial、最终源码和官方对照

完整 LH pial CPU→Torch 同输入配对已经完成，原始产物保存在服务器固定目录：
`runs/recon_torch_surface_20261009/full_pial_sub07_lh_v1/`。
持久会话为 headcw 的 `fnit-surface-full-pial-20261009`，每步保存坐标哈希、
SSE/RMS、dt、接受/拒绝以及四轮边界。完整报告将检查输出文件、有序面、
坐标与全部试步轨迹。此次冻结 v1 未在运行时改写。

| 范围 | CPU(s) | Torch(s) | 优化退化诊断 |
|---|---:|---:|---|
| sub07 LH完整四轮pial，v1 | 1396.651 | 1483.542 | 41步及全部决策、坐标、文件SHA相同；本对慢6.22%，继续opt-in |
| sub07 LH首轮white完整一步，final_v2 | 68.221 | 62.148 | 坐标、面、逐步决策相同；单对包含冷/热差异 |

完整 pial 双方清理均为 23→0；allocator/reserved 峰值为
141131264/197132288 bytes，整进程显存未采样。完整调用是共享节点的
单对观察，不能用静态正则梯度的2.2倍解释为整步提速。机器报告为
[完整pial](full_pial_sub07_lh_v1.json)。

独立最终源码 regularizer SHA
`9ce7501275af99be9619a1fa1f29355760f15e59f96e275291747373ee9058e8`
的 sub06/sub07 双侧共四组均每元素相同：
[sub07](regularizer_sub07_final_v2_gpu0.json)、
[sub06](regularizer_sub06_final_v2_gpu0.json)。
它们没有覆盖 v1 已冻结报告。完整 white 仍未实现；
[white首步](white_prefix_sub07_lh_final_v2_gpu0.json)只验证第一轮一步。

官方 FreeSurfer8.2同一输入重复两次，坐标与有序面0diff，墙钟
150.877/160.181秒；输出文件SHA不同，不把几何重现描述为字节重现。
与Python CPU/Torch均为1560个坐标元素不同，mean/P99/max顶点差
0.0000443/0/0.256908mm。现有Conda二进制44ad399...运行207.409秒，
相对Pythonmean/P99/max为0.005510/0.109754/1.210214mm；这些前置
输入SHA均匹配。报告为[官方](native_official_sub07_lh_v1.json)和
[Conda](native_conda_sub07_lh_v1.json)。官方重复几何稳定，不能将这些
局部误差解释成官方随机；还不能据三方差异认定某个编译器或库为原因。

原始 T1 整例、最终 white 完整实现、整体指标等效及干净环境隔离未验证；
最终源码完整pial、右侧及第二例完整pial、相交跨越扩展质量检查未完成。
整体等效维持`not_assessed`，不使用历史sub01的一致性为sub07背书。

## 复现入口

- [内核脚本](../../python_gpu_port/benchmark_placement_regularization.py)
- [完整 pial 配对](../../python_gpu_port/benchmark_placement_full_pial.py)
- [隔离原生参考及重复性](../../python_gpu_port/benchmark_placement_reference.py)
- [中文接口说明](../../../../docs/recon_all/PYTORCH_PLACEMENT_REGULARIZATION.md)

首次脚本因旧服务器法向模块缺少 `CoordinateNormalCache` 在 CUDA 之前
退出；原失败日志保留服务器 `regularizer_sub07_startup_import_failure.log`。
固定梯度 benchmark 改用旧 CPU 兼容法向入口；完整 pial 仅上传本轮模块和
必要的当前法向/斥力依赖到独立 workspace，未替换服务器正式 repo 或环境。
## 碰撞迁移补充（2026-10-09）

`collision_sub07_lh_v3_interrupted.json`保留gpucw1作业中断和实际检查点SHA，
没有最终报告、没有完成的精度或性能结论；不得把先前已写的检查点当作通过。

`collision_replay_cpu_v5.json`是headcw四线程CPU上部分真实动态面对的回放，
131,108对（864对源相交）逐对bool零差异。Torch CPU热中位数0.036486/0.033302秒，
源Numba0.034877/0.031787秒；CUDA未初始化。此文件不证明GPU性能、完整迭代、
完整pial或整例。脚本与输入/实际kernel版本见报告，中文说明见
[碰撞页面](../../../../docs/recon_all/PYTORCH_PLACEMENT_COLLISION.md)。
