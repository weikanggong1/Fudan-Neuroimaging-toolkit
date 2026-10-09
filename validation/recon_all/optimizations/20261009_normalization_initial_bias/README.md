# 初始aseg偏场复用：两例完整归一化ABBA

测试公开OpenNeuro ds000114 sub06/sub07的自产 `norm.mgz`、`aseg.presurf.mgz`、`brainmask.mgz`。八次完整文件API及一份追加显存诊断均完成，没有读取官方参考作为生产输入；原brain文件只在候选计算结束后比较。

## 对照范围

`initial_bias_abba_v7`固定四次顺序：CPU初始偏场→GPU初始偏场→GPU初始偏场→CPU初始偏场。两侧都用已通过16次回归的GPU控制点邻域，TF32启用、无半精度、CUDA缓存关闭，显式GPU1、CPU亲和4–7及四线程预算。只更换初始偏场函数，ridge和有序过滤保持原CPU实现。

| 输入 | 完整API中位数CPU→GPU / 秒 | 初始偏场中位数CPU→GPU / 秒 |
| --- | ---: | ---: |
| sub06 | 128.937→70.577 | 53.634→6.595 |
| sub07 | 134.061→82.456 | 52.122→5.803 |

8/8最终MGZ逐字节一致；不同体素、最大/P99误差均0，shape/dtype、affine、MGH头一致。初始float32源图/控制图/结果及后续两轮source/control SHA和选择细节都相同。ridge各次26.51–44.46秒，共享负载波动保留；不能把阶段比值当作原始T1整例加速或稳定吞吐。

`aggregate.json`汇总逐次和中位值，完整原始JSON/CSV/冷诊断进程收据均保留。`full_API.seconds`含读取、校验、传输、首次JIT、两轮和压缩写出；导入/CUDA初始化、SHA收集与结果比较除外。`process_time.json`另含整个诊断进程及比较/监控收尾，不能标成纯生产CLI冷启动时间。函数计时内的初始/轮间SHA诊断对旧新两侧均启用。

## 版本和GPU算法

祖先提交为 `0c8872b2ee15997f04e0eff5d424a6ba2d3a649d`；实际运行是冻结v4依赖加两个明确v7源模块的只读覆盖。`source_binding.json`记录候选模块、驱动、测试、绘图文件SHA，每次报告另记已加载14个归一化模块SHA。不能用祖先commit代替覆盖代码版本。

候选复用现有 `voronoi_fill_torch` / `smooth_bias_torch`。chessboard距离与稳定排序仍在CPU；层次传播、三轴sigma-8和double除后乘校正在GPU。初始平滑传全零controls，不能恢复原控制点强度。返回CPU NumPy float32后继续原完整归一化，不使用近似。

## 显存和验收边界

`initial_bias_memory_v8`是一份缓存启用的追加诊断，只用来读取有效PyTorch统计；输出/初始/轮间结果同样严格一致。allocated754,974,720字节、reserved773,849,088字节；其91.922秒单次API不用于宣称缓存提速。缓存关闭计数不可用，不填0。NVML宿主PID归属不明，进程树峰值null；整卡记录包含其他任务，采样不保证连续峰值。

本目录没有新的官方程序重复性结果、原始T1整例或整体指标等效结论。中文输入/输出、每项参数、具名示例、官方命令与脑图见[功能页](../../../../docs/recon_all/NORMALIZATION_ASEG_INITIAL_GPU.md)。运行 [reproduce.sh](reproduce.sh) 复现同输入完整API，不下载或更改检查点。

数据来源：[OpenNeuro ds000114](https://openneuro.org/datasets/ds000114)，CC0。公开PNG为该公开输入的自产brain原网格切片；没有发布MRI、许可证、凭据或服务器地址。私有前缀去为 `$FNIT`，主机去为 `A100_BENCHMARK_HOST`，原件与发布件SHA同时保留。
