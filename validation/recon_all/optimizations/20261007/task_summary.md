# recon-all GPU 优化五任务汇总（2026-10-07）

本轮目标是把端到端墙钟压到 600 秒以内，同时保持输出语义、网格拓扑、默认 TF32 和 20 GB 显存边界。当前冻结的九例 FNIT 中位数仍为 2669.520 秒；本轮没有把该历史数值改标成新整例结果。

## 任务结果

| 任务 | 处理范围 | 结果 | 证据和限制 |
|---|---|---|---|
| 1. white/pial | `place_surface_self_repulsion.vertex_buckets_current` | **已改代码**：`resolution != 1` 路径复用 Numba CSR 查询，保留 float32 桶键、顶点顺序和候选集合 | 160,000 顶点 headcw 微基准中位数 0.2821→0.0462 秒，6.10 倍，CSR 逐元素一致；仅为内核微基准，Python white/pial 和整例未重跑。默认整例仍选择 Conda C++ white/pial，不能把该收益外推到端到端。 |
| 2. sphere/register/remesh | 标准球面、球面配准、动态 remesh | **未改生产代码** | `sphere_register_gpu_audit.md` 记录：目标函数和 line search 依赖有序累加及串行候选接受；7-ring BFS 有动态候选和固定随机流；remesh 是 heap 顺序和原位拓扑更新。直接 CUDA reduction 或批量候选会改变结果。 |
| 3. 跨阶段 DAG | N4、SynthSeg、MNI 辅助链、半球组 | **未改生产调度** | 体积链存在文件依赖；MNI 辅助链与表面组理论上可重叠，但共享 GPU、失败回收和显存令牌尚未实测。既有左右半球阶段组已并行，不能用理论上限代替墙钟收益。 |
| 4. N4/EM | `fnit_n4_itk`、`mri_em_register` | **未改生产代码** | N4 观测 124.61–132.16 秒，EM 119.83–213.92 秒；当前分别为 Conda ITK FP32 和 `cpu_cached` C++。没有可证明等价的 GPU N4/完整 Python EM；线程扫参与 LTA/下游回归尚未在 gpucw1 完成。 |
| 5. 指标/IO/cache | `SurfaceStatsCache`、final white/pial 指标和脑区统计 | **未改生产代码** | 现有缓存已按表面版本和完整文件签名隔离，多个 atlas 共享网格/面积/曲率基础量，`-no-th3` 体积语义保持独立。进一步改动缺少实测收益，可能改变累加顺序或缓存失效边界。 |

## 验证边界

- headcw Conda：NumPy 1.26.4、Numba 0.61.2、Python 3.11.16；聚焦测试 `2 passed in 7.41s`。
- 真实 GPU 整例、双例自产连续链、父子进程同期显存和官方配对耗时：本轮未重跑；不能宣称达到 600 秒或完成整体指标等效。
- 生产没有开启 FP16/BF16，也没有用近似输出或参考文件替代必要阶段。

下一优先级是把白质放置/动态碰撞的完整 CUDA 内核做成同输入、同网格、双半球真实回归，再在 gpucw1 从原始 T1 空目录测整例墙钟和同期显存。只有该实测确认关键路径收益后，才应继续迁移 C++ white/pial、sphere 或跨阶段调度。
