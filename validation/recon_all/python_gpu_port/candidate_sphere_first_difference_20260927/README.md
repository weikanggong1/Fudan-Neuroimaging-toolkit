# 候选输入下 `mris_sphere` 首次采样分叉及修复

本次使用去标识真实 T1 `examples/data/sub-01_T1w.nii.gz`（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）。FNIT 连续前缀生成的 `lh.inflated`、`lh.smoothwm` 分别为 `f3a2128de469dc3c8d06e2649f85f420b26c84f95aea3c7912f677b4e57afd30`、`6d1a5d30639f3678c2e825d69b816f9bc35d5411546145f6c0bcab29a5fd0b91`。原生参照为 FreeSurfer 8.2 `mris_sphere`（SHA-256 `c34ca308a7fa03acdb3f689bf6125cf3d0198c37c3a62992a29f68e631c73612`），命令为 `mris_sphere -threads 4 -seed 1234 surf/lh.inflated surf/lh.sphere`。已安装的 FreeSurfer 只用于隔离验证，FNIT 运行不调用它。

## 函数与数据

`sample_standard_metric_matrix(vertices, faces, seed=1234, limit=None)` 从 `smoothwm` 的 `N×3` 浮点坐标和 `F×3` 有序三角面生成原始距离；返回长度 `N+1` 的 CSR 行偏移、邻居 ID、同长度的 `float32` 距离，以及顶点数、样本数和计时字典。`average_standard_metric(offsets, indices, distances)` 按原生行顺序平均互为邻居的距离，返回平均后距离、匹配次数和耗时。两者对应原生 `MRISsampleDistances` 的采样与对称化；`_angle(xyz, vertex, a, b)` 是采样角度的内部函数，输入同一网格及三个顶点 ID，返回弧度。球面完整命令的输入还包括同面序的 `inflated`，输出为有序三角表面 `lh.sphere`。

```python
from fnit.recon_all.sphere_standard_metric import average_standard_metric, sample_standard_metric_matrix

offsets, neighbor_ids, raw_mm, timing = sample_standard_metric_matrix(
    vertices=smoothwm_xyz,  # N×3，lh.smoothwm 坐标，单位 mm
    faces=faces,             # F×3，与 inflated 相同的有序面
    seed=1234,               # 等价命令的 -seed
    limit=None,              # 全部顶点；限制前缀仅用于诊断
)
target_mm, reciprocal_count, averaging_seconds = average_standard_metric(
    offsets=offsets,         # CSR 行偏移，N+1 项
    indices=neighbor_ids,    # 每项对应的邻居顶点 ID
    distances=raw_mm,        # 对称化前的 float32 距离
)
```

这例有 106,622 个顶点、8,268,920 个目标距离。生产 `_angle` 现用显式 `float64` 计算两个向量的长度，再按原生路径将夹角转回 `float32`；没有新增依赖。

## 首差与原因

GDB 在原生第一次 `logSSE` 后、优化更新前抓取目标距离、当前距离、邻居 ID、行偏移和球面坐标。[抓取清单](native_full_matrix_capture.json)包含数组 SHA-256；原始二进制留在 gpucw1 私有 scratch，未放入 Git。原生球面坐标 319,866/319,866 个 `float32` 分量与保存的 `sphere0000` 一致；CSR 各行长度全相同。抓取耗时 24.15 s，其中数组写出 2.90 s。原生日志与四份早期快照见[日志](native_first4_sse.log.gz)和[快照清单](snapshot_probe.json)。

[修复前全矩阵核对](comparison_corrected.json)显示目标距离 8,202,754/8,268,920 项逐位相同，第一个差异在顶点 37,432、行内第 73 项，邻居 ID 42,584：原生 5.4933853149 mm，Python 5.4882822037 mm。这里**不是原始距离计算的首差**：Python 该项对称化前后均为 5.4882822037 mm，因其第 42,584 行没有反向采样；原生第 42,584 行第 77 项有反向采样，并将两项平均为 5.4933853149 mm。真正首个邻居 ID 差异较晚，位于顶点 42,548、行内第 73 项，原生选 45,660，Python 选 46,382。

[采样探针](first_neighbor_id_probe.json)显示，顶点 42,548 的第七环中，Python 也先抽到了 45,660，却把它与先前选中的 44,218 之间的夹角算成 `0.7068582773208618`（`0x3f34f4aa`），比阈值 `0.7068583369255066`（`0x3f34f4ab`）低一个 `float32` ULP，因而拒绝。按 FreeSurfer `Vector3Angle` 的双精度长度公式和这三个真实顶点的原始 `float32` 坐标重算，夹角恰为 `0.7068583369255066`，原生接受 45,660。Numba 中原代码的 `float(np.float32)` 乘法仍按单精度执行；改为 `np.float64` 后，角度与原生逐位一致。更远处的目标距离差异是此次采样序列分叉和反向平均的后果。

修复前 Python **自身**邻居 ID 和目标矩阵求得距离 SSE `26.863064785355633`；原生自身数组重算为 `26.86393232857408`，与原生日志的 `26.863932` 一致。早期草算 `26.875918497687643` 把 Python 目标距离与**原生邻居 ID**的当前距离混用，不能作为任一实现的 SSE；[比较脚本](compare_initial_sse.py)已改为分别使用各自的邻居 ID。

## 同输入验收与耗时

| 同一候选输入上的检查 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 目标距离逐位相同 | 8,202,754 / 8,268,920 | **8,268,920 / 8,268,920** |
| 邻居 ID 逐项相同 | 8,228,725 / 8,268,920 | **8,268,920 / 8,268,920** |
| 原生邻居 ID 下当前距离逐位相同 | 8,268,920 / 8,268,920 | **8,268,920 / 8,268,920** |
| Python 自身距离 SSE | 26.863064785355633 | **26.86393232857408** |
| 全矩阵比较墙钟时间，含 JIT | 26.92 s | 27.50 s |

[修复后全矩阵 JSON](comparison_angle64.json)的 `source_sha256.metric` 为 `dde4f42db3f8f526c2ccb1c1379db4edc5369b0f41cbb72ed8b84dc14030ca92`，与提交的生产源完全相同。使用同一输入及原生 `sphere0000`–`sphere0004` 快照再跑保存的前四步，[逐步结果](angle64_first4_report.json)显示四次更新后各 **319,866/319,866** 个有序坐标分量与原生逐位相同；初始 SSE 与线搜索所选步长也恢复一致。四步 Python 诊断总耗时 **61.25 s**（含拓扑、目标矩阵、Numba JIT、梯度和比较），原生 `-w 1` 快照探针在写出 `sphere0004` 后 SIGTERM，墙钟 **30.03 s**。两种探针的初始化、诊断 I/O 和停止方式不同，不能据此计算完整球面的加速比。修复后的候选输入已完成[完整 LH `sphere` 单阶段验收](full_stage/README.md)：106,622 个有序顶点及 213,240 个有序面与同候选输入官方完全一致。`sphere.reg`、`aparc.annot` 和后续皮层统计尚未通过这条候选链验收。

聚焦回归测试 `tests/recon_all/test_sphere_standard_metric.py` 使用上述三个真实顶点的坐标核对阈值位模式；在 gpucw1 环境与现有对称化测试一起为 **2 passed**。原生全矩阵与前四步仅用于离线对照，没有成为 FNIT 的运行依赖。

## 复核文件

`compare_initial_sse.py` 的 CLI 必填项为：`--smoothwm`（候选表面）、`--sphere0000`（原生首次投影网格）、`--capture-dir`（原生二进制数组目录）、`--native-log`（原生 SSE 日志，可为 gzip）、`--report-json`（输出 JSON）。其 JSON 保留输入和实现哈希、矩阵长度、首差位置、逐项精度、两侧自身 SSE 和耗时。`probe_first_neighbor_id.py` 记录修复前的第 42,548 行采样试探，只适用于旧源哈希 `04a10a70277301cbf337b97513e6cfe9957845e89ca6a0c4045eea2dce70fb1d`；新源上首差不存在。[旧四步结果](python_vs_native_first4.json)仅保留作为修复前证据，当前验收以本报告的修复后 JSON 为准。
