# 拓扑候选面缓存与显式顶点法向

## 1．功能简介

`topology_candidate_normals` 用 NumPy/Numba 从当前候选的 orig 坐标计算面法向，以及显式调用时的顶点法向。它不运行或链接原版 FreeSurfer 阶段程序；不改变 TF32、线程、CUDA 分配器或文件。

源评分入口更新面法向，却刻意不重新计算顶点法向。两函数因此保持独立。面缓存的 `original_area` 也不是 `FACE.area` 或 `VERTEX.area`，不能用它替换统计面积。候选生成、完整 GA 与原始 T1 整例仍另行验证。

## 2．Python 调用、输入与输出

```python
from fnit.recon_all.topology_candidate_normals import (
    candidate_face_normals, candidate_vertex_normals,
)

face_cache = candidate_face_normals(
    original_vertices=current_original_vertices,  # float32 (N,3)，orig surface RAS，毫米
    faces=current_ordered_faces,  # 同版本整数 (F,3)，保留面和角点次序
    selected_faces=touched_face_indices,  # 整数 (K,)，控制器选出的待更新面
)

# 只有对应源调用实际更新顶点法向时才执行，不自动加入 fitness。
updated_vertex_normals = candidate_vertex_normals(
    faces=current_ordered_faces,  # 同版本 (F,3) 整数面
    face_normals=current_full_face_normals,  # 已更新的完整 float32 (F,3) 面缓存
    incident_offsets=current_incident_offsets,  # 原 vt.f 的整数 CSR 偏移 (N+1,)
    incident_faces=current_incident_face_indices,  # 原序关联面号 (E,)，保留重复项
    selected_vertices=matched_vertex_indices,  # 整数 (K,)，显式指定更新顶点
)
```

全部参数必填，无设备、线程或阈值默认值。函数不修改输入，不读写文件。非法数组、索引、CSR 或面归属抛 `ValueError`，非有限产物抛 `FloatingPointError`。

| 参数或输出 | 格式、空间与含义 |
| --- | --- |
| `original_vertices` | 有限 float32 `(N,3)`，tkregister/surface RAS，毫米 |
| `faces` | 整数 `(F,3)`，索引在 `[0,N)`，同一几何版本；不自动修网格 |
| `selected_faces` | 整数 `(K,)` 面号，可空、重复；输出保持给定顺序 |
| `face_cache['face_normals']` | 新 float32 `(K,3)`，无量纲；按原向量差和 cross 计算 |
| `face_cache['cached_original_area_mm2']` | 新 float32 `(K,)`，源面法向缓存长度除以二；不是统计面积 |
| `face_cache['fallback']` | 新 uint8 `(K,)`：0 普通；1 换角点、边放大 100；2 两次近零后长度设 1 |
| `face_normals` | 顶点函数的完整有限 float32 `(F,3)` 输入；不得只传选中面 |
| `incident_offsets` | 整数 `(N+1,)`，首项 0、末项 E、单调；当前原序关联面 CSR |
| `incident_faces` | 整数 `(E,)` 当前面号；关联面必须包含相应选中顶点；不排序或去重 |
| `selected_vertices` | 整数 `(K,)`，可空、重复；顶点函数仅返回这些顶点 |
| `updated_vertex_normals` | 新 float32 `(K,3)`，原面序累计后单位化；孤立或抵消时为零 |

源近零判断使用 `FLT_EPSILON`。面 cross 长度近零时，换角点并将边放大 100 重试；仍近零时长度设为 1。因此共线面缓存 `original_area` 会为 0.5，即使真实物理面积为零；保留这个源状态不代表放宽最终网格质量。顶点函数不跳过 ripped 面，近零长度也设为 1，不使用角度伪法向替代。

## 3．命令行调用

数组子函数没有文件 CLI；真实冻结阶段使用独立诊断工具：

```bash
python tools/benchmark_recon_topology_candidate_normals.py \
  --capture isolated_reference/repeat0/distance.bin \
  --native-repeat isolated_reference/repeat1/distance.bin \
  --patch-capture isolated_reference/same_candidate_patch_statistics.bin \
  --observer-receipt isolated_reference/receipt.json \
  --output new_diagnostic/candidate_normals \
  --threads 4
```

两个 `FNITDST2` 捕获为独立参考重复；`FNITFIT1` 指定相同完整候选的 patch 顶点，坐标和有序面先核对。工具按其原关联面序选择非 ripped 面，用自产法向替换冻结的其余缓存，再运行已有完整自产距离函数。参考距离只用于比较。`output` 须不存在，写私有 NPZ、JSON 和原 RAS 误差 PNG；`threads` 默认 4，作用域后恢复。严格比较失败保留报告并返回非零。

## 4．原软件调用与源码

对应固定 d932 源码的 `computeDefectFaceNormal_calculate_template`、`computeDefectFaceNormals` 和 `computeDefectVertexNormals`，属于 `mris_fix_topology` 内部步骤，无独立官方 CLI。完整参考命令：

```bash
mris_fix_topology -ga -seed 1234 -threads 4 reference_subject lh
```

当前 `mrisDefectPatchFitness` 的顶点法向重算被注释，不能接线时自行启用。原参考只在隔离 benchmark 路径运行，不进入生产求解。

## 5．当前真实精度、耗时与图

公开 CC0 ds000114/sub-07/ses-test 左半球，冻结自产 `79a41cdd` 上游。完整候选 110,534 顶点、216,860 面，99 个 patch 顶点实际触及 **261 个面**。自产面法向 783 个分量、完整 21×19×16 下游有向距离均与同输入参考逐位一致，最大/P99 误差为 0；自身重复及两次参考的相关字段也精确。

在 accelerator-b Xeon Platinum 8369B、四线程、CPU affinity 120–123：

| 已测范围 | 秒 |
| --- | ---: |
| 面法向完整 API 首次，契约之后 | 0.001905 |
| 面法向完整 API 热重复 | 0.001657 |
| 下游完整有向距离首次，含其 JIT | 4.107987 |
| 七个契约 | 0.958826 |
| 诊断 API：读取、计算、NPZ/PNG 写入，报告写入前 | 5.453895 |
| 工具从导入起到报告写入前 | 6.677995 |

面 API 首次在契约之后，不是冷 JIT。无原面法向子项独立计时，不计算原版加速比。操作未用 GPU，CUDA 前后未初始化、精度标志保持；未采样 GPU 进程占用，显存为 null，不填 0。没有整例提速或 600 秒达标结论。

源码 SHA `d9bedc28b8afd73a5816a343c7f6a6b1889b6ed473921bfc209afe05c8a1a84b`，实际上传快照 `420f98c3…` 加未提交新文件，584 个 payload 的 SHA 与大小全部校验，后续提交不反标为测试版本。当前真实参考未捕获缓存 `original_area`，其真实精度未判；顶点法向和两种近零回退仅有契约，本候选未触发回退。也未验证全缓存自产、完整 GA 或整体指标等效。

[完整收据、复现脚本和真实误差图](../../validation/recon_all/optimizations/20261010_topology_candidate_normals/README.md)。图仅显示触及面的 orig RAS 位置与法向误差，不是最终 T1 表面质量证明。

## 6．最近更新与 benchmark

- v1 上传包漏掉已有 `FNITFIT1` 解析器，导入失败、退出码 1；原目录保留，未执行数值比较。
- v2 在新空目录补齐既有解析器，算法代码不变；七个契约、261 面法向及完整距离传播通过，退出码 0。
- 面缓存、统计面积和顶点法向更新分别保留源调用语义；没有将合法候选的零统计面积重算为缓存面积，没有按被试或索引写特例。

## 7．参考文献与源码库

- [固定面缓存原实现](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp#L7633-L7712)。
- [固定候选法向与 fitness 调用](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp#L6654-L6740)。
- [FNIT 独立 Python/Numba 实现](../../src/fnit/recon_all/topology_candidate_normals.py)。
- Fischl B, Liu A, Dale AM. Automated manifold surgery. IEEE TMI, 2001. [DOI](https://doi.org/10.1109/42.906426)。
