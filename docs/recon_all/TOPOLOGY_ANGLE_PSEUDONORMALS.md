# 拓扑角度伪法向与有序相邻面

## 1．功能简介

`topology_angle_pseudonormals` 用独立 NumPy/Numba 实现拓扑距离阶段的两个内部步骤：按原 incident-face 顺序累计角度加权伪法向，以及查找一条边的原序第一个相邻面。不运行、加载或链接原版 FreeSurfer 阶段程序。它复用调用者当前几何版本的面法向和 CSR，不重新排序或重算法向。

这些是不归一化的拓扑伪法向，不能代替 white/pial 的成熟单位法向。当前只有两个子步骤，不包含候选重连、完整距离场或 GA 控制器。主流程继续使用既有 PyTorch GPU 模块；本函数不会改变 TF32、CUDA 分配器或调用方线程策略。

## 2．Python 调用、输入与输出

```python
from fnit.recon_all.topology_angle_pseudonormals import (
    angle_weighted_vertex_pseudonormals, other_edge_faces,
)

query_pseudonormals = angle_weighted_vertex_pseudonormals(
    original_vertices=original_surface_vertices,  # 当前候选 (N,3) orig surface RAS，毫米
    faces=current_surface_faces,  # 同一版本 (F,3) 整数三角面
    face_normals=current_face_normals,  # 当前 (F,3) float32 面法向；不在此重算
    incident_offsets=source_incident_offsets,  # (N+1,) 原顺序 CSR 偏移
    incident_faces=source_incident_faces,  # (E,) 原 vt.f 顺序，保留重复项
    incident_corners=source_incident_corners,  # (E,) 原 vt.n 的 0/1/2 角点
    selected_vertices=distance_query_vertices,  # (K,) 查询顶点号，保留顺序和重复
)
buffer_other_faces = other_edge_faces(
    faces=current_surface_faces,  # 同一版本 (F,3) 整数三角面
    incident_offsets=source_incident_offsets,  # (N+1,) 完整原序 CSR
    incident_faces=source_incident_faces,  # (E,) 完整原 vt.f
    selected_faces=distance_buffer_face_ids,  # (Q,) 待查面的原索引
)
```

所有参数必填，没有设备、文件、阈值或线程参数。所有输入数组保持不变；类型/形状/索引/CSR 非法时抛 `ValueError`。选中顶点的角点必须与当前面对应；零长度边造成非有限伪法向时抛 `FloatingPointError`。

| 输入或输出 | 格式、空间与含义 |
| --- | --- |
| `original_vertices` | `(N,3)`，转换为有限 float32；tkregister surface RAS/mm，与 faces 同版本 |
| `faces` | `(F,3)` 整数，范围 `[0,N)`；内部连续 int64 |
| `face_normals` | `(F,3)` 有限 float32，无量纲；当前面的既有法向，不因 ripflag 跳过关联面 |
| `incident_offsets` | `(N+1,)` 整数，首项 0、末项 E、单调；完整原序 CSR |
| `incident_faces` | `(E,)` 原面索引；不排序、去重或重建 |
| `incident_corners` | `(E,)` 整数 0/1/2，指出该顶点在关联面的角点 |
| `selected_vertices` | `(K,)` 原整数索引，可空或重复；输出保持相同顺序 |
| `query_pseudonormals` | 新 `(K,3)` float32 数组，按面顺序累计 angle × face_normal；角度为 rad，最终不单位化；孤立顶点为零 |
| `selected_faces` | `(Q,)` 原整数面索引，可空或重复 |
| `buffer_other_faces` | 新 `(Q,3)` int64 数组，依次对应 01、12、20 边；找不到相邻面返回自身面号 |

源算法把 cosine 夹到 `[0,1]`，包含钝角时也保留此规则。实际固定 C++17 编译单元的 `sqrt(float)`、`acos(float)`、点积和两次除法都为 float32；MIN/MAX 的比较表达式为 double，再写回 float32 alpha。修改前的强制 double 数学造成真实伪法向尾差，本次依据独立类型探针修复。

对于非流形中间候选，相邻面查找仍按两端各自的原面列表返回第一个公共非自身面，不擅自选最小索引。这不是最终网格质量许可；生产完整性、连通性、自相交等检查仍另行执行。

## 3．命令行调用

子函数为数组 API；真实阶段回归使用独立 benchmark 工具：

```bash
# 两个 capture 为同源码、同输入、固定四线程的独立参考重复；输出目录须不存在。
python tools/benchmark_recon_topology_pseudonormals.py \
  --capture benchmark/reference/repeat0/distance.bin \
  --repeated-capture benchmark/reference/repeat1/distance.bin \
  --observer-receipt benchmark/reference/receipt.json \
  --output benchmark/new-pseudonormals \
  --threads 4
```

`capture`/`repeated_capture` 是完整 `FNITDST1/2` 原序输入及参考输出；`observer_receipt` 绑定原程序、源码和输入 SHA；只在诊断目录使用，不进入生产调用。`output` 写自产 `candidate.npz`、严格数值 `report.private.json` 和查询位置/误差 PNG。`threads` 默认 4，为正整数 CPU 预算；作用域后恢复 Torch/Numba 设置。格式、参考重复性或严格位比较失败时保留已有报告并抛异常。GPU 不参与此脚本。

## 4．原软件调用与源码

两函数分别对应固定源码的 `computeVertexPseudoNormal` 和 `findOtherEdgeFace`，属于 `mris_fix_topology` 内部步骤，没有独立官方 CLI。参考完整命令是：

```bash
mris_fix_topology -ga -seed 1234 -threads 4 reference_subject lh
```

隔离观察器在首个实际候选距离步骤捕获并停止，不是完整 GA。原参考只用于 benchmark。类型探针在参考构建主机使用实际头文件、Conda 编译器与 Ninja target 参数，两次结果一致；CFFF 接收的只是去敏标量及哈希，原完整编译收据留在原机器。类型探针与 CFFF 算子计时分开，不声称跨机器性能比较。

## 5．当前真实精度、时间与示例

公开 CC0 ds000114/sub-07/ses-test，冻结 `79a41cdd` 自产左半球上游，参考 GA 首个实际候选包含 110,534 顶点、216,860 面；完整距离查询 903 顶点、buffer 1,628 面。不是 99 顶点的示意截断。两次完整参考的原序几何、CSR、全部伪法向及最终距离字节相同；OMP 的 buffer 追加行顺序不同，原始二进制 SHA 不相同，按 fno 对应后的字段精确。

| 项目 | 修复前 v1 | 修复后 v3 |
| --- | ---: | ---: |
| 伪法向不同元素 / 2,709 | 656 | 0 |
| 最大 / P99 分量误差 | 9.536743e−7 / 4.768372e−7 | 0 / 0 |
| 相邻面不同元素 / 4,884 | 0 | 0 |
| 自产重复不同位数 | 0 | 0 |
| 本地 / CFFF 边界契约 | 8 / 8 | 9 / 9 |

修复后在 CFFF Xeon Platinum 8369B、CPU affinity 120–123、四线程，完整伪法向 API 首次/重复为 **0.003581 / 0.003122 秒**，相邻面 **0.001369 / 0.001259 秒**。这些首次 API 在契约测试后运行，不是冷 JIT。九个契约耗时 0.802316 秒；driver 从导入后计，包括契约、输入读取、计算、NPZ/PNG/报告写入的已测段为 1.678432 秒，排除解释器启动、导入和末尾 COMPLETE 写入。没有单独测原参考的这两个子步骤，因此不计算加速比。

源码 SHA `562d7f5f617048b19d133fc74ebd4ed1800a4932f168a010a1c74ce54d76ddd0`。测试绑定 `ae868024…` 的实际脏树快照，580 个上传 payload 的大小和 SHA 全部校验；没有追标为后续干净提交。报告确认 CUDA 前后未初始化、线程预算恢复、TF32 标志未改；没有测进程树 GPU 占用，显存数值为 null，不是 0。

完整失败、修复及可复现报告见[本轮验证目录](../../validation/recon_all/optimizations/20261010_topology_angle_pseudonormals/README.md)，其中 PNG 显示全体查询点的 orig RAS 位置及误差；灰色背景仅为抽样顶点，不能用作最终表面质量图。完整距离、八个聚合场及全顶点反馈的验证属于[距离阶段](TOPOLOGY_SIGNED_DISTANCE.md)。本页不判完整 GA、原始 T1 整例、600 秒目标或整体指标等效。

## 6．最近更新与验证记录

- v1 保留原 double 数学实现的真实失败报告，严格零容差门不通过。
- 四组类型诊断分别比较 double/float 的 sqrt 与 acos；只有与实际固定编译单元一致的两种 float 运算给出全查询零差异。最终改动依据类型探针，未按被试、顶点或结果写例外。
- v2 的正式算子比较已通过，但 driver 最后误读了另一主机的类型探针路径，退出码 1；保留完整阶段报告及元数据失败，未补写假 COMPLETE。
- v3 在新空目录原样运行正式代码，并绑定可访问的去敏类型收据；最终退出码 0。新增通用非直角三角形 rawbits 契约，防止后续再次把 float 数学覆盖为 double。
- 严格 CUDA 状态说明另修正过时的“Python 仅预检查”“完整曲率统计未实现”；仍不把允许 CPU 的独立候选或本次子步骤通过称为严格全 GPU 配方完成。

## 7．参考文献与源码库

- [固定伪法向原实现](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp)。
- [固定相邻面原实现](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_topology.cpp)。
- [FNIT 独立实现](../../src/fnit/recon_all/topology_angle_pseudonormals.py)。
- Fischl B, Liu A, Dale AM. Automated manifold surgery. IEEE Transactions on Medical Imaging. 2001. [DOI](https://doi.org/10.1109/42.906426)。
