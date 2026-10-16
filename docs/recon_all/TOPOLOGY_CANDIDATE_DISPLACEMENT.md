# 拓扑候选的匹配位移初始化

## 1．功能简介

`candidate_displacement` 用独立 NumPy 实现 `computeDisplacement`：从匹配前保存的 fx/fy/fz 和匹配后的 orig 坐标计算全部未丢弃缺陷顶点的位移。它不运行或链接原版程序，不要求 GPU，不改变 TF32 或线程策略。

输入域是完整缺陷及其当前映射，不由 patch 面或保留的 inside 顶点猜测。函数只返回应写回的索引和值，由控制器更新自己的完整 `curvbak` 状态；丢弃及其他顶点保留原值。当前阶段不执行上游匹配、完整候选生成或 GA。

## 2．Python 调用、输入与输出

```python
from fnit.recon_all.topology_candidate_displacement import candidate_displacement

displacement = candidate_displacement(
    original_vertices=matched_original_vertices,  # float32 (N,3)，匹配后 orig surface RAS，毫米
    prematch_positions_mm=saved_prematch_positions,  # float32 (N,3)，匹配前保存的 fx/fy/fz
    translated_defect_vertices=current_defect_vertex_mapping,  # 整数 (K,)，完整原缺陷到当前网格映射
    discarded_defect_vertices=source_discarded_vertices,  # bool (K,)，原 status==DISCARD
)
# 控制器显式写回自己的状态，不从参考文件填充中间结果。
current_vertex_curvbak[displacement['vertex_indices']] = displacement['displacement_mm']
```

| 参数或输出 | 格式、空间与含义 |
| --- | --- |
| `original_vertices` | 有限 float32 `(N,3)`，当前候选 orig tkregister/surface RAS，毫米 |
| `prematch_positions_mm` | 同序有限 float32 `(N,3)`，匹配前保存的 fx/fy/fz，同一 RAS/mm |
| `translated_defect_vertices` | 整数 `(K,)`，完整原缺陷到当前 `[0,N)` 的映射；保留顺序和重复项 |
| `discarded_defect_vertices` | bool `(K,)`，源 status 是否 DISCARD，不以网格成员资格推断 |
| `vertex_indices` | 新 int64 `(A,)`，全部未丢弃条目的当前索引；顺序不变 |
| `displacement_mm` | 新 float32 `(A,)`，对应原有序 F32 平方和的 F32 平方根，毫米 |

所有参数必填，无设备、精度、线程、阈值默认值；A 可为零。输入不修改，函数不读取/写入文件。非法类型、数组或映射抛 `ValueError`；有限大坐标的平方溢出等非有限结果抛 `FloatingPointError`。它不返回伪造的全顶点旧状态，也不将未丢弃顶点限定为重连选中的 inside。

## 3．命令行调用

数组 API 没有生产文件 CLI；真实阶段回归用独立工具：

```bash
python tools/benchmark_recon_topology_candidate_displacement.py \
  --capture isolated_reference/repeat0/volume.bin \
  --native-repeat isolated_reference/repeat1/volume.bin \
  --observer-receipt isolated_reference/receipt.json \
  --output new_diagnostic/displacement \
  --threads 4
```

`capture` 与 `native_repeat` 为完整 `FNITVOL1` 独立参考重复，包含 orig、fx、完整映射/status 和参考初始位移。初始 active 参考位移只用于比较，不进入求解；自产 active 值替换冻结的 untouched 旧状态，再调用完整自产 unMRI 函数。该传播测试的 MRI/PDF/距离仍为冻结输入，不是整个候选自产。`output` 须不存在，写私有 NPZ、JSON、PNG；`threads` 默认 4，正整数且作用域后恢复。严格失败保存报告并返回非零。

## 4．原软件调用与源码

对应固定 d932 `mrisurf_defect.cpp` 的 `computeDisplacement`，属于 `mris_fix_topology` 内部步骤，无独立官方 CLI。完整参考：

```bash
mris_fix_topology -ga -seed 1234 -threads 4 reference_subject lh
```

原源码先计算 F32 坐标差、各轴平方与左结合求和，再求 F32 平方根写入 `curvbak`。位移在 unMRI 前初始化；不能直接将 unMRI 后的参考反馈当作初始位移。参考仅在隔离 benchmark 运行。

## 5．当前真实精度、耗时与图

公开 CC0 ds000114/sub-07/ses-test 左半球，冻结自产 `79a41cdd` 上游。完整候选 110,534 顶点；源缺陷共有 **67 条映射，38 条未丢弃、29 条 DISCARD**。38 个自产位移、下游八项体积字段和完整 110,534 个反馈值均与参考逐位一致，最大/P99 误差 0；自身和参考各两次重复稳定。

四线程、accelerator-b Xeon Platinum 8369B、CPU affinity 120–123：位移完整 API 首次 **0.001346 秒**，热重复 **0.000987 秒**；后续完整 unMRI API 首次含其 JIT **0.682083 秒**。四契约 0.023384 秒；工具从导入起到报告写入前 **2.402851 秒**，其中包含读取、计算、NPZ/PNG 写入的 API **2.197081 秒**。未计解释器启动及末尾报告/COMPLETE 写入，未测原位移子项独立时间，无加速比。位移函数无 JIT。

实际上传快照 `53469bbd…` 加未提交新文件，589 个 payload 大小/SHA 均校验；源码 SHA `a78d202203294a0a9935e5ec8389d1480e59865541e01f6b135727647bfaf908`。CUDA 前后未初始化，精度标志保持；GPU 未使用、进程树显存未采样，因此记录 null。没有原始 T1 整例、600 秒达标或整体指标等效结论。

[完整数值收据、复现脚本与真实位移误差图](../../validation/recon_all/optimizations/20261010_topology_candidate_displacement/README.md)。图中包含全部 38 个 active 顶点，不是最终表面质量图。

## 6．最近更新与 benchmark

- v1 新独立数组函数，返回显式写回索引；避免把冻结全状态或 patch 成员资格当作计算规则。
- 四个契约覆盖完整域、重复映射、discard、空 active、非法类型和有限坐标溢出；accelerator-b 完整真实 active 值及后续全部反馈通过。
- 保持原精度、输入、输出和质量门；候选连续生成及完整 GA 控制器待后续连接验证。

## 7．参考文献与源码库

- [固定原位移步骤](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp#L7395-L7415)。
- [FNIT 独立 NumPy 实现](../../src/fnit/recon_all/topology_candidate_displacement.py)。
- Fischl B, Liu A, Dale AM. Automated manifold surgery. IEEE TMI, 2001. [DOI](https://doi.org/10.1109/42.906426)。
