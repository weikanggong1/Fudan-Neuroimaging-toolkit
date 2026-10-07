# sphere / sphere.reg / register / remesh CUDA 审查（2026-10-07）

## 结论

本次审查没有接入新的 CUDA 后端。现有代码的安全边界要求保留 FreeSurfer 的有序 `float32` 累加、候选接受顺序、动态拓扑和 Gauss–Seidel 依赖；当前可见的 CPU/Numba 目标函数、距离计算、line search 和候选生成都至少有一项阻断条件。直接把这些循环换成普通 PyTorch reduction 会改变求和顺序或候选顺序，因此不能作为低风险优化提交。

已有的有序 CUDA gradient averaging 保留在 `RegistrationGradientAverager`，本轮没有重复实现。

## 静态证据

| 阶段 | 代码位置 | 阻断原因 |
|---|---|---|
| `mris_register` line search | `src/fnit/recon_all/mris_register_line_search.py:109-145` | `mean_delta` 按顶点顺序做 double `sqrt`/sum 后再决定 `min_dt`、`max_dt`；每个 decade、bracket 和 quadratic candidate 都依赖前一个 SSE。将候选批量化或用不保证顺序的 reduction 会改变步长接受结果。当前 objective 已经是 Torch，但 register 运行器默认 CPU。 |
| `mris_register` 数据流 | `src/fnit/recon_all/mris_register_sulc_run.py:54-121`、`src/fnit/recon_all/mris_register_smoothwm_run.py:58-146` | 输入、面、atlas 和中间曲率默认构造在 CPU；CUDA 只用于可选 averaging，并在 `RegistrationGradientAverager.__call__` 中把梯度搬到 CUDA 后再搬回 CPU。目标、line search、下一轮状态仍在 CPU，增加“全流程 CUDA”需要同时迁移 parameterization、atlas blur、rigid search、目标项和写出边界，不能只改一个函数。 |
| `sphere` metric sampling | `src/fnit/recon_all/sphere_standard_metric.py:54-179` | `_sample_rows` 是每顶点、每 ring 的动态 BFS；7-ring 候选集合、Dijkstra-like metric、固定随机流和拒绝采样都依赖前一项状态。候选数可变且 `MAX_NBHD_VERTICES` 溢出需抛错，普通 CUDA gather/排序无法保证同一候选和同一随机序列。 |
| `sphere` reciprocal averaging | `src/fnit/recon_all/sphere_standard_metric.py:227-243` | `_reciprocal_average` 按 CSR 行和邻居顺序寻找 reciprocal，命中后原位更新两项。可矢量化但需先建立稳定的 reciprocal 映射；当前阶段相对 7-ring sampling 不是已证实的主热点，暂无安全收益证据。 |
| `sphere` line search/SSE | `src/fnit/recon_all/sphere_standard_line_search.py:18-58,124-204` | 距离使用源顺序 `float32` spherical distance，行内 SSE 为 double，最终按顶点顺序串行合计；line search 还按原始 decade/bracket/quadratic 顺序评价。CUDA reduction 会改累计顺序，且没有 GPU 全链路可复用。 |
| `sphere` integration | `src/fnit/recon_all/sphere_standard_run.py:88-137` | 每轮 gradient 先 NumPy/Numba 计算，再可选 CUDA averaging 后 `.numpy()`，随后回到 NumPy line search/SSE；因此仅迁移距离或候选生成会反复 H2D/D2H，并不会降低主阶段墙钟。 |
| `remesh` | `src/fnit/recon_all/mris_remesh_python.py:188-250` 及 `split_edge` | remesh 以 heap 的长度降序/边编号降序取边，随后原位 split，更新 faces、edge index、edge faces 和 face_edges。拓扑在每次 split 后改变，且后续长度依赖前一次更新；并行候选或静态 CSR 会改变拓扑语义。 |

## 现有热点证据

用户提供的同输入阶段观测为 surface `791–1041 s`、register `233–368 s`。仓库内冻结 standard-sphere benchmark 还显示 rigid search 每例约 `24.00–29.23 s`、约 `3,228–3,395` 次评价（`validation/recon_all/optimizations/20261001_serial/whole/sub01/baseline_run.json` 等）。这说明 register 的刚体搜索虽然已是 Torch 算子，但运行器仍在 CPU；把单个 reduction 搬到 CUDA不能代表完整 register 提速。

## 本地验证边界

当前工作站没有 FNIT Conda 运行时：`/usr/bin/python3` 无 `torch`，且没有 `pytest`。因此本轮未执行数值回归或 CUDA benchmark；不能把静态审查写成运行通过。应在 gpucw1 的冻结环境中另行完成：

1. 同一 sphere/smoothwm/sulc/atlas 输入，CPU 与显式 CUDA 的 register 全阶段输出 hash、角度、SSE 轨迹和顶点差异；
2. `sphere_standard_metric` 完整 7-ring 候选表、VNL 随机状态和 reciprocal 距离逐元素比较；
3. remesh 每次 split 的 edge/face 数、面序、非流形检查和最终 surface hash；
4. 计时需包含 H2D/D2H、JIT、I/O，并与现有 791–1041 s / 233–368 s 观测同主机配对。

在上述证据前，不提交 CUDA 迁移，避免把非等价 reduction、候选批处理或动态拓扑并行误接入生产。
