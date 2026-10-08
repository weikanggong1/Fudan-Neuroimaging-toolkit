# Connectome 追踪的无损性能优化

## 目标

本页记录 `fnit.connectome.tracking` 在单 GPU 上的一个精度保持优化。优化只减少拒绝采样循环中的索引压缩，不改变 iFOD2、ACT、随机数生成、SIFT2、FA 采样或矩阵定义。

## 代码改动

`_initial_directions()` 和 `_grow()` 原来在每一轮拒绝采样中重新执行完整 batch 的 `nonzero()`。现在保留尚未接受的 seed 行索引，并在接受后从这个有序索引向量中删除对应行。索引顺序不变，所以同一输入、同一 seed 和同一 proposal 宽度仍消耗完全相同的随机数，第一条接受 proposal 也不变。

入口和追踪输出接口没有变化。输入仍是：

- `wm_sh`：`[X, Y, Z, C]` 的 float32 白质 FOD 球谐系数；
- `fod_affine`：FOD 体素到世界毫米坐标的 4×4 仿射矩阵；
- `five_tissue`、`five_tissue_affine`：`[X, Y, Z, 5]` ACT 五组织图和其仿射矩阵；
- `gmwmi`：与 ACT 网格一致的 GMWMI 播种权重；
- `n_seeds`、`seed`、`batch_size`、`arc_proposals`：播种、随机种子、批量和拒绝采样块参数。

输出仍为 `Tractogram`，包括 `paths`、`accepted_seeds`、`lengths_mm`、`endpoints`、FA 统计和后续的 SIFT2/连接矩阵输入。

## gpucw1 基准

基准使用当前 `origin/main` 的追踪代码与只加入索引复用的候选版本，在同一 H100、同一进程、同一固定随机种子和同一合成 5TT/FOD/GMWMI 输入上运行。为了避免共享 GPU 影响被误认为加速，先完成一次预热，再各取 7 次热调用；结果仅用于算子级回归，不能替代真实 DWI 的端到端基准。

| 版本 | 热调用均值 | accepted seeds | 路径点数 | 峰值 CUDA allocated |
|---|---:|---:|---:|---:|
| `origin/main` 基线 | 3.526307 s | 11,046 | 31,468 | 0.041482 GiB |
| 索引复用候选 | 3.485083 s | 11,046 | 31,468 | 0.041482 GiB |

候选相对基线减少 **1.17%**（1.0118×）。`accepted_seeds`、`lengths_mm`、`endpoints` 和每条路径均 `torch.equal=True`。测试运行时 GPU1 仍有其他进程，故该百分比是保守的算子级观察值；真实数据需在低负载 GPU 上复测。

历史真实 ds004666 100k 追踪基线仍见 [`tracking_100k_three_seed_20260929.md`](../../validation/connectome/ds004666/tracking_100k_three_seed_20260929.md)：MRtrix 为 67.73–84.42 s，FNIT 为 782.49–806.95 s。该比较包含不同硬件、负载和编译边界，不能把它解释为稳定的 9–12 倍算法差距。本次索引复用只减少小部分 Python/CUDA 压缩开销，不能解决主要的 `_grow()` arc/ACT kernel launch 和逐步拒绝采样瓶颈。

## 未采用的候选

- 合并 midpoint/end FOD 与 SH 采样：固定输入逐项相等，但热调用约慢 6.3%；
- 8-corner 五组织插值向量化：没有逐值/逐轨迹一致性，且无稳定加速；
- 跳过单向 seed 的反向传播：改变随机数消耗和轨迹群体，不属于无损优化；
- 将 `arc_proposals` 改为 32/64：速度提高但 accepted seed 和路径数量改变，不保持同输入输出；
- 把 FOD sampler 捕获到 `torch.compile` 闭包：输出相等，但因每个调用重新编译，热调用约慢 5 倍。

因此默认参数和追踪统计定义均保持不变。下一阶段若要接近 MRtrix 的速度，需要 fused CUDA/Triton arc+ACT kernel，并以单弧 oracle、固定轨迹和随机重复 envelope 逐层验收；在此之前不应宣称已达到 MRtrix 的速度。

## 复现

```bash
# 使用固定输入和随机种子运行追踪；--compile-arc 仍按原接口选择
python tools/benchmark_connectome_tracking_100k_matrices.py \
  --fod wm_fod_norm.nii.gz \
  --five-tissue five_tissue.nii.gz \
  --gmwmi gmwmi.nii.gz \
  --fa fa.nii.gz \
  --atlas atlas_dwi.nii.gz \
  --n-seeds 100000 \
  --batch-size 8192 \
  --seed 0 \
  --device cuda:0 \
  --compile-arc
```

MRtrix 的对应参考命令、输入哈希、矩阵定义和重复性指标见上述真实数据报告；FNIT 运行时不调用 MRtrix、FSL 或 FreeSurfer。
