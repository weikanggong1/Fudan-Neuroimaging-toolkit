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
  --official-dir reference_matrices \
  --output-dir tracking_comparison \
  --n-seeds 100000 \
  --batch-size 8192 \
  --seed 0 \
  --device cuda:0 \
  --compile-arc
```

MRtrix 的对应参考命令、输入哈希、矩阵定义和重复性指标见上述真实数据报告；FNIT 运行时不调用 MRtrix、FSL 或 FreeSurfer。

## 同机真实输入复测（2026-10-09）

本轮重新读取服务器 FNIT 索引，并验证 ds004666 的三个 NIfTI 输入哈希与历史 100k 报告一致。官方参考使用 MRtrix3 `3.0.3-103-g026e850d`，在同一测试服务器的 Xeon Gold 6430 上固定 8 个 CPU 线程；FNIT 拟使用该机单张 H100。

| MRtrix RNG seed | 播种数 | 保留流线 | 路径点数 | tckgen 墙钟（含读写） | 平均长度（mm） |
|---|---:|---:|---:|---:|---:|
| 0 | 100,000 | 27,685 | 1,153,474 | 16.1876 s | 40.2476 |
| 1 | 100,000 | 27,673 | 1,154,229 | 16.0321 s | 40.2914 |
| 2 | 100,000 | 27,693 | 1,151,332 | 16.2862 s | 40.1563 |

这三次是新的官方参考，不能沿用历史 67.73–84.42 秒作为本轮分母。统计和计时来自实际成功退出的 tckgen 命令，轨迹数量及长度由 nibabel 读取输出 TCK 后计算。参考原始日志和 TCK 留在私有服务器的统一 runs 目录；[公开汇总](../../validation/connectome/tracking_exact_20261009/mrtrix_same_host.public.json)不含服务器登录信息或原始影像。

FNIT 的本轮 100k 配对复测尚未完成：SSH 连接在官方测试结束后中断，原入口端口返回 Connection refused。新的“只校准活动行”候选仍未通过真实 GPU 逐轨验收，尚未加入生产实现。不能据此声称 FNIT 已加速或已与官方相等。

### 严格配对工具

`tools/benchmark_connectome_tracking_exact.py` 只加载指定的两个 tracking 源码和共同的 `fod.py`。输入是已有真实 NIfTI：45 通道 WM FOD、5 通道 ACT 五组织图、与五组织图网格及 affine 完全相同的 GMWMI。输出 JSON 分别记录哈希读盘、影像读取、CUDA 初始化、H2D、同步后的完整追踪、路径打包、D2H、摘要、严格比较及可选 TCK 写盘；这些时间不会混入 tracking 时间。输出还含逐轨 SHA-256、真正的 `torch.equal` 比较，以及 Torch allocated/reserved 峰值。

```bash
# 先固定两版源码；以下路径变量需指向真实文件。
BASELINE_TRACKING=/path/to/frozen/tracking.py        # 已验证的旧版追踪源码
CANDIDATE_TRACKING=/path/to/candidate/tracking.py   # 待验证的候选追踪源码
FOD_MODULE=/path/to/fixed/fod.py                   # 两版共同的 FOD/SH 实现
REAL_FOD=/path/to/fod_reference.nii.gz             # [X,Y,Z,45]，float32
REAL_FIVE_TISSUE=/path/to/five_reference.nii.gz     # [A,B,C,5]，float32
REAL_GMWMI=/path/to/gmwmi_reference.nii.gz          # [A,B,C]，float32
BENCHMARK_JSON=/path/to/results/tracking_exact.json

python tools/benchmark_connectome_tracking_exact.py \
  --baseline-tracking "$BASELINE_TRACKING" \
  --candidate-tracking "$CANDIDATE_TRACKING" \
  --fod-module "$FOD_MODULE" \
  --fod "$REAL_FOD" --five-tissue "$REAL_FIVE_TISSUE" --gmwmi "$REAL_GMWMI" \
  --n-seeds 10000 --batch-size 8192 --seed 0 --device cuda:0 \
  --repeats 2 --warmup-seeds 128 --memory-budget-gb 20 \
  --output "$BENCHMARK_JSON" --profile --profile-seeds 128
```

参数含义：`baseline-tracking` 和 `candidate-tracking` 是待比较源码；`fod-module` 是共享 FOD/SH 源码；三个影像参数是上面的真实输入；`n-seeds` 是尝试播种次数；`batch-size` 是相同的并行批量；`seed` 是两版相同的 PyTorch RNG seed；`device` 是单 GPU 或 CPU；`repeats=2` 给出 ABBA 顺序；`warmup-seeds=128` 单列小规模预热，不能保证全部编译形状已预热；`memory-budget-gb=20` 将 Torch allocator 限在预算的 90%，allocated/reserved 不含 CUDA 上下文；`output` 是 JSON 文件；`profile` 在正式计时后单独采集候选；`profile-seeds` 是 profiler 播种次数。

另有：`warmup` 控制每版预热次数；`cold` 单列同进程的首次完整调用，两版共享 CUDA/FOD/Inductor 缓存；`compile-arc` 对两版同时启用；`save-tck` 指定每次输出 TCK 的目录；`profile-dir` 指定 Chrome trace 目录；`baseline-commit`、`candidate-commit` 是调用者已核对的版本标签，实际源码仍由 SHA-256 绑定。100k 测试使用 `--n-seeds 100000 --compile-arc --cold --repeats 1 --warmup-seeds 0`。两版都从指定随机种子重新执行完整追踪，不复用既有轨迹。

对应官方命令：

```bash
MRTRIX_RNG_SEED=0 tckgen "$REAL_FOD" reference_seed0.tck \
  -algorithm iFOD2 -seed_gmwmi "$REAL_GMWMI" -act "$REAL_FIVE_TISSUE" \
  -seeds 100000 -select 0 -maxlength 250 -cutoff 0.1 \
  -samples 3 -power 0.5 -nthreads 8
```

MRtrix 的随机数发生器与 PyTorch 不同，跨软件采用接受率、长度和端点/TDI 分布及矩阵重复性验收；FNIT 基线与精度保持候选则要求相同输入、参数和种子的逐轨输出完全一致。

本工具的首次全量调用标记为 `first_full_call`，与 `cold` 和实际重复的 `hot` 分开。没有同规模 reference 的 profiler 记录标记 `assessed=false`、`all_equal=null`，不会计入严格通过的汇总。无热样本时不计算热加速比。
