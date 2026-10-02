# 当前 main 的十例真实 T1 版本回归

本次固定比较 `ac692bb4f7868a24ea4bd67180162e81726de9b4` 与 main 提交 `f436de588647a0de80735e4a98d53df5d88e502d` 的 `segment_4_subregions(structures="all", optimization="fast")`。两版各在同样十名受试者的原始公开 T1 上完整运行一次，包含内部 SynthSegPlus、TorchFAST、四类分区拟合与结果保存。运行成功 10/10，随后独立 CPU 审计成功 10/10。

这是两个 FNIT 版本的真实输出回归。[当前 main 与官方的最终配对](official_comparison.md)已关联同病例本轮完整官方流程：[110 分区 Dice](../analysis/per_region_dice.md)、[十例精度及真实脑图](../benchmark_results.md)。十例官方/FNIT 完整流程耗时比为 **29.010 ± 2.643**；新九例为 **29.231 ± 2.704**，均先逐病例计算再汇总。

## 数据与版本

- 数据为 OpenNeuro **ds000114 / 1.0.2 / ses-test / sub-01…sub-10**，固定 ID 顺序，未按分割结果筛选。上游 Git snapshot 为 `6299834614e9ae7df1e2fc5922545331b5c7f022`，许可证 CC0，DOI：[10.18112/openneuro.ds000114.v1.0.2](https://doi.org/10.18112/openneuro.ds000114.v1.0.2)。
- “原始公开 T1”指未修改的公开 snapshot 文件；数据发布者已做去面部处理。两版对每例使用同一文件 SHA、同一网格；本目录只归档文本，不归档 MRI。
- `sub-01` 曾用于开发，标记 `development_seen=true`；另外九例分别汇总。全部十例不能称为十例独立留出数据。
- 旧版源码清单含 432 个文件、428 个运行时 Python 文件；当前 main 清单含 439 个文件、435 个运行时 Python 文件。两版的 **24 个 GEMS 文件逐字节相同**，40 个权重与图谱资产身份相同。实际调用链审查见 [upstream_dependency_review.md](../upstream_dependency_review.md)，冻结源码身份见 [source_manifest.json](source_manifest.json) 与 [source_export_identity.json](source_export_identity.json)。
- 两版配置均为 `structures="all"`、`optimization="fast"`、4 CPU 线程、PyTorch 2.5.1 / CUDA 11.8；冻结配置采用 float32 和默认 TF32。预处理记录确认 float32；原观察器未记录每次网络前向的 autocast/TF32 状态。

## 输出一致性

服务器 CPU 审计实际读取了两版的保存标签与体积，并核对预处理观察器记录。本地另做纯文本独立审核，完整键、输入与源码 SHA、全部零差异及 10+9 两组 **480 个统计分布**均通过；统计重算采用样本标准差 `ddof=1`。

| 项目 | 完整比较数 | 结果 |
|---|---:|---|
| 原始 T1 网格上的标签图 | 10 对 | shape、affine、dtype、文件与数组 SHA 相同，0 个不同体素 |
| 四类高分辨率标签图 | 40 对 | shape、affine、dtype、文件与数组 SHA 相同，0 个不同体素 |
| 110 ROI 的软、硬体积字典 | 1100 对 | 全字典 SHA 相同，软、硬体积差均为 0 |
| prepared/raw 预处理记录 | 160 对 | 数组 SHA、几何与记录的内存步幅相同 |
| 原始输入、模型选择和预处理参数 | 10 例 | 两版身份和配置相同 |
| 拟合后四类网格最小 Jacobian | 40 个/版 | 全部有限且大于 0；范围针对拟合四面体网格 |

预处理数组通过观察器的 SHA 与几何记录核对；它们没有另存为影像。本地文本审核未重复读 MRI。由于本次两版标签逐体素相同，对同一官方输出、同一评价网格和同一统计规则计算的 Dice 也会相同。

逐条证据见 [标签对照](audit/map_pairs.tsv)、[体积对照](audit/volume_pairs.tsv)、[预处理对照](audit/context_pairs.tsv) 及 [完整汇总](audit/summary.json)。

## 实测运行时间

表中时间单位为秒。每名受试者每版只运行一次；`均值 ± SD` 的 SD 表示受试者间差异，不表示重复运行的随机波动。`API compute` 含内部预处理和分区计算；`API total` 是 API 调用总时间；`process` 从启动子进程前到独立 watcher 返回，包含导入、输入读取、计算、保存及观察器开销。输入等待、GPU 预算等待和启动前身份核验分别记录，未计入 process；没有从实测时间中扣除观察器开销。

### 全部十例

| 测量 | 旧版均值 ± SD | 旧版中位数；范围 | main 均值 ± SD | main 中位数；范围 |
|---|---:|---:|---:|---:|
| API compute | 287.338 ± 36.496 | 273.397；254.749–357.496 | 253.660 ± 6.308 | 254.652；245.684–261.887 |
| API total | 287.957 ± 36.529 | 274.088；255.329–358.231 | 254.260 ± 6.340 | 255.249；246.253–262.551 |
| process | 293.096 ± 36.580 | 279.182；260.286–363.505 | 259.366 ± 6.447 | 260.716；251.255–267.758 |
| 保存输出 | 0.561 ± 0.069 | 0.551；0.488–0.683 | 0.550 ± 0.046 | 0.545；0.481–0.644 |
| 本进程采样显存峰值 / MiB | 17102.000 ± 1541.963 | 17159；14748–18926 | 16856.200 ± 1405.989 | 16488；14748–18428 |

### 新九例，排除曾用于开发的 sub-01

| 测量 | 旧版均值 ± SD | 旧版中位数；范围 | main 均值 ± SD | main 中位数；范围 |
|---|---:|---:|---:|---:|
| API compute | 286.738 ± 38.658 | 270.736；254.749–357.496 | 254.459 ± 6.130 | 256.959；245.684–261.887 |
| API total | 287.363 ± 38.694 | 271.384；255.329–358.231 | 255.060 ± 6.166 | 257.572；246.253–262.551 |
| process | 292.444 ± 38.737 | 276.230；260.286–363.505 | 260.213 ± 6.221 | 263.232；251.255–267.758 |
| 保存输出 | 0.569 ± 0.068 | 0.555；0.489–0.683 | 0.552 ± 0.049 | 0.550；0.481–0.644 |
| 本进程采样显存峰值 / MiB | 16899.333 ± 1487.540 | 16488；14748–18448 | 16897.111 ± 1484.950 | 16488；14748–18428 |

### 分步骤记录

以下是各模块保存的真实计时器，单位为秒。全部 107 个计时键及其逐例值保存在 `audit/summary.json → cases → baseline/main → recipe_timings`；两组汇总位于 `cohort_all/cohort_new_subjects → measurements → recipe_timings`。部分计时器互相嵌套，不能把全部计时项相加。

| 模块及计时键 | 十例旧版均值 ± SD | 十例 main 均值 ± SD | 新九例旧版均值 ± SD | 新九例 main 均值 ± SD |
|---|---:|---:|---:|---:|
| 共享预处理 `shared_preprocessing/seconds` | 13.885 ± 0.980 | 13.105 ± 0.852 | 13.741 ± 0.921 | 13.010 ± 0.846 |
| 脑干 `brainstem/timing_seconds/total` | 20.402 ± 2.701 | 17.779 ± 0.709 | 20.089 ± 2.666 | 17.749 ± 0.745 |
| 丘脑 `thalamus/seconds` | 85.023 ± 11.761 | 74.884 ± 4.759 | 86.302 ± 11.713 | 75.665 ± 4.315 |
| 左海马与杏仁核 `hippo-amygdala-left/seconds` | 72.929 ± 13.749 | 64.916 ± 4.179 | 74.143 ± 14.004 | 65.515 ± 3.951 |
| 右海马与杏仁核 `hippo-amygdala-right/seconds` | 76.663 ± 18.774 | 64.931 ± 4.032 | 73.903 ± 17.629 | 64.449 ± 3.960 |

逐例主要时间见 [case.tsv](audit/case.tsv)。API 报告、驱动报告、观察器与 GPU 日志的路径、大小、SHA，以及每个计时器的来源绑定均在 [summary.json](audit/summary.json) 的各例 `baseline/main` 记录内；进程 watcher 时间另绑定 [fnit_raw_queue.json](fnit_raw_queue.json)。

## 硬件与共享负载

两版 raw 均在 gpucw1 的同一张 NVIDIA H100 PCIe 上运行，物理 GPU 1，UUID `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`，报告总显存 81559 MiB。本进程采样峰值上限为 **19073 MiB**（约 18.63 GiB）；十例均在上限内。峰值来自独立 GPU 日志对本进程 PID 的周期采样，不等于瞬时最高显存。PyTorch 分配峰值记录为 13.774–17.381 GiB，两版范围相同。

| 共享 GPU 条件 | 旧版 raw | main raw |
|---|---:|---:|
| 有效 GPU 样本数 | 561 | 502 |
| 其他 PID 显存，全部样本范围 / MiB | 804–27480 | 804–1274 |
| 其他 PID 显存，逐例采样均值的范围 / MiB | 804–19356.986 | 804–1005.429 |
| 整张 GPU 利用率，逐例采样均值的范围 | 31.745%–81.203% | 31.878%–35.647% |

两版运行时段不同，main 这批运行观察到的其他 GPU 负载更低。CPU 也使用共享节点，每个拟合进程设 4 线程，未进行独占 CPU 负载控制。因此表中时间是本次真实共享条件下的测量，不能归因为翻转 ensemble 缓冲复用带来的固定提速。没有使用多个运行的最好值替代这批结果。

## 文本归档与独立复核

最终八文件按原始字节复制，旧的六个启动与源码准备文件保留；完整服务器 `expanded.json` 留在服务器。副本身份和本地独立统计重算见 [copy_audit_receipt.json](copy_audit_receipt.json)。最终汇总 `audit/summary.json` 大小 897617 字节，SHA-256：

```text
cd497c2dd130c4238c1314a5b5cf7cb9f5f59a59b7910646459f03e1baafebf4
```

从仓库根目录可重做纯文本审核，`--evidence-root` 指向已回收的八个最终文本所在目录。该脚本核对 SHA、完整唯一键、零差异和所有统计，拒绝覆盖内容不同的最终文件，不读影像、不连接服务器、不启动拟合。

```bash
python validation/subregions/ten_public_t1_20261002/latest_main_regression/review_text_artifacts.py \
  --evidence-root /absolute/path/to/latest_main_f436de5_regression
```

## 记录

- **2026-10-02：** main `f436de5` 十例 raw 完整运行并通过 `zero_voxel_equivalence_passed`；与 `ac692bb` 的全部保存标签、110 ROI 体积及记录的预处理身份完全一致。
- **2026-10-02：** 最终八个文本归档；独立本地纯文本审核重算全部 480 个统计分布通过。十例官方均完整结束，已生成 [20 行配对记录](official_paired_runtime.tsv)和[最终对照](official_comparison.json)。

最终配对发布使用与远端统计相同的 **Python 3.11**。从仓库根目录复核时使用该版本；Python 3.10 的 `statistics.stdev` 存在末位舍入差异，会触发这里对来源统计的严格相等检查。原计时、整数计数和零差异门禁均保持不变。

```bash
python3.11 validation/subregions/ten_public_t1_20261002/publish_main_pairing.py \
  --root validation/subregions/ten_public_t1_20261002
```

发布脚本保护已有产物；实际重生成时使用包含已认证输入文件的独立输出副本。

## 参考资料

- [公开 dataset 与 snapshot DOI](https://doi.org/10.18112/openneuro.ds000114.v1.0.2)、[固定 snapshot 的元数据仓库](https://github.com/OpenNeuroDatasets/ds000114/tree/6299834614e9ae7df1e2fc5922545331b5c7f022)。数据许可与下载身份另见 [data_selection.md](../data_selection.md) 和 [data_manifest.json](../data_manifest.json)。
- [FNIT main 本次冻结提交](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/f436de588647a0de80735e4a98d53df5d88e502d)、[基线提交](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/ac692bb4f7868a24ea4bd67180162e81726de9b4)。
- 四类分区的原软件实现、论文和官方运行协议见 [official_protocol.md](../official_protocol.md)；本目录讨论版本回归，不另改变其评价协议。
