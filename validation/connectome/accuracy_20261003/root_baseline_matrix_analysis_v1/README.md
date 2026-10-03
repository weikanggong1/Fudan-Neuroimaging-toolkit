# 本轮实际 baseline 的矩阵与轨迹比较

## 1. 功能简介

从正式 `raw12` 中读取实际 baseline CON01/03，对各自已完成的官方五 seed 参考，复用冻结分析器的矩阵与轨迹验收。生产代码、配置、官方来源、指标和门槛不变；只在 nodecw10 CPU 读取已经保存的输出。CON01/03 均已完成本轮实际 baseline 分析，原 controller 的终态为 `analysis_completed`。两个病例的矩阵与轨迹整体科学状态均为 `failed`。

```mermaid
flowchart TD
    A[正式 configuration 与实际 baseline producer] --> B{本轮 baseline completed?}
    B -->|否| C[not_assessed 并等待30秒]
    C --> B
    B -->|是| D[完成状态 来源 SHA 实际 CLI 与 FS 核验]
    D --> E[实际矩阵和同次导出 TCK 标量]
    E --> F[原矩阵五次交叉比较]
    E --> G[TCK端点逐位回读与原轨迹比较]
    F --> H[原报告 8atlas表 逐字段5值与判定]
    G --> H
```

每例只有一个实际 FNIT seed，自重复性保留 `not_assessed`。`analysis_completed` 表示读取和比较完成；两例的 B/C 四臂整体科学状态仍均为 `failed`。

## 2. Python 调用、输入与输出

这是私有 CPU 控制器，没有新增公共库接口。实际解释器为 nodecw10 已有 Anaconda Python 3.11.7，NumPy 1.26.4、nibabel 5.4.0、SciPy 1.11.4、Matplotlib 3.8.0；没有安装新包。

```python
import os
import subprocess

analysis_environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # 显式隐藏GPU
analysis_command = [
    "/home1/gongwk/anaconda3/bin/python3.11",
    "/data/private_cpu_tools/analyze_actual_baselines.py",        # 本记录的控制器
    "--configuration", "/data/formal_frozen/accuracy_configuration.json",  # 原冻结配置
    "--frozen-tools", "/data/formal_frozen/tools_source/tools",   # 读取原科学工具
    "--output-dir", "/data/new_baseline_matrix_analysis",        # 必须不存在的新目录
    "--poll-seconds", "30",                                    # 仅状态等待，不增加科学预算
    "--timeout-hours", "24",                                   # 未完成时保存not_assessed
]
subprocess.run(analysis_command, env=analysis_environment, check=True)
```

输入分别为：原 configuration；本轮 `formal_accuracy_raw_v1/baseline/CASE/gpu_report.json` 与 wall；同次 `connectome` 四矩阵、atlas/nodes；同次导出的 `tracks.tck`、WM FOD 网格和 `track_metrics.npz`；逐例原官方 manifest 与五份 TCK。数据由原 `completed_run(config, case, 'baseline', actualFSdir)`、`verified_export`、`matrix.compare`、`population.compare` 和 `decisions` 验证。TCK 回读端点与返回对象保存的端点 dtype/数值逐位一致，weights/lengths/mean_fa/endpoints 数量必须与轨迹数相同。

服务器新输出为 `newroot/root_baseline_matrix_analysis_v1`，其中 `newroot=/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1`。目录包含逐例原 `matrix_envelope.json`、`population_envelope.json`、PNG、`case_report.json`、`eight_atlas_summary.csv` 和 `five_comparisons_by_field.json`。控制器 `report.json` 包含真实 argv、配置/辅助源码/TCK 的前后 SHA、原输入来源、两例状态与实际 CPU 时间。本目录按原字节保存两例 baseline 的原报告与 PNG；[初次 CON03 等待快照](evidence/first_controller_report.json) 保留当时的 `not_assessed`，另附 [实际完成终态](evidence/completed_controller_report.json)，不覆盖历史快照。原体积与 TCK 不复制进仓库。

## 3. 命令行调用及参数

实际命令和原 PID 11158 见 [启动收据](candidate_CON01_binding_receipt.json)。控制器已在 2026-10-03T08:37:52.018722+00:00 正常完成。复现须使用新的输出目录。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home1/gongwk/anaconda3/bin/python3.11 \
  /data/private_cpu_tools/analyze_actual_baselines.py \
  --configuration /data/formal_frozen/accuracy_configuration.json \
  --frozen-tools /data/formal_frozen/tools_source/tools \
  --output-dir /data/new_baseline_matrix_analysis \
  --poll-seconds 30 --timeout-hours 24
```

`--configuration`、`--frozen-tools`、`--output-dir` 均必填，含义如上一节。`--poll-seconds` 默认30，允许1–60秒；`--timeout-hours` 默认24，允许大于0且不超过48小时。两例固定为 CON01/03，不增加其他 baseline。正式 config SHA 必须为 `f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c`，科学 baseline fingerprint 必须为 `328c398496c5b90f459381ca462ba6597cbbe5f2e9700f0b1a273f1408962bac`；100k seeds、seed0 保持。等待时没有 GPU 作业或 SSH channel 长期占用。

完成后另执行 [CON03 来源复核脚本](verify_CON03_same_phase_receipts.py)，没有命令行参数，固定读取本轮两臂已完成报告、原冻结 helper 与既有输出，仅在私有 controller 目录新增收据。实际 CPU 命令为：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home1/gongwk/anaconda3/bin/python3.11 \
  /cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1/root_baseline_matrix_analysis_v1_controller/verify_CON03_same_phase_receipts.py
```

[原复核收据](candidate_CON03_binding_receipt.json) 保存实际 argv、脚本 SHA、B/C report/export 前后 SHA、源 fingerprint、五份官方 TCK SHA、两臂实际 TCK 端点逐位回读、track_metrics 数量和配置前后 SHA。它复用 `completed_run`/`verified_export`，未重算矩阵、FA、追踪、SIFT2或任何验收门槛；禁止覆盖现有收据。 最终采集命令、SSH helper前后SHA及冻结config/helper/report复核见 [采集收据](CON03_evidence_collection_receipt.json)；本目录文件的大小与SHA见 [原文件清单](artifact_manifest.json)。

## 4. 对应原软件调用

控制器只比较已有结果，没有对应的原软件求解命令。参考源由原官方 manifest 绑定为独立 raw DWI + official FS 全链，保留当时实际命令、全部输出 SHA 和五 seed 来源；原 MRtrix3 binary 为 `3.0.3-103-g026e850d`。对应追踪形式为：

```bash
MRTRIX_RNG_SEED=0 tckgen wm_fod_norm.nii.gz tracks.tck \
  -algorithm iFOD2 -seed_gmwmi gmwmi.nii.gz -act five_tissue.nii.gz \
  -seeds 100000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5
```

原五次使用各自已记录的 seed。实际参考目录是 `oldroot/task_04/official_raw10_cpu_group_A_v3/sub-CON01`，其中 `oldroot=/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002`；manifest SHA 为 `64b9066e465ddc1ae8e688c1fec48b2a5611a6312b8637a302fb5404e7d7ca5a`。CON03 对应 `oldroot/task_04/official_raw10_cpu_group_B_v3/sub-CON03`，manifest SHA 为 `60b4e0a48c194e0e7a0b9e764af153415b699b487c019632b325cb4937e6d85a`。正式 B/C 使用相同的已完成 candidate FS；独立 official 链使用其先前 baseline FS，本目录不宣称两边 FS 文件 SHA 相同。所有比较先检查 canonical raw 及官方 producer/实际执行来源；不以固定 FNIT 中间输入的 operator 参考替换官方 wholechain 来源。

## 5. 最新真实结果、耗时与图

### 本轮 CON01 baseline 对官方五次

[本轮运行收据](evidence/sub-CON01/run_receipt.json) 保存实际 argv、raw/FS来源及前后科学源；[原GPU报告](evidence/sub-CON01/original_gpu_report.json) 与 [原wall报告](evidence/sub-CON01/original_wall_report.json) 按原字节保留。实际 baseline 保留 **14,300** 条轨迹。矩阵 **180/240** 项通过、60失败；population **4/25** 项通过、21失败，均无未评估项。原报告分别为 [matrix](evidence/sub-CON01/matrix_envelope.json)、[population](evidence/sub-CON01/population_envelope.json)。完整五比较值/门槛/判定在 [48个字段记录](evidence/sub-CON01/five_comparisons_by_field.json)，每个字段含五个原官方 seed 的比较；[CSV](evidence/sub-CON01/eight_atlas_summary.csv) 每行一套atlas。

每格为通过数/5，合计分母30：

| Atlas | Count L1 | FBC L1 | Dice | Pearson | Length MAE | FA MAE | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aparc+tian-s1 | 5/5 | 5/5 | 3/5 | 5/5 | 5/5 | 2/5 | 25/30 |
| aparc.a2009s+tian-s1 | 3/5 | 3/5 | 4/5 | 1/5 | 5/5 | 4/5 | 20/30 |
| fs-aparc | 5/5 | 3/5 | 4/5 | 5/5 | 5/5 | 2/5 | 24/30 |
| glasser+tian-s1 | 2/5 | 5/5 | 2/5 | 5/5 | 3/5 | 3/5 | 20/30 |
| glasser+tian-s4 | 2/5 | 5/5 | 2/5 | 5/5 | 3/5 | 2/5 | 19/30 |
| schaefer1000+tian-s4 | 3/5 | 5/5 | 1/5 | 1/5 | 5/5 | 2/5 | 17/30 |
| schaefer200+tian-s1 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | 29/30 |
| schaefer500+tian-s4 | 5/5 | 5/5 | 3/5 | 4/5 | 5/5 | 4/5 | 26/30 |

### 本轮 CON01 baseline 与已绑定 candidate

[C 原报告](candidate_CON01_report.json) 的 actual GPU/wall、配置、科学源和输出 SHA 已再次只读核验，见 [绑定收据](candidate_CON01_binding_receipt.json)；它保留 **14,431** 条轨迹。双方都来自当前正式 `formal_accuracy_raw_v1`，不是前轮 baseline。下表由原报告的判定直接汇总；[比较摘要](CON01_same_phase_comparison_summary.json) 保留数值、门槛身份与原 C 报告 SHA。

| 矩阵字段（各/40） | 本轮B | 本轮C | 通过数变化 |
|---|---:|---:|---:|
| count_relative_l1 | 30 | 25 | -5 |
| sift2_fbc_relative_l1 | 36 | 29 | -7 |
| count_support_dice | 24 | 33 | +9 |
| count_pearson | 31 | 30 | -1 |
| mean_length_common_normalized_mae | 35 | 38 | +3 |
| mean_fa_common_normalized_mae | 24 | 32 | +8 |
| 合计（/240） | 180 | 187 | +7 |

两臂48组矩阵字段的官方 pair 值与门槛完全相同。Count L1、FBC L1 和 Pearson 通过数下降；Dice、Length、FA 通过数增加，没有所有指标都改善的结论。

| Population字段（各/5） | 本轮B | 本轮C |
|---|---:|---:|
| accepted_fraction_absolute_difference | 2 | 5 |
| length_ks | 1 | 5 |
| endpoint_8mm_histogram_pearson | 0 | 2 |
| 原网格保存点访问分布 Pearson（`tdi_native_voxel_pearson`） | 0 | 1 |
| 四体素块保存点分布 Pearson（`tdi_four_voxel_block_pearson`） | 1 | 0 |
| 合计（/25） | 4 | 13 |

轨迹工具的8mm endpoint bins 由每次比较中全部运行的端点 extrema 独立构建，其他病例或运行可不同。本例实际逐项核对 B/C `endpoint_bin_edges_mm`、公共网格 shape/affine 及五字段官方 pair 值/门槛，均完全相同。接受比例差上限为 `.00223`，length KS 上限为 `.01620673531708`。本例 `+9` 是原五字段比较的通过判定数变化，不能称为总体准确率或完全匹配。原网格保存点访问分布 / 四体素块保存点分布仍由保存采样点逐个 `np.rint` 到公共网格得到，定义未改成 tckmap。

本次 baseline CPU 分析 **6.832 s**；candidate 已有 CPU 分析 **7.917 s**，二者读取范围/前后复核不同，不作为性能比较。正式 raw-DWI CLI B为 **2200.811 s**，C为 **2072.133 s**；导出在计时结束后另计 **5.383/5.080 s**。这是共享负载下的当前CON01配对观察，整体速度判断由正式两例AB/BA及后续汇总负责。三类GPU显存 ledger/采样数据保留原 GPU/wall 报告，本 CPU 比较不产生新的GPU显存测量。

![本轮CON01 baseline保存轨迹的长度与真实点访问图](evidence/sub-CON01/population.png)

图中 `fnit_0` 为本轮baseline；中间FOD voxel z附近五层 point visits 原样汇总，只作展示，所有判定使用完整轨迹和网格。

### 本轮 CON03 baseline 与 candidate

实际 B 保留 **11,606** 条轨迹，C为 **11,582** 条。原 B 报告：[matrix](evidence/sub-CON03/matrix_envelope.json)、[population](evidence/sub-CON03/population_envelope.json)、[48字段五比较](evidence/sub-CON03/five_comparisons_by_field.json)、[8atlas CSV](evidence/sub-CON03/eight_atlas_summary.csv)。实际 B/C 的原 GPU/wall 报告均保留，见 [B运行收据](evidence/sub-CON03/run_receipt.json)、[C运行收据](evidence/candidate_sub-CON03/run_receipt.json)；[C原CPU报告](candidate_CON03_report.json) 仍如实保留当时 B 运行中的 timing 快照，当前完整配对另存为 [两例 wall/memory 汇总](actual_two_pair_wall_memory_summary.json)。

B矩阵 **199/240**，C为 **190/240**；B population **5/25**，C为 **3/25**。下列199来自本轮实际completed producer，未使用历史199替代。两臂无未评估的交叉比较，但整体均失败。

每格为 B→C 通过数（各字段分母5），合计分母30：

| Atlas | Count L1 | FBC L1 | Dice | Pearson | Length MAE | FA MAE | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aparc+tian-s1 | 3→3 | 2→4 | 4→5 | 4→2 | 5→5 | 5→5 | 23→24 |
| aparc.a2009s+tian-s1 | 2→5 | 5→5 | 3→4 | 1→0 | 5→5 | 5→5 | 21→24 |
| fs-aparc | 3→2 | 4→4 | 5→5 | 5→2 | 5→5 | 5→4 | 27→22 |
| glasser+tian-s1 | 5→5 | 5→5 | 5→4 | 4→1 | 4→2 | 5→4 | 28→21 |
| glasser+tian-s4 | 5→5 | 5→5 | 5→2 | 1→2 | 5→4 | 5→3 | 26→21 |
| schaefer1000+tian-s4 | 5→5 | 5→5 | 2→3 | 1→4 | 4→4 | 5→4 | 22→25 |
| schaefer200+tian-s1 | 5→5 | 5→5 | 5→5 | 4→4 | 5→5 | 5→5 | 29→29 |
| schaefer500+tian-s4 | 5→5 | 5→5 | 3→3 | 0→1 | 5→5 | 5→5 | 23→24 |

| 矩阵字段（各/40） | 本轮B | 本轮C | 通过数变化 |
|---|---:|---:|---:|
| count_relative_l1 | 33 | 35 | +2 |
| sift2_fbc_relative_l1 | 36 | 38 | +2 |
| count_support_dice | 32 | 31 | -1 |
| count_pearson | 20 | 16 | -4 |
| mean_length_common_normalized_mae | 38 | 35 | -3 |
| mean_fa_common_normalized_mae | 40 | 35 | -5 |
| 合计（/240） | 199 | 190 | -9 |

| Population字段（各/5） | 本轮B | 本轮C |
|---|---:|---:|
| accepted_fraction_absolute_difference | 1 | 0 |
| length_ks | 0 | 0 |
| endpoint_8mm_histogram_pearson | 0 | 0 |
| 原网格保存点访问分布 Pearson（`tdi_native_voxel_pearson`） | 3 | 3 |
| 四体素块保存点分布 Pearson（`tdi_four_voxel_block_pearson`） | 1 | 0 |
| 合计（/25） | 5 | 3 |

[本轮 CON03 配对摘要](CON03_same_phase_comparison_summary.json) 保存全部48字段的双方五比较值、判定和门槛身份。48组矩阵官方 pair 值和门槛完全相同；五项 population 的官方 pair 值和门槛也相同。实际 B/C endpoint bins、grid shape/affine完全相同；bins一般仍依各次全部端点 extrema构建，本例仅报告已核验的同值关系。Count/FBC通过数各增加2，Dice/Pearson/Length/FA分别减少1/4/3/5，没有全字段改善的结论。

本次B03 CPU分析 **6.038 s**。两臂端点与返回对象逐位一致、weights/lengths/mean_fa/endpoints数量一致、原官方五TCK和同次导出SHA前后不变。真实脑图如下；`fnit_0`仅表示当前baseline，point visits沿用原定义，展示层不参与取舍。

![本轮CON03 baseline长度与真实point visits](evidence/sub-CON03/population.png)

### 两例实际配对耗时与显存

| 病例 | B CLI秒 | C CLI秒 | C−B秒 | C/B变化 | B/C计时外导出秒 |
|---|---:|---:|---:|---:|---:|
| sub-CON01 | 2200.811 | 2072.133 | -128.678 | -5.847% | 5.383/5.080 |
| sub-CON03 | 2033.586 | 2043.839 | +10.253 | +0.504% | 5.081/4.849 |
| 两例合计 | 4234.398 | 4115.972 | -118.426 | -2.797% | 单独导出，不并入CLI |

这是共享GPU负载下两例实际AB/BA观察。CON03候选 **+0.504%** 的单例增时完整保留；两例合计 **-118.426秒**（-2.797%）不构成受控速度证明。其余八例没有本轮fresh baseline，不以旧baseline补齐十例配对。

| 实际运行 | Allocated GB | Reserved GB | Process-tree GB |
|---|---:|---:|---:|
| sub-CON01 baseline | 14.681594880 | 17.607688192 | 19.795017728 |
| sub-CON01 candidate | 14.681651200 | 17.607688192 | 19.795017728 |
| sub-CON03 baseline | 14.679795712 | 17.607688192 | 19.795017728 |
| sub-CON03 candidate | 14.679790592 | 17.607688192 | 19.795017728 |

四次三类观测值均小于 **20e9字节**，没有新GPU测量。Allocator ledger保留每次既有reset前与退出时的峰值；process-tree为实际PID及当时子进程同时显存的0.5秒目标间隔采样，最大观测间隔和monitor状态保留原报告，采样最大值不是连续显存上界。

## 6. 更新记录与benchmark记录

- 科学起点 `7af34e6d`，正式配置 `f8eba1cb`，两例baseline科学fingerprint `328c3984`。
- 当前独立worktree从 `b2d1476f93614c9359025a969c50d3c2ac5879ac` 开始，只增加私有 CPU 控制器与本目录证据。
- CON01：实际完成且所有原报告/export/helper/config/TCK前后SHA通过，端点和标量数量通过。控制器SHA `30a11ba5c31d5ab023bd73479961ff710a1cacbaf052e822345b559b5ffdbfdc`。
- 首次证据提交 `2b593167`：CON03当时未完成，原快照保留 `not_assessed`。
- 本次CON03收尾：实际baseline completed后，原PID11158 CPU控制器完成且配置/helper前后SHA相同，另保留原matrix/pop/PNG与B/C wall/GPU报告、严格复核收据和全部逐字段配对。没有重跑GPU、修改科学源码/门槛或新增其余八例baseline。
- 两例实际同阶段矩阵判定总数 B **379/480**、C **377/480**；population B **9/50**、C **16/50**，不能把它们解释为总体准确率。其余八例比较须保持各自实际来源与历史参照边界。

## 7. 工具、原实现及参考依据

- [冻结分析器的仓库来源](../../../../tools/analyze_connectome_accuracy_cohort.py)：`completed_run`、同次导出与原判定聚合。
- [矩阵比较](../../../../tools/benchmark_connectome_raw_cohort_envelope.py)、[重复范围](../../../../tools/connectome_repeat_common.py)、[轨迹比较](../../../../tools/benchmark_connectome_tracking_population.py)：定义、原始来源核验与有限官方重复范围。
- [本轮总说明](../../../../docs/connectome/ACCURACY_OPTIMIZATION_20261003.md)、[十例实际来源说明](../../../../docs/connectome/actual_cohort_comparison.md)：raw/FS来源及完整pipeline科学验收。
- [MRtrix3原代码库026e850d](https://github.com/MRtrix3/mrtrix3/tree/026e850d)、[iFOD2原实现](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/algorithms/iFOD2.h)、[ACT原实现](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/ACT/method.h)。模型文献与参数的完整说明见总说明参考文献；本 CPU 控制器未移植或重新执行原求解器。
