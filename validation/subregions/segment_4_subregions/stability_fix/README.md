# segment_4_subregions：丘脑精度修复与完整 benchmark

[功能、参数与流程图](../../../../docs/subregions/README.md) · [修复前记录](../README.md) · [v16 C6 基线](../../speed_v16/README.md)

2026-10-01，在同一例真实公开 T1 上完成冻结版本 **回溯修复版** 的四类分割复测。官方同阶段输入的丘脑细核加权 Dice 从修复前的 0.910995 恢复至 **0.960523**；C6 基线为 0.964068，当前仍相差约 0.003545。下面同时报告原始 T1 和全部结构的结果。

## 修复与范围

窄 Gaussian 合成标签模型产生较大的数据总代价，FP32 标量汇总可能舍入有效的小幅下降，影响 L-BFGS 的 Wolfe 线搜索。修复限定在**丘脑的合成标签与强度拟合**：图谱 alpha 统计、数据项及几何顶点梯度按固定顺序归约，贡献用 FP64 累加后恢复 FP32；数据和形变先验总代价、L-BFGS 方向、历史及线搜索的标量判断使用 FP64。丘脑使用原生 L-BFGS 的二环方向和有界 Armijo 回溯：以实际 FP32 位移计算方向内积及曲率，单次顶点位移最多 0.5 个工作网格体素，步长依次减半、最多 20 个试探。只接受代价充分下降、代价与梯度有限、全部 tetrahedron Jacobian 为正的试探；失败或报错时恢复接受点及有效缓存。网格目标函数前向和反向的小矩阵运算使用完整 FP32，结束或异常时恢复调用者的 TF32 设置。影像、顶点、Gaussian 参数及顶点梯度保持 FP32，网络、其余阶段和其他 recipe 保留默认精度。采样、owner hints、工作网格、平滑日程和迭代上限沿用原设置。

契约测试覆盖 FP64 代价、消去敏感的方向内积、Armijo 对上升试探的拒绝、接受位置的缓存与 FP32 参数/梯度；CUDA 测试另验证固定梯度归约顺序。没有新增依赖，生产流程继续使用原生 PyTorch；此次修复发生在项目已有的共享 GEMS recipe 子函数，公共函数及输出结构一致。此例的精度恢复支持该修复；其他结构仍沿用其原数值路径，当前不保证完整四结构流程逐比特重复。

先评估了通用 synthetic FP64 候选 C1：同阶段丘脑恢复至 0.965569，但右海马加权 Dice 从 0.831696 变为 0.772371，右杏仁核从 0.929489 变为 0.898656。随后仅丘脑标量修复 C2 的完整 stage 加权 Dice 为 0.902469，表明标量汇总单独不够稳定。C3/C4 的合成拟合重复一致，但 C4 后续强度拟合仍使官方 Dice 在 0.885167–0.958684 间波动。C4 接受步记录显示两次合成阶段的零步停止时梯度 L2 仍为 488.898 / 1,924.620，方向内积分别为 −1,112.756 / −234.444。C5 据此加入零步重试，并稳定全部丘脑阶段。完整同阶段加权 Dice 为 0.949945，计算 634.443 s 未达 10 分钟；原始 T1 为 0.775745、552.424 s。耗时拆分为丘脑约 170 s、双侧海马/杏仁核约 410 s。最终回溯修复版保留稳定数值路径，以有界回溯替换丘脑强 Wolfe 线搜索；其他 recipe 保留原精度与算法。[范围评估阶段](c1_scope_evaluation/full_stage/summary.json)及[原始 T1 记录](c1_scope_evaluation/full_raw/summary.json)保留原始身份。

## 数据、运行范围与耗时

一例 OpenNeuro ds000114 公开去面部 `sub-01_T1w.nii.gz`（CC0），形状 `256×156×256`，[来源清单](../../../../examples/data/SOURCES.json)。原始 T1 SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。

- 原始 T1 全流程：内部一次 SynthSeg+、DK68 分区及白质代理，再拟合脑干、双侧丘脑、左/右海马与杏仁核。
- 官方同阶段输入：同一人的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`；FNIT 读取已保存 FreeSurfer 8.2 参考用于对照。

两种输入均运行唯一公开函数 `segment_4_subregions`，完整四项 recipe，`optimization="fast"`、4 个 CPU 线程，共享 H100 物理 GPU 1。每约 5 秒记录负载与本进程显存，本进程停止上限 19,073 MiB。图谱和模型沿用已核验资源，没有重新调用官方软件。

| 输入范围 | 计算 | API 含保存 | 监控进程总墙钟 | PyTorch 分配峰值 | 本进程显存采样峰值 |
|---|---:|---:|---:|---:|---:|
| 原始 T1 全流程 | 410.721 s（6.85 min） | 411.237 s | 435.155 s | 15.47 GiB | 18,930 MiB |
| 官方同阶段输入 | 533.773 s（8.90 min） | 534.373 s | 578.304 s | 4.91 GiB | 9,718 MiB |

计算包括影像读入、共享准备、全部拟合与合并；API 另含保存。监控进程总墙钟还含 Python 导入、CUDA 初始化和验证记录。首次下载、图谱安装及官方独立运行不计入。完整 FNIT CPU 整例尚未测量；共享卡的实测时间不直接用作归因明确的加速比。官方四结构阶段的固定历史记录为脑干 240.10 s、丘脑 596.84 s、左右海马/杏仁核 1,005.17 s，合计 1,842.11 s（30.70 min），历史文档报告 4 线程、不含 recon-all；记录混合不同服务器，目前只找到脑干原始计时日志，未能重新核验整段合计。本次未重新运行官方程序，见[日志审计](official_timing_evidence/audit.json)和[固定历史文档](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/blob/d90020e/validation/subregions/unified.md)。

## 丘脑细核与官方对照

逐核硬 Dice 按官方标签体素数加权：`sum(reference_voxels × Dice) / sum(reference_voxels)`；平均 Dice 为可评估标签的算术平均。严格通过要求同一标签 **Dice ≥ 0.95 且硬体积相对误差 ≤ 5%**。两张标签均为空的标签单独记录，其 Dice 不计为 1；原网格不同使原始 T1 与同阶段输入的参考体素数和可评估标签数不同。

| 输入范围 | 可评估核 / 参考体素 | C6 加权 Dice | 修复前加权 Dice | 修复版加权 Dice | 修复版平均 Dice | 严格通过 |
|---|---:|---:|---:|---:|---:|---:|
| 同阶段输入 | 44 / 10,123 | 0.964068 | 0.910995 | **0.960523** | 0.908637 | 17/44 |
| 原始 T1 | 46 / 7,793 | 0.764980 | 0.776716 | **0.775793** | 0.630122 | 0/46 |

## 全部结构的逐标签指标

下表同样使用**细标签 Dice**，而非将整个家族合并后的外形 Dice。

| 结构 | 原始 T1 标签数 | 原始 T1 加权 Dice | 原始 T1 平均 Dice | 同阶段标签数 | 同阶段加权 Dice | 同阶段平均 Dice |
|---|---:|---:|---:|---:|---:|---:|
| 脑干 | 4 | 0.924751 | 0.882705 | 4 | 0.991306 | 0.983000 |
| 双侧丘脑 | 46 | 0.775793 | 0.630122 | 44 | 0.960523 | 0.908637 |
| 左海马 | 19 | 0.678531 | 0.639700 | 19 | 0.870763 | 0.848035 |
| 左杏仁核 | 9 | 0.790483 | 0.582631 | 9 | 0.941614 | 0.862876 |
| 右海马 | 19 | 0.634352 | 0.595625 | 19 | 0.813436 | 0.773765 |
| 右杏仁核 | 9 | 0.760340 | 0.413409 | 9 | 0.917690 | 0.646729 |
| 全部可评估标签 | 106 | 0.833303 | 0.612754 | 104 | 0.955452 | 0.849160 |

原始 T1 自动粗分割与官方输入不同，其整体指标与同阶段拟合分别解释。本次仅验证一例开发病例；速度和丘脑恢复已实测，全部亚区的官方等价尚未达到。110 项软体积均有限且非负，四个最终最小 Jacobian 均为正；逐标签硬/软体积、空标签及严格阈值结果保存在下方原始表中。

## 软体积与官方对照

软体积为工作网格后验积分。下面按标签报告 `abs(FNIT − 官方) / 官方` 的平均值及误差在 5% 内的项数；即使原网格硬标签为空，仍可评价其软体积。逐标签硬体积、软体积及相对误差见 [同阶段表](full_stage/comparison.tsv)与[原始 T1 表](full_raw/comparison.tsv)。

| 家族 | 同阶段平均绝对相对误差 | 同阶段误差 ≤5% | 原始 T1 平均绝对相对误差 | 原始 T1 误差 ≤5% |
|---|---:|---:|---:|---:|
| 脑干 | 1.00% | 4/4 | 6.95% | 2/4 |
| 丘脑细核 | 2.67% | 45/50 | 11.83% | 14/50 |
| 左海马 | 3.84% | 14/19 | 7.66% | 9/19 |
| 右海马 | 4.54% | 10/19 | 6.74% | 6/19 |
| 左杏仁核 | 2.70% | 8/9 | 18.36% | 3/9 |
| 右杏仁核 | 4.73% | 6/9 | 14.30% | 2/9 |

## 六层轴位脑图

每张图从上至下为**官方、FNIT、标签差异**；红色表示任意细标签不同，包含结构边界和核内分界。显示时 T1 先重排为 RAS；原始 T1 含斜切方向，另用线性插值建立 RAS 轴对齐显示网格，并保留原各向异性体素比例。两张标签均用最近邻插值到同一显示网格，Dice 仍在原始输入网格计算；专门丘脑图按官方丘脑范围等分选取六层，使用原图谱细核颜色，左右标记和世界坐标随图保存。

### 同阶段输入：丘脑细核

![同阶段丘脑细核：官方、FNIT 与标签差异](full_stage/thalamus_nuclei_vs_official_axial.png)

### 原始 T1 全流程：丘脑细核

![原始 T1 丘脑细核：官方、FNIT 与标签差异](full_raw/thalamus_nuclei_vs_official_axial.png)

### 完整四类结构

![同阶段四类结构与官方对照](full_stage/vs_official_axial.png)

![原始 T1 四类结构与官方对照](full_raw/vs_official_axial.png)

## 重复性、源码与输出核对

前期单丘脑两次重复的官方加权 Dice 为 0.958382 / 0.963259，相互加权 Dice 为 0.989980、相差 129 个标签体素；[重复汇总](c1_runs/summary.json)保存完整记录。这两次属于 C1 的标量精度试验；未采用的 C2 表明仅此修改不足以稳定恢复。[C3](c3_runs/summary.json)、[C4](c4_runs/summary.json)和[C5](c5_runs/summary.json)保留各自重复性、代价和阶段数组核对记录。[回溯修复版的两次同阶段单丘脑](c6_runs/summary.json)官方加权 Dice 均为 0.961959、平均 Dice 为 0.910199，标签相互 Dice 为 1.0、差异体素为 0；六个拟合阶段的 alpha、顶点、Gaussian 参数和代价轨迹逐字节相同，耗时 145.819 / 143.637 s。此重复证据限定在该病例的丘脑；完整 pipeline 合并还与其他结构比较置信度，其最终指标以本页的两次整例运行为准。最终源码另校正 Adam 不使用线搜索的报告字段，实际 pipeline 全部使用 L-BFGS。

独立复核另完成 [63 项本地归档与指标检查](independent_final_audit.json)和 [16 项服务器 CPU 标签统计复算](independent_remote_metric_audit.json)，直接读取保存的标签与官方参考，结果与报告一致。原始 T1 的 46 个可评估丘脑标签中，44 个参考非空；另两个标签各有 1 个预测假阳性体素，均值与严格通过数将其计入，加权时参考权重为 0。

最终上传包包含 **443 个文件、412 个运行时 Python 文件**，两次实际运行逐文件检查大小与 SHA-256。每次 8 项实际输出均有服务器文件身份、dtype、形状、affine 与网格检查；原始影像、图谱、权重和网格诊断数组保留在服务器。

| 可复核记录 | 原始 T1 | 同阶段输入 |
|---|---|---|
| 完成、计时、质量与检查 | [summary](full_raw/summary.json) | [summary](full_stage/summary.json) |
| 官方逐标签硬/软体积对照 | [comparison.tsv](full_raw/comparison.tsv) | [comparison.tsv](full_stage/comparison.tsv) |
| 原生函数报告 | [api_report](full_raw/api_report.json) | [api_report](full_stage/api_report.json) |
| 逐组件耗时 | [timing_summary](full_raw/timing_summary.json) | [timing_summary](full_stage/timing_summary.json) |
| GPU 占用记录 | [gpu_load](full_raw/gpu_load.jsonl) | [gpu_load](full_stage/gpu_load.jsonl) |
| 实际源码与 8 项输出核对 | [audit](full_raw/actual_source_and_output_audit.json) | [audit](full_stage/actual_source_and_output_audit.json) |
| 图层、网格、切片及 SHA | [fine-nucleus metadata](full_raw/thalamus_nuclei_vs_official_axial.json) | [fine-nucleus metadata](full_stage/thalamus_nuclei_vs_official_axial.json) |
| 下载文件大小及 SHA | [archive](full_raw/archive_manifest.json) | [archive](full_stage/archive_manifest.json) |

[最终冻结源清单](final_source/source_manifest.json)、[源改动](final_source/source_patch.diff)和[发布源码依赖检查](final_source/publication_source_verification.json)分别记录实测快照与最终整合状态。后续 main 更新涉及 fMRI、MCFLIRT、SynthStrip 及依赖说明；本入口不调用 SynthStrip，分割使用的 37 项运行时及公开导出依赖与最终冻结源完全相同。全包中其他模块的差异在发布核对文件逐项列出。

最终源码的 GEMS 和统一 CLI 测试 **350 passed**；原生回溯优化器与作用范围的 CPU/CUDA 契约 **59 passed**，包含在完整测试中；正 Jacobian 接受守卫及真实融合路径集成 **5 passed**（[日志](tests_c6_integration.log)）。完整测试[日志](tests_gems.log)；最终源码的公共入口、CLI 与输出测试 **67 passed**（[日志](tests_final_public.log)）。[真实重复驱动](../../thalamus_stability.py)及[细核绘图脚本](../../plot_thalamus_nuclei.py)支持复核，完整调用沿用[验证命令](../../README.md#复核)。

## Reference

原软件命令、源码及四类分割文献见[功能文档末尾](../../../../docs/subregions/README.md#reference)。
