# segment_4_subregions：真实 T1 验证

[功能、参数和流程图](../../docs/subregions/README.md) · [十例完整 benchmark](ten_public_t1_20261002/README.md) · [当前 main 官方配对](ten_public_t1_20261002/latest_main_regression/official_comparison.md)

`segment_4_subregions` 从一张三维 T1 完成共享粗分割、DK68 皮层分区、白质代理，以及脑干、丘脑、左/右海马与杏仁核四项 recipe 的原生 PyTorch 拟合。统一输出原 T1 网格标签、110 项硬/软体积和四份高分辨率标签。

## 数据及比较范围

本轮使用 OpenNeuro ds000114 snapshot **1.0.2** 的 **ses-test，sub-01–sub-10**，十例选择在测试前固定；另列排除开发用 sub-01 的新九例。公开快照已去脸，本次未追加处理。CC0 许可、输入文件及 SHA-256 见[数据记录](ten_public_t1_20261002/data_selection.md)。这批公开 T1 与原单例开发派生 T1 分开记录。

- **raw 完整流程**：从公开 T1 内部完成 SynthSeg+、TorchFAST、白质归一化、工作网格与全部四项分割；原网格为公开 T1 网格。
- **stage 条件测试**：读取同一病例本轮 fresh FreeSurfer 8.2.0-1 的 `norm/aseg/wmparc`；原网格为其 norm 网格，不把官方预处理计入 FNIT stage。

官方均从同一公开 T1 新执行完整 `recon-all -all -openmp 4` 和三个细分割命令，实际协议见[官方完整流程](ten_public_t1_20261002/official_protocol.md)。FNIT 运行时使用项目原生 PyTorch 实现；官方结果只作为独立评分参考。逐标签硬 Dice、硬/软体积、几何、有限值、Jacobian、模型/资产身份和自身显存均另行核对。

## 最新十例精度与完整耗时

当前 main `f436de5` 对十例完成独立 raw 全流程，API compute **253.66 ± 6.31 [245.68, 261.89] 秒**，API total **254.26 ± 6.34 [246.25, 262.55] 秒**，进程 wall **259.37 ± 6.45 [251.26, 267.76] 秒**；均值 ± 样本 SD [最小值, 最大值]，n=10/10，NA=0。新九例 raw 进程 wall 为 **260.21 ± 6.22 [251.26, 267.76] 秒**，n=9/9，NA=0。自身 PID 显存采样峰值 14,748–18,428 MiB，限制 19,073 MiB；共享 H100 PCIe、4 线程。

同病例本轮官方完整 `recon-all`＋三个细分割实际 wall 为 **125.45 ± 12.28 [102.13, 138.90] 分钟**，n=10/10，NA=0。[同病例配对对照](ten_public_t1_20261002/latest_main_regression/official_comparison.md)分别统计官方完整流程/current main raw 和官方三项细分割/ac692bb stage，不把不同病例极值相除。

同病例官方/FNIT wall 比值：当前 main raw 全部十例 **29.010 ± 2.643**，新九例 **29.231 ± 2.704**；ac692bb stage 分别为 **5.212 ± 0.457** 与 **5.293 ± 0.399**。每例先计算比值再汇总，样本 SD 使用 ddof=1；两组各 n=10/10、9/9，NA=0。raw 比较官方完整流程，stage 比较三个官方细分割命令之和；[逐例配对、范围与中位数](ten_public_t1_20261002/latest_main_regression/official_paired_runtime.tsv)和[完整统计](ten_public_t1_20261002/latest_main_regression/official_comparison.md)保留实际来源。

raw 精度原计量版本为 `ac692bb`；main 独立审计证实 50 张原网格/高分辨率标签图的值与几何完全相同，1,100 条硬/软体积字典及 160 条上下文记录相同，因此 raw Dice 与脑图可继承。[实际 main 审计](ten_public_t1_20261002/latest_main_regression/audit/summary.json)保留逐例来源。stage 结果及耗时来自 `ac692bb` 的实际测试，两版 24 个 GEMS 文件逐字节相同。

[六区域十例/新九例精度](../../docs/subregions/README.md#六区域原网格精度)给 raw-native 与 stage-native 的家族加权 Dice 均值、样本 SD、范围和 n/NA；[完整 110 分区 Dice](ten_public_t1_20261002/analysis/per_region_dice.md)、[880 行 ROI 汇总](ten_public_t1_20261002/analysis/cohort_roi_summary.tsv)、[4,400 行逐例 ROI](ten_public_t1_20261002/analysis/cohort_roi.tsv)、[逐例家族表](ten_public_t1_20261002/analysis/cohort_family.tsv)保留全部分区与计划病例。单方缺失 Dice 为 0；双方均空和失败为 NA，病例间 SD 不称为重复波动。

当前 main 四项 recipe 总计和共享预处理见[实际计时对照](ten_public_t1_20261002/latest_main_regression/official_comparison.json)；原冻结 stage 与官方的[步骤明细](ten_public_t1_20261002/analysis/steps/cohort_steps.tsv)、[步骤汇总](ten_public_t1_20261002/analysis/steps/cohort_steps_summary.tsv)及[recon-all FSTIME](ten_public_t1_20261002/analysis/steps/cohort_reconall_fstime.tsv)使用实际记录。未独立计时及不可评分的中间阶段 Dice 为 NA。真实[六层 axial 脑图](../../docs/subregions/README.md#官方对照脑图)及[脑图清单](ten_public_t1_20261002/brain_figures/plot_manifest.json)来自本轮 raw-native，含开发例、中位例和最低例。

[v2 HR 网格守恒审计](ten_public_t1_20261002/official_hr_grid_conservation_v2.json)通过 40 张官方 HR 图和 1,100 条标签记录核查，raw_hr/stage_hr 计数零差异，独立最近邻复查一致，原分析元数据未改。v2 修复首轮报告写入的 NumPy 标量 JSON 兼容性；算法、计数和网格规则保持。

## 历史版本

[单例开发重复性与精度修复](reproducibility_20261002/README.md)保留官方/FNIT 三次 all 流程重复、丘脑完整积分、脑干梯度归约及稳定连通域选择的原记录。原单例指标、分步骤、脑图和历史 recon-all 时间仅作为开发历史，不代表这批十例的最新结果。

[4178a48 完整 benchmark](segment_4_subregions/raw_precision_analysis/README.md)、[丘脑回溯修复](segment_4_subregions/stability_fix/README.md)、[入口整合](segment_4_subregions/README.md)与[v16 C6](speed_v16/README.md)继续保留来源身份。旧官方存档缺少完整生成记录；本轮十例均以 fresh 官方结果评分。

## 复核

```bash
# 原始 T1：--weights 为已核验的 SynthSeg+ 模型目录
python validation/subregions/run_unified.py \
  --t1 /absolute/path/sub-01_ses-test_T1w.nii.gz \
  --atlas-root /absolute/path/subregion_atlases \
  --weights /absolute/path/weights \
  --structures all --device cuda:0 --optimization fast \
  --output-dir /absolute/path/segment_4_subregions

# stage：另加 --aseg 和 --wmparc，将 --t1 改成本例 fresh norm.mgz
# 官方评分参考：--reference-brainstem/--reference-thalamus/--reference-left/--reference-right
# 分别指定独立官方已保存的四份 FSvoxelSpace 标签。
```

完整十例运行、终态数值审计、精度统计、步骤提取与图像协议见[benchmark 协议](ten_public_t1_20261002/benchmark_protocol.md)和[发布证据](ten_public_t1_20261002/benchmark_evidence.json)。当前 main 另建独立输出，与原冻结结果按同输入逐值审计，再绑定同病例官方时间。

### Reference

[原软件完整命令、实现链接及四项算法文献](../../docs/subregions/README.md#reference)。
