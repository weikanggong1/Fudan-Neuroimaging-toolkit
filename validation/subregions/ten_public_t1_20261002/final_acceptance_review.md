# 十例公开 T1：准备阶段验收清单

本文件保留终态前的验收条件和准备阶段审查。最终十例官方与 FNIT 已全部完成；实际结果见[当前 main 配对对照](latest_main_regression/official_comparison.md)、[终态精度复核](final_accuracy_review.json)及[110 分区 Dice](analysis/per_region_dice.md)。准备审查只读取已有文本与元数据，没有读取 MRI。

下方第 5 项所预定的首轮 HR 报告写入遇到 NumPy 标量 JSON 序列化错误，原脚本与失败记录保留。仅修复报告序列化的 [v2 审计](official_hr_grid_conservation_v2.json)已通过 40 张图和 1,100 条标签记录的严格计数与独立重采样门禁，原分析元数据未改；[失败记录](hr_conservation_attempt1_failure.json)与[重跑身份](hr_conservation_v2_launch.json)记录过程。最终发布前已人工核验 v2 终态。

## 已独立核对的档案

| 档案 | 已核对内容 | 适用范围 |
| --- | --- | --- |
| `first_case_interface_audit.json`，1,079,580 B，SHA `1ca4ee901d9b70ab8c390b6fce7932d37a8f98e6ce1ca0a2af9cfba556d5bc94` | 本地与远程扩展档案一致；440 ROI 与 24 family 唯一键；全部 Dice/Jaccard、交并集、差异体素、硬体积及 family 加权/平均 Dice 独立重算通过；raw/stage native 网格匹配；四张 HR 的 union shape/affine 独立重算完全相同，整数相位残差最大 1.43e−14 voxel | sub-01 接口检查；不是最终队列精度或 HR 原图计数守恒结论 |
| `raw_all_live_audit.json`，230,894 B，SHA `820774cb5af76224b427f7495d8bdcbfc75f87473e3d95d3b5018c6fdb476059` | 10 名/10 输入 SHA；全部 raw 原生和 HR 数值、标签支持、体积、四张 mesh 最小 Jacobian、冻结源码观察器、GPU 与计时字段完整；计时 mean/median/sample SD/range 重新计算一致；sub-01 来源与首例档案一致 | 10 例 raw FNIT 完成情况；未比较十例官方标签 |
| `acceptance_preparation_metadata_review.json` | 以上元数据核验的来源 SHA、核验数量、首例四张 HR 网格与剩余证据 | 准备阶段审查记录 |

`raw_all_ten_numeric_gate.json` 不是实际文件名；本轮完整十例 raw 数值门禁保存在 `raw_all_live_audit.json`。其中 `raw_input_sha256` 是 `{绝对输入路径: SHA}` 映射，不能当作 SHA 字符串使用。最终 analyzer 的来源字段也保留输入路径映射。

## 最终必须逐项通过

1. **终态与身份**：三队列终态；`analysis_status.json` 与 `cohort_analysis.json` 均 `final_outcome_ready=true`，`not_completed_attempts=[]`。固定 sub-01–sub-10 和 snapshot 1.0.2，输入 SHA 与原始公开下载清单一致；sub-01 的 `development_seen=true`，另九例 false。环境准备失败、依赖阻塞及成功重跑的历史单列，不伪装为算法失败，也不从计划名单删除。
2. **完整性与唯一键**：10 例全完成时应有 4,400 ROI、240 family、40 ALL110 case 行。分别按 `(case_id,space,label)`、`(case_id,space,family)`、`(case_id,space)` 唯一；四空间为 raw/stage × native/hr，每例每空间 110 ROI，家族为 BS、TH、HIP-L/R、AMY-L/R。若最终有失败，仍保留 4,400 ROI 和 40 case 行及 NA 原因；family 行可能少于 240，必须公开缺测，不能补造行或声称十例全部成功。
3. **计数与指标**：非负整数计数；交集不超过双方计数，union=a+b−intersection，different=a+b−2×intersection；Dice=2×intersection/(a+b)，Jaccard=intersection/union。双方均空为 NA；单方缺失为 0。硬体积=计数×实际网格体素体积。软体积字段及 NA 原因与原始 API/官方体积来源一致。
4. **网格和命名**：raw native=公开 T1 原网格，stage native=该例 fresh recon-all norm 网格；官方 nearest/order0→目标，完整 affine 而非轴顺序假设。HR 使用官方轴、spacing、整数相位 union FOV，raw/stage 和同侧 HIP/AMY 共用相应网格。官方右 HA 原始 ID 只加 10000 一次，FNIT 已全局编号不再偏移。
5. **HR 守恒**：`official_hr_grid_conservation.json` 最终 `state=passed`；40 图、1,100 label-map 行，110 标签覆盖十例；`nonconserved_label_map_rows=0`、`independent_resample_disagreements=0`、`metadata_unchanged=true`，原图 SHA 与 finalanalysis 相同；每个 geometry gate 通过，count tolerance=0。已启动任务带 `--resample`，应保存独立 order0 重采样计数。任何差异保留并定位源头，不能以近似容差接受或修补结果。
6. **两组汇总**：`cohort_all` 固定 planned=10；`cohort_new_subjects` 固定 planned=9 且只排除 sub-01。每项保留 valid/NA 数，样本标准差 ddof=1；mean/median/range 重算。受试者等权汇总；每例 family/ALL110 的 reference-weighted Dice 仅以官方细分区计数加权，另保留 mean-label/micro/foreground 及每 ROI，不能把加权结果当成小核平均或随机复现范围。
7. **冻结源码与数值门禁**：最终 20 个 raw/stage 记录均通过 saved-array finite/integer/support、有限非负体积、native 硬计数及每结构正 mesh Jacobian 最小值；input SHA、observer before/after 运行模块 SHA、source manifest 和 428 runtime Python 模块一致；`solver_options_changed=false`、`reference_used_by_observer=false`。不存在未记录的 voxelwise Jacobian 图，不据此宣称做过该检查。
8. **显存与资源**：逐个检查 20 个 raw/stage own-PID monitor：PID/启动时间/GPU UUID 正确、样本数>0、sampling_errors=0，实测 sampled peak≤19073 MiB（20 GB 十进制上限）。现有 10 raw sampled peak 最大 18926 MiB。当前证据是周期采样的本进程峰值，不是未记录的瞬时全局峰值；共享 GPU 占用和等待与自身用量分开，stage 可使用另一个固定 UUID。
9. **计时配对**：API compute/save/total、runner process wall、observer overhead、输入/GPU 等待分开；official end-to-end 包括 fresh recon-all，stage 对比只使用官方三个细分割命令之和。速度倍数按同一例逐例配对后汇总，不以独立极值相除。solver 子计时不与包含它的 recipe 总计重复相加；缺失步骤 timer 为 NA。
10. **实测脑图**：`postprocess_status=completed`，steps/plots exit code=0，`plot_manifest.status=completed`；绘图绑定同一 finalanalysis/manifest，逐 ROI count audit 通过。固定 sub-01 连续性例，以及 raw-native ALL110 reference-weighted Dice 中位（偶数取较低中央值）和最差病例；完整排名、重复角色、NA 和开发标识公开。局部 ROI crop、多 axial slice、同窗/同切片；PNG header/size/SHA 与清单一致，不输出全头脸部影像。图上 Dice 为 native 统计，显示栅格不重新定义评价指标。

## 结果终态后最小回收集

先回收文本与 SHA，再按 `plot_manifest` 回收真实局部 PNG。已有 manifest、下载/许可/资源/硬件审计和冻结脚本不重复下载。缺陷排查前不回收 MRI、权重、图谱或源码快照。

| 用途 | 新增最小文件 |
| --- | --- |
| 精度与两组统计 | `analysis/analysis_status.json`、`analysis/cohort_analysis.json`、`analysis/cohort_summary.json`、`analysis/cohort_roi.tsv`、`analysis/cohort_family.tsv`、`analysis/cohort_case.tsv` |
| 运行终态及历史 | 最终 `fnit_queue.json`、`official_queue.json`、`fnit_stage_queue.json`；以 `cohort_analysis.queue_sources` 的实际路径/SHA 为准。归档准备历史已嵌入分析时不重复回收原日志；出现疑问只取对应文本记录 |
| HR 守恒 | `official_hr_grid_conservation.json`；若失败，再取对应审计日志及已记录原图边界/affine/count delta；不先重算或修补 |
| 步骤与计时 | `postprocess_status.json`；`analysis/steps/cohort_steps.json`、`cohort_steps.tsv`、`cohort_steps_summary.json`、`cohort_steps_summary.tsv`、`cohort_reconall_fstime.tsv` |
| 实际脑图 | `brain_figures/plot_manifest.json` 及其列出的 PNG（不要假设 PNG 文件名或数量） |

`cohort_analysis.json` 已包含完整 source audit、20 个 FNIT 数值门禁、GPU monitor 指纹/峰值/样本数、官方身份、timing 和 observer 内容。正常验收无需再回收 20 套 API/report/observer/monitor 原文；若任一字段缺失、不一致或 sampling error，则只回收对应的文本文件及前后 SHA。

## 冻结接口指纹

一次远程文本核验确认下列指纹均一致；已有等待进程保持原状，没有另启任何 waiter。

| 文件 | SHA-256 |
| --- | --- |
| `analyze_cohort.py` | `ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc` |
| `plot_cohort.py` | `f135af099a67d543bfbefbbabbe8a71f1c6ba044133a7b5767d8368787e5b4ab` |
| `extract_cohort_steps.py` | `ca08aad54b81b3d6e7f6e826d205c7043081db6dc5136cb455b39374729e9eaa` |
| `run_cohort_postprocess.py` | `34454aae59b30910c29289324845599468af300bb7e1ca5f5e0b8c61dc728ebd` |
| `audit_official_hr_grid_conservation.py` | `b5a30f56f97533918b12271ed118de975549021739d0ad5f26455e9f0fb08c13` |

守恒结果是额外人工验收门禁，冻结发布脚本并未自动把它纳入 `benchmark_evidence`。父任务应在发布前明确核对该 JSON 通过，并保留其最终 size/SHA 和链接；不需要因此修改本轮冻结脚本。
