# 十例 benchmark 文档整合审阅

2026-10-02，仅核对当前文件和已冻结脚本。此文件是更新计划，不是最终结果；不填尚未生成的 Dice、耗时或脑图。主任务提供的当前进度为 raw 十例完成、官方及 stage 各四例完成；本地保存的 `analysis_status.json` 仍为 `waiting`。最终分析、步骤和脑图全部就绪后再更新三处公开入口。

## 当前文件身份

仓库 HEAD：`ac692bb4f7868a24ea4bd67180162e81726de9b4`。

| 只读文件 | SHA-256 |
|---|---|
| `docs/subregions/README.md` | `6c0901a42d8a4b2266ae318b242504d854a19132a02f943dce005e89587cfcd1` |
| `validation/subregions/README.md` | `79a2026fc3b249ee1dc16e010dcb4483fa09c326d3db0a3d9f5f5673fe10dab9` |
| 主页 `README.md` | `16e447bd9cd2d248c0c63f62d76f424c74589b5cd1239a3446c037ae1fded2db` |
| publisher F8 `publish_cohort_summary.py` | `f8c433c1451c3c45c3a1afee43738bd874788199434cdf24bc9803f028b1b011` |

本次只新增此审阅记录；公开文档、publisher、统计脚本、拟合脚本和生产代码均未修改。

## 最小更新范围

### 1. 功能页保持现有七节

当前 `docs/subregions/README.md` 的七个二级标题各出现一次。前四节保留功能和调用结构，第五节更新为最新十例，第六节继续记录历史，第七节保留原实现和文献。

| 现有节与当前行号 | 最小更新 |
|---|---|
| 从一张 T1 到完整结果，9 | 保留现有 end-to-end Mermaid、共享预处理及四项 recipe 说明。 |
| Python 调用、输入输出与参数，47 | 保留全部参数、输入/输出结构、完整变量名和注释；本轮未改变 API 或默认 fast 参数。 |
| 命令行，150 | 保留现有 FNIT CLI。 |
| 原软件调用，164 | 在三个细分割参考命令前补本轮 fresh `recon-all -i ... -s ... -sd ... -all -openmp 4`，并链接十例 `official_protocol.md` 的实际 CPU 环境和输出目录。 |
| 最新精度、运行时间与脑图，177–276 | 替换当前单例开发 benchmark 主体，按下述五小节列十例/新九例结果；保留 `### 官方对照脑图` 标题，维持现有锚点。 |
| 最近版本 benchmark 记录，278 | 新增“十例公开 T1，另列未参与开发的九例”记录；将现有重复性、完整积分、拓扑修复明确标为单例开发历史并链接原记录，不将旧数字复制到最新表。 |
| Reference，289 | 保留四类算法、SynthSeg、GEMS 的参考文献和固定原实现链接；数据版本 DOI 可补充，或通过本轮数据来源链接读取。 |

第五节的具体待更新结构：

1. **数据、输入与统计口径。** 使用 `ds000114` snapshot `1.0.2`、`ses-test` 的固定 sub-01–sub-10；公开快照已有去脸，本次未追加。全部十例与 `development_seen=True` 的 sub-01 排除后的九例并列。raw 从公开 T1 内部完成共享预处理；stage 从本例本轮新官方 norm/aseg/wmparc 开始，排除官方预处理时间。
2. **六区域原网格精度。** 两张六行表：全部十例、新九例；每行分别给 raw native 与 stage native 的均值 ± 样本标准差 [最小值, 最大值]、有效数/计划数和 NA 数。六区域为脑干、双侧丘脑细核、左/右海马、左/右杏仁核。家族在受试者内按官方细标签体素数加权，再对受试者等权汇总；标准差为受试者间差异，不能称为重复波动。高分辨率和全部 ROI 链接完整表。
3. **完整流程 benchmark。** FNIT raw 的实际 process wall、API compute/save/total；官方本例实测 full recon-all 加三个细分割完整 wall；stage 条件下分别列 FNIT stage wall 与三个官方细分割命令 wall。官方双侧海马/杏仁核只有一个命令 wall，不能左右重复相加。若显示速度，先对同一受试者算比值，再统计这些比值的均值/样本 SD/范围及配对数；不能把独立极值相除。共享 CPU/H100 PCIe 和四线程条件明确。
4. **分步骤对照。** 链接实际 recon-all FSTIME 来源、四项 recipe 总计、合成准备/拟合、工作图准备、强度准备/拟合与 solver prepare/fit/postfit 明细。官方 4/4/8 个整数阶段 timer 与 FNIT timer 范围按记录列出；没有独立 timer 或可比中间标签的步骤保持 NA/未测。子计时、阶段总计与 recipe 总计不重复相加。
5. **官方对照脑图。** 本轮图来自 raw-native 的 sub-01、总体加权 Dice 中位例和最低例，四组局部脑区；每幅六张 axial slice、官方/FNIT/差异三行共切片、共裁剪和共灰阶窗。选择角色重复时只绘一次。可嵌入热图及一组代表性图，其余链接本轮独立结果文档；本轮 plot 不生成 stage 图，旧 stage 图只能作为历史图链接。

功能页第 3 行的“完整 benchmark”改指本轮 `ten_public_t1_20261002/README.md`；单例重复性通过第六节历史记录继续进入。当前旧历史 recon-all 拼接总时间（233 行）和旧单例结果图不再置于“最新”段落。

### 2. 验证索引成为最新结果入口

`validation/subregions/README.md`：

- 3 行保留功能页链接，增加本轮完整 benchmark 与独立结果文档入口。
- 9–14 行将旧单例派生 T1 的 shape/SHA 介绍替换为本轮固定十例选择、数据版本和 [数据来源](data_selection.md) 的链接；说明 raw/native 与 fresh norm/stage 的评分网格。旧单例 SHA 留在原历史记录中。
- 16–22 行改为“最新十例完整精度与耗时”，摘要引用全部十例/新九例结果和实际 E2E，提供完整 ROI、家族、步骤与脑图链接。旧三次重复数字放到“历史版本”，保留“开发病例官方/FNIT重复性”独立链接。
- 24–26 行新增历史单例重复性入口，避免用它代表十例受试者分布。
- 28–42 行保留公开验证调用，补本轮完整队列/分析/步骤/绘图/发布协议入口；不把等待耗时混入拟合 wall。
- Reference 锚点保留。

### 3. 主页仅改一个功能行

主页 `README.md:42` 的 `segment_4_subregions` 行保持函数名、原软件名和功能介绍。第三列旧单例耗时和 Dice 改善数改为“十例全流程 benchmark（另列未参与开发的九例）”的最新链接；官方/FNIT单例重复性另作短链接，不与十例跨受试者标准差合并。主页无需铺开全部指标。

154–191 行资源准备、Mermaid 与 API 示例保留；195–197 行通用验证索引可保持原状。本轮不涉及其他功能或依赖说明。

## 待链接的正式产物

以下路径以仓库根目录为基准；从功能页使用 `../../`，从验证索引使用 `ten_public_t1_20261002/`，主页使用 `validation/subregions/` 前缀。发布前应确认文件存在并与机器可读清单的 SHA 一致。

| 内容 | 正式链接 |
|---|---|
| 本轮入口 | `validation/subregions/ten_public_t1_20261002/README.md` |
| 完整中文结果和脑图 | `validation/subregions/ten_public_t1_20261002/benchmark_results.md` |
| 每例全部110 ROI、两输入和两网格，保留失败，4,400行 | `validation/subregions/ten_public_t1_20261002/analysis/cohort_roi.tsv` |
| 六区域逐例/整例统计 | `analysis/cohort_family.tsv`、`analysis/cohort_case.tsv` |
| 跨病例ROI与家族统计 | `analysis/cohort_summary.json` |
| 建议新增的880行跨病例ROI表 | `analysis/cohort_roi_summary.tsv`，见下一节 |
| 实际分步骤及汇总 | `analysis/steps/cohort_steps.tsv`、`analysis/steps/cohort_steps_summary.tsv` |
| recon-all显式elapsed与实际日志行 | `analysis/steps/cohort_reconall_fstime.tsv` |
| 同受试者配对耗时 | `paired_runtime.tsv` |
| 本轮真实脑图 | `brain_figures/plot_manifest.json`，其实际PNG及 `cohort_raw_native_family_dice.png` |
| 发布来源、源码、图像和文本SHA | `benchmark_evidence.json` |
| 官方新完整流程与准备阶段记录 | `official_protocol.md`、`benchmark_protocol.md` |
| 历史单例重复性 | `validation/subregions/reproducibility_20261002/README.md` |

表中缩写路径均相对于本轮 `ten_public_t1_20261002/`。脑区 PNG 文件名中的病例由最终已定义排名决定，不预先填入某个“中位/最低”编号。

## 全110 ROI跨病例统计已存在，但缺专用TSV

`analyze_cohort.py` 的 `cohort_summary()` 已对每个 `(space, label)` 汇总：

- `cohort_summary.json["cohort_all"]["roi"]`：全部十例，110 × 四种评价空间，共440项。
- `cohort_summary.json["cohort_new_subjects"]["roi"]`：排除开发病例的九例，同样440项。
- 每项 `measurements.dice` 已有 `mean`、`std_sample`、`min`、`max`、`median`、`defined_subjects`、`missing_or_na_subjects`、`planned_subjects`、`std_ddof=1`。
- `hard_status_counts` 记录双方均空、单方缺失、正常评价和未评价的类别。单方缺失的0 Dice参与统计，双方均空和失败为NA，仍占计划分母。

F8会复制这个 JSON，当前输出的 `benchmark_evidence.statistics` 仅另列 native 六区域、耗时和步骤；`cohort_roi.tsv` 是逐例4,400行，不是跨病例聚合表。

建议最终发布后用**独立CPU后处理**将经SHA核验的既有 `cohort_summary.json` 平铺为 `analysis/cohort_roi_summary.tsv`，无需重新计算指标、修改任何冻结统计脚本或拟合：

| 推荐列 | 来源 |
|---|---|
| `cohort`、`space`、`mode`、`resolution` | 两组名称及现有space；raw/stage × native/hr |
| `label`、`name`、`family` | 既有ROI元数据 |
| `planned_n`、`defined_n`、`na_n` | 现有planned/defined/missing_or_na_subjects |
| `dice_mean`、`dice_sample_sd`、`dice_min`、`dice_max` | 现有measurements.dice |
| `dice_median`、`std_ddof` | 可附既有字段 |
| `hard_status_counts` | 原类别计数，可序列化JSON或平铺明确列 |

产物固定为两组 × 四space × 110 ROI，共 **880 行**，检验唯一键 `(cohort, space, label)`、分母10/9、`defined_n + na_n = planned_n` 和样本SD口径。未定义数值用空值/NA，不生成零值替代。

新增 `roi_summary_artifacts.json` 记录后处理脚本、原 `cohort_summary.json`、F8 `benchmark_evidence.json` 与新TSV的大小/SHA。保留F8原生成文件及其hash；新增独立衍生清单即可。功能文档和索引同时链接逐例4,400行及汇总880行，便于查询每个分区的Dice。

## Publisher F8接口核查与发布顺序

当前只读源码接口未发现与分析/步骤/绘图schema的直接冲突；尚无完整最终实测分析可执行终态发布验证。需要按以下顺序完成：

1. 三个队列均终态，分析 `final_outcome_ready=true`。失败病例仍保留十例/九例分母及4,400行，不能只发布已成功的四例。
2. 从同一次分析生成步骤和脑图；两者引用的分析、manifest和状态SHA与发布器一致。分析重跑后状态SHA改变，旧plot不能直接复用。
3. F8可从服务器目录或外部完整**文本**镜像读取；必须带完整分析、输入manifest、五个分析产物、步骤、绘图清单、实际PNG及协议。纯curated目录只有摘要/TSV不能作为完整 `--root`。
4. `--output` 指向正式验证目录；`--copy-evidence`只复制allowlist摘要、TSV、步骤、图和协议，不复制fullanalysis、API/观察器、MRI/atlas/weights/license或source snapshot。
5. 已有同名curated文件若SHA不同，F8拒绝覆盖；准备文本镜像时需与正式协议/硬件记录对齐，不能用静默覆盖解决。四个生成文件已存在也会拒绝，应使用新输出目录保存下一次发布。
6. 当前图都是raw-native。公开文档必须这样标注；stage精度和计时来自本轮fresh norm分析，不能借旧stage图充作本轮图。
7. F8的“准备阶段记录”从实际 `run_preparation_summary.case_outcomes` 计数，独立保留启动失败；当前stage依赖阻塞另列，不累加为算法失败。最终计数、脑图和速度都从本轮正式证据取值。
8. 对完整结果运行F8后，生成独立ROI聚合TSV与清单，再更新三个公开入口及历史链接，最后核验Markdown锚点、文件存在/尺寸/SHA和六层图的实际PNG。

发布器不自动修改主功能文档或主页。本次没有修改F8或统计脚本；待最终结果完成再由主任务授权文档整合。
