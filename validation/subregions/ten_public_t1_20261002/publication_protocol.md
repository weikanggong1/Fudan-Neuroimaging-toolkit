# 本轮结果发布脚本

`publish_cohort_summary.py` 读取本次十例最终文本分析、实际分步骤计时和衍生脑图，生成 `README.md`、`benchmark_results.md`、`paired_runtime.tsv` 和 `benchmark_evidence.json`。不读取 MRI，不拟合，不调用官方软件，不导入 FNIT 或 GPU 库；Python 标准库即可运行。

## 必须先完成的真实证据

- `analysis/analysis_status.json` 和 `cohort_analysis.json` 均须 `final_outcome_ready=true`，各运行队列必须终态，无未完成尝试。部分快照和运行中状态直接拒绝，在创建输出目录前退出。
- 分析必须保留预先固定的 sub-01–sub-10，以及 110 分区 × 10 例 × raw/stage × native/hr 的 4,400 行。失败行仍保留，不能按精度或成功状态删除计划受试者。
- `analysis/steps/cohort_steps.json` 须来自同一分析和输入清单；没有实测的步骤保持 NA。
- `brain_figures/plot_manifest.json` 必须完成，且来自同一分析、状态和输入清单。发布前验证每张实际 PNG 的大小、SHA 和 PNG 头；正文只引用真实生成的脑图。
- JSON/TSV 的大小与 SHA、十例身份和冻结源码清单一致；发布脚本重新计算两组 native 六区域分布，并与分析摘要交叉核验。

## 汇总口径

全部十例的计划分母固定为 10；未参与开发的九例只排除预声明 `development_seen=true` 的 sub-01，分母固定为 9。每个指标分别列有效数和缺失数，不把缺测替换为零。单方细分区缺失的 Dice 为零并参与汇总；双方均空才为 NA。

区域 Dice 为每例内部按官方细分区体素数加权，再对受试者等权汇总。报告均值、样本标准差（ddof=1）和有定义受试者的最小/最大值；少于两例时标准差为 NA。这是受试者间差异，不是重复运行随机波动。

速度严格逐例配对后汇总：

- raw：本例官方实际完整 T1 流程 wall / 本例 FNIT raw 实际进程 wall。
- stage：本例三个官方细分割命令进程 wall 之和 / 本例 FNIT stage 实际进程 wall；排除 recon-all。

每一倍数使用同一受试者的分子和分母；范围来自逐例比值，不使用独立耗时极值相除。官方 CPU 和共享 H100 PCIe 的实测流程比值单列硬件与阶段条件。API、保存、进程墙钟和等待分列；多层 solver 子计时、阶段总计和 recipe 总计不重复相加。

准备阶段另列实际 `run_preparation_summary.case_outcomes` 的归档条数、失败记录、已启动的失败记录及涉及受试者；当前合并队列的失败/依赖阻塞/预检查类别另列。官方环境启动失败不与因其阻塞的 FNIT stage 再相加为算法失败，也不改变最终逐例分母。缺少准备摘要时记为未提供，不写历史失败为零。发布证据保留原准备摘要、来源 SHA 与派生计数；启动原因及环境修复过程链接本轮官方协议。

## 服务器或本地完整文本镜像调用

先运行现有分析、步骤提取和脑图脚本。十例未终态时不调用结果发布。

```bash
python publish_cohort_summary.py \
    --root /absolute/path/ten_public_t1_text_evidence \
    --output /absolute/path/curated/ten_public_t1_20261002 \
    --copy-evidence
```

`--root` 可以是服务器任务目录，也可以是仓库外的完整文本镜像。`--analysis-dir`、`--steps-dir`、`--figures-dir` 可指定已下载的文本和 PNG 目录；原文件路径可迁移，但内容大小和 SHA 必须一致。`--output` 指向发布目录。

`--copy-evidence` 只复制指定的分析摘要/TSV、步骤 JSON/TSV、绘图清单与实际 PNG，以及已有协议/硬件记录。完整 `cohort_analysis.json`、逐例 API/观察器、MRI、图谱、权重、许可证和源码快照不会复制到发布目录；完整分析的路径、大小与 SHA 保留在 `benchmark_evidence.json`。不指定该选项时，输出目录中的相对链接目标必须已经存在并通过相同 SHA 核验。

发布目录已经有四个生成文件时拒绝覆盖；使用新的输出目录保存下一次结果。脚本不修改 `docs/subregions/README.md`。父任务在真实结果核验后更新功能文档。

## 验证记录

`test_publication_contract.py` 只验证发布数学和拒绝条件：样本标准差、零 Dice/缺测分母、同一受试者配对比值、迁移文本 SHA、重复观测、非终态以及非法数值。测试数值是独立契约样例，不是分割 benchmark。实际运行中的 `analysis_status.json` 拒绝发布记录见 `publication_validation.json`。
