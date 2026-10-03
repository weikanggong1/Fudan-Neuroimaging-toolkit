# 十人配对比较与汇总工具

`compare_public10.py` 读取 FNIT 和独立原软件流程已经完成的结果。它不运行重建，不重采样影像，也不修改结果。真实数据来源、固定被试选择和采集条件见 [DATASET.md](DATASET.md)。本工具的单元测试只检查统计计算和报告规则，不能代替真实数据 benchmark。

本次主比较固定使用bf339a0的20个新FNIT完整结果，见[修复源码绑定](source_binding_memoryfix.public.json)：433文件清单SHA为`f13a40989b96d9e3608a427a1fe10d1960b20f146c768a3dd101f84fe4deae1e`，只有AMICO solver临时内存调度相对5d84c7变化。原5d的case01两个成功与case02 MMORF失败保留为legacy，不混入新20个主候选。case01/02的checkpoint组件检查也不进入端到端时间。每人两个分支和20配对分母保持固定。

截至 2026-10-03，本轮冻结版 FNIT **20/20** 完成，选定原参考与配对比较均为 **19/20**（TBSS 10/10、MMORF 9/10）；格式检查为 270/270 与 162/180，合计 **432/450**。case05/10 TBSS 的新 raw 恢复已完成并比较；case10 MMORF 初次、R1、R2 均在原 EDDY 分配阶段失败，缺少 18 张完整官方指标图。该位置保留缺失状态，本轮停止进一步恢复，最终状态为 `incomplete`。原参考共 25 次（19 成功、6 失败），FNIT 共 21 次（20 成功、1 失败），历史成功另列；失败时钟不进入完整配对时间比，见[恢复证据](OFFICIAL_FAILURES.md)。

20 项候选报告的 433 文件清单均绑定 `bf339a0368a7711d2c6ca3477c8d7dc1fc17e75a`。本轮开发树的合并核验 `adf74371` 与 `origin/main` 的 `7af34e6d` 另见 [合并核验](merge_integration.public.json)：73 项 CPU 回归通过，没有重新运行十人影像。冻结结果不能改标为当前合并 main 的 20 例实测。

新候选根目录为用户私密 benchmark 目录下的 `memoryfix_cohort/results/`；原参考位于原 `results/`，case01 MMORF 选择完整恢复的 `official_recovered`，初次失败保留。最初三个队列共享 GPU 锁，实际顺序按 UTC 记录，不保证相邻 AB/BA。新主候选先执行 case02 MMORF，再执行固定其余 19 项；只有新 FNIT controller 设置释放锁后的 0.25 秒间隔，两个原 controller 使用冻结旧版，没有该间隔。初始队列现已结束，三个失败参考随后在新目录按同一全作业锁执行恢复；两项 TBSS 成功，case10 MMORF 的 R1/R2 失败后停止。排队、锁等待及该间隔均在作业时钟外。

## 输入与输出

每个比较对应一名匿名被试和一个分支。两个流程必须使用相同的原始 AP、PA、梯度、T1 和模板。两侧各自估计 TOPUP、SynthStrip 掩膜、EDDY、DTI、NODDI 和配准，不能共用估计结果。

| 内容 | FNIT 相对路径 | 独立原软件相对路径 |
|---|---|---|
| 九张 native 图 | `native/dti_{FA,MD,L1,L2,L3,MO}.nii.gz` 与 `native/NODDI_{ICVF,OD,ISOVF}.nii.gz` | 同名路径 |
| TBSS 标准图 | `registration/standard/{NAME}.nii.gz` | `tbss/stats/all_{NAME}.nii.gz` |
| TBSS skeleton 图 | `registration/skeleton/{NAME}.nii.gz` | `tbss/stats/all_{NAME}_skeletonised.nii.gz` |
| MMORF 标准图 | `registration/standard/{NAME}.nii.gz` | `mmorf/standard/{NAME}.nii.gz` |
| dMRI 掩膜与完整 DWI | `eddy/nodif_brain_mask.nii.gz`、`eddy/data.nii.gz` | 同名路径 |
| TOPUP 场和校正 b0 | `topup/fieldmap_fout.nii.gz`、`topup/fieldmap_iout.nii.gz` | 同名路径 |
| MMORF T1 掩膜、变换与 Jacobian | `registration/t1_brain_mask.nii.gz`、`mmorf_warp.nii.gz`、`mmorf_jacobian.nii.gz` | `mmorf/` 下同名文件 |

TBSS 比较 27 对指标图，MMORF 比较 18 对。JSON 包括影像形状、affine、qform/sform、文件大小和 SHA-256，主要 ROI 的相关、MAE、RMSE、归一化 RMSE、平均有符号差、p95、最大差，以及非零支持 Dice/Jaccard。还比较 TOPUP 输入对、acqp/index、完整 EDDY DWI 和旋转梯度；MMORF 另报告位移差、非正 Jacobian 比例、T1 掩膜 Dice 和两个 FLIRT 矩阵。缺失或不同网格的影像保留失败原因，不进行隐式重采样。

## 主要比较范围

Native 使用独立原软件 EDDY 掩膜 `>0.5`。标准空间使用调用者提供的模板或固定 mask `!=0`。TBSS skeleton 使用同一模板范围与原始 skeleton `>=2000` 的交集。两侧都包含主要 ROI 内的零值，避免遗漏支持较少的结果。共同非零支持及两个脑掩膜的交集只作补充，并与主要结果分开记录。

输入通过 nibabel 解码为 float32，验证统计按卷用 float64 累计中心矩。3D 指标图的 p95 使用所有 ROI 体素；较大的 4D 数据的 p95 使用固定种子 1729、不放回均匀抽样至多一百万个 voxel-volume 元素，抽样方法与数量写入报告。其余误差和相关使用全部选定元素。常数或空 ROI 的相关系数为 `null`，不记作 1。

形状、有限值和 affine（绝对容差 `1e-5`，不使用相对容差）是格式检查；通过这些检查不代表数值等价。本工具不预设精度验收阈值。高相关也不能代替 MAE、RMSE、范围覆盖和变换差异。

## 当前差异观察（19 个配对）

以下为 2026-10-03 本轮最终的 19 个完成配对；case10 MMORF 没有完整参考，仍保留十人分母。标准 FA 的主要 ROI 是固定 MNI152 T1 brain 1 mm 非零范围，包含两侧零值。表中是逐人指标的中位数 [Q25,Q75]，NRMSE 定义为 `RMSE / 原参考 RMS`；没有应用数值等价阈值。

| 分支；有效人数 | Pearson r | MAE | NRMSE |
|---|---|---|---|
| TBSS；10/10 | 0.995808245 [0.991750767,0.996729561] | 0.010437746 [0.008492381,0.015245841] | 0.060759006 [0.052699201,0.085105294] |
| MMORF；9/10 | 0.929446724 [0.919110521,0.960836104] | 0.044948335 [0.030315001,0.050054118] | 0.217009355 [0.160364724,0.241314446] |

`case08` 两分支的以下配准前统计相同。主要 ROI 使用该例独立原软件 EDDY 脑 mask 的 168,935 个体素；相关和误差累计全部选定元素，没有抽样。

| 结果 | 元素数 | Pearson r | NRMSE |
|---|---:|---:|---:|
| TOPUP 校正 AP/PA 两张 b0 | 337,870 | 0.957562846 | 0.137970132 |
| EDDY 完整 117 帧 DWI | 19,765,395 | 0.982441559 | 0.136383849 |
| native FA | 168,935 | 0.856050004 | 0.285848576 |

因此差异在配准前已经存在，不能把标准图差异全部归因于 FNIRT/MMORF；这也尚未隔离 TOPUP、mask、EDDY 或梯度中的具体原因。第一人的较好脑图是固定展示例，不能代表十人均有同样精度。各图仍应结合 MAE、NRMSE、分位误差、支持范围及上游检查解读，格式通过或较高相关不能单独形成数值等价结论。

[保存输入追踪](b0_input_trace_20261003.public.json)通过全部解码体素确认，case02/08 的两个分支均为冻结 FNIT AP0/PA0、原参考 AP21/PA0，候选为 AP `[0,21,52]`。原参考记录 EDDY `reference_scan_no=21`；FNIT 报告未直接持久化此参数，按冻结 pipeline 的选择传递代码推断为 0，不能把推断写成直接记录。独立[十人接受点评分修复验收](b0_accepted_point_20261003.public.json)没有改变十人选择，因而未消除这两例的选择方法差异。该组件与[同 pair 场求解诊断](PRECISION.md)不进入整链时钟，也没有更新 bf339a0 的 20 个输出。

## 命令行调用

```bash
# 两侧是各自从相同原始数据运行后得到的新目录。
benchmark_root_dir="$PWD/public10_work"
candidate_output_dir="$benchmark_root_dir/memoryfix_cohort/results/case01/tbss/fnit/output"
original_output_dir="$benchmark_root_dir/results/case01/tbss/official/output"
standard_roi_file=/resources/MNI152_T1_1mm_brain.nii.gz
fa_skeleton_file=/resources/FMRIB58_FA-skeleton_1mm.nii.gz
raw_ap_bvals_file="$benchmark_root_dir/inputs/case01/raw/AP.bval"
comparison_report_file="$benchmark_root_dir/comparisons/case01_tbss.json"

python validation/dmri_pipeline/public10_20261002/compare_public10.py compare \
  --case-id case01 --backend tbss \
  --candidate-dir "$candidate_output_dir" --reference-dir "$original_output_dir" \
  --reference-roi "$standard_roi_file" --fa-skeleton "$fa_skeleton_file" \
  --bvals "$raw_ap_bvals_file" --report "$comparison_report_file"
```

本次十人 benchmark 的两个分支统一使用 MNI152 T1 brain 1 mm 非零 ROI；TBSS skeleton 再与原 skeleton `>=2000` 相交。MMORF 将 `--backend` 改为 `mmorf`，保留相同 `--reference-roi`，省略 `--fa-skeleton`。其他独立实验可事前固定不同模板或 mask，但应明确记录，不能混入本次汇总。

| 参数 | 含义 |
|---|---|
| `--case-id` | 匿名 `caseNN` 标识 |
| `--candidate-dir` | FNIT 完整结果目录 |
| `--reference-dir` | 独立原软件完整结果目录 |
| `--backend` | `tbss` 或 `mmorf` |
| `--reference-roi` | 标准网格的模板或固定 mask；非零处进入主要 ROI |
| `--fa-skeleton` | TBSS 原始 skeleton；MMORF 不需要 |
| `--skeleton-threshold` | 默认 2000，按原始 skeleton 数值比较 |
| `--bvals` | 原始完整 AP bval，用于选择 `b>100` 的旋转梯度 |
| `--report` | 新 JSON 路径；存在时拒绝覆盖 |

报告的 `complete` 表示所需 18/27 对指标图可比较且格式检查通过，不能解读为逐点一致。上游与可选变换检查的结果各自记录。格式检查失败时程序仍写完整报告，并返回退出码 2。

## Python 调用

```python
import importlib.util
from pathlib import Path
from types import SimpleNamespace

comparison_module_file = Path("validation/dmri_pipeline/public10_20261002/compare_public10.py")
module_spec = importlib.util.spec_from_file_location("public10_comparison", comparison_module_file)
comparison_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(comparison_module)

comparison_arguments = SimpleNamespace(
    case_id="case01", backend="mmorf",
    candidate_dir=Path("public10_work/memoryfix_cohort/results/case01/mmorf/fnit/output").resolve(),
    reference_dir=Path("public10_work/results/case01/mmorf/official_recovered/output").resolve(),
    reference_roi=Path("/resources/MNI152_T1_1mm_brain.nii.gz"),
    fa_skeleton=None, skeleton_threshold=2000,
    bvals=Path("public10_work/inputs/case01/raw/AP.bval").resolve(),
)
comparison_report = comparison_module.compare_case(comparison_arguments)
# 返回匿名字典；保存时使用 json.dump(..., allow_nan=False)。
```

## 完整计划汇总

汇总清单必须在观察结果前固定。每行指定 `case_id`、`backend` 和三个报告路径，路径可相对于清单文件所在目录。完整计划有十人 × 两个分支共 20 行。`candidate_process_metrics` 与 `reference_process_metrics` 可指向 GNU `/usr/bin/time -v` 文本，或包含 `wall_seconds`、`max_rss_kib`、`exit_status`（也支持 `exit_code`）的 JSON。队列的 `process_metrics.json` 还可提供本进程及其子进程、以及其他进程的采样显存峰值；若同目录有 `time.txt`，使用 GNU time 的完整进程时钟、RSS 和退出码，另保留 observer 时钟，避免把监测线程退出时间算入运行时长。

```json
{
  "planned_cases": [
    {
      "case_id": "case01",
      "backend": "tbss",
      "candidate_report": "memoryfix_cohort/results/case01/tbss/fnit/report.json",
      "reference_report": "results/case01/tbss/official/report.json",
      "comparison_report": "comparison.case01.tbss.json",
      "candidate_process_metrics": "memoryfix_cohort/results/case01/tbss/fnit/process_metrics.json",
      "reference_process_metrics": "results/case01/tbss/official/process_metrics.json"
    }
  ]
}
```

上例只演示一行结构；实际清单写入全部 20 行。即使清单缺行，汇总也保留预期的 20 个位置，并将缺行记为未完成。

有明确初次失败/恢复时，可在相同行追加`initial_failed_reference_report`、`initial_failed_reference_process_metrics`、`reference_rerun_reason`，或FNIT侧的`initial_failed_candidate_report`、`initial_failed_candidate_process_metrics`、`candidate_rerun_reason`。case01 MMORF参考行指向`official_recovered`，并保留初次original报告；case02 MMORF候选行指向新的memoryfix结果，另保留旧20 GB失败。保留的初次尝试不参与选定主运行的配对速度比。legacy case01成功进度表另列于[协议](PROTOCOL.md#原5d84c7版本进度legacy回归记录)，不作为新的主候选成功计数。

若同一参考需要多次从 raw 完整恢复，在该行追加 `intermediate_reference_attempts`，按实际发生顺序列出最终选定参考之前的中间失败。每项的 `report` 必填，`process_metrics` 和 `reason` 可选；路径仍相对于清单目录，时钟文件必须与该次报告在同一目录。

```json
{
  "intermediate_reference_attempts": [
    {
      "report": "recovery_attempt1/case10/mmorf/official/report.json",
      "process_metrics": "recovery_attempt1/case10/mmorf/official/process_metrics.json",
      "reason": "原 EDDY 在 GPU 启动分配阶段失败；同输入完整恢复"
    }
  ]
}
```

中间报告须为失败终止，且匿名病例、分支、原始输入和资源 SHA-256、已记录的 EDDY seed 与选定病例绑定。重复报告路径或 SHA-256，包括与初次或选定报告重复，均产生 `binding_errors`，不增加尝试次数，该行也不进入完整配对耗时汇总。中间记录公开为 `retained_intermediate_reference_attempts`，只保留状态、错误类型、报告 SHA、处理/GNU/observer 时钟、RSS、采样显存和公开原因；私密路径、原始异常消息和命令不发布。

`reference_attempts.initial` 统计每个病例的最初运行，`intermediate` 统计保留的中间失败，`rerun` 统计最终选定的替代运行，`selected_primary` 统计当前主参考，`all` 为 initial + intermediate + rerun。中间组无记录时省略，不把缺失当成零时钟。失败时钟独立列出，永不加入选定成功参考的时钟或速度比。`cases.csv` 的 `retained_intermediate_reference_attempts_json` 保存逐次时钟和完整 SHA，Markdown 也列出逐次失败。改变恢复历史后创建新的清单和 summary 目录，保留之前的进度报告；已完成比较只有在候选、参考和比较报告 SHA 均未改变时才可复用。

多次恢复记录新增 12 项计数、去重、输入绑定、隐私与失败时钟排除的 CPU 契约检查，与原比较工具的 18 项共 30 项通过。它们不重新计算旧真实比较，也不改变图像统计或 ROI 算法；小型报告 fixture 不进入性能表。

```bash
benchmark_plan_file="$benchmark_root_dir/comparisons.private.json"
aggregate_report_file="$benchmark_root_dir/comparisons/aggregate.manual.json"
python validation/dmri_pipeline/public10_20261002/compare_public10.py aggregate \
  --manifest "$benchmark_plan_file" --report "$aggregate_report_file"
```

`--expected-case-count` 默认 10，仅用于工具开发或明确另定的队列；本次十人 benchmark 使用 10。重复标识、未知分支或超出计划的病例拒绝汇总。缺失报告、错误绑定、失败运行及外部非零退出码均保留，不能得到完成状态。完成运行还须匹配 AP 影像/bval/bvec/JSON、PA 影像/bval/JSON、各分支实际使用的模板、SynthStrip 权重和 MMORF 的 T1 输入 SHA-256；两侧都不存在的可选 `PA.bvec` 不计缺失，任一侧存在则纳入检查。输入或资源不同、缺少必要哈希或 EDDY seed 不同的运行不进入配对耗时汇总。

已完成原参考 case01 TBSS 的 GNU time 为 **2480.47 秒**；MMORF 恢复已完成 18 图，GNU time 为 **2113.98 秒**、API 为 **2097.248599635903 秒**、observer 为 **2113.9841028 秒**，GNU 最大 RSS 为 **9,507,700 KiB**。三个时钟分列，GNU 比与 API 比分别计算。来源或必要时钟不完整时保留缺失状态，不能借用另一种时钟拼出本次主要时间比。本轮最终完成 19 配对；20 个病例行和 450 个指标图行仍是完整计划分母，不能把所有行都算为完成。432 张已比较图通过的是格式检查，报告未应用数值等价阈值。

处理时长和完整进程时长分别汇总；加速比先在同一被试内计算 `原软件秒数/FNIT秒数`，再给出十人的 median、q25、q75、IQR、范围和均值，不用两个群体 median 的比值。耗时汇总保留每个完成的配对运行，即使它的精度检查未通过；因此不能把这一比值直接称为等价实现的加速。每张图报告实际 `n` 和格式检查通过数，不把失败病例静默删除。

FNIT 的嵌套 instrumentation 记录不能与五个完整 pipeline 阶段相加。内存区分 FNIT 全流程 allocator peak、队列每 5 秒采样的两侧完整进程树显存、原 MMORF 独立 PID 显存，以及 GNU time 的 RSS。另报告所选 GPU 上其他进程的采样显存，识别共同负载。缺失显存观测写 `null`，不写 0；没有采样到短时峰值不能解读为峰值不存在。

## 原实现与记录

2026-10-02 新增双分支比较及完整计划汇总；18 项统计契约测试通过，记录在 `tests/dmri_pipeline/test_public10_comparison.py`。41 项 AMICO 测试与两例真实 NODDI 组件回归分别记录，不作为十人整链完成证据。2026-10-03 最终核对冻结 bf339a0 的 20 个 FNIT 完整流程、19 个完整选定原参考和 19 个配对、432/450 张已比较图。case05/10 TBSS 恢复成功；case10 MMORF 初次及 R1/R2 EDDY 失败保留并停止继续恢复。25 次原参考的 19 成功/6 失败和 21 次 FNIT 的 20 成功/1 失败分列，不把失败时间加入配对比。合并后的 73 项回归是代码/工具检查，不改标实际整链源码。新 b0 接受点评分修复单独完成十人选择器验证；没有据此重跑或替换整链。结果不声明数值等价。原软件运行和参数见 [benchmark_official.py](benchmark_official.py) 与 [run_official_mmorf.py](run_official_mmorf.py)，代码与参考文献见 [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/)。
