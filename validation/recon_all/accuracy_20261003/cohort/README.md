# 10例公开原始T1w及统一评估

## 功能与流程

本工具冻结10个不同被试的原始3D T1w，只下载所选解剖影像，保留公开数据的许可、快照、来源、大小和哈希，为协调者的基线、官方和最终候选整例比较提供同一输入。冻结名单后不按重建结果替换失败被试。

```mermaid
flowchart LR
 A[固定公开快照与CC0元数据] --> B[预先冻结两个数据集各5人]
 B --> C[匿名S3下载并匹配快照MD5和大小]
 C --> D[冻结SHA256和原始NIfTI几何]
 D --> E[本轮只读复验10例]
 E --> F[协调者冻结程序资源版本]
 F --> G[分别从原始T1和空目录运行三版本]
 G --> H[138项诊断、逐标签和表面及脑区指标]
```

ds000030固定1.0.0 / `4070b6eea231517bfeff42527a46d9d166ac4e13`，选择sub-10159、10171、10189、10193、10206；ds000114固定1.0.2 / `6299834614e9ae7df1e2fc5922545331b5c7f022`，选择sub-04至08的ses-test。选择规则及预先排除旧示例sub-01/02/03和重复retest的理由见`cohort_frozen.json`。均按固定快照被试ID排序选择，无影像质量或重建结果筛选。

## Python调用、输入与输出

`download_public_t1w.acquire(case, destination, timeout, attempts)`输入一条冻结case字典和下载根目录，返回此例校验后的字典或保留失败原因的字典。`verify_cohort.py`再读`cohort_verified.json`校验实际文件，独立输出`read_only_verification.json`。`prepare_evaluation.prepare`仅写配置。

```python
from pathlib import Path
from prepare_evaluation import prepare

# 运行前由协调者填入实际提交、资源路径和程序SHA，不使用占位值启动整例。
case_configs = prepare(
    manifest_path=Path("cohort_verified.json"),  # 含10例固定输入SHA的公开manifest
    bindings_path=Path("runtime_bindings.private.json"),  # 私有运行环境，不提交许可证配置
    output_dir=Path("configs"),  # 必须为新目录，避免覆盖旧报告
    server_root=Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003"),
)
```

输入影像为未裁切、未重新采样的公开原始`.nii.gz`，shape为3D，存储dtype均float32；ds000030为176×256×256，ds000114为256×156×256。所有scanner affine、qform/sform代码与矩阵、体素尺寸和单位逐例保留，未假定两数据集具有同一网格。路径和SHA见`cohort.csv`。

输出包括：`cohort_frozen.json`（选择与许可）；`cohort_verified.json`（下载来源/HTTP对象版本、固定快照匹配、SHA及几何）；`download_status.json`（逐例完成/失败）；`read_only_verification.json`（本轮实际文件复验）；`evaluation_schema.json`（评估项和失败保留规则）；`task_status.json`（阶段完成及未运行事项）。原始影像仅在服务器私有目录，不能提交Git。图像receipt及HTTP头保留在原始影像旁。

参数如下：

| 工具 | 参数 | 含义、默认与异常 |
|---|---|---|
| 下载 | `--manifest` | 必填；冻结或已校验10例JSON，恰好10个不同被试 |
| 下载 | `--destination` | 必填；服务器私有影像根目录 |
| 下载 | `--state-dir` | 必填；逐例状态、校验manifest输出目录 |
| 下载 | `--timeout` | 单次curl超时秒，默认300，必须正数 |
| 下载 | `--attempts` | 每例尝试次数，默认3，必须正数；失败保留原例并非零退出 |
| 复验 | `--manifest` | 必填；含实际SHA256和大小的已校验manifest |
| 复验 | `--output` | 必填；写本轮只读校验JSON；任何大小/哈希/3D/affine/有限值失败均保留并非零退出 |
| 配置 | `--manifest` | 必填；所有10例须已下载校验 |
| 配置 | `--bindings` | 必填；协调者实际固定的baseline/candidate/runtime/official/comparison绑定；不完整则报错 |
| 配置 | `--output` | 必填；本地或服务器新目录，不覆盖 |
| 配置 | `--server-root` | 必填；三个版本隔离输出根目录及cohort/configs部署根目录 |

bindings中的baseline/candidate字段均沿用`execute_whole_case.py`：`python`为主页Conda解释器；`code_root/code_commit/source_archive_sha256`固定实际源码；`weights/assets/native_bin_dir`为校验资源目录；`fs_license`仅私有环境填写；`invocation`为`initialized_cuda_api`或`cli`；`pipeline_kwargs`保留成熟后端设置并固定`hemisphere_workers=2`。线程固定总4，GPU固定本轮GPU0 UUID，device为cuda:0。official字段需固定8.2.0 d932c45及程序/资产清单SHA，`recon_all`为独立benchmark官方入口。comparison字段为解释器、源码根、已有比较脚本目录及LUT路径；需要原工具支持的额外字段可以保留。

每例生成baseline/candidate执行JSON和单例comparison JSON，另生成全10例比较配置及`evaluation_plan.json`。生产配置仅输入原始T1、声明资源和自产目录，官方输出只在比较配置中出现。官方命令属于独立benchmark计划，须由root确认固定程序和私有许可证、线程及资源后运行。

## 命令行复现

以下命令在服务器cohort目录执行，解释器示例需由协调者确认实际Conda环境。

```bash
FNIT_COHORT_ROOT=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003
FNIT_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
# 再次校验已存在10例，不发起下载。
"$FNIT_PYTHON" verify_cohort.py --manifest cohort_verified.json --output read_only_verification_new.json
# 如需独立复现下载，使用已冻结SHA的manifest；匹配的已有文件直接复用。
"$FNIT_PYTHON" download_public_t1w.py --manifest cohort_verified.json --destination "$FNIT_COHORT_ROOT/public_t1w" --state-dir "$FNIT_COHORT_ROOT/cohort/download_reproduction" --timeout 300 --attempts 3
# 按实际版本填写template中的null并仅保存在私有目录，再生成配置。
"$FNIT_PYTHON" prepare_evaluation.py --manifest cohort_verified.json --bindings runtime_bindings.private.json --output configs --server-root "$FNIT_COHORT_ROOT"
```

根协调者复用`optimizations/20261002_parallel/run_whole_queue.py`、`execute_whole_case.py`和`run_monitored.py`，锁统一`/tmp/fnit-shared-benchmark.lock`。顺序队列全部10例失败仍保留，不由任务1启动。比较复用`compare_whole_cases.py`与`compare_subject.py`，每例分别启动进程：`--config configs/comparison_CASE.json --output comparison/CASE --lock /tmp/fnit-shared-benchmark.lock`。正式比较前必须校验对应官方完成记录、输入SHA及程序/资源SHA，不能因为目录存在而视为官方执行完成。

现有compare_whole_cases有两项协调者需处理的接入事项：多例在同进程重复设置Torch interop线程，建议单例进程；已有官方provenance描述硬编码旧sub02跨节点及归档计时，不能用于本轮新10例计时事实。已向root说明，不修改共享比较工具。

## 原软件调用

下载和manifest工具没有对应独立FreeSurfer命令。官方完整重建只在隔离benchmark目录执行，示例：

```bash
# 输入变量必须来自cohort_verified.json；官方入口已固定为8.2.0 d932c45。
"$FNIT_OFFICIAL_RECON_ALL" -all -i "$FNIT_INPUT_T1W" -s "$FNIT_CASE_ID" -sd "$FNIT_OFFICIAL_SUBJECTS" -openmp 4
```

不用`-parallel`增加额外线程。保存运行log、done、实际程序和资源SHA以及线程、硬件和输入哈希。

## 当前验证与历史

2026-10-03旧任务冻结并下载10例完成，下载时间00:59:24至01:00:59 UTC，影像合计107,511,490字节。本轮medium接续保留所有影像和未提交文件，以新`read_only_verification.json`实际复核10/10大小、SHA256、快照MD5及nibabel几何，失败0，未重新下载。固定来源annex MD5/大小先校验，再冻结SHA；两者匹配才能绑定公开快照。

本阶段没有运行整例、没有新精度或运行时间结论、没有候选脑图。旧两例仅提供诊断线索，不能计入新10例。正式结果需包含端到端和分步骤时间、逐标签Dice、双向点到三角面距离、逐脑区厚度/面积/体积偏差、max/P99和局部异常脑图；138项严格诊断保留。执行完成、输出完整、网格质量、严格复现、相对baseline退化分别记录。整体等效保持`not_assessed`，不自行添加容差。

## 许可、文献与源码

本轮再次浏览固定提交的[ds000030元数据](https://raw.githubusercontent.com/OpenNeuroDatasets/ds000030/4070b6eea231517bfeff42527a46d9d166ac4e13/dataset_description.json)及[ds000114元数据](https://raw.githubusercontent.com/OpenNeuroDatasets/ds000114/6299834614e9ae7df1e2fc5922545331b5c7f022/dataset_description.json)，两者License均CC0；没有独立LICENSE文件，明确许可依据为快照dataset_description.json。元数据原文及SHA保存在source_metadata和manifest。固定版本官方页面：[ds000030 1.0.0](https://openneuro.org/datasets/ds000030/versions/1.0.0)、[ds000114 1.0.2](https://openneuro.org/datasets/ds000114/versions/1.0.2)。

文献：[UCLA数据说明](https://www.nature.com/articles/sdata2016110)；[test-retest数据说明](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3641991/)。[CC0许可](https://creativecommons.org/publicdomain/zero/1.0/)。原软件源码：[FreeSurfer](https://github.com/freesurfer/freesurfer)，实际benchmark版本必须验证8.2.0 d932c45；[recon-all使用说明](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)。

本工具仅使用Python标准库、现有nibabel/numpy及curl，不新增项目依赖。
