# FNIT 与官方的单对真实诊断

`evaluate_pair.py`比较同一公开原始T1w产生的baseline与official整例。参考身份为官方FS8.2.0 d932c45，被评估身份为冻结main `816e5610417a4c587caf321049438a9554139016`。默认角色保持baseline；可选`startup_only_candidate`只评估启动处理候选，完整精度候选仍单独保留、未评估。底层工具沿用`compare_whole_cases.py`、`compare_subject.py`及`python_gpu_port`既有比较/质量/绘图函数，没有修改生产算法或诊断门槛。

流程：绑定输入、源码与资源哈希及两套完整回执 → 138项严格诊断与几何对应 → 原统计与全部分区、逐标签Dice、局部误差/no-th3 → 网格质量与脑图 → 分阶段双向三角面距离。每段分别取得并释放`/tmp/fnit-shared-benchmark.lock`。前面的短报告在长距离计算前写盘，方便协调者使用。比较时间和锁等待时间分列，均不并入此前整例时间。

## 输入与输出

CLI参数`--config`必填，为该例JSON配置路径。`case`是冻结被试ID；`baseline_config/official_config`指向已完成整例的实际配置；`baseline_commit`为完整40位提交；`code_root`为固定基线源码；`source_archive`为实际源码包；`baseline_resources`为固定资源SHA报告；`cohort_manifest`为任务1真实输入清单；`whole_driver/scripts_dir`是复用比较工具入口和目录；`python`为主页Conda解释器；`label_table`为固定LUT；`gpu_uuid/lock`为共享协议；`output`为该单对诊断目录。路径均为服务器绝对路径。脚本从基线整例配置读取真实原始输入和两版自产结果，不修改影像、表面或分区。

输出`execution_binding.json`保留两套整例回执、源码包和资源报告哈希、程序/资产校验、比较器哈希及同机/线程配置。`checkpoint.json`明确reference_role=official、evaluated_role=baseline和candidate未冻结状态，逐段记录等待、运行、完成或错误、独占计算时间；相同脚本/config可断点续跑，任一绑定发生改变则拒绝续跑。任何一段错误写入检查点后非零退出，保留此前已完成文件。

机器结果包括138项严格诊断、几何网格对应门控、`region_baseline_vs_official.json`（aparc/aseg/wmparc/全局）、`all_regions_baseline_vs_official.json`（aparc/DKT/a2009s/pial全部同名分区的厚度mm、面积mm²、体积mm³和曲率）、分割逐标签Dice、no-th3同算法比较、局部max/P99/差异位置、两版质量报告、逐阶段表面距离及公开新T1脑图。no-th3使用既有CPU诊断工具计算每版自产white/pial/厚度/注释，原stats的GrayVol另行保留。相同索引比较先检查完整网格对应；不对应时不报告同索引距离，计算双向顶点到完整三角面距离，单位surface RAS mm，不把它称为连续Hausdorff。

质量工具本身保留穿越检测180秒限制和20,000,000候选包围盒上限；达到限制时报告覆盖不足，不能称完整质量通过。整体等效仍为`not_assessed`。等待中的阶段不能计为已校验或已完成。

## Python与CLI

```python
# main从显式配置读取实际文件；设置命令行参数后调用。
import sys
from evaluate_pair import main
sys.argv = ["evaluate_pair.py", "--config", "ds000030_sub-10159.json"]
main()
```

```bash
# 在服务器task_01/pair_tools目录；只读取已完成整例，不重新启动重建。
FNIT_PAIR_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
"$FNIT_PAIR_PYTHON" evaluate_pair.py --config ds000030_sub-10159.json
```

服务器gpucw1没有tmux，首次启动用nohup持续执行，PID存于`pair.pid`，stdout/stderr在`pair-launch.log`；不要重复启动已运行PID。检查点路径为`accuracy_20261003/task_01/pair_baseline_official_v1/ds000030_sub-10159/checkpoint.json`。

## 当前真实回执及更新记录

2026-10-03首例两套完整回执核对：baseline输出138/138完整，入口2565.5213555628434秒，实际源码包SHA `0107e059153ff46ace5ab477539c232810f3309db0d569c861a02b6c69b460e7`；official完成，入口5811.068942066282秒，程序清单SHA `bb5725f4ce71fd4edf06f0890a42e0b6a5924a520e7bba6f53dd86c7e382e0a2`，done/log哈希保留在官方回执。两配置同输入SHA `0cb8a28928917b452b516e4e4694b3b79eb96b4c558f47ac63603035d0072e8b`、launch host=gpucw1、总线程4及GPU0 UUID一致。后台入口PID32848已运行，首次检查等待共享锁，数值结果尚未产出；运行后verify_binding阶段进一步重算影像、源码包、官方程序/资产及基线资源哈希。

这是第一版单对入口，继承完整三方工具的算子定义与138诊断，后续新病例可使用同schema分别比较baseline与official、候选与official及候选与baseline；必须真实冻结候选版本后再生成候选评估。当前没有新精度结论或脑图结果。

## 原软件与文献

此单对评估是FNIT诊断工具，没有对应独立FreeSurfer命令。此前官方整例命令、数据来源/许可、原始T1输入结构和引用见上层cohort/README.md。原软件：[FreeSurfer源码](https://github.com/freesurfer/freesurfer)、[recon-all说明](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)。原始公开输入：[ds000030 1.0.0](https://openneuro.org/datasets/ds000030/versions/1.0.0)，[数据论文](https://www.nature.com/articles/sdata2016110)。本轮没有新增依赖。


## 2026-10-04：独立启动处理候选角色

新增可选配置 `evaluated_role`，默认 `baseline`，原 `baseline_config/baseline_commit/baseline_resources`、原文件名及数值算子不变。可选值仅为 `startup_only_candidate`；不接受笼统 `candidate`，避免误把启动补丁认作合并精度候选。本轮合并精度候选仍是 0/10；本工具不会更新该计数。

启动角色新增必填字段：

- `evaluated_config`：已真正完成的 retry `retry_config.json`；不得使用 nominal准备配置。
- `evaluated_commit`：实际整例配置的完整提交，不能填历史816提交。
- `evaluated_resources`：这次实际运行的 `admission.json`，必须完整成功、绑定实际retry配置且包含资源inventory。
- `evaluated_resources_kind`：固定 `admission_inventory`。
- `code_root/source_archive`：实际执行的8f启动处理候选源码及归档，不是合并精度候选。

其余 `official_config/cohort_manifest/whole_driver/scripts_dir/python/label_table/gpu_uuid/lock/output` 继续使用同一已核对协议。`output` 应为独立新评估目录，不能写入旧baseline评估目录。

例如，用既有评估配置仅作比较器/官方入口模板，从实际完成回执填入新角色：

```python
import json
from pathlib import Path

# 这三个路径由服务器实际完成回执确认；不使用nominal准备配置代替retry。
existing_pair_configuration_path = Path('/absolute/path/old_pair_config.json')
completed_retry_configuration_path = Path('/absolute/path/attempt_01/retry_config.json')
startup_guard_configuration_path = Path('/absolute/path/startup_guard_v2.json')
configuration = json.loads(existing_pair_configuration_path.read_text())
actual_run = json.loads(completed_retry_configuration_path.read_text())
startup_guard = json.loads(startup_guard_configuration_path.read_text())
configuration.update(
    evaluated_role='startup_only_candidate',
    evaluated_config=str(completed_retry_configuration_path),
    evaluated_commit=actual_run['code_commit'],
    evaluated_resources=startup_guard['admission_report'],
    evaluated_resources_kind='admission_inventory',
    code_root=actual_run['code_root'],
    source_archive=startup_guard.get('source_archive', actual_run.get('source_archive')),
    output='/absolute/path/new_startup_only_sub06_vs_official',
)
# baseline资源不是本次candidate资源；不改标签或复制历史manifest。
for old_field in ('baseline_config', 'baseline_commit', 'baseline_resources'):
    configuration.pop(old_field, None)
Path('/absolute/path/new_startup_pair_config.json').write_text(
    json.dumps(configuration, indent=2) + '\n')
```

新输出前缀是 `startup_only_candidate_vs_official`，例如 `strict_startup_only_candidate_vs_official.json`、`geometry_startup_only_candidate_vs_official.json`、`region_startup_only_candidate_vs_official.json` 及相应Dice/local/no-th3/surface报告；质量目录为 `quality_startup_only_candidate`，官方质量目录和figures沿用既有结构。checkpoint中的pair/evaluated_role和执行绑定键同步使用独立角色；`overall_metric_equivalence`继续 `not_assessed`，阶段完成不构成精度等效声明。

启动角色额外检查实际 config/launch全部字段、真实开始时间与完整completion、原始T1 SHA、同host/GPU/总线程4、逻辑cuda:0、cache-disabled及源码环境。实际 admission须绑定同一retry配置、原准备config/launch SHA、`prepared_not_executed`计划和全部当前资源；prepared与实际admitted inventory必须完全一致，再核对所有文件SHA。归档逐文件比对当前源码，并固定native_free及两个scheduler，拒绝额外Python源码。历史baseline资源格式不会被改标签接受。

所有既有18阶段、逐阶段共享锁、几何门控、strict138、no-th3、Dice、局部误差、质量、图和时间范围保持。新角色断点续跑时重新核对执行/资源绑定，并校验helper SHA；不改变任何数值算子。默认baseline仍保留原断点协议：升级脚本会触发已有脚本SHA保护，需要独立新评估目录或保留旧脚本完成旧任务，不能覆盖旧检查点。

CPU回归验证角色默认兼容、独立命名、prepared不能冒充完成、历史资源报告不能改标签、输入/权重/scheduler/archive漂移拒绝、另一attempt拒绝、主机/GPU/线程环境绑定及不完整completion拒绝。这些临时小文件仅验证证据绑定，不是影像精度benchmark。实际8f整例须完成后再启动评估，现阶段没有该候选的新精度结论。


## 仅绑定核验与元数据纠正

`--verify-only`运行相同的完整输入/源码/资源/执行回执核验后立即退出。输出必须是不存在的新目录；checkpoint状态为`binding_verified_only`，不执行138项数值比较、几何、统计或绘图，不标记比较complete。输入SHA变量遮蔽的历史回执保留，独立纠正流程见[输入SHA元数据纠正](PAIR_BINDING_METADATA_CORRECTION.md)。


## 精度候选的独立实际执行角色

新增 `evaluated_role="precision_candidate"`，用于本轮冻结源码 `3a0c9aba6321b4981fd8174b4b191515459aa38b`。它使用 `evaluated_config`、`evaluated_commit`、`evaluated_resources` 与 `evaluated_resources_kind="admission_inventory"`；不能把启动处理的8f整例、historical whole报告或准备/准入状态改名作为精度候选完成证据。原baseline和startup-only角色保持原行为。

```python
# 实际原始T1整例完成后，再填写actual_config_path和actual_admission_report_path。
precision_comparison_configuration = dict(existing_official_pair_configuration)
precision_comparison_configuration.update(
    evaluated_role="precision_candidate",  # 独立被评估身份，不叫baseline或startup-only。
    evaluated_config=actual_config_path,  # 已实际完成的本轮原始T1配置。
    evaluated_commit="3a0c9aba6321b4981fd8174b4b191515459aa38b",  # 与执行回执/源码归档相同。
    evaluated_resources=actual_admission_report_path,  # 同一次实际执行的资源清单。
    evaluated_resources_kind="admission_inventory",  # 不用历史baseline资源报告。
    code_root=actual_executed_source_path,  # 与actual_config中的执行源码完全一致。
    source_archive=actual_executed_source_archive_path,  # 冻结源码归档SHA绑定。
    output=new_precision_comparison_output_path,  # 独立新比较目录，保留旧结果。
)
```

`verify_admitted_candidate_binding`复用原启动候选核验流程；`verify_startup_binding`保留为兼容入口，不复制另一套归档/资产核验。精度角色额外读取本次 `fnit-native-free-run.json`：完成状态、原始input/subject_dir/device/总线程4、138项输出及每个输出自身路径、完成回执及总时长、无FP16/BF16或autocast、缓存禁用须一致。报告SHA保留并确认核验期间稳定。官方程序/资源SHA、log/done与两侧host/GPU/原T1/cohort绑定仍通过同一个evaluate_pair核验。

输出前缀为 `precision_candidate_vs_official`，执行绑定键为 `precision_candidate` 和 `precision_resource_verification`，质量目录为 `quality_precision_candidate`。helper SHA与resume配置绑定沿用候选协议，核验阶段每次重做。`--verify-only`同样只标`binding_verified_only`；完成比对也不自动建立整体指标等效结论，阈值、138项诊断与所有既有数值算法不变。

2026-10-04：新增精度候选角色支持，与此前输入SHA变量修复独立。8项新CPU回归覆盖外置benchmark工具精确键/绝对路径/冻结SHA与inventory准入、角色/3a提交绑定、源码输入及资产漂移、prepared/admitted或历史报告拒绝、138输出路径门控、旧角色兼容、比较器命名。原10项角色核验及4项SHA/只核验回归仍通过。临时fixture只验证工具，真实两例annotation另行准备；精度候选原始T1整例尚未开始，当前没有新的实际精度比较或等效声明。
