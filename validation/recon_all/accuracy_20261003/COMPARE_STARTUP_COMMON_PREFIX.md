# 失败基线与启动处理候选的现存共同前缀 CPU 比较

## 功能与范围

`compare_startup_common_prefix.py`只读比较sub06原失败基线816与完成后的8f启动处理候选自产输出，定位连续上游artifact漂移。不是官方比较，不读取官方中间结果，也不构成整例精度等效。基线必须绑定原失败completion，报告明确 `baseline_whole_case_complete=false`；候选必须已有真实complete/pipeline complete回执，未结束就拒绝比较。

流程：固定索引及实际config/launch/completion/源码归档绑定 → 同原始T1 SHA、同机/GPU/线程4协议 → 共同体积/LTA → 两侧surface/register、顶点对应证明 → 对应门控下的map/cortex label → 重算文件与回执SHA确认比较期间稳定。

不重新运行重建，不修补或重采样输出，不把缺失项当零差异。不同值不会自动归因调度、CUDA、随机数或算法；同输入12组annotation一致已有独立证据，这里只做连续上游漂移排查。

## 输入参数

唯一CLI参数 `--config` 是冻结JSON。所有路径是实际服务器绝对路径：

- `baseline_config/candidate_config`：两次实际执行的 `attempt_01/retry_config.json`，不是 nominal准备config。
- `baseline_archive/candidate_archive`：两次实际源码包，与各config SHA一致。
- `baseline_commit/candidate_commit`：两次完整实际提交（816基线及8f启动处理版本）。
- `baseline_completion_sha256/candidate_completion_sha256`：根任务从两次真实diagnostics/completion.json字节冻结的SHA256，必填。
- `baseline_pipeline_sha256/candidate_pipeline_sha256`：根任务从两次实际output/fnit-native-free-run.json字节冻结的SHA256，必填。不得从其他case复制回执再生成假绑定。
- `comparison_code_root`：冻结FNIT源码，用于成熟 `compare_subject._topology`；不替换运行源码。
- `scripts_dir`：已有 `compare_volume_prefix.py` 所在目录。
- `fnit_index`：可选，默认 `/cwStorage/home/gongwk/Notebook_code/FNIT/INDEX.json`；运行时读取并记录SHA。
- `lock`：共享锁 `/tmp/fnit-shared-benchmark.lock`；整个CPU比较取得锁后进行。
- `output`：不存在的新比较目录。

示意配置（提交、归档和工具路径从实际回执填入，不使用文件夹名推断）：

```json
{
  "baseline_config": "/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/baseline_resource_replays_v1/ds000114_sub-06/attempt_01/retry_config.json",
  "candidate_config": "/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/startup_whole_sub06_v2/attempt_01/retry_config.json",
  "baseline_archive": "/absolute/path/verified_baseline_816.tar.gz",
  "candidate_archive": "/absolute/path/verified_startup_8f.tar.gz",
  "baseline_commit": "ACTUAL_BASELINE_FULL_COMMIT",
  "candidate_commit": "ACTUAL_STARTUP_FULL_COMMIT",
  "baseline_completion_sha256": "ACTUAL_BASELINE_COMPLETION_SHA256",
  "candidate_completion_sha256": "ACTUAL_CANDIDATE_COMPLETION_SHA256",
  "baseline_pipeline_sha256": "ACTUAL_BASELINE_PIPELINE_SHA256",
  "candidate_pipeline_sha256": "ACTUAL_CANDIDATE_PIPELINE_SHA256",
  "comparison_code_root": "/absolute/path/frozen_comparison_source",
  "scripts_dir": "/absolute/path/frozen_comparison_source/validation/recon_all/python_gpu_port",
  "lock": "/tmp/fnit-shared-benchmark.lock",
  "output": "/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/common_prefix_sub06_v1"
}
```

## Python与命令行

```python
from compare_startup_common_prefix import main
# 仅在根任务批准、候选整例已结束后调用。
comparison_configuration_path = '/absolute/path/frozen_common_prefix_config.json'
exit_code = main(argv=['--config', comparison_configuration_path])
```

```bash
# 保持已验证Conda解释器；工具自己禁用CUDA并设置各库CPU线程预算4。
FNIT_COMPARISON_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
"$FNIT_COMPARISON_PYTHON" compare_startup_common_prefix.py --config /absolute/path/frozen_common_prefix_config.json
```

2026-10-04 已以 v3 工具完成 sub-06 真实共同前段比较；不重建、不改变生产输出。无对应原软件命令，本工具复用FNIT CPU诊断实现，不调用FreeSurfer/FSL或其封装。

## 输出内容与门控

新目录内 `common_prefix.json` 是严格JSON，逐项写盘并保留具体异常。包括config/脚本/比较器/固定索引SHA、两边实际执行回执、commit及archive/核心源码SHA、文件存在性及SHA、总墙钟与锁等待。两次launch每项config绑定、实际开始时间、同host/总线程4/逻辑cuda:0和GPU/cache/source环境均核对；读取完整归档与当前源码/Python集合确认冻结。

体积：orig/001、rawavg、orig、nu、T1、brainmask、norm、brain、wm.seg、wm.asegedit、wm、filled、aseg.presurf。记录shape/dtype及双方完整affine、affine差异数/max/P99；只有相同shape和完全相同affine才计算同体素差异数/max/P99。wm/filled/aseg离散文件拒绝非整数、负值或非有限值，复用 `_dice` 给每标签Dice和foreground Dice；brainmask强度与foreground Dice分开，不把强度体积冒称标签。

LTA：talairach及talairach_with_skull的原文件SHA、4×4矩阵、差异元素数/max/P99。矩阵来自现有LTA解析器；这只是原始数值比较，不证明变换头的type/源目标几何语义相同。

表面：两侧orig.nofix/orig.premesh/orig、white.preaparc、smoothwm/其nofix、inflated/其nofix、qsphere.nofix、sphere、sphere.reg。记录顶点/面shape、dtype和拓扑。原geometry函数依赖final white，不适合本例失败前缀，因此复用 `_topology`，以各自产white.preaparc作anchor；conform affine/TkRAS、surface footer（排除filename）、有效闭合连通genus-zero拓扑、有序面及顶点数、与各自产anchor索引一致均通过后才算同索引坐标差异数/max/P99和顶点欧氏距离mm。未证明对应时同索引结果为null；不新增长时间最近三角面搜索。

curv/sulc/avg_curv/H/K及共有smoothwm.*.crv采用相应表面门控且长度必须等于anchor顶点数，再比较值；cortex.label使用证明对应的anchor，按既有 `compare_subject.compare_cortex` 语义分别比较每顶点entry multiplicity（差异数/max/P99）及unique membership（Dice/成员差）。记录双方entries、unique_vertices、duplicate_entries、repeated_vertex_count；原始文件SHA、ID序列是否一致另行保留，不通过dedup掩盖文件或重复次数差异。负值、越界和非整数ID仍严格拒绝。shape不符或无法证明对应即明确not_assessed。

**候选finish阶段可能重写同名map。** 文件共有只证明现存artifact可比较，不能证明它是annotation前同输入冻结快照。报告明确该限制，后写map不一致不能作为启动调度改变上游的证据。没有纳入失败基线尚未产生的annotation、最终pial/white、stats和其他下游输出。

v2按现场实际pipeline schema，读取同一批bytes并计算SHA，再解析JSON。核对run.input/subject_dir/threads/device及TF32/禁低精度/禁autocast/allocator策略，要求实际CLI argv的原始输入、运行输出和资源参数逐项一致。基线exit_code和child_exit_code必须是非零严格integer且相同，不能缺失/null/bool；run.status必须failed、failed_stage为annotation_hemisphere_group，且stage/error/失败worker traceback相互吻合。候选要求严格双零退出、run.status complete、output_validation passed138/138与completion相同、mesh双侧passed、numeric_validation状态及未评估总体精度状态保留；completion.pipeline_total_seconds须等于run.total_seconds，138条outputs必须全部指向本run目录且文件存在。

旧baseline completion本身没有pipeline路径或pipeline SHA，只有泛化child错误；因此v2配置必须额外冻结各自真实completion与pipeline SHA，不能仅靠同commit/archive推断case。基线pipeline字段与失败证据一并保存；候选numeric_validation=not_run不会变成精度通过。

比较期间任何artifact或执行回执（包括pipeline）SHA变化即失败；存在但解析失败的文件保留error，退出非零。`comparison_complete`表示计划项已检查，不代表全部相同；缺失、对应门控未通过、非有限值及文件错误另列。总体精度等效始终 `not_assessed`，合并精度候选0/10不变。

## CPU回归与依赖

依赖为现有环境的NumPy/Nibabel/SciPy；没有新增包。回归覆盖失败baseline身份、候选完成门控、prepared不能冒充执行、输入/源码/archive漂移、额外Python源码、缺失artifact不当作零差异，及错case回执、非法退出码、路径与precision、138及mesh/numeric状态。表面和计数门控在既有Conda依赖可用时执行；本地无这些包，明确 skip，未安装依赖。v2 在 gpucw1 已有 Conda 环境隐藏 CUDA 的 15 项测试通过（0.584 秒），v3 扩展至 19 项，全部通过（0.747 秒）。临时小文件只验证比较工具；真实 benchmark 单独报告。

原实现：`validation/recon_all/python_gpu_port/compare_volume_prefix.py` 和 `src/fnit/recon_all/compare_subject.py`；数据读取使用Nibabel。官方参照来源和公开数据许可沿用本轮cohort文档，本工具没有新官方计算或新精度结果。


## v3 验证器重复标签语义修复

v2真实CPU比较仅因LH cortex.label重复ID失败，原失败回执保持不变。现场两边文件SHA相同：`ac9d1d572549401a86b2f257e81622865e8cbe6608680a7f141d62d34d831778`；各121822 entries、121694 unique，128 duplicate entries，索引0至130344，LH anchor130346，无负值/越界。RH无重复且索引范围有效。

这是验证器错误新增唯一性条件。成熟 `label_cortex_fix_ga_python.py` 的 `np.concatenate(base_ids, ga_ids)` 会保留重叠ID；成熟 `compare_subject.py` 已用幂等membership与per-vertex multiplicity比较。这些重复不是启动候选引入的差异，也不是生产label需要修补的证据。

v3移除错误的唯一性条件，保留每条entry计数。新增小fixture复现128个重叠GA ID，验证记录序列重排时multiplicity/membership一致但不声称bytes一致；改变重复次数时membership可仍相同而multiplicity必须不同；负数/越界/非整数仍失败。额外检查ASCII原始ID都是integer、声明entries与实际行数匹配，Nibabel读出的ID序列与原始整数序列一致，防止读入时隐式截断。

生产baseline和candidate输出完全不改。根任务审阅后使用 fresh v3 工具与输出目录，真实比较 `comparison_complete`，77.351 秒、54 个共同文件、15 个计划缺失路径、10 张未证明对应的表面、0 解析错误；v2 失败保持。13 张体积的体素、affine、dtype 一致；唯一共同 talairach.lta 的 16 元素一致；12 张通过对应门控的双侧表面坐标/有序面一致；双侧 cortex 成员、重复次数、字节一致。talairach_with_skull.lta 双方都缺失，不计为相同。现存 finish 重写曲率差异仍保留，不能推断整例终端输出不退化。

实际 v3 工具 SHA `ab89a5624cb3c2132c73f72f618f1b959e675fdd06d01550dfb1e7badac0ad6a`；配置 SHA `385a38d98f171c52580bb5f4e725f7d74af156ce0b88c00354a0abd8ab734242`；原始比较 JSON SHA `a24bab1f526d5fa79829a225732505c9dfbc60bcf12018cdb446b456c2b58187`。逐项结果见 [RESULTS.json](runtime/startup_prefix_sub06_v2_v3/RESULTS.json) 和其绑定的 v2/v3 原回执。没有整体指标等效结论。
