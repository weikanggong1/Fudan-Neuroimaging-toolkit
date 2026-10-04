# 阶段通过后准备新候选整例

## 功能与边界

`after_startup_stage_whole.py` 等待阶段队列完整成功，使用已有真实阶段输出做 CPU 数值比较，全部 `exact_regression_pass=true` 后才准备新候选配置绑定，然后交给原 `resource_admission.py`。不复写历史 launch，不把准备计划当成已执行。原 admission 的历史重跑绑定与GPU准入逻辑不变。

## 配置、调用与输出

入口：`python after_startup_stage_whole.py --config /absolute/path/new_guard.json`。

原配置字段：`output`（不存在的新 guard 目录）、`stage_queue`（真实阶段队列JSON）、`stage_wait_seconds`、`stage_root`（既有真实阶段输出）、`cases`（非空必需比较例列表）、`comparison_script`、`whole_case_config`（新候选原始T1配置）、`resource_script`、`retry_root`（不存在的新 attempt）、`admission_report`（不存在的新报告）。

新增字段：

- `comparison_output_root`：可选独立新比较输出父目录；默认 `output/comparisons`。每例 `<case>_comparison` 必须不存在。可重复读取旧实测数据做新 CPU 比较，不覆盖旧比较回执，也不重跑阶段GPU。
- `source_archive`：候选源码归档路径，可放 guard 配置或整例配置；必须与整例 `source_archive_sha256` 匹配。
- `expected_source_sha256`：可选相对于 `code_root` 的文件到预期SHA映射，供显式复核候选补丁。
- `expected_resource_sha256`：可选实际资源绝对路径到预期SHA映射；与实际inventory核对。历史权重真实性仍需独立原始清单，当前登记不能补造历史证据。

整例配置每个字段均原样复制到 nominal `diagnostic_root/launch.json`；准备前核对 T1 SHA、解释器和许可证存在性（不读取许可证）、资源目录及逐文件SHA、归档SHA、归档每个文件与源码一致性、Python文件集合一致性，以及 native_free 和两个半球调度模块SHA。归档使用与源码根相对的安全普通文件路径；不接受链接或缺失核心模块。已有 nominal diagnostics、原配置 output、retry attempt 或 admission报告一律拒绝。

准备回执写 `status=prepared_not_executed`、`algorithm_entered=false`、`prepared_utc`、整例configSHA、archiveSHA、三个核心源码SHA和当前资源manifest；没有 `started_utc`、历史执行命令或成功声明。目录独占新建。实际整例由 admission 改为独立 retry输出；执行入口在 retry diagnostics 写真正 launch/completion，不能与 nominal准备回执混淆。

Guard `guard.json` 记录阶段回执SHA、准备回执SHA及具体失败文件/阶段/traceback。开始交给 admission 后，guard的 `algorithm_entered=null`：派发状态无法证明 pipeline函数进入，需实际 admission/执行报告。总整例完成和精度评估各有自己的回执。

## 验证与使用限制

专属CPU测试覆盖计划字段绑定、三核心源码和资源manifest、已有目录拒绝、T1/源码/归档漂移、额外Python源码、旧比较保留、新比较目录、数值gate失败不派发和缺失归档准确错误报告。测试使用临时小文件及mock派发，仅验证调度；不声称影像benchmark或GPU成功。

运行期间须保持候选冻结源码和资源不变。原 admission 将在进入等待时重新登记、锁前复核其实际inventory；新准备记录与 admission实际资源SHA也须在结果审查时一致。没有新 GPU kernel、算法fallback或外部软件功能调用，因此没有独立原软件命令和脑图对照。
