# 最终raw矩阵reader v2：来源路由与metadata guard修复

## 1. 当前状态

v2仅修复metadata wrapper，不改原raw `120a3e…` 数学、科学guard、641实际生产源或fixed5FNIT结论。旧nodecw10 PID52792经start_ticks、完整argv、实际pending观察核对后，于01:17:10 UTC停止。01:26:09 UTC实际hostname=nodecw10、PID不存在、无final_report/failure；旧launch、source/config/output/停止证据全部保留。

最终实际运行已完成，现无active waiter。完成map SHA a12765a9绑定旧九例与真实new-CON11；冻结工具与配置经逐版本修复后在项目外独立v7输出目录完成20组/4800判定及9099文件结束SHA复验。2782/4800被接受，20组整体均failed，详见[最终结果](FINAL_RAW_MATRIX_RESULTS.md)。

## 2. 必须交接的official map

新增配置字段：`official_origin_binding={path,sha256}`、`official_mapper={path,sha256}`，以及`selected_protected_roots`明确v3/v4实际run根。路径和SHA只能来自实际产物，不填预计值。

绑定JSON必须为`scope="explicit actual per-case official source hand-off only"`、state completed、execution_completed=true、exactly canonical10cases。CON11指定launch的实际case_ids必须恰为`["sub-CON11"]`，拒绝旧cohortB waiting11来源。view中的完成symlink允许作为metadata来源路由，但必须逐项指向该launch原始真实case目录；真实producer case不能本身是替代alias，不复制科学产物。

## 3. 全map只读审计

v2复用原mapper的`configuration/completed_case/verify_manifest_contract/referenced_files`，不调用publish。逐项核对mapper及imported origin helper来源、全10canonical输入、同一冻结controller/worker/helper、五seed参数、实际launch/controller选中完成记录、DWI/anatomy producer、raw T1/DWI、MRtrix reader、198条planned/executed命令、TCK/scalar/八atlas四矩阵/node与保存file ledger。

完整actualoriginmap、source identities、当前controller observations、各例file verification写入新ready observation；合法symlink只作为路由记录，不被说成新producer输出。旧controller若开始执行被交给新launch的CON11，或选中completed行发生改变，均拒绝。

## 4. 命名空间与完整结束ledger

首次mkdir前保护canonical实际input_files父目录；保护明确selected run根与已绑定官方actual producer路由。资格齐备后，在actual_inputs/envelope写入前再次保护实际root/job/replacement、frozen source/config/report目录、官方manifest解析后的producer与model/anatomy目录，执行双向ancestor检查。每个immutable文件本身也受保护，单个共享archive不被错误当成整个共享存储producer树。

20条chain不再丢弃`validate_gpu_run`的touched，而是逐链保留并合并完整required-output SHA。全raw、helper/scientific source、static reports、嵌套configs、v3 receipt、GPU/origin map和官方consumed/saved file ledger均纳入immutable snapshot。矩阵统计后、final_report写入紧前再次复验，变化时拒绝completion并保留partial/failure。所有数学定义仍由原工具执行。

## 5. 调用与输出

入口仍是`benchmark_connectome_final_raw_envelope.py --config … --config-sha256 … --output-root …`。`--watch`等待已配置来源，`--check-only`只读核验，poll为1–60秒，timeout为正值至168小时。v2配置必须绑定真实official map及mapper和selected根，旧v1九例配置不能启动v2。

最终输出仍为20arm×8atlas，每臂official5seed×singleFNITseed0，self/population保持not_assessed。qualified observation附完整official审计，immutable_input_snapshot保存结束ledger；每臂actual_inputs/envelope/execution，以及全部实际完成后的final_report。本次完整10例map、20条实际资格、20份矩阵统计及结束复验全部执行完毕。

## 6. 证据与回归

v2初版源码SHA `bc2961583a544c7ad28a69c54bfc857be2f6bd288eb26abf8c97168c6e371f24`。六项stdlib metadata回归通过：raw父目录、selected actual job、resolved官方producer双向拒绝；缺explicit mapper拒绝；嵌套config/source及required-output mutation拒绝；curator统计stub中source变化在终结检查被拒绝且不生成final_report。它们不是科学benchmark。

[旧两项缺口复现](../../validation/connectome/tenraw_20261002/task_04_final_raw_matrix_preparation_v2/fnit-task04-final-reader-review-tests-v1.json)与[旧冻结源码](../../validation/connectome/tenraw_20261002/task_04_final_raw_matrix_preparation_v2/frozen_v1_tool.py)保持原SHA绑定。实际[nodecw10 v9 import核验](../../validation/connectome/tenraw_20261002/task_04_final_raw_matrix_preparation_v2/actual_v9_imports.json)确认原helper SHA，未导入FNIT或torch，无GPU/MRI action。完整20链资格与official新map审计已实际完成，完整证据见最终结果。

## 7. 版本与后续

2026-10-03：封存旧v1 waiter；只在task_04分支修metadata guard；保留curator旧source+复现；真实source/map/schema交接已完成；最终工具SHA9be317df，九项metadata回归通过。[准备记录](../../validation/connectome/tenraw_20261002/task_04_final_raw_matrix_preparation_v2/preparation.json)列出具体字段和界限。

原raw科学代码、Task1/2/4 producer、Task5 selecteddriver及root共享tools/docs/payload不改。早期七份rawfailed报告原样保留；fixed5FNIT self与raw singleFNIT self分别报告。官方版本、实现和文献沿用实际原manifest及项目说明，不新增资源或计算依赖。

最终修订区分源码树清单、显式七份历史比较记录和本次typed运行契约：前两类自身完整SHA保护；本次launch/resource/config/report及输出完整展开。mutable官方controller保存前后观察SHA并检查选中完成行、固定工作量及交接病例不重复派发。v9 map identity变量覆盖在wrapper中独立safe_json绑定；root另修成熟子函数，原冻结v9未热改。所有v2–v6失败记录与原v1停止/缺口证据保留在交付包和既有准备记录。
