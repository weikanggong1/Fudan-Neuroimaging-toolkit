# 单对比较输入 SHA 元数据纠正

## 功能与原因

旧 `evaluate_pair.py` 在输入核验后使用同名 `sha` 遍历官方程序清单，写入 `execution_binding.json` 时误用了最后一个程序的 SHA。输入验证在遮蔽前执行；本次修改只修复绑定回执字段，不修改算法、原比较数值或门槛。

原配置、checkpoint、execution binding 和全部数值结果保持只读。新配置只能改变 `output`；新核验输出及纠正 JSON 使用独立路径。没有重跑整例、数值比较或官方程序。

## 先完整核验绑定

`evaluate_pair.py --verify-only` 复用完整 `verify_binding`，包括两侧实际 config/launch/completion、原始输入/cohort、候选源码归档和成熟启动资源验证、官方 program/resource manifest 及 log/done。它仍取得原共享锁；仅 CPU 文件核验，不调用重建或 GPU 算法。

```bash
# FNIT_PAIR_PYTHON使用实际冻结Conda解释器；工具目录使用本次独立验证路径。
"$FNIT_PAIR_PYTHON" "$FNIT_PAIR_TOOLS/evaluate_pair.py" \
  --config "$FNIT_PAIR_VERIFICATION_CONFIG" --verify-only
```

`FNIT_PAIR_VERIFICATION_CONFIG`是原配置的副本，除不存在的新`output`外全部字段保持相同。已有输出目录会被拒绝。产物只有 `checkpoint.json` 和 `execution_binding.json`。checkpoint状态是 `binding_verified_only`，阶段只有已完成的 `verify_binding`，`numerical_comparison_executed=false`；没有 `strict_138`，不把核验称作比较完成或精度通过。

## 再审计唯一纠正

`audit_pair_binding_correction.py` 的唯一参数 `--config` 指向冻结 JSON，包含下面六份回执的实际路径与 SHA-256，以及不存在的新纠正文件路径。所有 SHA 须在运行前由协调者冻结；不能省略或在脚本内自动接受漂移。

```json
{
  "original_config": {"path": "/absolute/original_pair_config.json", "sha256": "64位原配置SHA"},
  "original_checkpoint": {"path": "/absolute/original_result/checkpoint.json", "sha256": "64位原checkpoint SHA"},
  "original_binding": {"path": "/absolute/original_result/execution_binding.json", "sha256": "64位原binding SHA"},
  "verification_config": {"path": "/absolute/new_verify_config.json", "sha256": "64位新配置SHA"},
  "verification_checkpoint": {"path": "/absolute/new_verify_result/checkpoint.json", "sha256": "64位新checkpoint SHA"},
  "verification_binding": {"path": "/absolute/new_verify_result/execution_binding.json", "sha256": "64位新binding SHA"},
  "output": "/absolute/separate_correction.json"
}
```

```bash
"$FNIT_PAIR_PYTHON" "$FNIT_PAIR_TOOLS/audit_pair_binding_correction.py" \
  --config "$FNIT_PAIR_CORRECTION_CONFIG"
```

审计要求原比较checkpoint为complete、新checkpoint仅绑定核验；配置仅output不同、路径不重叠；新核验使用当前审计的 evaluator/helper SHA；两份binding字段集合相同且唯一变化是顶层`input_sha256`。两侧subject、config/launch/completion、资源/源码/比较器等原绑定值都须不变。

脚本重读实际原始输入、cohort、候选config/launch、官方config/completion，核对全部输入SHA一致；官方manifest须与completion和原binding一致。旧错误值必须等于该冻结manifest中最后一个程序SHA，且程序文件当前哈希正确、不同于原始输入SHA。全部已读文件及工具在写出前复核SHA。

新`correction.json`状态为`metadata_correction_verified`，记录旧/新值、唯一差异、错误来源、六份冻结回执及其他实际读文件SHA、工具SHA。它只解释元数据纠正，`overall_metric_equivalence=not_assessed`。文件已存在则拒绝，不覆盖原回执或新核验结果。没有新增依赖，使用标准库与项目原验证工具。

## CPU 回归与实测边界

4项输入绑定/verify-only回归覆盖不同输入与两个程序SHA、两个评估角色、程序漂移拒绝、仅核验退出且拒绝已有输出。另4项纠正审计回归覆盖唯一纠正且原文件不变、额外绑定字段变化拒绝、冻结回执及错误来源拒绝、伪称数值完成拒绝。连同10项既有角色绑定回归，共18项CPU测试通过（0.204秒）。临时小文件只验证工具逻辑；真实资源重核单独报告。

2026-10-04 gpucw1 实际部署 `4c084d8a4714beccbcf670d54236d3454af5ee5d` 工具，原 sub-06 官方对照配置只改变 output 后，完整 `--verify-only` 入口 21.501 秒、exit 0。audit 再核验六份原/新实际回执和当前输入，`metadata_correction_verified`、exit 0，唯一差异为 input_sha256：原错误值 `8aa986c658663d2c26fa482a2ce303c4eecc28fbd69e4f6e9bace438be058200` 是原程序清单末项 SHA，正确输入为 `7e33afb28f631fac31d81e2428a6144f61aba4102859c694eeb49ee584de04f7`。没有重跑数值比较，原输入、运行和数值报告保持。完整[真实回执](../../runtime/startup_pair_binding_correction_v1/correction.json)与原/新 checkpoint、binding、配置、工具 SHA 一并归档，归档 SHA `fe809462507f630957060676061f06bdeee0767a5ca25c71ce9d58fed612ecba`。

2026-10-04：修复SHA变量遮蔽，新增只核验模式与独立元数据纠正审计；不改生产源码，不改变已有数值比较和结论。


## v4 角色协议补全（2026-10-04）

现场检查发现，冻结 `startup_pair_binding_tools_v3`（`4c084d8a`）的 evaluator 已支持 baseline，但其 audit 仅支持 `startup_only_candidate`。上述18项旧测试中的“两个角色”指 evaluator，不代表 baseline audit 已通过；此前 sub-06 的真实审计属于启动候选，仍按原版本保留。10159/10171 的 baseline 修正尚待新工具部署、共享锁内真实核验和审计，不能引用旧启动候选结果替代。

新 audit 明确支持 `baseline`、`startup_only_candidate` 和 `precision_candidate`。baseline checkpoint 按真实 evaluator 协议校验 `script_sha256`，不要求它未写出的 `role_helper_sha256`。baseline 资源证明使用 binding 的 `baseline_resources_sha256` 冻结清单，再核对 `mismatches=[]`、原比较/config/completion/binding 的 code commit 一致，逐项重新哈希 weights、assets、binaries 的 `resolved_path`。启动/精度候选继续要求对应非空 `startup_resource_verification` / `precision_resource_verification` 以及准确 helper SHA。

两侧实际 config 和 launch 必须明确含 `input_sha256`，并与真实原始输入一致；不从缺失字段推测输入身份。新增 CPU 小文件测试实际经过 evaluator 的 baseline verify-only，再执行 audit，覆盖唯一纠正、三类资源实际重新读取、资源文件漂移、冻结资源清单漂移、官方 launch 缺少输入声明、候选 helper/资源证明缺失。测试用于元数据协议与拒绝门槛，不是脑数据 benchmark。

```bash
python3 -m unittest discover \
  -s validation/recon_all/accuracy_20261003/cohort/pair \
  -p 'test_*binding*.py' -v
```

新 v4 必须使用独立工具目录、冻结完整现有 evaluator/helper/audit SHA，以及新核验/审计输出目录；旧 v3 队列、失败日志和原18阶段数值均保留。审计不获取锁，调用方须持 `/tmp/fnit-shared-benchmark.lock`；verify-only 自行按原配置取得同一锁。当前文档是本地工具更新说明，不宣称两例真实纠正已经完成。
