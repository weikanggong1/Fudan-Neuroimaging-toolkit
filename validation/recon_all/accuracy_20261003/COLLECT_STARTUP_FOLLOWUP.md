# Startup 运行 metadata 快照

`collect_startup_followup.py` 仅用 Python 标准库，CPU一次读取已有运行的指定metadata并复制原字节。输入 `--run-root` 为既有运行目录；`--output` 为全新、与输入分离的输出目录，不覆盖任何旧文件。

```bash
python collect_startup_followup.py \
  --run-root /absolute/startup_whole_sub06_v2 \
  --output /absolute/new_startup_followup_snapshot

# 仅CPU fixture单测，不是真实影像benchmark。
python -m unittest discover -s validation/recon_all/accuracy_20261003 \
  -p test_collect_startup_followup.py -v
```

复制白名单：根config/guard.config/launch/admission JSON；guard直接下JSON/log；nominal/diagnostics/launch.json；attempt_01下retry_config.json、diagnostics直接下JSON/log、monitor/monitor.json和gpu_samples.csv、subject/fnit-native-free-run.json、scripts直接下hemisphere-group及startup report/worker.log/request.json。不递归MRI/mesh/assets/weights，不读取license命名文件，不跟随文件或目录软链接。JSON内的资源/程序路径只记录，不打开外部文件。

输出 `snapshot/` 原始字节副本、`startup.followup.json` 和 `README.md`。每个文件读取一次，复制/解析/hash都使用同一bytes；报告observed UTC、size/SHA、读取期间size/mtime是否变化。JSON半写入或解析失败标注invalid_or_incomplete并保留字节；状态不能通过。文件之间不是原子快照，动态报告可能处于不同瞬间。

无completion.json始终completion_state=pending；可另记录guard/preflight已失败或metadata声明仍running/waiting，不能把guard exited当成功，也不通过PID推断运行结束。只有completion complete+exit0+pipeline complete+output_validation passed且expected=present=138、missing为空+mesh_validation passed+admission complete/exit0/finished_utc且cleanup不处于waiting/error+已有monitor exit0且monitor_thread_finished=true，快照无无效/变动文件，才passed=true。mesh完整性仅按已有pipeline报告判断，本工具不重新读取mesh验证；官方等效not_assessed。脚本退出0只代表快照写成，验收须看JSON的passed和run_state。

每组按真实startup_attempts列表记录每半球次数与每次operation_entered；缺标记为unknown，不推成false；startup次数不等于算法调用次数。独立startup worker报告另列以供核对。metadata中的source/code commit/archive SHA、程序文件SHA绑定原样记录为declared，**不读引用源码再次校验**。collector自身源码SHA另外记录。

2026-10-04：14项CPU fixture通过，涵盖pending/guard exited、failed preflight、exit0缺mesh、完整报告、严格目录范围和软链接、半写JSON原字节、startup多尝试与一次算法进入。随后在既有 Conda 环境隐藏 CUDA 运行同一测试，14 项通过（0.065 秒），工具已上传并由协调者执行真实 metadata 快照；这不属于 GPU 计算或影像数值验证。

补充真实schema：admission query_or_validation_failed和guard whole_case_exit_code非零均记录失败，缺completion仍pending；startup等待输出用startup_wait_actual_seconds，兼容旧startup_wait_seconds_actual并记录source_field。collector不计算显存，不会将两个采样器的峰相加。

## 当前真实整例快照

2026-10-04，隔离启动修复 `8f3e51f5` 的 sub-06 原始 T1 整例完成后，在全新目录保存最终快照。138 项完整、已有 mesh 验收、完成/admission/monitor 的全部门禁通过，`passed=true`。归档 SHA-256 为 `4a720c3a54bd715665bed4e06740618345ccb003e1d4a775b56838ccece8d67a`。原字节报告见 [最终快照](runtime/startup_whole_sub06_v2_final_snapshot/)，详细耗时和显存见 [本次中文结果](../../../docs/recon_all/CUDA_STARTUP_BENCHMARK_20261004.md)。
