# 十例精度候选准备（2026-10-04）

## 版本与范围

候选工作分支以创建时真实 `origin/main` 的 `cc9402734faeba93b3a13c29932fa1392eaccf62` 为固定基线。合入启动补丁 `5f75ed5c` 后的冲突解决提交为 `9f7ed77308224ef1bd5d5c7d3eb89cd648a1c949`，五个精度文件合入后的源码提交为 `127eb30656ae60a639e44634b642bbb945bc93d1`。真实 GPU 阶段和后续整例实际使用的冻结生产源码为 `3a0c9aba6321b4981fd8174b4b191515459aa38b`。本次文档及证据归档提交没有重新冻结、修改或替换该运行源码。

保留 main 的完整父进程 CUDA cache 释放函数与单次调用、设备解析、取消逻辑及诊断测试。启动 worker 的 AST 与已验证 5f 版本一致，加入 READY/GO 屏障和可信的 pre-callable bootstrap OOM 有限 fresh-exec 重试。main 的 `native_free.py`、`segstats_wmparc_python.py`、`volmask_python.py`、`mris_register_average_numba.py` 保持逐文件字节一致。原生产 API/CLI 不变。

## 五个精度文件的真实阶段依据

| 文件 | 实际源文件 SHA-256 | 同输入证据及范围 |
|---|---|---|
| `input_talairach_chain.py` | `ab0a02eca9a738cf9c43e0cbcf179165fe187529bbc34ad1929b619fd60bc8d3` | `task_02/synthstrip_writer.json`：旧两例冻结真实网络输出的 ABBA writer 回放，值和几何差异 0；输出恢复 uint8；非整数输入明确拒绝。未验证当前 main 的完整 Talairach 输入链。 |
| `ca_normalize_python.py` | `167be9d72496b7279b98e52b8f0d6c0808d2c09d02df24af1fed0fd331100b9b` | `task_03/production_fix_abba_completed.json`：旧两例冻结输入 ABBA 8 次调用，候选 norm/ctrl 与官方差异 0。原 driver 未记录退出码，sub-02 性能稳定性未确认。 |
| `place_pial_python.py` | `2377f847610a935859935e7ecafdbbd52a74bd4e27ad372a8324b49249d18e33` | `task_05/placement_complete_v1/completion_summary.json`：一例真实输入双侧完整 pial 四轮，11 个输入未变，有序面未变，mean/P99/max 位移均 0；该阶段集合 14 complete、0 failed。未采集同刻进程树显存峰。 |
| `surface_stats_cache.py` | `6f64b40773c61d46fd167c34a74530d9475dd7992b6607c99df14511826ab2a1` | `task_05/roi_lh_v1.json`：历史 LH 真实相同 surface/pial/thickness/annotation，threads=2、cuda:0，ABBA，候选相同 face-area oracle P99/max=0。 |
| `surface_roi_gpu.py` | `5842a1a57ab0983b5735d3631826e9b89bd7123c271d18a4e4ad1161fb0ee758` | 本文件仅 docstring 修改；`task_05/first_phase.json` 绑定 SHA，当时的 queued 不能称为已完成测试。其 cache/ROI 行为的后续完成证据为上行 `roi_lh_v1.json`。 |

以上是旧 `816e5610417a4c587caf321049438a9554139016` 冻结输入的阶段诊断，五个候选文件 SHA 与各回执准确对应。阶段依赖及完整调用链没有因合入新 main 自动获得验证。CA 历史 README 保留诊断参考作用。当前主线整合候选已完成下述两例 annotation 阶段验证。新的十例连续链尚未完成，整体官方等效为 `not_assessed`；既有 8f 的整例和官方 18 phase 结果保持原版本绑定。

## CPU 测试

原 Conda Python 隐藏 CUDA 运行启动调度标准库测试 30/30 与 CA 坐标测试 3/3。main 17 个测试与 5f 28 个测试的函数名并集为 30 个，候选全部保留。原 Conda 缺 pytest，独立 `--system-site-packages` 测试 venv 安装固定 pytest 8.3.5；pial/cache 使用原始 pytest fixtures，四套原始测试最终 40 passed、exit=0、pytest 内部 43.81 秒（外部 44.948272 秒）。结果与轮子、程序、源码完整 SHA 保存于独立运行回执，不能把这些人工小输入单测称为真实 benchmark。

独立 CPU workspace：`FNIT/workspaces/recon_accuracy_20261003/ten_case_candidate_cpu_v1`；运行回执：`FNIT/runs/recon_accuracy_20261003/ten_case_candidate_cpu_v1`。复现聚焦测试：

```bash
# 仅 CPU 测试；环境复用原 Conda 包，不启动生产整例。
candidate_workspace=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/ten_case_candidate_cpu_v1
cd "$candidate_workspace"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$candidate_workspace/src" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$candidate_workspace/test_venv/bin/python" -m pytest -q \
  tests/recon_all/test_hemisphere_parallel.py tests/recon_all/test_ca_coordinate_contract.py \
  tests/recon_all/test_pial_rejected_stop.py tests/recon_all/test_surface_stats_cache.py
```

## 新主线候选的真实 annotation 阶段验证

canonical 运行目录为 `FNIT/runs/recon_accuracy_20261003/ten_case_candidate_annotation_stage_v1`。基线为已完整验证的 8f `startup_overlay_gpu1_v2`，候选为固定 3a 主线整合源码。四臂按 sub06 AB、sub07 BA 串行执行，全部 `stage_complete`、退出码 0、自有子树清理完成。两臂同 15 项数据/资产 SHA、线程总预算 4（LH/RH 各 2）、startup wait 30 秒、禁用 CUDA caching；GPU1 实际 UUID 为 `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`，锁为 `/tmp/fnit-stage-gpu1-benchmark.lock`，启动前并持锁复查可用显存至少 `10,000,000,000` bytes，阶段超时 900 秒。

sub06 两臂父进程在组调用前建立标量 CUDA context；sub07 两臂未预初始化。父进程和 worker 实际 allocator 为 disabled，worker 实际 TF32 开启，父进程 TF32 开启、autocast 关闭。该预初始化 API 对照只覆盖标量 context 与本次明确的 cache 策略，不覆盖任意长期已初始化 CUDA 或 caching 配置。每半球实际 startup attempt 均 1、operation_entered 均 True。

| arm | group 实际墙钟 s | 阶段含汇总 s | monitor 命令墙钟 s | 资源等待 s | 同刻父子外层采样峰 bytes | 组内独立采样峰 bytes | 整卡采样占用峰 bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| ds000114_sub-06_baseline | 136.727012 | 146.210083 | 146.759675 | 0.892343 | 4475322368 | 4475322368 | 30584864768 |
| ds000114_sub-06_candidate | 139.556746 | 146.592467 | 147.189874 | 0.873425 | 4475322368 | 4475322368 | 23610785792 |
| ds000114_sub-07_candidate | 118.888445 | 128.245192 | 128.745292 | 0.927675 | 3984588800 | 3986685952 | 40446722048 |
| ds000114_sub-07_baseline | 125.419761 | 132.387994 | 132.836339 | 0.866464 | 3984588800 | 3984588800 | 68828528640 |

group 墙钟包含组内启动、运算、发布和清理；阶段墙钟另含输入核验、复制、网格读取及语义输出与汇总。monitor 命令墙钟从进程启动至退出，资源等待不计入该命令/算法墙钟。另有共享 CPU 任务预算 4，本阶段预算 4；这些时间是共享环境观测，不作为干净配对提速或完整整例耗时。

| arm | 外层采样数 | 查询失败 | 请求间隔 s | 实际最大间隔 s |
|---|---:|---:|---:|---:|
| ds000114_sub-06_baseline | 248 | 0 | 0.5 | 0.872296 |
| ds000114_sub-06_candidate | 250 | 0 | 0.5 | 0.838343 |
| ds000114_sub-07_candidate | 218 | 0 | 0.5 | 0.772959 |
| ds000114_sub-07_baseline | 217 | 0 | 0.5 | 1.785358 |

显存峰来自有限离散采样，不是连续峰。外层和组内是两套独立采样器，每套均按同一采样时刻合计父子进程占用，两套峰不相加。整卡峰包含外部任务。组内 sub07 候选峰 `3,986,685,952` bytes 与外层 `3,984,588,800` bytes 不同，保留各自 scope。

候选 sub06 的 `parent_idle_cuda_cache` 实际为 complete，耗时 `0.0016404860652983189` 秒，parent allocator allocated/reserved 的 before/after 均 0；该计数不含 CUDA context、worker 和外部 allocator，不能推导进程显存为 0。同臂的同期父子进程采样峰实际为 `4,475,322,368` bytes。候选 sub07 helper 报告 `not_applicable`，没有用该 helper 创建 context。原 8f 基线没有 main 的此项报告，按缺字段保留。

### 12 套同输入分区比较

两例 CPU 原始比较均 `exact_regression_pass=true`。sub06 LH/RH 顶点数为 130346/132837，sub07 为 114342/114824；每侧比较 aparc、aparc.a2009s、aparc.DKTatlas 三套分区，共 12 pairs。每套 original packed IDs、table indices、vertex indices、color table 和 names 数组精确相同；smoothwm/sphere.reg 坐标和有序面相同；原 annot 和 NPZ 文件字节也相同。差异顶点均 0，所有存在标签 Dice=1（min/P05/median 均 1），不混淆 packed ID 与解剖类编号，不统计两臂均不存在的标签。全部逐标签 Dice 仍保存在两份 `annotation.pair.json`。官方整体等效保持 `not_assessed`。

### 证据归档及整例状态

87 个 JSON/CSV/log/Markdown 元数据文件归档在 `validation/recon_all/accuracy_20261004_candidate_preparation/annotation_stage_metadata_v1/`。不含 MRI、mesh、NPZ、权重、资产或许可证内容。源包、完整源文件清单、CPU 回执及 main diff 另列于同一 preparation 目录；运行 source/archive/commit、config、程序及工具 SHA 在 plan/launch 和各 inventory 中重新核验。

原预检失败 `FileNotFoundError run_monitored.py` 发生在 GPU 启动前，保留 `preflight_failure_v1.json` 和 `plan.preflight_v1.json`。随后通过显式 monitor_script 复用已验证的外部监控工具，两臂算法仍使用各自冻结源码，没有回退候选算法或覆盖原产物。

协调者已于 `2026-10-04T12:24:49.294967Z` 启动 3a 的原 T1 sub06 空目录整例：`FNIT/runs/recon_accuracy_20261003/precision_candidate_3a_whole_sub06_v1/attempt_01`，GPU0、总线程 4、CLI。收到该启动通知时尚无 completion，本证据提交不能记为整例完成，也不把旧 8f 结果归到新 3a。

原函数的详细输入、输出、参数、原软件调用与参考文献见各模块功能页；本页记录版本整合和真实证据范围。本分支未推送 main，冻结 8f 与 3a 运行源码均未改动。
