# 同输入左侧顶点指标与脑区统计阶段诊断

## 功能与当前范围

使用已完成3a sub-06的自产surface RAS/mm white/pial、有序面和注释，比较冻结8f与3a的实际成熟函数。每臂独立新进程执行cold一次、同进程warm一次；只运行左半球顶点图和六套ROI表，不运行原始T1整例，不读取官方结果生成候选，不重试算法错误。两次重放各新建SurfaceStatsCache，在每次六表内复用；warm表示同进程、OS/Python/CUDA已调用，并不意味着沿用上一遍ROI张量缓存。

流程：验证10输入SHA → 双锁和GPU1空闲≥20,000,000,000字节 → 复制到新目录 → 成熟metrics函数 → 六ROI表 → 保存数值数组 → 同进程第二遍 → 原输入/复制white/pial不变检查 → 回收owned进程树 → 释放锁。

## 输入、参数及输出

`replay_metrics_roi_stage.py` 参数：

- `--checkpoint`：已完成自产subject目录，只读。10文件为white/pial、cortex.label、五个annot、brainvol.stats与talairach.xfm；manifest逐项绑定大小/SHA。
- `--input-manifest`：JSON，`files`必须恰含上述10项，不能以准备记录代替已完成执行。
- `--assets`：现有FNIT资产根路径。CUDA metrics直接调用成熟PyTorch算子，不运行native metrics二进制。
- `--output`：独立新目录；包含subject副本、report.json、cold/warm metrics.npz、六个stats文本及按名字保存的ROI npz。
- `--gpu-uuid`：完整物理GPU UUID；必须与CUDA_VISIBLE_DEVICES sole UUID一致，并核验实际cuda:0 device UUID。

report记录输入/复制SHA、解释器/Torch/nibabel、FP32与TF32、loaded_source SHA、坐标/有序面、逐函数墙钟和父/子CPU、CUDA同步、内部thickness/area/curv秒数、每遍ROI cachecounter、输出dtype/shape/有限性及文件SHA。全入口从输入验证开始，包含复制/加载/写出；算子阶段由已有StageProfiler同步计时。第一遍cold可能包括CUDA上下文首次启动，与整例在后期已初始化的父进程历史不同，不用于直接补算整例时间。

外层 `metrics_roi_dual_lock_controller.py --config /绝对路径/A8f.json` 接收已有stage schema，新增 `benchmark_kind=metrics_roi_stage` 与 `gpu_lock=/tmp/fnit-stage-gpu1-benchmark.lock`，固定 `lock=/tmp/fnit-shared-benchmark.lock`、threads=4和最低20GB。复用stage_benchmark_controller与resource_admission的owned树追踪/取消/清理，run_monitored以0.5秒采样；显存不足时释放共享锁继续等待。算法进入后失败停止队列。

`run_metrics_roi_pair.py --run /规范run路径 --controller /规范workspace路径/metrics_roi_dual_lock_controller.py` 顺序运行A8f、B3a，用户取消传至controller回收，任一臂失败不启动下一臂。 本地更新补齐 Popen 返回前取消的信号窗口：新 controller 赋值后若已收到取消，立即发送 SIGTERM 并等待其 owned-tree 清理退出，B 臂不启动。当前服务器冻结 v1 尚未热替换；正在运行的工具保持原字节，本修复供后续独立冻结版本使用。

## 本轮实际状态及版本边界

### v1：算法前诊断脚本接口失败

规范工具目录 `FNIT/workspaces/recon_accuracy_20261003/metrics_roi_sameinput_tools_v1`，运行目录 `FNIT/runs/recon_accuracy_20261003/metrics_roi_sameinput_sub06_v1`。队列PID28336已退出，queue.status=failed；仅A8f controller exit1，B3a未启动。A8f等待共享锁974.794秒后获准入，child在 `thread_budget(4)` 抛出 `TypeError: thread_budget() takes 0 positional arguments but 1 was given`，失败发生在所有metrics/ROI数值算子之前，cold/warm完成遍数均为0；监控命令墙钟2.915秒、5样本、owned采样峰0，自有树全部退出。

这是**新增诊断脚本调用成熟接口的错误，不是FNIT成熟metrics实现bug、不是OOM、不是性能退化证据**。v1的operation_entered曾在进入thread_budget之前过早标True；不能据这个字段声称算法执行，应结合空last_stage、TypeError栈和调用边界判断。v1源码、配置、原始失败报告/日志保留，不覆盖。

### v2：修正工具冻结，真实两臂已完成

- 工具：`FNIT/workspaces/recon_accuracy_20261003/metrics_roi_sameinput_tools_v2`。
- 产物：`FNIT/runs/recon_accuracy_20261003/metrics_roi_sameinput_sub06_v2`。
- source/checkpoint保持原实体路径，只读；8f与3a源码不变。
- runner改为 `thread_budget(threads=4)`；operation_entered初值False，仅在真实 `profiler.run(..._finish_cortical_metrics...)` 调用前设True。
- runner SHA：`f5cfa7b5ea9c9f056602db4de25a6f6063766e2a825bee6d8ecfe3eef22abe50`。
- 冻结tool_manifest SHA：`84ab6ef8f44c3ca12600605b1a5c58929ae3a935f2c23be0f29b74bfd6dd4286`。
- v2队列脚本 `run_metrics_roi_pair_v2.py` SHA：`a5bef543343b0ef77c418e72d55b41afff99012be4a645326a59f34738d5b84b`。
- config SHA：A8f `595679f22f0f0245d9c19ac217d5a6330a4b5767fa8e0024d1eb688503314bcb`；B3a `91e9fe1b3e08e81c517f92360fb7ab0bfb8fa629e7c570ea400a7bfb190dcaa4`。
- 队列PID41582，start_ticks=294502774，终态status=complete；两臂controller_exit_code=0，all_tracked_owned_exited=true，current_arm/current_child_pid=null。

同一v2队列先取得共享锁和GPU1锁、核验20GB空闲，然后在两套实际冻结环境中执行 `test_metrics_roi_controlflow.py`。真实全局CUDA_VISIBLE_DEVICES为空，CUDA lazy_init硬禁止；使用真实thread_budget/StageProfiler/六表dispatcher和成熟算子签名autospec，所有数值算子mock。检查两遍metrics调用、十二个ROI表调用、线程恢复、保存结构。这是**CPU接口/控制流检查，不是benchmark**；临时测试数据与mock数值不列入真实结果。双边预检通过才执行A8f/B3a，任一算法失败停止，不自动重试算法。

实际冻结环境两臂CPU接口预检均exit0、cuda_initialized=false，覆盖两遍metrics、十二表dispatcher、线程恢复和输出结构；预检数值算子mock，不计benchmark。随后真实两臂cold/warm均完成。新v2是修正算法前诊断runner错误后的独立运行；原v1失败保持。

根任务管理的既有协调guard PID45867，于2026-10-04 14:23:32UTC启动：仅暂停总队列编排父114563；当前sub-07整例child继续运行，未暂停任何算法或修改资源预算。v2队列终态/退出或2400秒限时二者任一满足便自动SIGCONT恢复编排父，PID/start_ticks精确核验，回执另存。stage自身不创建第二套暂停guard、不绕过双锁；2026-10-04 14:52:33UTC自然获准入，GPU1空闲32,501,661,696字节。协调guard是否恢复以其独立回执为准，不根据阶段报告代替核验guard状态。

### 实际不变的输入与资源

- 原输入：`FNIT/runs/recon_accuracy_20261003/precision_candidate_3a_whole_sub06_v1/attempt_01/subject`，complete/138；精确10输入SHA绑定。
- GPU1：`GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`，逻辑cuda:0，总线程4、默认TF32、无half/autocast、allocator disabled。
- source：8f3e51f58c37d28e1283d5bb8e248e93530cd398 与3a0c9aba6321b4981fd8174b4b191515459aa38b。

远程v2工具已冻结，后续实测按其真实runner/config/report SHA写新独立回执；不热换远程工具，不修改既有报告。实际结果详见下节及独立机器回执；不据单次阶段观察建立整例性能结论。

## 数值解释

同输入white/pial与有序faces不变，所以各morph可按索引比较。沿用 `docs/recon_all/SURFACE_METRICS.md` 既有门槛：area绝对0.001 mm²，其余成熟thickness/curv绝对0.005，另加0.001相对项；该门槛不扩展为未经定义的area.mid/TH3volume整体等效判定。ROI以同脑区名字和九列对应，报告逐列差异，不要求字节相同：3a ROI面积修正官方逐面float32分摊后double累加的定义，旧版先float32顶点舍入；未为此新设或事后放宽阈值。六表格式仍按成熟函数精度写出，保存数组记录该文本已舍入的语义，不冒称未舍入底层值。

## 原软件对应与参考

对应FreeSurfer `mris_place_surface --thickness WHITE PIAL 20 5 OUT`、`--area-map SURFACE OUT`、`--curv-map SURFACE 2 10 OUT`；ROI对应`mris_anatomical_stats -no-th3`。本轮只调用FNIT既有Python/PyTorch实现，原软件命令是定义参照。源与引用沿用项目 `SURFACE_METRICS.md`、成熟anatomical_stats函数说明及[FreeSurfer官方源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。新测试不建立整例等效门槛，不将阶段观测宣称整例提速或五项改动的因果代价。

## 成功实测与报告分析

仅左半球、同一3a自产sub-06的10个输入：8f/3a冷metrics为33.557/33.154秒，暖metrics为30.219/30.547秒；整遍含输出序列化为56.066/55.484秒和52.014/52.908秒。每遍重算并新建ROI缓存。两臂采样显存峰均2,046,820,352字节，188/189样本，最大采样间隔0.710/0.668秒，不能当连续峰。

五个既有形态门槛在四组比较全部通过、越界点0；area.mid/volume只报告差异，不设新等效阈值。六ROI表命名行与九列已舍入文本数值在四组均完全相等；不代表未舍入累加相等。仅此左半球同输入阶段结论，不代表另一半球或原始T1整例等效。本次未复现原整例约25.8秒metrics版本差，原因仍不能由单次观察指定。

仅读取已完成数组的比较不执行影像算子：原双锁等待控制器80035未进入command，按授权SIGTERM取消后interrupted/130、all_tracked_owned_exited=true；其取消回执保留。随后同一冻结compare脚本在comparison_cpu1_v2、隐藏CUDA、CPU库1线程完成报告分析。该分析耗时不纳入任何GPU臂/整例benchmark。新目录不覆盖原等待产物。

本地成功结果目录为 `metrics_roi_sameinput_sub06_v2_results`：计时来源receipts_20261004T1459.json、timing_observation.json；数值来源numerical_receipts_20261004T1507.json、comparison.json、numerical_summary.json；完整现存工具/输入/源库存及输出绑定为evidence_binding.json。中文解释见NUMERICAL_RESULTS.md。

## 函数与具名参数示例

诊断runner的 `verify_inputs(manifest, checkpoint)`：manifest是含files的JSON对象；checkpoint是自产subject路径；校验恰好10项大小和SHA，返回已验证项；`main()`解析下述具名CLI参数，在新输出内调用既有函数。`compare_metrics_roi_stage.main()`的 `--left`、`--right` 是两臂已完成output目录，`--output`必须是新目录；先核验controller终态、输入与同索引空间，再分析保存数组。控制器 `--config` 指向显式JSON配置，stage queue的 `--run` 是含A8f.json/B3a.json的新运行根，`--controller` 是已冻结双锁控制器路径。

```bash
# 示例须由双锁控制器调用，不直接绕过资源准入运行GPU算法。
checkpoint_directory=/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/precision_candidate_3a_whole_sub06_v1/attempt_01/subject
input_manifest_path=/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/metrics_roi_sameinput_sub06_v2/inputs.json
assets_directory=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/assets
output_directory=/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003/metrics_roi_example_new/output
gpu_uuid=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
# 下列argv放入已说明的controller config；source/PYTHONPATH由config绑定。
# replay_metrics_roi_stage.py --checkpoint "$checkpoint_directory" \
#   --input-manifest "$input_manifest_path" --assets "$assets_directory" \
#   --output "$output_directory" --gpu-uuid "$gpu_uuid"
```

## 现有测试及边界

- `test_metrics_roi_pair.py`：Popen内部触发SIGTERM，验证赋值后补转发、等待退出、不启动B臂；本地复核1测试通过。测试对象是run_metrics_roi_pair.py兼容队列，不是v2完整owned-tree队列，模拟cleanup返回，不能冒称已测试真实操作系统进程树回收。
- `test_metrics_roi_controlflow.py`：实际冻结8f和3a环境隐藏CUDA执行，真实thread_budget和StageProfiler/dispatcher，数值算子autospec mock，均通过；明确不是影像算法精度或速度测试。
- v2真实成功回执核验owned后代退出；Popen后取消补发逻辑存在于v2，尚无单独v2取消竞争动态单测，作为当前测试覆盖缺口保留，不用成功未取消运行代替取消测试。
