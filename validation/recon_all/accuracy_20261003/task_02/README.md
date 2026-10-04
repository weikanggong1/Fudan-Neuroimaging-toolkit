# 任务2：前段精度诊断与修复

本页所有前段结果对应历史816输入/该任务writer修复，未重标为3a或本集成分支的新整例；当前3a整例结果见主精度报告。

当前完成状态（2026-10-03 05:28 UTC）：十例完整前段成对运行20/20成功，N4交叉2/2成功，sub-04 AB/BA重复4/4成功。整体recon-all等效`not_assessed`，生产修改仅`c24b3d6d`的SynthStrip保存dtype修复。

## 当前交付

- `cohort_prefix_collected.json` / `cohort_prefix_times.csv`：完整10例输入/程序/权重/模板/实际runtime SHA、体积/空间/XFM/LTA比较、阶段时间、采样显存。首例8d相关模块与816SHA一致，其他九例正式816依赖路径，真实来源逐次记录。
- `prefix_repeat_collected.json` / `prefix_repeat_times.csv`：sub-04四次exit0，真实输出比较全部0差；AB候选慢3.062秒、BA快1.212秒，当前不能宣称稳定前段加速。
- `n4_cross.json`：现代编译器换旧库仍与现代四字段精确一致，未复现官方尾差；旧GCC205秒仅诊断，生产N4未变。
- `completion_receipts.json`：完整回执、已退出PID、日志SHA/尾部、无Traceback。nohup父进程退出码未采集，子运行码与完整回执证明计算完成，未虚构父退出码。
- `synthstrip_dtype_brains.json`及中文文档脑图：两公开真实图像保存前后绝对差0，原始影像不入Git。

完整接口、分步骤/总耗时、现有FP32例外和参考见`docs/recon_all/SYNTHSTRIP_MGH_DTYPE.md`。固定基线公开816e5610；整体原始T1→Talairach回放使用GPU0 H100/线程4/共享锁，默认TF32及已有阶段FP32例外不变。

## 诊断与可复现脚本

`frozen.json`为旧两例产物重新读取：orig/SynthStrip值与官方一致，SynthStrip旧保存float32而官方uint8，nu差2和29体素；不冒充十例。`synthstrip_writer.json`为真实冻结图像AB/BA保存回归。`n4_diagnostic.json`、`n4_environment.json`与`refinement.json`分别记录同输入ITK版本、工具链/官方当前float、小矩阵算子排除。`prepare_n4_diagnostic.py`从本项目公开816的`tools/n4_itk/n4_itk.cpp`校验SHA生成字段导出诊断；不复制原软件生产源码/二进制。

`cohort_prefix.py`、`repeat_prefix_timing.py`运行前段并释放每case共享锁；`collect_prefix_results.py`、`collect_completion.py`只读取产物。`inspect_frozen.py`、`verify_synthstrip_writer.py`、`run_n4_diagnostic.py`、`run_n4_environment.py`、`run_n4_cross.py`固定使用授权gongwk@gpucw1既有真实数据。隔离官方程序只用于benchmark，不进入FNIT生产链。影像、raw导出、权重、license均留服务器。

服务器诊断目录：`/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic`；完整前段：`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_02/cohort_prefix`。

## 历史交接快照

`phase1_checkpoint.json`/`cohort_prefix_snapshot.json`为3/10交接；`phase2_checkpoint.json`为7/10交接；`progress_8_pairs.json`/`n4_cross_snapshot.json`为8/10时的只读快照。其pending描述仅对应历史时间，当前以完整回执为准。完整空目录整例、138项严格诊断和最终统计由协调者统一完成。
