# 任务2：前段精度诊断与修复

基线：公开 `816e5610417a4c587caf321049438a9554139016`。本任务工作树中唯一生产修复是 `input_talairach_chain.py` 的 SynthStrip MGH writer；GPU网络、conform与N4计算未改变。具体接口与中文调用见 `docs/recon_all/SYNTHSTRIP_MGH_DTYPE.md`。

## 已复验

- `frozen.json`：本轮读取旧两例真实阶段。orig 与官方体素/空间一致；SynthStrip 数值一致但候选float32、官方uint8；nu分别差2和29体素。报告明确标记历史产物，不冒充新十例整例。
- `synthstrip_writer.json`：同冻结真实SynthStrip输出的AB/BA保存回归。修复后两例uint8与官方数值/空间相同，拒绝不能无损表示的整数转换；写盘0.314–0.328秒，对照0.387–0.417秒。文件约减半。此计时只包括writer，不能与完整GPU前段相加。
- `n4_diagnostic.json`：同真实官方orig、同几何、拟合1/重建4线程。相同Conda编译器ITK4/5的lattice、logfield、expfield和corrected逐元素一致；ITK5重复两次也一致。含诊断导出的墙钟为121.262/128.348/121.664秒（5/4/5）。这排除了这两个安装的ITK版本差异本身。
- `n4_environment.json`：官方FS8.2程序SHA `9d77c96f...`重新执行同orig float，与历史官方全bit一致。独立公开源码GCC4.8.5/ITK4/floatmask诊断也与本轮官方全bit一致；与Conda首差已在lattice（1316/1331,max5.66e-7）。旧环境204.96秒、官方208.31秒，相比Conda诊断明显慢，**只保留诊断，不接默认生产**。编译器、模板和已编译库仍用交叉试验分开核验，不凭版本字符串归因。

## 复现与文件结构

脚本固定使用用户授权 `gongwk@gpucw1` 的已有真实检查点。影像、raw导出、权重、许可证均留在服务器，不提交Git。所有benchmark使用 `/tmp/fnit-shared-benchmark.lock`；N4拟合1线程、重建4线程；GPU前段使用CUDA0、CPU线程4、默认TF32和现有逐阶段FP32例外。

`inspect_frozen.py` 只读旧阶段，不含运行计时。`verify_synthstrip_writer.py` 在相同真实网络输出上AB/BA。`run_n4_diagnostic.py` 对相同公开源码派生的诊断程序分别链接已存在的ITK4与5。`run_n4_environment.py` 仅在隔离benchmark调用官方程序，不作为生产后端。`run_n4_cross.py` 每个variant单独取锁并释放。`cohort_prefix.py` 对固定十例原始T1运行完整前段AB/BA，保留失败，不替换被试，并按同刻父子进程采样显存。

`prepare_n4_diagnostic.py` 生成的诊断源 `n4_diagnostic.cpp` 从公开基线 `tools/n4_itk/n4_itk.cpp` 生成，添加raw字段导出与ITK4编译兼容，不包含FS原软件代码或二进制。生成的 `gcc48/n4_diagnostic.cpp` 为同一诊断的C++11/floatmask版本；它只用于定位旧工具链，不能直接安装到生产。

服务器隔离目录：`/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic`。新十例前段：`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_02/cohort_prefix`。完整网络与导入模块使用协调者SHA验证的 `baseline_runtime_816e5610/src`；首例此前已运行的8d750e2严格保留真实来源，各前段/Synth模块SHA与816一致。

基线入口在部署时由下面命令生成，不在本报告目录复制整份旧生产源码：

```bash
git show 816e5610417a4c587caf321049438a9554139016:src/fnit/recon_all/input_talairach_chain.py \
  > /tmp/input_talairach_chain_baseline.py
```

执行时显式指定正式基线PYTHONPATH；每次worker记录真实模块路径及SHA、固定seed=0、资源SHA、前向设备/dtype/TF32/autocast和GPUUUID。`run_monitored.py`采样1秒，失败样本不填0，报告采样间隔和同刻父子占用；采样最大值不代表连续峰值。

## 当前与最终验收

新十例阶段报告完成后与本目录同步。此目录的前段AB/BA验证不是十例完整recon-all；完整空目录整例、138项严格诊断、脑图和最终统计由协调者统一完成。整体等效保持 `not_assessed`。
