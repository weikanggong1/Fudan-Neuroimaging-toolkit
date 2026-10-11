# 2026-10-11 编排候选验证

验证基线：`aeabc5aba9341f0358772eb00b1f07eb2cbaf643`。实际候选版本：`46bf4a13fcaba1130b8d61339691b530cb6ad144`；生产模块与 `7636284e140ef7a7b6bb432df88e915481343eda` 相同。冻结测量脚本SHA-256：`994abf6c56a174a979badd866bb16061c91fffe316b0adcd99ee44c07ee9a28c`。

`contracts.public.json` 为96项stdlib unittest通过记录，包含9个模块；pytest专用测试未运行。真实数据为两例自有原始T1重建的完整冻结annotation输入，使用同一输入、资源、GPU和四线程预算，case06按基线→候选、case07按候选→基线。各worker2线程，父调用CUDA预先初始化，三个图谱串行执行，两个半球并行发布。

| 真实输入 | 基线完整annotation组墙钟 / 秒 | 候选完整annotation组墙钟 / 秒 |
|---|---:|---:|
| case06 | 115.793719 | 106.671905 |
| case07 | 90.504133 | 97.943524 |

两例共12份新annotation逐字节一致，变化顶点数均为0，逐标签最小/平均Dice均为1，颜色表和名称相同；orig、smoothwm、sphere.reg的顶点及面顺序逐项相同。见 `comparison.public.json`。本结果只证明本次同输入annotation调度修订没有改变输出。

实际CLI拒绝非法Torch N4并行输入，退出码1且未创建输出；预先初始化CUDA的公开API也拒绝相同组合，当前设备、线程、TF32、autocast和环境标志保持。`preflight.public.json` 另记录SimpleITK2.5.6实际导入及共享库/metadata SHA；未修改共享环境。这是预检失败路径验证，未执行MRI阶段或模拟晚期计算失败。

组墙钟包含私有目录复制、worker启动、计算、I/O、发布和回收，并在退出前同步目标设备。GPU共享，每输入只有一对；两个方向的耗时变化不同，不能称为稳定加速或归因于单个后端。命令主体另含导入、清单校验和基准副本准备，未包含解释器启动。冻结脚本的 `command_scope` 曾错误写明外层另测启动时间，见 `failures.public.json` 的计时说明更正。

候选worker实测TF32启用、CPU/CUDA autocast均关闭，八个半球子步骤均同步目标设备，峰值范围为外层重置或CUDA初始化以来；嵌套Profiler不重置外层峰值。NVML查询可用，但驱动PID到本地任务进程的归属未解决，自身进程树峰值为null。设备总使用量及计算进程之和另列，不可视为本任务峰值，也不构成连续显存上限证明。

原始日志、资源、私有中间影像、重新生成的annotation和脑图只留私有运行目录。仓库只保留typed报告、标量及SHA。脑图生成记录不代替视觉QC或完整质量门。早期缺pytest、旧配方合同预期、未配置SimpleITK目标、输入不全及设置错误均保留。

本次未运行原软件annotation新对照，未进行原始T1整例、138项整例验收或完整科学等效评估。完整EM、GA及最终pial质量门未由本报告批准；默认后端保持native。完整EM的first-line-search路径不能认定为完整EM算法验收。

后续 `e900e725d0df6113d6e01714b00543033faec4ae` 仅修订基准输入预检、副本历史报告清理及计时文字，经过Python语法检查；本次实测仍绑定此前冻结脚本，未替换正在运行的源码。
