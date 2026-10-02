# 历史官方参考与本轮重复结果

## 已核实的事实

[只读审计记录](old_vs_fresh_reference_audit.json)覆盖历史官方文件、当前三项输入、当前官方源码的大小、SHA、mtime，以及旧/新输出的完整网格。四个结构各自的 native、HR 共 **8 对输出，shape 和 affine 均逐位一致**。右 AAA 的 atlas ID 为 7010；在相同网格上，旧官方 native/HR 有 **0/1** 个硬标签体素，本轮官方有 **24/626** 个。这是标签内容变化，不能用网格相位解释。

服务器 `plus/subjects/fs_sub01/mri` 当前不存在。已恢复 `/tmp/fnit_official_gpucw1_full_{thal,hippo}_20260929.py`、对应 `.log` 及 launcher `.sh`。驱动声明 `set_thread_count(4)`，使用与本轮相同的 `reconall_reference_gpucw1/fs_sub01/mri/{norm,aseg,wmparc}.mgz` 路径；launcher 选择实际 FreeSurfer 8.2 的 `fspython`，日志显示 FreeSurfer 8.2.0-1 加载并正常完成。不能将其当作 threads=1 的试验。

## 尚无法恢复的生成条件

旧驱动和日志没有保存当时输入、导入的 Python/native library 或生成输出的 SHA。当前旧参考文件的 mtime 早于恢复的同名驱动和日志，不能由文件名及路径认定这些日志就是当前保存文件的产出者。历史输入路径可核实，**历史输入内容、源码/二进制身份及输出对应关系未形成哈希闭环**。launcher 未显式记录 BLAS/OMP 线程上限；这也不能单独证明其造成了标签差异。

目录名中的 `threads1`、当前源码 mtime、官方版本加载消息均不能补足缺失的生成清单。旧/新差异的原因目前记为 **unknown**，不归因于随机 seed、线程变化或 atlas 版本。当前 atlas 与安装官方资源 12/12 文件完全一致，见[资源身份审计](atlas_identity_comparison.json)。

本轮精度、重复性和可接受范围采用重新执行、有输入/源码/资源/输出校验的三次官方结果。旧指标保留为历史观测，不纳入官方随机波动范围；先前根据旧官方 AAA 硬标签为空作出的接受解释撤销。
