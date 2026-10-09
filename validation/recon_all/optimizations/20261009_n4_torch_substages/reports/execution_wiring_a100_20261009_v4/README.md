# 完整 N4 显式缓存隔离接线

## 实际版本与范围

`3f021ec3`加[source_v4_patch_manifest.json](source/source_v4_patch_manifest.json)列出的7个补丁；全部生产接线源码与已通过37项测试的v3逐SHA相同。v4只修正诊断中rawavg原始网格与conform掩膜的混用。完整693个Python文件SHA见[v4_frozen_source.json](source/v4_frozen_source.json)。测试3.85s见[unit_v3.xml](logs/unit_v3.xml)，未把契约fixture当成真实benchmark。

数据为已有公开ds000114 sub-06/sub-07原始T1；每例在新空目录连续到nu，没有参考复制或手补检查点。A100 GPU0、CPU0–3、4线程、TF32默认，保留SynthStrip/Talairach已验证FP32例外，不启用半精度。实际输入、权重、资产、原生参考程序SHA及完整200轮worker报告见[raw_isolated/summary.json](raw_isolated/summary.json)。

## 结果

| 病例 | 到nu完整API，s | N4含exec，s | 旧完整in-process到nu，s | 新增体素差 |
|---|---:|---:|---:|---|
| sub-06 | 71.881 | 13.556 | 152.322 | rawavg/orig/strip/nu0/nu全0 |
| sub-07 | 76.739 | 11.962 | 150.198 | rawavg/orig/strip/nu0/nu全0 |

shape/affine/dtype及两种LTA矩阵相同。包括校验、两链和诊断的研究墙钟202.524s；不是recon-all整例。旧完整in-process来自同硬件/线程前组，当前GPU0有其他任务14,658MiB驻留，时间不能视为同期ABBA稳定吞吐。

相对native的nu0仍为4016/3259体素全+1，nu为3709/9230体素不同、max3。严格复现未通过；隔离执行没有新增退化；整体脑区/表面指标等效未判定。原生默认保持。当前只把`n4_backend="torch", n4_execution="isolated"`作为显式候选。

子allocated/reserved各1,295,297,024/1,384,120,320字节。目标卡观测峰24,920,457,216字节，全部计算进程上界24,899,485,696字节；tree归属未解决为null，实际最大间隔7.827s、1次SMI查询超时。这些包含共享驻留，不是精确FNIT父子峰或整例20GB验收。

## 诊断失败与重跑

[diagnostic_failed_v3/summary.json](diagnostic_failed_v3/summary.json)保留原v3失败：sub-06原始链API已写出，随后诊断将256³conform脑mask用于rawavg原始网格而IndexError。v3不标为两例完成。v4只修诊断，rawavg仅按自身网格比较全体积；重新生成两例新目录，完成全部体素/几何/LTA检查。失败日志[logs/wiring_raw_v3.log](logs/wiring_raw_v3.log)与新日志[logs/wiring_raw_v4.log](logs/wiring_raw_v4.log)均保留。

## 收据与复现

16份白名单机器报告/日志通过目录与主机去敏感公开，1752个数值/布尔/null字段与原件不变，影像/权重/许可证均未打包。原始私有包SHA`963f2a13fb4297d15fea790975912ff962ff37881a6654e596e6534bbe64ae5f`。原件/公开SHA映射见[public_export_manifest.json](public_export_manifest.json)、[publication_receipt.json](publication_receipt.json)。

复现脚本`benchmark_input_n4_execution.py`在本优化目录上两级：`--config`绑定原始case/input/SHA/公开来源，`--previous-raw-pair`提供计算完成后的既有诊断参考，`--weights/--assets/--native`为声明资源，`--output`新目录，`--profiling-module`固定采样器，`--device cuda:0 --threads 4 --seed 1729 --code-version <frozen-label>`。GPU API边界同步；API不含父初始导入，子exec导入/退出包含；官方/旧结果不进入计算路径。

接口、全部参数/空间与具名示例见[输入链](../../../../../../docs/recon_all/INPUT_N4_CHAIN.md)、[缓存worker](../../../../../../docs/recon_all/N4_CACHED_WORKER.md)。纯python-gpu全流程仍未开放，不把N4阶段迁移称为整个recon-all已移除原生程序。
