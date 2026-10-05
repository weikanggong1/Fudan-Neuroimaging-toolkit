# FNIT、fMRIPrep 与 DeepPrep：十例正式结果

公开 CC0 OpenNeuro ds001226 v5.0.1，CON01、CON03–CON11；每例配对真实 T1w 与完整180帧BOLD。十例三对均已严格比较完成，`all_threeway_cases_compared`；report和collector输入SHA守卫均为true。FNIT为原冻结0.16.0／`1128bc52c7a0233266e5b8a8d7dc0b382994e676`，这张表不代表本轮新robust参考或当前main重测。

| 耗时：相同十例中位数 | 分钟 | 边界 |
|---|---:|---|
| FNIT 整例 API | **76.74** | imports/CUDA初始化之后至返回并同步 |
| FNIT 完整 Python 进程 | **77.03** | 子进程创建至退出，含初始化与外层验证 |
| DeepPrep25.1.0 | **28.74** | 冷容器创建至退出 |
| 官方 fMRIPrep25.2.4 | **204.81** | 冷容器创建至退出 |

| 一致性：病例内逐位置时间r均值，再取十例中位数 | MNI 2 mm r | NRMSE | 完整91k CIFTI r | NRMSE |
|---|---:|---:|---:|---:|
| FNIT → fMRIPrep | 0.5457 | 22.39% | 0.7249 | 8.90% |
| DeepPrep → fMRIPrep | 0.7135 | 12.70% | 0.5999 | 12.56% |
| FNIT → DeepPrep | 0.4940 | 20.67% | 0.5815 | 12.03% |

NRMSE=RMSE/箭头右端参考RMS，不拟合幅度或偏置。完整180帧、固定MNI脑mask全部228483点、91282灰坐标及21结构；原物理网格与精确CIFTI轴核验，不取交集、不删低相关或零值。r未定义的常数序列另计，仍纳入误差：三对CIFTI未定义位置累计分别为6164、8163、8028；volume均为0。

DeepPrep volume对参考的一致性高于本轮旧FNIT，CIFTI则旧FNIT更高；这不能单凭汇总指标归因于某一个步骤。DeepPrep25.1.0内部使用作者fMRIPrep24开发版fork，参考是官方25.2.4。

时间为各自实际硬件和范围下的观测：参考为nodecw10 Xeon Gold6418H CPU（8进程、OMP4，内部额外MNI2009c配准），两候选为gpucw1 H100与CPU，共享GPU活动记录。FNIT额外计算ICA/AROMA clean分支。GPU等待、资源准备、后QC、比较与绘图不并入容器时间；不同硬件、范围和边界不能解释为通用加速倍数。仍包含CPU、FreeSurfer或其原生节点、Workbench步骤，未建立整体数值等价。

前六例参考容器exit0但初次wrapper文件名QC失败，原容器时钟与后续独立QC分别保留；CON09参考NSS首帧标记保留，不截掉该帧。新旧失败分析、冻结MRI成品及原报告均未覆盖。

完整机器可读数据：[summary.public.json](summary.public.json)、[metrics.csv](metrics.csv)、[timings.csv](timings.csv)、[case_table.csv](case_table.csv)。30份逐例比较报告和CSV按三个pair目录保存。下载94项逐个校验大小和SHA：[DOWNLOAD_SHA256.public.json](DOWNLOAD_SHA256.public.json)。

正式摘要SHA：`6e623312729e643e34bc875e60fdfe78576ba5161ffb23577fab5a663cc75751`。本文件由该摘要生成；脑图使用CON01原完整比较与实际MRI，来源另存。

CON01真实脑图：[FNIT→fMRIPrep](figures/CON01/fnit_fmriprep/volume_consistency.png)、[DeepPrep→fMRIPrep](figures/CON01/deepprep_fmriprep/volume_consistency.png)、[FNIT→DeepPrep](figures/CON01/fnit_deepprep/volume_consistency.png)。各图已验证其comparison SHA与本次十例相同，原完整180帧MRI及固定脑mask均只读；图的逐体素NRMSE与表中pooled NRMSE分别定义。
