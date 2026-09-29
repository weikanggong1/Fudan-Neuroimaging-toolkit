# recon-all 验证记录

[功能说明与安装](../../../docs/recon_all/README.md) · [阶段索引](../../../docs/recon_all/CONDA_CPP_STAGES.md) · [验收门槛](RELEASE_GATES.md)

主要真实影像为去标识的 `examples/data/sub-01_T1w.nii.gz`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。记录分为三种：冻结官方上游的同输入算子比较、FNIT 自产上游的连续链、从原始 T1 开始的整例。三种结果不得互相替代。

## 当前单 T1 标准流程

入口已固定执行拓扑修复、真实 `white.preaparc`、标准球面与配准、最终 white、Conda 源码构建的四轮 pial 以及体积/顶点后处理，不再通过多个开关组合近似表面。Python pial 保留为可单独调用的同输入验证函数。[同一自产输入的双引擎比较](native_pial_candidate_20260929.json)给出输出差异与耗时。文件完整性与数值验收是运行报告中的不同字段。必要的 11 个原生程序已在 2026-09-29 的新主页 Conda 环境从固定源码归档构建并安装；全新联网检出与无预装软件环境中的整例隔离验证仍待完成。[安装说明](../../../docs/recon_all/CONDA_CPP_BUILD.md)。

旧版 v5 开发链从原始 T1 完成了 59 个阶段，双侧 white/pial 的[独立网格检查](mesh_validation_20260929.json)通过，但因使用早期代码快照缺少固定清单中的 18 项，运行报告为 `incomplete`，耗时 11,651 秒。与归档官方整例严格比较仅 5/138 项通过，通过项均是 MRI；它不能代表现版交付。新版 v6 主页安装链、固定源码快照的 v7 标准链及另一例真实 T1 仍在运行；完成后记录当前 138 项整例结果。

## 2026-09-29 的现版同输入记录

| 范围 | 实测与边界 | 记录 |
| --- | --- | --- |
| 缺陷体积映射 | 真实 T1 冻结输入，16,777,216 个体素全部一致；新 Conda 自编译 `mri_label2vol` | [精度、时间和输入](defects_volume_20260929.json)、[函数说明](../../../docs/recon_all/DEFECTS_VOLUME.md) |
| Talairach affine 子进程 | 冻结真实 SynthStrip 输入，XFM、LTA 与直接路径逐字节一致；需整例显存复核 | [记录](talairach_child_20260929.json) |
| SynthSeg 显存 | 同一 T1、同一 GPU 的候选分割与旧路径逐体素一致，体积 CSV 逐字节一致；独立阶段进程采样 18,450 MiB | [记录](synthseg_memory_20260929.json)、[资源说明](../../../docs/recon_all/GPU_MEMORY.md) |
| BA/VPnl 注释 | 修正相同统计值时的标签顺序后，六张注释的双侧顶点编码全部与冻结参考一致 | [记录](exvivo_wiring_20260929.json)、[注释说明](LABEL2ANNOT.md) |
| 图谱曲率 | 冻结球面上的 `avg_curv` 相关性超过 0.999999999999；四张曲率附图超过 0.999999999999999 | [函数说明及计时](../../../docs/recon_all/CURVATURE_OUTPUTS.md) |
| Jacobian、灰白对比、SNR | 双侧 Jacobian 相关性超过 0.9999999999999；百分比图逐顶点一致；70 行 SNR 数据行一致 | [记录](surface_metrics_wiring_20260929.json)、[函数说明](../../../docs/recon_all/SURFACE_EXTRA_METRICS.md) |

以上多数使用冻结官方上游，不能当作 FNIT 自产上游的整例吻合。`white.preaparc` 的候选输入首差和最终 white 冻结输入下的 57 个大于 0.1 mm 的顶点仍须在现版整例中追踪；见[预白质](../../../docs/recon_all/WHITE_PREAPARC_CONDA_CHAIN.md)与[最终 white](../../../docs/recon_all/FINAL_WHITE_CONDA.md)。[真实 T1 的非零自相交输入](intersection_positive_real_20260929.json)经 Conda 源码构建程序修复到零，相同有序面；尚无对应的官方同输入修复输出。

## 比较方法

[比较器](compare_complete_subject.py)与生产入口共用[138 项输出清单](../../../src/fnit/recon_all/expected_outputs.py)：39 张 MRI/其他体积图、18 个表面、46 张顶点图、12 张注释和 23 个统计文件。官方存档自比是 138/138；人为把厚度或面积改动 0.02 单位时仅对应输出失败，这只证明比较器能检出差异。

```bash
python validation/recon_all/python_gpu_port/compare_complete_subject.py \
  /data/reference-sub01 /data/subjects/sub01 \
  --report /data/sub01-comparison.json
```

完整门槛见[RELEASE_GATES.md](RELEASE_GATES.md)。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
