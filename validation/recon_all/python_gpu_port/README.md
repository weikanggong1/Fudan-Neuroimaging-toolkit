# recon-all 验证记录

[功能说明与安装](../../../docs/recon_all/README.md) · [阶段索引](../../../docs/recon_all/CONDA_CPP_STAGES.md) · [验收门槛](RELEASE_GATES.md)

主要真实影像为去标识的 `examples/data/sub-01_T1w.nii.gz`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。记录分为三种：冻结官方上游的同输入算子比较、FNIT 自产上游的连续链、从原始 T1 开始的整例。三种结果不得互相替代。

## 当前单 T1 标准流程

入口固定执行 MNI152 非线性变换、拓扑修复、white.preaparc、标准球面与配准、最终 white、四轮 pial 和完整后处理。GPU 的第二次归一化、厚度、white/pial 面积及曲率复用现有 PyTorch 函数；MNI 非线性使用 FP32 CUDA 例外。CPU 路径保留 Conda 源码构建组件。Python pial 仍用于独立同输入验证，未因 GPU 名称替换更快的 C++ 放置路径。[阶段说明](../../../docs/recon_all/CONDA_CPP_STAGES.md)。

主页安装的证据覆盖[固定源码编译安装的 14 个程序](build_full_14_20260929.json)、[98 项资产新安装及续传校验](asset_fresh_install_20260930.json)和[11 项已安装权重校验](weights_verify_20260930.json)。无预装软件的整例物理隔离尚未验证。源码、输入和二进制哈希按各报告的实际执行版本解释，不能把安装或同输入结果当作当前整例结果。

## 本轮性能与精度

[2026-09-30 性能配对](performance_20260930/README.md)记录原始 T1 整例、同输入算子和完整资源采样。基线为 b8cd17b；中间版本 bb28e0c 两例输出及网格均完整，与基线的 138 项诊断全部通过，七张分割图所有分区 Dice=1，最终脑区统计差为零。耗时分别由 5884.96/6080.80 s 变为 5951.02/6055.82 s，未见稳定整例提速。8957e07 的主设备漏传已由 279e09f 修复，失败运行与修复回归单独保留；后续完整版本从原始 T1 和新空目录运行。

[前段数值定位](../../../docs/recon_all/VOLUME_PREFIX_PARITY_20260930.md)已修复 conform 单精度矩阵乘法顺序，两例 orig 的体素与仿射对官方均一致；sub-02 的 195 个灰度首差不再存在。N4 的 ITK 版本、浮点尾差、四组 EM 输入及误差传播保留真实报告，不把全部差异归因于随机性。官方同主机 N4/EM 重放与跨主机差异分别记录，未完成官方整例全面重复性测试。

[两例完整指标与图示](performance_20260930/sub01/full_279e09f_pair/summary.json)及[sub-02](performance_20260930/sub02/full_279e09f_pair/summary.json)包含对官方的严格诊断、分区 Dice、逐脑区误差和双向点到三角面距离。旧版本的整例摘要由现版配对替换；原始阶段参考记录保留其冻结输入边界。

本轮最终 `279e09f` 两例已连续完成；GPU 6303.79 s、CPU 6099.80 s，相对基线慢 7.12%/0.31%，没有观察到整例加速。分割、white/pial 几何与主要脑区统计相对基线无改变；GPU 五项新增严格差异及厚度传播诊断详见性能目录。

严格复现、优化是否引入退化、整体指标等效分别报告。[138 项门槛](RELEASE_GATES.md)不变，整体指标等效阈值尚未正式确认，当前为 not_assessed。不能用平均相关性或文件通过数代替局部检查，也不因未判定总体等效而阻止已通过相应回归的性能优化。

[复现方法](../../../docs/recon_all/BENCHMARK_METHODS.md)列出所有比较脚本的输入、输出、单位、参数与具名示例，以及已初始化 CUDA API、指定 GPU 统计和外层命令计时范围。

## 保留的冻结同输入参考记录

| 范围 | 实测与边界 | 记录 |
| --- | --- | --- |
| N4 与 EM 输入敏感性 | 自产 `nu` 相对官方仅差 2 个体素；对冻结官方 `nu`、`brainmask` 重跑 Conda `mri_em_register` 时 LTA 16/16 元素一致。两处差异在连续链中放大，不能据此断言 N4 是唯一原因 | [N4 源码构建对照](n4_source_probe_20260929.json)、[EM 配对](em_register_input_sensitivity_20260929.json) |
| N4 同主机参考重放 | 在 gpucw1 上官方程序的 `nu` 与归档官方逐体素一致，FNIT 仍差 2 个体素；官方程序换到 headcw 后差 34 个体素 | [同主机与跨主机配对](n4_same_host_replay_20260930.json) |
| 缺陷体积映射 | 真实 T1 冻结输入，16,777,216 个体素全部一致；新 Conda 自编译 `mri_label2vol` | [精度、时间和输入](defects_volume_20260929.json)、[函数说明](../../../docs/recon_all/DEFECTS_VOLUME.md) |
| Talairach affine 子进程 | 冻结真实 SynthStrip 输入，XFM、LTA 与直接路径逐字节一致；需整例显存复核 | [记录](talairach_child_20260929.json) |
| SynthSeg 显存 | 同一 T1、同一 GPU 的候选分割与旧路径逐体素一致，体积 CSV 逐字节一致；独立阶段进程采样 18,450 MiB | [记录](synthseg_memory_20260929.json)、[资源说明](../../../docs/recon_all/GPU_MEMORY.md) |
| SynthSeg CPU 后端 | nodecw10 上 PyTorch 2.5.1 的 MKLDNN 路径段错误；关闭该后端后独立真实 T1 推理及完整 CPU runner 的 SynthSeg 阶段均通过，后续仍在复跑 | [阶段与测试记录](synthseg_cpu_backend_20260930.json) |
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
