# ds004666 单被试结构连接组配对验证

这是 `sub-01/ses-2mm` 同次 T1w 与 DWI 的固定输入实验。[OpenNeuro ds004666](https://github.com/OpenNeuroDatasets/ds004666) 的原始 AP-DWI 为 105 个体积（5 个 b0、50 个 b=1000、50 个 b=2000），使用公开原始 bval/bvec。T1 经 SynthSeg 2.0 分割后，生成同一个 20 区 GM atlas；MRtrix 与 PyTorch 读取相同 atlas NIfTI 和相同 RAS 世界坐标。原始文件的下载地址、字节数、SHA-256 见 [download_manifest.tsv](download_manifest.tsv)，atlas 与 T1 分割哈希见 [report.public.json](report.public.json)。源数据许可和元数据见 [dataset_description.json](dataset_description.json)。

## 验证范围

MRtrix3 3.0.3-103-g026e850d 参考臂使用 Dhollander 响应、默认 lmax=8 MSMT-CSD、mtnormalise、FSL 6.0.7.4 的 5TT、GMWMI 播种、iFOD2/ACT、tcksift2，以及四次 `tck2connectome -symmetric -assignment_radial_search 4`。追踪参数为 `-seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5`，随机种子 0。原 UKB 脚本使用 FreeSurfer+FIRST 5TT；本次 FSL 5TT 是明确的替代。对照的 PyTorch 版本使用 lmax=4 自拟响应/FOD、离散方向的双向概率追踪、二值 GM/WM 终止、近似 SIFT2，以及同样的矩阵赋值规则，分别用种子 0、1、2 各尝试 10,000 次。均传入已有 T1 分割和单位 RAS 世界变换，自动分割/配准不在这次计时和数值比较内。

公开处理后 DWI 没有与之对应的 eddy-rotated bvec 文件，因此此次数值比较使用**同一原始 AP-DWI 及其原始梯度**。它检验算法在同输入下的结果，不能代表代码接口要求的已校正 UKB DWI 结果，也不能验证 TOPUP/eddy。示例图使用处理后 b0 检查 T1/DWI 几何；这不改变数值比较的输入。对照也没有复现原仓库的皮层加 Tian atlas。两臂的 FA 估计与沿纤维采样方法不同。

## 输出一致性

下表为 PyTorch 种子 0 对固定 MRtrix FSL-5TT ACT 参考；20×20 矩阵的严格上三角排除对角线。count/FBC 在全部 190 条边计算相关与误差；长度/FA 只在两臂 count 都非零的边计算数值指标。归一化 MAE 除以被比较边上参考的非零值绝对均值。Dice 比较各矩阵非零边集合。[机器可读报告](report.public.json)记录全部 3 个种子、Spearman、RMSE、自连接与文件哈希；旁边的五组 CSV 保留原始 20×20 输出。

| 矩阵 | Pearson r | 归一化 MAE | 非零边 Dice | 数值比较边数 |
|---|---:|---:|---:|---:|
| 流线 count | 0.689 | 0.595 | 0.549 | 190 |
| SIFT2 FBC | 0.739 | 2.035 | 0.549 | 190 |
| 加权 mean length | 0.878 | 0.292 | 0.549 | 31 |
| 加权 mean FA | 0.651 | 0.131 | 0.549 | 31 |

MRtrix 在 10,000 次播种中生成 3,021 条流线，其中 2,915 条获得 atlas 双端分配；PyTorch 种子 0 生成并分配 6,319 条。三个 PyTorch 种子两两比较的平均 Pearson r 为 count 0.994、FBC 0.993、length 0.917、FA 0.816；平均连接支持 Dice 0.862。这说明本实现自身的矩阵在三次采样中相对稳定，**不表示与 MRtrix 输出一致**。

把 MRtrix 的 `-seed_gmwmi` 单独换成 PyTorch 同款二值边界 mask 的 `-seed_image` 后，MRtrix 获得 3,348 条 atlas 分配；与 PyTorch 种子 0 的 count r=0.669、Dice=0.586，FBC r=0.735。此受控臂用于定位播种差异，不是原脚本条件。

只把 PyTorch 四处 SH 阶次从 lmax=4 提至 lmax=8 的[敏感性诊断](lmax8_diagnostic.public.json)，count/FBC Pearson 升至 0.765/0.817，但 count 归一化 MAE 升至 0.685、连接支持 Dice 降至 0.492，FBC 归一化 MAE 升至 2.315。提高 SH 阶次未解决输出差距；该诊断不替代正式 lmax=4 三种子结果。

![相同 DWI 与 atlas 的四类结构连接矩阵及差值](../../../docs/connectome/figures/connectome_comparison.png)

![矩阵相关性、支持和归一化误差](../../../docs/connectome/figures/connectome_metrics.png)

## 默认自动 T1 接口检查

同一公开 T1w 与原始 AP-DWI 另以 100 次种子尝试跑通默认自动 SynthSeg 与 TorchFLIRT（不传分割、atlas 或变换）：内部调用 35.11 秒，接受 62 条流线；20×20 四矩阵均有限且对称。自动组织前景与公开 DWI 脑掩膜 Dice 0.940，和上述固定世界仿射的 SynthSeg 前景 Dice 0.958。此[接口检查记录](auto_interface_smoke.public.json)只证明自动链能运行；原始 DWI 仍不符合生产接口的预处理条件，也不参与上述数值对照。

## 运行时间

计时均在 gpucw1 上，PyTorch 2.5.1 使用 NVIDIA H100、float32 和 TF32。PyTorch 的三次端到端调用（已有分割和单位变换，包含 NIfTI 读取、FOD、追踪、近似 SIFT2、四张 CSV 与中间 NIfTI 写出）为 **25.41、25.24、27.38 秒**；峰值进程 RSS 分别约 1.06、1.05、1.04 GB，未记录 GPU 峰值显存。MRtrix 各阶段是独立进程的墙钟时间：Dhollander 22.88 秒，MSMT-CSD 182.98 秒，mtnormalise 3.25 秒，FSL 5TT 859.56 秒，iFOD2/ACT 13.55 秒，tcksift2 20.75 秒，四张矩阵赋值合计 0.32 秒。MRtrix 的 FSL 5TT 包含 T1 解剖处理，PyTorch 计时则复用已有 SynthSeg 分割；这些数字**不能直接构成完整流程的加速倍数**。10,000 次播种的实测不能外推到默认 10,000,000 次的耗时或内存。

同一组 3,021 条 MRtrix 真实流线、SIFT2 权重、长度、FA 和 atlas 输入下，另做单独的矩阵赋值逐元素比较：2,915 条双端分配在两臂一致，count 的 400/400 元素完全一致；FBC、mean length、mean FA 的最大绝对误差分别为 `8.99e-6`、`5.63e-6 mm`、`2.96e-8`。H100 上 10 次计时的中位数分别为 MRtrix 四个进程 `0.172 s`（含启动与 I/O）和本包已驻留 GPU 计算 `0.0746 s`。该[真实轨迹赋值报告](../ds004666_fsl_act_real_tracks_assignment_report.json)把追踪/FOD 差异从矩阵映射误差中分离，计时不代表完整流程加速。

![固定真实流线的矩阵赋值对照](../../../docs/connectome/figures/ds004666_real_assignment_matrices.png)

![真实流线矩阵赋值逐元素误差](../../../docs/connectome/figures/ds004666_real_assignment_metrics.png)

## 复查

[report.public.json](report.public.json) 记录各 CSV 的 SHA-256、每种子结果、归一化定义及各阶段时间。数值运行所用远端快照的 8 个 connectome `.py` 文件与合并到 FNIT 0.12.0 后的本地源码 SHA-256 逐一相同，哈希也在报告中。用仓库脚本可在这些已公开的小矩阵上重新计算指标：

```bash
python tools/compare_connectome_matrices.py \
  --reference-dir validation/connectome/ds004666/mrtrix_fsl5tt_act \
  --candidate-dir validation/connectome/ds004666/pytorch_seed_0 \
  --candidate-dir validation/connectome/ds004666/pytorch_seed_1 \
  --candidate-dir validation/connectome/ds004666/pytorch_seed_2
```

**结论：** 相同输入下的完整追踪与 SIFT2 矩阵差异仍大，未达到“同输入输出一致”的推送条件。矩阵赋值阶段的一致不能代替完整流程一致。
