# ICA、ICA-AROMA 和混杂回归 CPU 对照准备

2026-10-04 核对起点：`cc9402734faeba93b3a13c29932fa1392eaccf62`。此页记录功能覆盖与验证入口；完整 PICA／AROMA 和混杂 CPU 官方精度已汇总在[说明文档](../../../docs/fmri/CPU_ICA_BENCHMARK_20261004.md)与本目录的两个机器可读报告。服务器固定入口、索引和现有源码已读取；既有运行目录和冻结源码保持各自身份。

| 功能 | 本轮覆盖 | 官方或独立参照 | 当前输入 |
|---|---|---|---|
| `decompose_spatial_ica` | 自动 Laplace PPCA、固定成分数；`pow3` 对称 ICA；500 次上限、0.001 收敛阈值、种子 0；后验阈值 0.5/0.75；分块输出、收敛状态、残差缩放、频谱 | 原 FSL MELODIC 2601.1，自动/固定 `--dim`，相同 `--dimest=lap --nl=pow3 --eps=0.001 --maxit=500 --seed=0 --Ostats --nobet` | 一例真实完整 490 帧，88×88×64；同一官方 FEAT 高通 BOLD 和脑掩膜 |
| `run_melodic_bids` | BIDS 来源链接、全部成分图/混合矩阵/JSON、overwrite 与来源实体校验 | BIDS 输出契约；分解精度仍由上述同内核官方对照验证 | 源数据集存在时复用全帧 derivatives；未核实目录的项目不冒充已执行 |
| 运动相关 | 同一 90% 帧抽样序列、1000 次抽样、当前/导数/前后移位，直接与平方相关 | 原作者 `feature_time_series`，种子 0 | 原 490×95 mixing 和原 490×6 运动参数；另检查路径与内存矩阵输入 |
| 高频比例 | 半功率频率及 0.01 Hz 下限 | 原作者 `feature_frequency` | 原 MELODIC FTmix |
| 空间比例 | edge、CSF、空成分、原 MNI 掩膜及阈值图 | 原作者 `feature_spatial`，隔离调用原 FSL roi/maths/stats | 原 91×109×91×95 阈值 IC 图；原三张 MNI 2 mm 掩膜 |
| 噪声分类 | 固定超平面、CSF/高频阈值、零起始返回 | 原作者 `classification`；使用同一四特征 | 同分解的全部 95 成分 |
| IC 回归 | nonaggr、aggr、空噪声、重复索引、参数校验；保留均值和默认强度掩膜 | 原 `fsl_regfilt`，1 起始噪声编号 | 全体积完整 490 帧及实际官方噪声集 |
| 混杂回归 | 单独二次趋势、motion6/12/24、WM/CSF、global、全部混杂、0.01–0.1 Hz 联合带通、秩退化与单位不变性 | 独立 NumPy float64 SVD；不称 MATLAB/AFNI 实测 | 完整真实 AROMA BOLD；同网格真实脑/WM/CSF 掩膜与运动参数 |
| `run_aroma_pipeline` | 自产 PICA→原生或 MNI 分类→两模式回归→可选混杂；收敛及变换/掩膜契约 | 同输入阶段官方对照；端到端由协调者统一验证 | 本任务阶段产物，不读取官方结果帮助生产估计 |

PICA 当前公开接口读取 NIfTI 路径，非线性固定为 `pow3`；这两项不是可切换参数。AROMA 的 mixing、FTmix 和 motion 支持路径或 NumPy 矩阵。ICA 内存 BOLD、其它对比函数、其它 ICA 模式和额外 CLI 不属于现有实现，不能写成已支持功能。

现场确认 FSL 6.0.7.22、MELODIC 2601.1 的 `h46763b2_0` 包及 `fsl_regfilt/fslroi/fslmaths/fslstats` 可读。原作者文件从已有独立 benchmark 外部目录读取。Python 3 兼容文件只删除 `future` 导入，并将 `past.utils.old_div` 替换为 `operator.truediv`；原文件和兼容文件均单独记录 SHA。三张掩膜已记录大小和 SHA。该外部目录没有许可文件，本任务不复制发布原源码或资源。

新测量使用 [benchmark.py](benchmark.py)。私有 manifest 留在仓库外，包含匿名 dataset alias 和实际输入路径；公开报告只有输入、源码和适配器 SHA、形状、核组、线程和聚合数值。调用示例：

```bash
# 协调者给出 FNIT_ENV_PREFIX、FNIT_WORKSPACE、PRIVATE_MANIFEST、CPU_SET。
# 线程环境必须在解释器启动前设定；1/8 线程分别用独立空输出目录。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
PYTHONPATH="$FNIT_WORKSPACE/src" taskset -c "$CPU_SET" "$FNIT_ENV_PREFIX/bin/python" \
  "$FNIT_WORKSPACE/validation/fmri_cpu_20261004/task02_ica_aroma/benchmark.py" \
  --manifest "$PRIVATE_MANIFEST" --dataset real_run_01 --output-dir "$NEW_OUTPUT_DIR" \
  --phase pica --backend fnit --device cpu --threads 8

# 原 MELODIC 阶段。原程序只能在独立 benchmark 中调用。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
taskset -c "$CPU_SET" "$FNIT_ENV_PREFIX/bin/python" \
  "$FNIT_WORKSPACE/validation/fmri_cpu_20261004/task02_ica_aroma/benchmark.py" \
  --manifest "$PRIVATE_MANIFEST" --dataset real_run_01 --output-dir "$NEW_OFFICIAL_OUTPUT_DIR" \
  --phase pica --backend official --threads 8 --trace-original
```

阶段还包括 `aroma-features`、`denoise --mode nonaggr/aggr`、`confounds --case drift/motion6/motion12/motion24/tissue/global/all/all_bandpass` 以及 BIDS 包装。混杂独立参照使用 `--backend independent`。`--n-components 95` 固定阶数，省略则自动定阶；两者均保留完整混合模型。

原 FSL launcher 的退出行为必须由原执行程序退出证据核对。适配器复用项目独立验证工具；启用 trace 的耗时包含 trace 成本，公开报告明确标注。FNIT API 时间包含读写和输出检查、排除解释器导入及哈希；协调者另记录外部进程端到端墙钟。带 HTML 的原 MELODIC 使用 `--official-report` 单列，避免与核心输出的时间混为一项。

预计首轮全功能约 20–60 分钟，实际取决于核组、文件系统和自动定阶；这是调度估计。静态热点是 PICA 多遍 float64 voxel block、每成分 EM、AROMA 1000 次抽样的矩阵计算、IC/混杂回归反复转换与转置。优化候选先按同线程 CPU profile 决定；CUDA 分支需要原/候选相同输入和配置的独立回归。

参考：[FSL MELODIC 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)、[MELODIC 原代码](https://git.fmrib.ox.ac.uk/fsl/melodic)、[ICA-AROMA 原代码](https://github.com/maartenmennes/ICA-AROMA)、[Pruim 等 2015](https://doi.org/10.1016/j.neuroimage.2015.02.064)、[Beckmann 与 Smith 2004](https://doi.org/10.1109/TMI.2003.822821)。

最新 GPU 混杂回归：十个完整调用、六组全图核对完成；默认数据与影像头一致。见 [聚合报告](gpu_projection_20261005.public.json)。PICA 非默认阈值与完整 BIDS 输出的五项调用及数值核对已完成，见 [聚合报告](pica_features_20261005.public.json)。
