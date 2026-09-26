# ProbtrackX 配对验证

[功能、参数和结果解读](../../docs/probtrackx/README.md) · [CPU 聚合指标 JSON](report.public.json) · [GPU 聚合指标 JSON](report.gpu.public.json) · [比较脚本](../../benchmark/probtrackx_validation.py)

## 实验设置

- 2026-09-27 在 `gpucw1` 上运行。原版为 FSL 6.0.7.22 的 `probtrackx2` CPU；FNIT 为 `src/fnit/probtrackx/pipeline.py` SHA-256 `87ae878bc076329712f23bad0980fc14610ecd73f659aa220050397fdafb9afb` 的 CPU 路径。FNIT Python 进程设 `OMP_NUM_THREADS=8`、`MKL_NUM_THREADS=8`。 实现时参考的上游 ptx2 源码是 tag `2608.0`，并非这台服务器所装二进制的逐字节同版本源码。
- 共同输入是一例 UK Biobank dMRI 已有的 FSL BEDPOSTX 后验：104×104×72 网格，50 帧，三纤维。该后验生成日志不可得。ROI 1 是扩散空间 FA>0.5 掩膜内最靠近掩膜中位点的体素中心所成 7 体素球。
- ROI 2 先由 FA 与 dyad 方向选择，初始网络只得到一条非零连接。为检查非零连接数值，再以 FSL seed-to-voxel pilot 图在距 ROI 1 至少 4 体素、FA>0.25 的候选中，选出密度和最大的 7 体素球。**因此后一个网络目标是按 FSL 输出选的，不能当成独立的无偏测试集。** 两个 ROI 不重叠，均与 posterior mask 共网格。
- 两边设置每 seed 体素 200 条轨迹、总步数 400、步长 0.5 mm、曲率阈值 0.2、随机种子 20260927；FNIT `batch_size=256`。`/usr/bin/time` 墙钟计时含载入、追踪、写盘。FSL 进程虽然返回 255，但各预期输出完整且通过 NIfTI 几何、计数与矩阵读取检查；不能据其退出码单独判断测试失败。

同一输入还在 H100 GPU1 上运行了 FSL `probtrackx2_gpu`（报告的二进制标识 `2412.6-dirty`）和 FNIT CUDA；GPU 同时有其他任务驻留，运行时间仅是这次环境下的墙钟观测。两边 seed `waytotal` 均为 1400；网络矩阵的两个非零元素 FSL 为 1194/1061、FNIT 为 1190/1067。GPU 的配对图像与时间见 `report.gpu.public.json`。

报告中的 Pearson 在两张图的非零体素并集上计算；support Dice 衡量全部非零体素；top-tenth Dice 将 FSL 非零体素数的 10% 作为两图相同的 top-*k*。低计数尾部会降低 support Dice，因此同时报告密度和、相关、top-*k* Dice 和原始矩阵。不同随机数流不要求逐体素相同。

## 复现入口与数据边界

`benchmark/probtrackx_validation.py` 对一对 FSL/FNIT seed 与网络目录计算汇总指标和三联图。FSL 用 `-s BEDPOSTX/merged -m BEDPOSTX/nodif_brain_mask.nii.gz -x ROI --opd -P 200 -S 400 --rseed=20260927`；网络模式将 `-x` 换成两行 ROI 列表并增加 `--network`。FNIT 对同一后验使用 `--nsamples 200 --nsteps 400 --rseed 20260927` 及相应 `--seed` 或 `--roi-list`。网络矩阵按 ROI 列表顺序输出。真实 UK Biobank 输入、后验、ROI 和单被试图像均只保存在授权服务器上；公开报告仅含汇总数值。 [UK Biobank 的影像公开使用指引](https://community.ukbiobank.ac.uk/hc/en-gb/articles/16594178325277-Submitting-publications-and-use-of-UK-Biobank-images)要求公开展示被试影像前联系其团队。

## 可公开的合成示例

![合成 posterior 上 FSL 与 FNIT 的 seed-to-voxel 和 ROI 网络密度图](fsl_fnit_synthetic_comparison.png)

这个示例仅使用 [生成脚本](generate_synthetic_posterior.py) 构造的 40×40×40 单纤维管状 posterior（每体素 20 帧）和两个 7 体素 ROI，不含真实被试图像。[一键复现脚本](run_synthetic_comparison.sh) 以每 seed 体素 200 条轨迹、总步数 240、随机种子 20260927 运行 FSL 与 FNIT；[绘图脚本](plot_synthetic_comparison.py) 生成上图。

在 gpucw1 的这次 CPU 运行中，两边 seed-to-voxel `waytotal` 都为 1400，密度图在非零体素并集上的 Pearson *r*=0.9875。FSL 的有向矩阵非零元素为 1369/1368，FNIT 为 1366/1369；完整计数、Dice 和计时见 [合成数据指标 JSON](synthetic_reference.public.json)。小样本计时受程序启动开销影响，不代表大规模速度。

```bash
FSLDIR=/absolute/path/to/fsl bash validation/probtrackx/run_synthetic_comparison.sh /absolute/output/dir
```

运行目录须是新目录；脚本会生成 posterior、运行两套追踪、检查 FSL 输出并写出 JSON 与图。此处图像与真实 UK Biobank 数值验证分别报告。
