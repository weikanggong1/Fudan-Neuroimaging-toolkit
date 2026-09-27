# SynthSR 当前源码验证

[返回主页](../../README.md) · [功能说明](../../docs/synthsr/README.md) · [机器可读报告](report.public.json)

本页记录 2026-09-27 对当前 Nibabel 接口和默认 TF32 设置的真实数据回归。候选运行不调用 FreeSurfer；同一批病例的 FreeSurfer 8.2.0-1 CPU/CUDA 输出作为固定参考。

## 数据与源码绑定

验证集为 12 幅真实临床 T1w，健康对照与抑郁症病例各 6 例。原图、病例标识和生成影像不进入 Git；报告只保存匿名病例编号和聚合值。候选基于提交 `9146b0004468ffc2a30cacad0ab4fa7b6fdb3b48` 的工作树，SynthSR 功能源码树 SHA-256 为 `7b5bc19e1afa806fe8698ea70b6358bacaab23b19877d20d21d2e7c5f3560543`。四臂使用同一个官方 `synthsr_v20_230130.h5`，SHA-256 为 `a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b`。

## 实际命令

12 例各用全新进程，在 H100 GPU1 顺序运行当前默认 TF32：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fnit.cli synthsr \
  --i INPUT_DIR/case-01/T1w.nii.gz \
  --o work/current/case-01__T1w_synthsr.nii.gz \
  --device cuda:0 --threads 8 \
  --weights /path/to/synthsr_v20_230130.h5
```

对应原版命令为：

```bash
mri_synthsr --i INPUT_DIR/case-01/T1w.nii.gz \
  --o case-01__T1w_synthsr.nii.gz --threads 8
```

CPU 单例把候选改为 `--device cpu --cpu`。显存测量命令为：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python validation/model_io_current/measure_peak.py \
  --feature synthsr --input INPUT_DIR/case-01/T1w.nii.gz \
  --weights /path/to/synthsr_v20_230130.h5 --device cuda:0 --threads 8 \
  --output-dir work/peak/synthsr --report work/peak/synthsr.json
```

## 数值与时间

12/12 例的 shape、`uint8` dtype 和数值 affine 均与原版 CPU、CUDA 输出一致；affine 最大绝对差为 0。

| 当前 FNIT GPU 比较对象 | 平均 MAE（灰度级） | 平均 NRMSE | 最低完全相同体素比例 | 最大体素差 |
|---|---:|---:|---:|---:|
| 原版 CPU | 0.0231395 | 0.00158027 | 96.3994% | 9 |
| 原版 CUDA | 0.0231430 | 0.00158043 | 96.3987% | 9 |

当前 CPU 单例与原版 CPU 的完全相同体素比例为 99.99497%，MAE 为 0.0000503，最大差 1。GPU 差异高于 CPU，来自当前要求启用的 TF32：输出几何不变，但靠近量化或组织边界的预测会产生更多 1–9 灰度级变化，因此不能声称逐体素等价。

| 运行 | 病例数 | 完整命令墙钟时间 |
|---|---:|---:|
| FreeSurfer 原版 CPU，固定同批参考 | 12 | 中位数 103.60 s |
| FreeSurfer 原版 CUDA，固定同批参考 | 12 | 中位数 53.27 s |
| FNIT 当前源码 H100 GPU | 12 | 中位数 14.45 s，范围 13.05–16.88 s |
| FNIT 当前源码 CPU | 1 | 31.36 s |

原版与当前候选不是同一时段运行，因此时间用于描述已观察到的完整命令耗时，不计算稳定加速倍数。独占 GPU 单例 Python API 为 12.442 s，完整测量进程 16.72 s；Torch 峰值 allocated 13,780 MiB、reserved 19,074 MiB。

12 例当前默认 TF32 候选的逐例结果、shape、affine、dtype 和输出文件哈希见 [`report.public.json`](report.public.json)。

## 当前公开示意图

下图使用仓库公开 `sub-04` FLAIR，依次显示输入、固定 FreeSurfer 参考和当前默认 TF32 的 FNIT 输出。12 例临床统计与这幅公开图不是同一数据集。公开图两幅输出的 shape、`uint8` 和 affine 一致，完全相同体素比例为 97.8558%，MAE 为 0.02145 灰度级，最大差为 2。

![公开 FLAIR 的 FreeSurfer 与当前 FNIT SynthSR 输出](../../docs/synthsr/figures/synthsr_flair_comparison.png)
