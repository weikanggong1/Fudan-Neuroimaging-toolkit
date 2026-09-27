# WMH-SynthSeg 当前源码验证

[返回主页](../../README.md) · [功能说明](../../docs/wmh_synthseg/README.md) · [机器可读报告](report.public.json)

本页记录 2026-09-27 对当前 Surfa→Nibabel 版本的真实数据回归。候选程序只导入 FNIT、PyTorch、Nibabel、NumPy 和 SciPy；推理时没有调用 FreeSurfer。FreeSurfer 8.2.0-1 的 `mri_WMHsynthseg` 输出只作为固定参考。

## 数据、实现与输出

输入为仓库 `examples/wmh_data/` 中三幅公开 FLAIR 衍生图，来源是 OpenNeuro ds003592 的 `sub-02`、`sub-03` 和 `sub-04`。这三幅图只把脑外置零，来源与 SHA-256 见[示例说明](../../examples/WMH.md)。它们没有人工 WMH 真值；本实验衡量 FNIT 对原版程序的复现程度，不衡量病灶检出准确率。

候选运行绑定到基础提交 `9146b0004468ffc2a30cacad0ab4fa7b6fdb3b48` 上的工作树。WMH 功能源码树 SHA-256 为 `39cd9b4c380a93370c1244e72eea4c3eaa9f277dd8d70ef44a4ec2293e6a5860`；逐文件哈希保存在报告的 `candidate_source.files`。

每例输出一幅 `float32` 标签图、一幅同网格 `float32` WMH 概率图和软体积 CSV。三例输出 shape 分别为 `192×216×126`、`192×216×120`、`192×216×126`；候选与参考的数值 affine 均完全相同。

## 实际命令

三例均用全新进程、8 个 CPU 线程和 H100 GPU1 顺序运行，保留默认 TF32：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fnit.cli wmh-synthseg \
  --i examples/wmh_data/sub-02_FLAIR.nii.gz \
  --o work/current/sub-02_seg.nii.gz \
  --csv_vols work/current/sub-02_volumes.csv \
  --device cuda:0 --threads 8 --crop --save_lesion_probabilities \
  --weights /path/to/official/weights
```

CPU 检查把 `--device` 改为 `cpu`，其余参数和 `sub-02` 输入不变。显存记录使用：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python validation/model_io_current/measure_peak.py \
  --feature wmh-synthseg --input examples/wmh_data/sub-02_FLAIR.nii.gz \
  --weights /path/to/official/weights --device cuda:0 --threads 8 \
  --output-dir work/peak/wmh --report work/peak/wmh.json
```

对应原版参考命令为：

```bash
mri_WMHsynthseg --i sub-02_FLAIR.nii.gz --o sub-02_seg.nii.gz \
  --device cpu --threads 8 --crop --save_lesion_probabilities \
  --csv_vols sub-02_volumes.csv
```

## 数值与时间

| 指标 | 当前结果 |
|---|---:|
| shape / 数值 affine 一致 | 3/3 |
| 最低全标签逐体素一致率 | 0.99997810 |
| 最低 WMH 标签 77 Dice | 0.99981002 |
| 病灶概率 NRMSE，中位数 [最大值] | 0.00091963 [0.00115553] |
| 病灶概率最大绝对差，三例最大值 | 0.009606 |
| CSV 软体积最大绝对差，三例最大值 | 62.62 mm³ |
| CPU 单例：标签、概率、CSV | 全部数值相同 |

| 运行 | 病例数 | 完整命令墙钟时间 |
|---|---:|---:|
| FreeSurfer 8.2 原版 CPU 参考 | 3 | 中位数 109.99 s，范围 105.75–171.00 s |
| FNIT 当前源码 H100 GPU | 3 | 中位数 11.34 s，范围 10.17–11.67 s |
| FNIT 当前源码 CPU | 1 | 64.44 s |

原版三例任务所在时段有重叠，只用于生成数值参考；上表不据此计算稳定加速倍数。独占单例 Python API 的阶段时间为权重加载 2.277 s、推理 1.937 s、保存 1.369 s，总计 5.584 s；完整测量进程为 9.85 s。Torch 峰值显存为 allocated 29,730 MiB、reserved 35,828 MiB，超过项目期望的 20 GB 上限，当前不能声称低于 20 GB。

三例 GPU、单例 CPU 的逐例指标、输出合同和当前源码哈希见 [`report.public.json`](report.public.json)。

## 当前示意图

下图使用公开 `sub-04` FLAIR；中、右列分别叠加原版和当前源码的 WMH 标签 77。当前三维比较有 97 个标签不同体素，WMH Dice 为 0.99981002。

![公开 FLAIR 的 FreeSurfer 与当前 FNIT WMH-SynthSeg 输出](../../docs/figures/wmh_synthseg_comparison.png)
