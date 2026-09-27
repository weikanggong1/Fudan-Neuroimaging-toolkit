# 33 类 SynthSeg 当前源码验证

[返回主页](../../README.md) · [功能说明](../../docs/synthseg/README.md) · [机器可读报告](report.public.json)

本页记录 2026-09-27 对独立 33 类 SynthSeg 的当前源码回归。候选推理不调用 FreeSurfer；FreeSurfer 8.2.0-1 `mri_synthseg --noaddctab` 只用于产生同输入参考输出。

## 数据与源码绑定

输入为仓库 `examples/data/` 中三幅真实、去面容 T1w。三例没有人工解剖分割真值，因此指标表示对原版实现的复现程度。候选基于提交 `9146b0004468ffc2a30cacad0ab4fa7b6fdb3b48` 的工作树；SynthSeg 功能源码树 SHA-256 为 `39fa204aea7674ad7c6e09652d0f8750dd2872b1b78799812ab0d71b5b6c8972`，核心输出实现 `synthseg.py` 为 `db1dfcb3e79abe29d39ee535fbc87ec559b4e54c86ca7c07e7b5c154e6c7605d`。

当前实现按原版输出合同写入 `int32` 标签、qform code 0 和 sform code 2；默认网格和 `keep_geometry=True` 两条路径使用同一契约。针对性测试保存并重新读取 NIfTI，结果为 `2 passed`；WMH-SynthSeg、SynthSR 和 TorchFAST 的跨模块回归为 `28 passed, 4 skipped`。另在一幅真实 T1w 上运行 `--keep-geometry`：输出 shape `256×156×256` 与输入相同，affine 最大差为 0，dtype/qform/sform 为 `int32`/0/2。

## 实际命令

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fnit.cli synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o work/current/sub-01_seg.nii.gz \
  --csv-vols work/current/sub-01_volumes.csv \
  --device cuda:0 --threads 4 --weights /path/to/official/weights
```

CPU 检查把 `--device` 改为 `cpu` 并使用 8 个线程。对应原版命令为：

```bash
mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_seg.nii.gz \
  --vol sub-01_volumes.csv --threads 4 --cpu --noaddctab
```

显存与阶段时间用下面的单例 Python API 包装器记录：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python validation/model_io_current/measure_peak.py \
  --feature synthseg --input examples/data/sub-01_T1w.nii.gz \
  --weights /path/to/official/weights --device cuda:0 --threads 8 \
  --output-dir work/peak/synthseg --report work/peak/synthseg.json
```

## 输出一致性与时间

| 指标 | 当前三例结果 |
|---|---:|
| shape / 数值 affine | 3/3 一致；affine 最大差 0 |
| dtype / qform / sform | 3/3 为 `int32` / 0 / 2，与原版一致 |
| 最低逐体素标签一致率 | 0.99998817 |
| 全部病例、前景标签的最低 Dice | 0.99902629 |
| 各病例标签 Dice 中位数的中位数 | 0.99998952 |
| CSV 共同数值列 | 33 |
| CSV 最大绝对误差 | 104.06 mm³ |

CPU 单例与原版具有同一 shape、affine、`int32`、qform/sform；13,355,776 个输出体素中仅 1 个标签不同，逐体素一致率为 0.999999925，CSV 最大差 0.24 mm³。

| 运行 | 病例数 | 完整命令墙钟时间 |
|---|---:|---:|
| FreeSurfer 8.2 原版 CPU 参考 | 3 | 中位数 304.23 s，范围 217.55–389.24 s |
| FNIT 当前源码 H100 GPU | 3 | 中位数 9.48 s，范围 8.84–10.13 s |
| FNIT 当前源码 CPU | 1 | 55.57 s |

原版三个 CPU 任务所在时段有重叠，只用于数值参考，不能与候选时间计算稳定加速倍数。独占 GPU 单例的 Python API 总计 4.990 s，完整测量进程 8.83 s；Torch 峰值 allocated 19,769 MiB，reserved 22,820 MiB。有效张量分配低于 20 GiB，但缓存预留超过 20 GiB。

三例当前源码的逐例记录、输出 affine、dtype 和文件哈希见 [`report.public.json`](report.public.json)。

## 当前示意图

下图使用公开 `sub-02`。两列标签图在同一网格显示，最后一列标出不同体素；该例逐体素一致率为 0.99998881。

![FreeSurfer 与当前 FNIT 33 类 SynthSeg 输出](../../docs/synthseg/figures/synthseg_comparison.png)
