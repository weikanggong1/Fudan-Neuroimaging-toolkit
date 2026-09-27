# TorchFAST 当前源码验证

[返回主页](../../README.md) · [功能说明](../../docs/fast/README.md) · [机器可读报告](report.public.json)

本页记录 2026-09-27 对当前 Nibabel 输入输出链的真实数据回归。候选只运行 FNIT/PyTorch，不调用 FSL；同一幅 brain-only T1w 的 FSL 6.0.7.4 FAST4 2111.3 输出作为固定参考。

## 数据与源码绑定

主回归使用一幅真实 brain-only T1w；私有路径和病例标识不写入仓库，报告保存输入 SHA-256 `f1b6f58edb3ac9996282d3fcbadcb94626785347481f27cf5a3b9419bf640c12`。示意图另用仓库公开 `sub-02` 的参考 SynthStrip 脑图。

候选基于提交 `9146b0004468ffc2a30cacad0ab4fa7b6fdb3b48` 的工作树。FAST 功能源码树 SHA-256 为 `433723da856c798faca9a66ea95784bdbb5ad5b13f66f451ce32e7030559943b`；算法文件 `algorithm.py` 为 `9ffed0035a618ec4d6dc1eea8deeadaddd071ced9b04b66593cb6bd4d1102638`。

## 实际命令

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m fnit.cli fast \
  --image T1_brain.nii.gz --output-prefix work/current/T1_brain \
  --device cuda:0 --threads 8 -b -B --overwrite
```

CPU 单例把 `--device` 改为 `cpu`。对应原版参考命令为：

```bash
fast -n 3 -t 1 -b -B -o T1_brain T1_brain.nii.gz
```

本次固定参考实际保存的是 `fast -b` 的七个共同输出；`restore` 只检查 FNIT 内部恒等式，不参与与 FSL 文件比较。显存和阶段时间使用：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python validation/model_io_current/measure_peak.py \
  --feature fast --input T1_brain.nii.gz --device cuda:0 --threads 8 \
  --output-dir work/peak/fast --report work/peak/fast.json
```

## 输出与数值一致性

八幅 FNIT 输出均保留输入的 `208×253×226` shape 和数值 affine。三张 PVE、bias 和 restore 为 `float32`；hard segmentation、PVE segmentation 和 mixel type 为 `int32`。与 FSL 对应文件的 shape、dtype、affine 及 qform/sform code 全部一致。

| 输出 | Pearson | MAE | Dice 0.5 | FNIT/FSL 体积比 |
|---|---:|---:|---:|---:|
| CSF PVE | 0.988556 | 0.008464 | 0.987640 | 1.013900 |
| GM PVE | 0.984627 | 0.015411 | 0.992328 | 0.989954 |
| WM PVE | 0.993731 | 0.007022 | 0.997031 | 1.004929 |

log-bias Pearson 为 0.99999999936，MAE 为 2.85×10⁻⁶。hard segmentation、PVE segmentation 和 mixel type 的逐体素一致率分别为 0.999899、0.990349 和 0.939031。FSL 使用逐体素原地 HMRF 更新，本实现使用 GPU 同步更新，因此不声明逐体素等价。

| 运行 | 病例数 | 完整进程墙钟时间 |
|---|---:|---:|
| FSL FAST CPU，同病例参考运行 | 1 | 305.46 s |
| FNIT 当前源码 H100 GPU 测量进程 | 1 | 12.55 s |
| FNIT 当前源码 CPU CLI | 1 | 36.24 s |

时间记录不在同一时段，不能解释为隔离负载下的稳定加速倍数。GPU Python API 中构造 0.517 s、推理 2.448 s、保存八幅压缩 NIfTI 4.365 s，总计 7.329 s；Torch 峰值 allocated 2,352 MiB、reserved 3,444 MiB。

本页单例验证覆盖当前 I/O、数值、时间与显存；完整机器记录见 [`report.public.json`](report.public.json)。

## 当前公开示意图

公开 `sub-02` 的当前重跑得到 GM Pearson 0.978172、MAE 0.017553、Dice 0.987420。下图展示相同 brain-only 输入上的 GM PVE 和 bias-corrected 图。

![FSL FAST 与当前 TorchFAST 的公开真实 T1w 对照](figures/fast_comparison.png)

图的数值记录见 [`figures/metrics.json`](figures/metrics.json)，绘图命令使用 [`tools/plot_fast_comparison.py`](../../tools/plot_fast_comparison.py)。
