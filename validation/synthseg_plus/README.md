# SynthSeg+ 真实 T1w 对照（2026-09-29）

[功能与用法](../../docs/synthseg_plus/README.md) · [GPU 逐项结果](report.real.json) · [CPU 逐项结果](report.cpu.real.json) · [耗时记录](benchmark.volumes.real.json)

输入为仓库公开的去面容 T1w `examples/data/sub-01_T1w.nii.gz`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。两端均使用 FreeSurfer 8.2.0-1 的 SynthSeg 2.0 主网络和皮层分区权重；FNIT 推理不调用 FreeSurfer。测试在 gpucw1 的 H100 PCIe 上进行，CPU 线程数为 4。

仓库附带原版生成的[合并标签图](reference/sub-01_official_synthseg_plus.nii.gz)和[软体积 CSV](reference/sub-01_official_volumes.csv)，SHA-256 分别为 `cc98218cdf55cec648612d492a70e80f08710730286f8867b6a6e09c7ce7e9bc` 与 `cadb6fd4f46f59ccc410acdc0836ee5d64f96939a1ed8aa850a3b81c848c7040`。安装 FNIT 和官方权重后，可直接用这两个参考文件做输出核对，无须安装 FreeSurfer。
原版 `--cpu` 与 GPU 在本例生成的 NIfTI 和 CSV 均逐字节相同。

## 复现命令

```bash
# 官方参考：--i 是单幅 T1w；--o 是合并标签图；--parc 开启皮层分区；--vol 写软体积 CSV。
/usr/bin/time -f 'wall_seconds=%e' mri_synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o official_vol.nii.gz --parc --vol official_vol.csv --threads 4

# FNIT：输入、合并图和 CSV 与上一命令逐项对应；--weights 指官方模型目录。
PYTHONPATH=src python -m fnit.cli synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o fnit_vol.nii.gz --parc --csv-vols fnit_vol.csv \
  --weights /absolute/path/models --parc-weights /absolute/path/models \
  --device cuda:0 --threads 4

# 同一输出范围的 CPU 计时：官方 --cpu 与 FNIT --device cpu 都使用 4 个线程。
/usr/bin/time -f 'wall_seconds=%e' mri_synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o official_cpu_vol.nii.gz --parc --vol official_cpu_vol.csv \
  --cpu --threads 4
/usr/bin/time -f 'wall_seconds=%e' env PYTHONPATH=src python -m fnit.cli synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o fnit_cpu_vol.nii.gz --parc --csv-vols fnit_cpu_vol.csv \
  --weights /absolute/path/models --parc-weights /absolute/path/models \
  --device cpu --threads 4

# --official/--fnit 指同网格标签图；后两个参数分别指两份软体积 CSV。
python validation/synthseg_plus/compare.py \
  --official validation/synthseg_plus/reference/sub-01_official_synthseg_plus.nii.gz \
  --fnit fnit_vol.nii.gz \
  --official-volumes validation/synthseg_plus/reference/sub-01_official_volumes.csv \
  --fnit-volumes fnit_vol.csv \
  --output report.real.json
```

`benchmark.py` 对同一个 Python 对象连续调用两次，分别记录首轮和权重已载入后的时间。其计时包含模型调用与 NIfTI 对象构建，不含写出合并图/CSV。运行方法：

```bash
PYTHONPATH=src python validation/synthseg_plus/benchmark.py \
  --input examples/data/sub-01_T1w.nii.gz \
  --weights /absolute/path/models \
  --official validation/synthseg_plus/reference/sub-01_official_synthseg_plus.nii.gz \
  --volumes --save-combined fnit_vol.nii.gz --save-csv fnit_vol.csv \
  --output benchmark.volumes.real.json
```

## 输出一致性

| 指标 | 本例结果 |
|---|---:|
| 标签图 shape / affine | `(256, 203, 257)` / 数值完全一致 |
| 类型 / qform code / sform code | 两端均为 `int32` / `0` / `2` |
| 合并图逐体素一致率 | `0.9999758157`；不同体素 `323` 个 |
| 98 个前景标签的最低 / 中位 Dice | `0.9983427` / `0.9997051` |
| 软体积 CSV 的列名和顺序 | 完全一致；颅内容积 + 32 个主分割 + 68 个皮层区，共 101 个数值列 |
| 颅内容积绝对 / 相对差 | `45.20 mm³` / `0.0035%` |
| 主分割软体积平均 / 最大绝对差 | `5.86 / 100.36 mm³`；最大项为 CSF，相对差 `0.052%` |
| 68 个皮层区软体积平均 / 最大绝对差 | `2.11 / 7.55 mm³`；最大相对差 `0.164%` |

FNIT CPU 完整命令与原版 CPU 完整命令使用同一输入及输出参数：合并图仅有 `6` 个体素不同，101 列软体积平均 / 最大绝对差为 `0.028 / 0.293 mm³`。这些数值见[CPU 逐项结果](report.cpu.real.json)。

皮层分区软体积需重现原版的 channels-last NumPy 归约顺序。用 PyTorch 在 GPU 上直接求和会使本例皮层区平均绝对差升至约 `366 mm³`；改为原版顺序后降至 `2.11 mm³`。这反映两种浮点累加顺序的差异，不是硬标签体积与软体积的换算。当前 GPU 推理仍默认 TF32，未使用 float16 或 bfloat16。

FreeSurfer 默认在合并 NIfTI 中写入颜色表扩展，FNIT 的 NIfTI 未写该扩展；上述一致性指标针对标签数据和空间信息。图像显示采用相同切面与色表，右列标出不同体素：

![原版与 FNIT 的公开 T1w 皮层分区](../../docs/synthseg_plus/figures/synthseg_plus_comparison.png)

## 时间

| 实现及调用范围 | 本例时间 |
|---|---:|
| FreeSurfer 8.2.0-1 GPU，`mri_synthseg --parc --vol` 完整命令 | `542.08 s` |
| FreeSurfer 8.2.0-1 CPU，同一完整命令加 `--cpu` | `241.48 s` |
| FNIT CPU，`fnit synthseg --parc --csv-vols --device cpu` 完整命令 | `210.19 s` |
| FNIT H100，`SynthSegPlus(...)(volumes=True)` 首轮 / 同对象第二轮 | `16.68 / 13.10 s` |
| FNIT H100，`SynthSegPlus(...)(volumes=False)` 首轮 / 同对象第二轮 | `7.30 / 1.97 s` |

CPU 的两条完整命令计时范围相同，本轮 FNIT 用时比原版少 `31.29 s`。GPU 原版完整命令与 FNIT Python 调用的计时范围不同，不能直接相除作为加速倍数。H100 与其他作业共享，原版 GPU 在这次运行中比 CPU 慢也不代表一般情况；FNIT 两次 GPU 调用的少量边界体素会随 CUDA 算法选择而变化。当前只验收了一幅完整 T1w；`fast=True`、QC、robust 路径和其他采集方案未纳入本次对照。
