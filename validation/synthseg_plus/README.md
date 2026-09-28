# SynthSeg+ 真实 T1 对照（2026-09-28）

测试输入为仓库公开的去面容 `examples/data/sub-01_T1w.nii.gz`，SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。FNIT 与 FreeSurfer 8.2.0-1 `mri_synthseg --parc` 使用同一输入、同版主网络和皮层分区 H5，分别在 H100 上运行；FNIT 不调用 FreeSurfer。权重组五个文件在官方安装目录通过大小及 SHA-256 校验，其中皮层 H5 的 SHA-256 为 `83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684`。

```bash
# FNIT：--i 输入 T1；--o 合并标签；--parc 开启皮层分区；--weights 是权重目录
/usr/bin/time -f 'elapsed_sec=%e' fnit synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o fnit_synthseg_plus.nii.gz --parc \
  --weights /absolute/path/models --parc-weights /absolute/path/models \
  --device cuda:0

# 仅作为参考的软件；FNIT 运行环境无需安装
/usr/bin/time -f 'elapsed_sec=%e' mri_synthseg \
  --i examples/data/sub-01_T1w.nii.gz \
  --o official_synthseg_plus.nii.gz --parc --threads 4

# --official/--fnit 是两幅同网格标签图；--output 是逐标签 JSON
python validation/synthseg_plus/compare.py \
  --official official_synthseg_plus.nii.gz \
  --fnit fnit_synthseg_plus.nii.gz \
  --output report.real.json
```

两幅输出均为 `(256, 203, 257)`，NIfTI affine 数值一致。逐体素一致率 `0.9999759655`，差异 `321` 个体素；98 个至少在一幅图中出现的前景标签，最低 Dice `0.9983427`，中位 Dice `0.9997124`。68 个皮层分区中最低 Dice `0.9983427`；非皮层标签最低 Dice `0.9990263`。逐标签体素数、Dice 和硬标签体积差见[机器报告](report.real.json)。最大绝对硬标签体积差约 `94 mm³`。这些指标衡量与原版输出的复现程度，不是对人工真值的准确度。

FNIT 完整 H100 命令墙钟 `10.26 s`；官方 H100 命令 `378.99 s`，官方 CPU 命令 `625.83 s`。FNIT 模型调用本身 `4.11 s`；本次 Torch 峰值 allocated `19.30 GiB`、reserved `22.31 GiB`。三个完整命令均含各自启动、模型读入、读写与运行环境开销；单次运行且官方 GPU/CPU 任务与其他作业共享节点，不能把 `378.99/10.26` 当作稳定的模型推理加速倍数。FNIT H100 完整输出 SHA-256 `d42e8c3b519ad63dc6ea6d46d4a77c94e3f2925bc6e57b052411fc605f830a22`；官方 H100 输出 `cc98218cdf55cec648612d492a70e80f08710730286f8867b6a6e09c7ce7e9bc`。

默认 CLI 保留官方的预处理输出网格；Python API 默认 `keep_geometry=True`，会再以最近邻重采样到原 T1 网格。比较时必须把该参数设成 `False`，或在 CLI 中不加 `--keep-geometry`。本轮未对 `fast=True`、其他病例、官方软体积 CSV、QC 或 robust 路径做验收。
