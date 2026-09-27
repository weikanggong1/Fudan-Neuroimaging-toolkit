# WMH-SynthSeg：12 例 FLAIR 的原版复现与耗时

[返回主页](../../README.md) · [功能说明](../../docs/wmh_synthseg/README.md) · [官方源码](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg/WMHSynthSeg)

本页记录旧版 Surfa 输出路径的 12 例验证。当前无 Surfa 实现的同例新旧逐体素、字节与耗时核对见[迁移报告](../wmh_no_surfa_20260928/README.md)；本页数字不代表新版 12 例重测。

本实验比较独立包与 FreeSurfer 8.2.0-1 WMH-SynthSeg 的输入、输出和运算结果。原版已使用 PyTorch；独立包保留官方 checkpoint、33 类标签、WMH 概率图及软体积计算，同时提供无需 FreeSurfer 安装的单被试 Python 接口。公开病例没有人工 WMH 标注，因此一致性结果不能衡量临床准确率。

## 病例与来源

固定选择 [OpenNeuro ds003592](https://openneuro.org/datasets/ds003592) 中 `sub-02` 至 `sub-13` 的 `ses-1` FLAIR，共 12 例，均为公开 CC0 源文件；本仓库只发布其 [SHA-256 清单](SHA256SUMS) 和三例脑外清零的[测试衍生文件](../../examples/WMH.md)，不在 Git 仓库存放这 12 例原图。固定文件名及下载指令如下，从仓库根目录运行：

```bash
mkdir -p work/wmh_reproduce/raw
for n in $(seq -w 2 13); do
  curl --fail --location --retry 3 \
    "https://s3.amazonaws.com/openneuro.org/ds003592/sub-${n}/ses-1/anat/sub-${n}_ses-1_FLAIR.nii.gz" \
    --output "work/wmh_reproduce/raw/sub-${n}_FLAIR.nii.gz"
done
cd work/wmh_reproduce/raw
sha256sum --check ../../../validation/wmh/SHA256SUMS
cd ../../..
```

来源 DOI：`10.18112/openneuro.ds003592.v1.0.13`。三个随仓库发布的 FLAIR 是单独的**脑外清零衍生样例**；下述 12 例基准使用从上述 URL 下载、通过 SHA-256 核验的原始 FLAIR。原始病例的标签 77 没有人工审阅真值。

## 固定实现与运行条件

测试主机 gpucw1 配有 2 × NVIDIA H100 PCIe（每卡 81,559 MiB，驱动 535.216.03）及 Intel Xeon Gold 6430。原版源码 CUDA 臂和本包使用 Python 3.11.7、PyTorch 2.5.1/CUDA 11.8、Nibabel 5.4.0、NumPy 1.26.4、Surfa 0.6.3；原生 `mri_WMHsynthseg` 使用 FreeSurfer 自带的 Python 3.8.13 和 PyTorch 2.1.2+cpu。FreeSurfer build 为 `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。原版源码和 checkpoint 的 SHA-256 见[来源清单](../../docs/provenance.json)。本包使用同一个 `WMH-SynthSeg_v10_231110.pth`，没有重训练或修改参数。

每例、每臂使用**全新进程**，8 个固定 CPU 核心，CUDA 臂使用物理 GPU 0；设 `NVIDIA_TF32_OVERRIDE=0`，避免 TF32 设置造成无关差异。四臂统一运行 `--crop`、病灶概率输出和软体积 CSV。时间是完整命令的墙钟秒数，包含 Python 启动、加载权重、读图、两遍定位/预测、保存图像与 CSV，不含下载权重和 FLAIR。原生 CPU 臂通过真实 `mri_WMHsynthseg` 启动器；该 gpucw1 安装的 `fspython` 无 CUDA，故 **“原版 CUDA”** 臂以相同 FreeSurfer 安装中**未经修改**的 `inference.py` 在上述 CUDA Python 环境运行。每例输出文件存在且退出码为 0 才算成功；原版内部会捕获单图异常，不能只凭进程返回码判断。

CPU 和 CUDA 分别进行同设备比较：原版 CPU 对本包 CPU，原版源码 CUDA 对本包 CUDA。指标包括输出尺寸、分割与概率图仿射矩阵、全标签体素一致率、WMH 标签 77 的硬 Dice、病灶概率 NRMSE，以及 CSV 每类软体积的绝对误差。原版分割 NIfTI 默认使用 float32；本包也以 float32 保存整数值标签。Nibabel 与 Surfa 写盘时的 qform/sform *code* 可能不同，因此比较数值仿射和体素值。另用一例检查 `.mgz`：原版 Nibabel 与本包 Surfa 的所有标签体素一致，仿射矩阵最大绝对差 **2.29 × 10⁻⁵ mm**，来自 MGZ 几何序列化的浮点差异。12 例主基准使用 `.nii.gz`。

## 复现命令

先安装本包、下载/配置权重，并令 `FREESURFER_HOME` 指向 FreeSurfer 8.2.0-1 的安装目录。下列脚本自动在原版两臂建立仅包含 `bin`、`python` 和官方权重链接的临时 FreeSurfer home，不修改原版源码或模型。每条命令对 12 例逐个启动新进程，输出放在被 Git 忽略的 `work/`：

```bash
python -m pip install .
python tools/setup_weights.py --model wmh-synthseg
export WMH_WEIGHTS="$HOME/.cache/fnit"
export FREESURFER_HOME=/path/to/freesurfer-8.2.0-1

for arm in official-cpu official-cuda torch-cpu torch-cuda; do
  python tools/benchmark_wmh.py \
    --input-dir work/wmh_reproduce/raw \
    --output-dir "work/wmh_reproduce/${arm}" \
    --weights "$WMH_WEIGHTS" --arm "$arm" \
    --freesurfer-home "$FREESURFER_HOME" --threads 8
done

cases=($(printf 'sub-%02d ' {2..13}))
for device in cpu cuda; do
  python tools/compare_wmh_outputs.py \
    --reference-dir "work/wmh_reproduce/official-${device}" \
    --candidate-dir "work/wmh_reproduce/torch-${device}" \
    --cases "${cases[@]}" \
    --output "work/wmh_reproduce/comparison-${device}.json"
done

python tools/summarize_wmh_validation.py \
  --official-cpu work/wmh_reproduce/official-cpu/execution.json \
  --official-cuda work/wmh_reproduce/official-cuda/execution.json \
  --torch-cpu work/wmh_reproduce/torch-cpu/execution.json \
  --torch-cuda work/wmh_reproduce/torch-cuda/execution.json \
  --comparison-cpu work/wmh_reproduce/comparison-cpu.json \
  --comparison-cuda work/wmh_reproduce/comparison-cuda.json \
  --output work/wmh_reproduce/report.public.json
```

`tools/benchmark_wmh.py` 保存逐例 `execution.json`、日志、分割、概率图与 CSV；`tools/compare_wmh_outputs.py` 生成逐例误差；汇总脚本只在四臂和两种同设备比较均完整时写匿名报告。若想只执行其中一臂，可单独运行相应命令；已有部分输出时需使用该脚本的 `--resume` 选项或改用新目录。

## 数值与耗时

四臂均完成 12/12 例。CPU 与 CUDA 的同设备原版/本包比较，在每一例均得到：全标签体素一致率 **1.000000**、WMH 标签 77 的 Dice **1.000000**、病灶概率 MAE/NRMSE/最大绝对差 **0**、分割与概率图仿射矩阵最大差 **0**，以及所有 CSV 软体积最大绝对差 **0 mm³**。输出位于同一处理后网格。压缩文件或 NIfTI 元数据不保证逐字节相同；例如 qform/sform code 会受写盘库影响。

| 完整单例命令 | 中位数（秒） | 平均数（秒） | 逐例范围（秒） |
|---|---:|---:|---:|
| FreeSurfer 原生 CPU | 97.38 | 98.33 | 67.39–124.30 |
| FreeSurfer 未改动源码 CUDA | 8.25 | 8.41 | 7.92–9.66 |
| 本包 CPU | 70.69 | 71.05 | 69.92–74.67 |
| 本包 CUDA | 8.41 | 8.97 | 8.06–13.39 |

以中位数比较，本包 CPU 命令少 **26.70 秒**（约 27%），12 例中 11 例短于原生 CPU；CUDA 中位数比官方源码多 **0.16 秒**，属于同量级。CUDA 本包第一例为 13.39 秒，其余例为 8.06–9.65 秒。计时反映**整条命令及各自 Python/PyTorch 运行环境**：原生 CPU 用 FreeSurfer 的 Torch 2.1.2，而本包 CPU 用 Torch 2.5.1，因此不能把 CPU 差异全归因于改写代码或网络本身。不同设备之间的运行时间与数值不要混作同设备移植误差。

[report.public.json](report.public.json) 给出 12 例匿名逐例秒数、比较指标和汇总分布；只包含匿名病例序号，没有服务器路径或账号。单例运行示例见[公开 FLAIR](../../examples/WMH.md)，影像对照见[并排图](../../docs/figures/README.md)。
