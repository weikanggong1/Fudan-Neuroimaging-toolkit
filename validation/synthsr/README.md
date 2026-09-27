# SynthSR：12 例 T1w 的原版对照

[返回主页](../../README.md) · [SynthSR 输入输出与调用](../../docs/synthsr/README.md) · [匿名汇总数据](../../benchmark/synthsr_validation_2026-09-23.json)

本页记录旧版导入 Surfa 时的 12 例 T1w 基准。当前无 Surfa 路径的同一真实 FLAIR 核对见[迁移报告](../synthsr_no_surfa_20260928/README.md)；本页数字不代表新版 12 例重测。

本次在相同输入、相同官方权重和默认处理选项下，比较 FreeSurfer `mri_synthsr` 与本包 `fnit synthsr`。四组分别是原版 CPU、原版源码 CUDA、本包 CPU、本包 CUDA。比较的是**对原版输出的复现程度**和完整命令耗时；这些病例没有用于评价合成图像质量的人工真值。

## 病例、权重和环境

从用户持有的临床 T1w 中固定选择 12 例，健康对照与抑郁症病例各 6 例。每例为一幅 3D `T1w.nii.gz`。原始影像、病例标识和生成的影像均不进入 Git；仓库仅保存不含路径的[四臂汇总结果](../../benchmark/synthsr_validation_2026-09-23.json)。复现脚本可在其他 T1w 数据上运行，但这里的数值和耗时只对应这 12 例。

测试机为 `gpucw1`，CPU 是 Intel Xeon Gold 6430，GPU 是 NVIDIA H100 PCIe 80 GB，驱动 535.216.03。FreeSurfer 为 8.2.0-1；原版源码 CUDA 臂使用 TensorFlow 2.13.1（CUDA 11.8、cuDNN 8），本包使用 PyTorch 2.5.1（CUDA 11.8）。四臂均读取官方 `synthsr_v20_230130.h5`，SHA-256 为 `a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b`，没有重训练或转换权重文件。

该 FreeSurfer 安装中的 TensorFlow 在默认动态库搜索路径下看不到 CUDA GPU。配置 `LD_LIBRARY_PATH` 和 `TF_FORCE_GPU_ALLOW_GROWTH=true` 后，使用 FreeSurfer 自带的 `python/bin/python3` 执行**未修改**的安装版 `python/scripts/mri_synthsr`，TensorFlow GPU 可见性预检通过，原版 CUDA 臂完成 12 例。原版 CPU 臂使用 `bin/mri_synthsr --cpu`。两臂使用同一个安装目录中的源码和 `.h5`。基准脚本在 CUDA 计时前检查 TensorFlow 是否能看到 GPU，检查失败会停止。

每例、每臂均新启一个进程。CPU 臂固定 8 个 CPU 核，GPU 臂固定另 8 个 CPU 核；脚本设置 `NVIDIA_TF32_OVERRIDE=0`，GPU 选择由 `--physical-gpu` 指定。计时从启动单例命令前到进程结束，包含 Python 启动、读图、加载 `.h5`、重采样、默认的左右翻转双次推理、锐化和 NIfTI 写盘。权重下载和 TensorFlow GPU 可见性预检不计入逐例时间。这是共享节点上的墙钟时间，负载变化会影响复测。

## 12 例复现命令

以下命令从仓库根目录运行。先将自己的 12 幅影像放在 `INPUT_DIR/case-01/T1w.nii.gz` 等独立子目录，安装本包，并设置真实的 FreeSurfer 路径。`TF_PYTHON` 指向 FreeSurfer 自带 Python；它需能导入 TensorFlow、Surfa 等脚本依赖，并通过 GPU 可见性预检。gpucw1 实测的动态库搜索路径由站点 Anaconda 库目录和 CUDA 11.7 库目录组成；下面用占位路径表示，须按本机安装位置替换。

```bash
python -m pip install .
export INPUT_DIR=/path/to/12-t1w-cases
export FREESURFER_HOME=/path/to/freesurfer-8.2.0-1
export TF_PYTHON="$FREESURFER_HOME/python/bin/python3"
export SR_WEIGHT="$FREESURFER_HOME/models/synthsr_v20_230130.h5"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH="/path/to/anaconda/lib:/path/to/cuda/lib64:${LD_LIBRARY_PATH:-}"

python tools/benchmark_synthsr.py --input-dir "$INPUT_DIR" \
  --pattern '*/T1w.nii.gz' --output-dir work/synthsr_bench/official-cpu \
  --weights "$SR_WEIGHT" --freesurfer-home "$FREESURFER_HOME" \
  --arm official-cpu --threads 8

python tools/benchmark_synthsr.py --input-dir "$INPUT_DIR" \
  --pattern '*/T1w.nii.gz' --output-dir work/synthsr_bench/official-cuda \
  --weights "$SR_WEIGHT" --freesurfer-home "$FREESURFER_HOME" \
  --official-script "$FREESURFER_HOME/python/scripts/mri_synthsr" \
  --tf-python "$TF_PYTHON" --arm official-cuda --threads 8 --physical-gpu 1

python tools/benchmark_synthsr.py --input-dir "$INPUT_DIR" \
  --pattern '*/T1w.nii.gz' --output-dir work/synthsr_bench/torch-cpu \
  --weights "$SR_WEIGHT" --arm torch-cpu --threads 8

python tools/benchmark_synthsr.py --input-dir "$INPUT_DIR" \
  --pattern '*/T1w.nii.gz' --output-dir work/synthsr_bench/torch-cuda \
  --weights "$SR_WEIGHT" --arm torch-cuda --threads 8 --physical-gpu 1
```

`--input-dir` 和 `--pattern` 选择输入；每个 `--output-dir` 内保存逐例 NIfTI、日志和 `execution.json`。`--weights` 在四臂均固定同一个官方文件；`--arm` 决定调用 FreeSurfer 还是本包以及 CPU/CUDA 设备。`--threads 8` 和脚本中的 `taskset` 固定每个进程的 CPU 并行度。若中途停止，同一参数重跑时加 `--resume`；完整的新实验应使用新的输出目录。

只运行本包时不必安装 FreeSurfer：`python tools/setup_weights.py --model synthsr` 可从官方地址下载并校验相同的 `.h5`，随后可省略 `--weights` 或指定下载文件。原版对照仍需 FreeSurfer 的脚本和运行环境。

用下列命令逐例核对形状、`uint8` 类型、仿射矩阵和体素误差。`compare_synthsr_outputs.py` 的 NRMSE 为体素 RMSE 除以参考图像的 RMS 强度；只有输出形状相同时才计算体素误差。

```bash
python tools/compare_synthsr_outputs.py \
  --reference-dir work/synthsr_bench/official-cpu \
  --candidate-dir work/synthsr_bench/torch-cpu \
  --output work/synthsr_bench/compare-official-cpu-torch-cpu.json
python tools/compare_synthsr_outputs.py \
  --reference-dir work/synthsr_bench/official-cuda \
  --candidate-dir work/synthsr_bench/torch-cuda \
  --output work/synthsr_bench/compare-official-cuda-torch-cuda.json
```

## 数值与耗时

四臂各完成 12/12 例。每一项输出均为相同形状的 1 mm 合成 T1w、`uint8` 体素，数值仿射矩阵最大绝对差为 **0**。下表的精确比例取 12 例中的最低值，MAE 是各例体素 MAE 的平均值；最大体素差在四种比较中均为 **1 灰度级**。

| 比较 | 平均体素 MAE | 最低逐体素相同比例 |
|---|---:|---:|
| 原版 CPU 对本包 CPU | 0.0000556 | 99.9929% |
| 原版 CUDA 对本包 CUDA | 0.0000269 | 99.9968% |
| 原版 CPU 对本包 CUDA | 0.0000643 | 99.9924% |
| 原版 CPU 对原版 CUDA | 0.0000631 | 99.9923% |

因此本包与原版在输出网格和绝大多数体素上相同，但不应称为逐体素完全一致。原版本身的 CPU 与 CUDA 输出也有少量 1 级灰度差；跨设备的数值差不能全部归给 PyTorch 改写。

另用公开 `sub-04` FLAIR 分别检查 `--lowfield`、`--v1`，以及 `--ct --disable_flipping --disable_sharpening` 组合。原版 CPU 与本包 GPU 的输出形状及仿射完全相同，三组体素完全一致比例分别为 **99.9901%、99.9904%、99.9947%**，最大差值均为 1。这里给 FLAIR 加 `--ct` 仅用于核对该处理分支，不代表 CT 应用效果。

`.npz` 路径另用一幅固定随机种子生成的 `uint16` 32³ 测试阵列核对。在关闭翻转和锐化时，两版写出的 `vol_data` 均为 32³ `float32`；平均绝对差为 **0.0000386**，最大差为 **0.000736**。这项测试检验整数 `.npz` 输入的 dtype 保留及浮点 `.npz` 输出规则，不用于评估临床影像质量。

`.mgz` 路径使用另一幅固定随机阵列检查读写。两版输出均为 32³、nibabel 读出存储 dtype `float32`、仿射矩阵相同；平均绝对差 **0.000122**，最大差 **1**。原版和本包均先量化为 0–255，再由 nibabel 写入 MGZ。

| 完整单例命令 | 中位数（秒） | 平均数（秒） | 逐例范围（秒） |
|---|---:|---:|---:|
| FreeSurfer 原生 CPU | 103.60 | 105.14 | 43.99–140.26 |
| FreeSurfer 未修改源码 CUDA | 53.27 | 55.58 | 24.03–134.03 |
| 本包 PyTorch CPU | 42.32 | 42.05 | 38.62–44.64 |
| 本包 PyTorch CUDA | 13.80 | 14.04 | 12.94–15.54 |

以每例原版/本包时间比计算的中位数，CPU 对 CPU 为 **2.49 倍**，CUDA 对 CUDA 为 **3.93 倍**；原版 CPU 对本包 CUDA 为 **7.61 倍**。这些是完整命令的观测比值，包含 TensorFlow/PyTorch 各自的启动、权重加载和图像 I/O；尤其原版 CUDA 的逐例范围很宽，不宜解释为网络内核的稳定加速倍数。逐例运行日志和临床影像仅保留在测试机上，Git 中的 JSON 提供可公开的汇总统计。

## 三例公开 FLAIR 的运行与图示

仓库还提供 `examples/wmh_data/` 中三例脑外清零的公开 FLAIR 衍生样例。来源、处理方式及文件 SHA-256 见[示例数据说明](../../examples/WMH.md)和 [SOURCES.json](../../examples/wmh_data/SOURCES.json)。这些文件便于下载仓库后检查 SynthSR 调用；**12 例 T1w 的计时和上表数值来自另一组未发布病例**，不能与三例 FLAIR 混为同一基准。

```bash
python examples/check_wmh_data.py
export FREESURFER_HOME=/path/to/freesurfer-8.2.0-1
export SR_WEIGHT="$FREESURFER_HOME/models/synthsr_v20_230130.h5"

python tools/benchmark_synthsr.py \
  --input-dir examples/wmh_data --output-dir work/synthsr_public/official-cpu \
  --weights "$SR_WEIGHT" --freesurfer-home "$FREESURFER_HOME" \
  --arm official-cpu --threads 8
python tools/benchmark_synthsr.py \
  --input-dir examples/wmh_data --output-dir work/synthsr_public/torch-cuda \
  --weights "$SR_WEIGHT" --arm torch-cuda --threads 8 --physical-gpu 0
python tools/compare_synthsr_outputs.py \
  --reference-dir work/synthsr_public/official-cpu \
  --candidate-dir work/synthsr_public/torch-cuda \
  --output work/synthsr_public/comparison.json

python -m pip install matplotlib
python tools/plot_synthsr_comparison.py \
  --original examples/wmh_data/sub-04_FLAIR.nii.gz \
  --official work/synthsr_public/official-cpu/sub-04_FLAIR_synthsr.nii.gz \
  --torch work/synthsr_public/torch-cuda/sub-04_FLAIR_synthsr.nii.gz \
  --input-label 'Original FLAIR' \
  --output work/synthsr_public/sub-04_comparison.png
```

`plot_synthsr_comparison.py` 在同一组 RAS 坐标取轴位、冠状位和矢状位切片，避免把 3 mm 层厚的输入与 1 mm 输出按数组索引硬对齐。已生成的[三列对照图](../../docs/synthsr/figures/synthsr_flair_comparison.png)依次显示原始 FLAIR、FreeSurfer 输出和 PyTorch 输出。`matplotlib` 仅用于制作图片，模型推理不需要它。
