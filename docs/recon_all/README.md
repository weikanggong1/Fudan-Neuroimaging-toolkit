# Python recon-all：原始 T1 到皮层指标

[返回首页](../../README.md) · [实现](../../src/fnit/recon_all/) · [整例验证](../../validation/recon_all/python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)

`fnit-recon-all` 从一幅 T1w 生成脑体积分割、双侧 white/pial/sphere 表面、厚度、面积、顶点体积、曲率、aparc/a2009s/DKT 标注和脑区统计。默认调用 Python 包及其 CPU/CUDA 算子；可选调用用户在 Conda 中从 FreeSurfer 源码编译的六个 C++ 程序。两条路径均不要求安装官方 FreeSurfer、FSL 或原生 recon-all 运行包。模型权重和模板单独下载并校验。

**当前为近似核心重建。** 默认 Python 路径的拓扑修复、white/pial 放置和球面配准尚未与 FreeSurfer 8.2 对齐；只生成核心输出的子集。旧的纯 Python 配置曾在一个真实 T1 上跑通整例，严格固定配置的 138 项官方输出中 6 项通过、52 项缺失、80 项存在差异。这是该旧配置的结果，不是可选 C++ 路径的验收结果。C++ 路径仍使用近似的上游几何，尚未证实皮层逐顶点厚度、面积、体积、曲率及脑区统计与官方数值一致。旧配置的差异和计时见[整例报告](../../validation/recon_all/python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)；根因、容差及修复顺序见[差异分析](../../validation/recon_all/python_gpu_port/DISCREPANCY_AND_TOLERANCE_20260927.md)。

## 安装外置数据

Python ≥3.10；CUDA 运行需与驱动兼容的 PyTorch。仓库根目录安装：

```bash
python -m pip install '.[recon-all-python-stages]'
fnit-setup-weights --model recon-all --dest /path/to/weights
fnit-setup-recon-all-assets --dest /path/to/assets
```

默认下载 **6 个权重文件**（SynthStrip、SynthMorph affine、SynthSeg 及三个标签表；合计 135,394,037 字节）和 **15 个模板/图谱文件**（217,497,770 字节，约 207 MiB）。15 个文件包含双侧 folding atlas，供可选原生球面配准使用。安装器检查每个文件的大小和 SHA-256。`fnit-setup-recon-all-assets --all` 才下载全部 102 个阶段研究资源。已有文件可加 `--verify-only` 核验；两个目录均在仓库与 wheel 外。权重的来源和许可见[权重说明](../WEIGHTS.md)，模板清单见[资产验证](../../validation/recon_all/python_gpu_port/ASSETS_VALIDATION.md)。

仓库 [`environment.yml`](../../environment.yml) 也会安装 Python 阶段依赖。SimpleITK/ITK、Numba 和部分 NumPy/Surfa 操作运行在 CPU；PyTorch 网络及部分体素、表面算子可在 CUDA 上运行。模型张量为 float32，默认允许 TF32；SynthSeg 阶段为已验证的硬标签一致性关闭 cuDNN TF32。没有自动启用 float16 或 bfloat16。

## 单被试

```bash
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4
```

`subject_dir` 必须不存在或为空。没有 GPU 时传 `--device cpu`。Python 内调用：

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    "subject_T1w.nii.gz", "/scratch/subjects/sub01",
    "/path/to/weights", "/path/to/assets",
    device="cuda:0", threads=4,
)
```

返回报告包含阶段耗时；被试目录内的 `fnit-native-free-run.json` 记录状态、设备及近似步骤。`--n4-python /path/to/python` 可指定另一已安装 SimpleITK 的 Python 解释器执行 N4；该选项仍只启动 Python 算子。

## 可选 Conda C++ 阶段

按[独立 Conda 编译与验证说明](CONDA_CPP_BUILD.md)准备构建环境，将 FreeSurfer 源码固定在提交 `d932c45b7941662ea380a05efef580568b98d41a`，运行仓库的 [`build_recon_all_fs_cpp_conda.sh`](../../tools/build_recon_all_fs_cpp_conda.sh)：

```bash
conda activate /path/to/conda-build-env
bash tools/build_recon_all_fs_cpp_conda.sh /path/to/freesurfer-source /path/to/native-build
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 --native-bin-dir /path/to/native-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

指定 `--native-bin-dir` 即启用外部 `mri_em_register` GCA 配准；四个开关分别调用 `mris_fix_topology`、`mris_inflate` 与 `mris_sphere`、`mris_place_surface` 生成五张顶点图，以及 `mris_register`。`--native-sphere` 和 `--native-registration` 都要求 `--native-topology`。Python API 使用同名参数 `native_bin_dir=...`、`native_topology=True`、`native_sphere=True`、`native_surface_metrics=True`、`native_registration=True`；下述批量 Python API 也支持这些参数。入口检查二进制可执行并记录 SHA-256，不证明任意指定目录中的程序均由上述脚本编译。许可证从外部 `FS_LICENSE` 环境变量继承；Git 仓库和 wheel 均不包含二进制、许可证或外置权重/模板。原生阶段会用到 CPU，整例并非全 GPU；其输出仍需与官方同输入逐文件验收。

## 多被试 Python API

每个设备一次运行一例；同设备的病例顺序执行。每例由独立 Python 进程运行，结果保持输入顺序。批量调用没有独立命令行。

```python
from fnit.recon_all import run_recon_all_python_batch

reports = run_recon_all_python_batch(
    [
        {"t1": "/data/sub01_T1w.nii.gz", "subject_dir": "/scratch/subjects/sub01"},
        {"t1": "/data/sub02_T1w.nii.gz", "subject_dir": "/scratch/subjects/sub02"},
    ],
    "/path/to/weights", "/path/to/assets",
    devices=("cuda:0", "cuda:1"), threads=4,
)
```

## 与官方结果比较

```bash
python -m fnit.recon_all.compare_native_free \
  /path/to/official_subject /scratch/subjects/sub01 \
  --output /scratch/comparison.json
```

[比较器](../../src/fnit/recon_all/compare_native_free.py)输出体素、表面、逐顶点图和逐脑区统计差异。当前候选表面的顶点数与官方不同，空间最近点误差只能用于定位差异，不能证明同源顶点一致。逐文件严格门槛和待完成步骤见[发布验收](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)；独立阶段的配对结果见[移植记录](../../validation/recon_all/python_gpu_port/README.md)。
