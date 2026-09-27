# Python recon-all：原始 T1 到皮层指标

[返回首页](../../README.md) · [实现](../../src/fnit/recon_all/) · [整例验证](../../validation/recon_all/python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)

`fnit-recon-all` 从一幅 T1w 生成脑体积分割、双侧 white/pial/sphere 表面、厚度、面积、顶点体积、曲率、aparc/a2009s/DKT 标注和脑区统计。运行时调用 Python 包及其 CPU/CUDA 算子，不要求安装 FreeSurfer、FSL 或原生 recon-all 运行包。模型权重和模板单独下载并校验。

**当前为近似核心重建。** 拓扑修复、white/pial 放置和球面配准尚未与 FreeSurfer 8.2 对齐；只生成核心输出的子集。一个真实 T1 的整例流程已跑通，严格固定配置的 138 项官方输出中 6 项通过、52 项缺失、80 项存在差异。皮层逐顶点厚度、面积、体积、曲率及脑区统计不能用于需要官方 recon-all 数值一致的分析。详细差异和逐阶段计时见[整例报告](../../validation/recon_all/python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)。

## 安装外置数据

Python ≥3.10；CUDA 运行需与驱动兼容的 PyTorch。仓库根目录安装：

```bash
python -m pip install '.[recon-all-python-stages]'
fnit-setup-weights --model recon-all --dest /path/to/weights
fnit-setup-recon-all-assets --dest /path/to/assets
```

默认下载 **6 个权重文件**（SynthStrip、SynthMorph affine、SynthSeg 及三个标签表；合计 135,394,037 字节）和 **13 个模板/图谱文件**（211,781,658 字节，约 202 MiB）。安装器检查每个文件的大小和 SHA-256。`fnit-setup-recon-all-assets --all` 才下载全部 102 个阶段研究资源。已有文件可加 `--verify-only` 核验；两个目录均在仓库与 wheel 外。权重的来源和许可见[权重说明](../WEIGHTS.md)，模板清单见[资产验证](../../validation/recon_all/python_gpu_port/ASSETS_VALIDATION.md)。

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
