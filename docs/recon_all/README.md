# Python recon-all：原始 T1 到皮层指标

[返回首页](../../README.md) · [实现](../../src/fnit/recon_all/) · [最新整例验证](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md)

`fnit-recon-all` 从一幅 T1w 生成脑体积分割、双侧 white/pial/sphere 表面、厚度、面积、顶点体积、曲率、aparc/a2009s/DKT 标注和脑区统计。调用 Python 包的 CPU/CUDA 算子和在 Conda 中从固定 FreeSurfer 源码编译的三个必需 C++ 程序；另有三个可选表面程序；快速/标准球面和球面配准调用 Python API。无需安装官方 FreeSurfer、FSL 或原生 recon-all 运行包。模型权重和模板单独下载并校验。

**当前仍是近似核心重建。** 最新从原始 T1 完成的 v3 整例运行 39 个阶段、进程墙钟 2903.79 秒；固定 138 项中 19 项通过、47 项缺失、72 项存在差异。[整例精度和时间](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md)及[13 张上游体积图](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/prefix_13.json)给出详细结果。随后[双侧拓扑同输入验证](TOPOLOGY_CONDA_GA.md)已得到逐点一致的 `orig`；但[精确 `orig` 的逐顶点试验](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/exact_orig_vertex_probe_20260927/README.md)仍有 1.080/1.103 mm 厚度 MAE，原因集中在 [最终 smoothwm 输入及 white/pial 放置](SMOOTHWM_FINAL_PARITY.md)。另一次[真实 T1 的 CPU 下游重放](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/prefix_surface_replay_20260927/README.md)完成 21 阶段，但沿用旧拓扑二进制，19 个通过项全来自复制的 MRI 前缀，不能视为当前代码的整例验收。新拓扑、white/pial、后处理和最终脑区指标仍待连续验收。

## 安装外置数据

Python ≥3.10；CUDA 运行需与驱动兼容的 PyTorch。仓库根目录安装：

```bash
python -m pip install '.[recon-all-python-stages]'
fnit-setup-weights --model recon-all --dest /path/to/weights
fnit-setup-recon-all-assets --dest /path/to/assets
```

默认下载 **10 个权重文件**（SynthStrip、SynthMorph affine、SynthSeg、EntoWM、MCA/dura、静脉窦及相关标签表；合计 145,283,019 字节）和 **19 个模板/图谱文件**（217,851,651 字节，约 208 MiB）。19 个文件包含双侧 folding atlas、填充所需的 `SubCorticalMassLUT.txt`，以及可选预白质链所需的三个 MCA/dura、静脉窦先验。该链还需 cropped/full 两张 MNI152 图像，可用 `--asset` 单独选择；下载器目前从约 515 MB 的上游归档提取，故未加入默认下载组。另两张目录中的 MNI LTA 不参与该链。安装器检查每个文件的大小和 SHA-256。`fnit-setup-recon-all-assets --all` 才下载全部 102 个阶段研究资源。已有文件可加 `--verify-only` 核验；两个目录均在仓库与 wheel 外。权重的来源和许可见[权重说明](../WEIGHTS.md)，模板清单见[资产验证](../../validation/recon_all/python_gpu_port/ASSETS_VALIDATION.md)。

仓库 [`environment.yml`](../../environment.yml) 也会安装 Python 阶段依赖；完整 recon-all 及 Conda 编译工具使用 [`environment-recon-all-cpp.yml`](../../environment-recon-all-cpp.yml)。SimpleITK/ITK、Numba 和部分 NumPy/Surfa 操作运行在 CPU；PyTorch 网络及部分体素、表面算子可在 CUDA 上运行。模型张量为 float32，默认允许 TF32；SynthStrip 和 SynthSeg 阶段为已验证的体素一致性局部关闭 cuDNN TF32。没有自动启用 float16 或 bfloat16。

## 单被试

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --device cuda:0 --threads 4
```

`subject_dir` 必须不存在或为空。没有 GPU 时传 `--device cpu`。Python 内调用：

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    "subject_T1w.nii.gz", "/scratch/subjects/sub01",
    "/path/to/weights", "/path/to/assets",
    device="cuda:0", threads=4,
    native_bin_dir="/path/to/recon-cpp-build/bin",
)
```

### 输入与输出结构

`run_recon_all_python(t1, subject_dir, weights_dir, assets_dir, ...) -> dict` 的 `t1` 是单幅 T1w 影像路径；`subject_dir` 是不存在或为空的输出目录；`weights_dir` 和 `assets_dir` 是经 SHA-256 校验的外置权重、模板目录；`native_bin_dir` 包含至少 `mri_em_register`、`mri_segment`、`mri_edit_wm_with_aseg`。`device` 是 `cpu` 或 `cuda:N`，`threads` 为正整数；五个 `native_*` 布尔值控制表面阶段。`FS_LICENSE` 在环境中指向私有许可证。

函数写入 FreeSurfer 风格目录：`mri/*.mgz` 和 `mri/transforms/*.lta` 存放体积分割及变换；`surf/lh.*`、`surf/rh.*` 存放有序三角网格及每顶点的厚度、面积、体积、曲率图；`label/*.annot` 为每顶点脑区标签；`stats/*.stats` 为逐脑区表；`scripts/` 为阶段日志；根目录的 `fnit-native-free-run.json` 是运行报告。当前并非每个官方文件都生成，文件级范围以[138 项比较器](../../validation/recon_all/python_gpu_port/compare_complete_subject.py)为准。

返回的 `dict` 与 JSON 内容相同：`status` 为 `complete` 或 `failed`，`input`、`subject_dir`、`device`、`threads` 和 `profile` 记录调用；`stages` 为按执行顺序排列的 `{name, seconds}` 列表；`surfaces` 以 `lh/rh` 为键，记录网格顶点/面数、分步耗时及近似步骤；`gca_registration`、`white_matter_chain`、`white_preaparc`、`topology_repair`、`sphere_generation`、`surface_metrics`、`sphere_registration` 记录所选实现及二进制哈希；成功时有 `total_seconds`，失败时另有 `failed_stage` 与 `error`。

`run_recon_all_python_batch(jobs, weights_dir, assets_dir, devices=(...), ...) -> list[dict]` 接受 `jobs=[{"t1": 路径, "subject_dir": 空目录}, ...]`；其他参数与单被试一致。每个设备一次处理一例，同设备顺序执行；返回列表按输入顺序包含各例上述报告，任一例失败则抛出 `RuntimeError`。批量入口没有 CLI。

`--n4-python /path/to/python` 可指定另一已安装 SimpleITK 的 Python 解释器执行 N4；该选项仍只启动 Python 算子。

## Conda 编译程序与可选表面阶段

按[独立 Conda 编译与验证说明](CONDA_CPP_BUILD.md)准备构建环境；三个必需程序和三个可选表面程序的功能、官方命令和精度/时间证据见[逐阶段说明](CONDA_CPP_STAGES.md)。将 FreeSurfer 源码固定在提交 `d932c45b7941662ea380a05efef580568b98d41a`，运行仓库的 [`build_recon_all_fs_cpp_conda.sh`](../../tools/build_recon_all_fs_cpp_conda.sh)：

```bash
conda activate /path/to/conda-build-env
bash tools/build_recon_all_fs_cpp_conda.sh /path/to/freesurfer-source /path/to/native-build
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 --native-bin-dir /path/to/native-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

`--native-bin-dir` 为必填项，提供 `mri_em_register`、`mri_segment` 和 `mri_edit_wm_with_aseg`；`--native-topology` 调用 [Python 居中球和 Conda 拓扑变体](TOPOLOGY_CONDA_GA.md)，再用 Python remesh 生成 `orig`，`--native-sphere` 调用 `mris_inflate` 并使用 Python quick/standard sphere，`--native-surface-metrics` 调用 `mris_place_surface` 生成五张顶点图，`--native-registration` 调用 Python `run_register_sphere`。后两个球面开关保留原名称以兼容已有 CLI/Python 调用；`--native-sphere`、`--native-registration` 和 `--native-white-preaparc` 都要求 `--native-topology`。Python API 使用同名参数 `native_bin_dir=...`、`native_topology=True`、`native_sphere=True`、`native_surface_metrics=True`、`native_registration=True`、`native_white_preaparc=True`；下述批量 Python API 也支持这些参数。入口检查二进制可执行并记录 SHA-256，不证明任意指定目录中的程序均由上述脚本编译。许可证从外部 `FS_LICENSE` 环境变量继承；Git 仓库和 wheel 均不包含二进制、许可证或外置权重/模板。原生阶段会用到 CPU，整例并非全 GPU；其输出仍需与官方同输入逐文件验收。

### 可选的预白质表面链

下列两张 MNI152 图像是 `--native-white-preaparc` 的额外资源；三个模型权重和 MCA/dura、静脉窦先验已在默认下载组。两张图像目前都从约 515 MB 的归档提取，按需安装：

```bash
fnit-setup-recon-all-assets --dest /path/to/assets \
  --asset average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.cropped.nii.gz \
  --asset average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.nii.gz
```

```bash
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --native-bin-dir /path/to/native-build/bin --device cuda:0 --threads 4 \
  --native-topology --native-white-preaparc
```

该开关在 `filled.mgz` 之后用 CPU 生成被试 MNI152 LTA、MCA/dura 与静脉窦标签及 `brain.finalsurfs.mgz`；每侧精确 `orig` 产生后，用 Python 计算 gray/white 阈值，调用 Conda `mris_place_surface --white` 生成 `white.preaparc`，再用 Python CPU 三轮平滑生成最终 `smoothwm`。启动时检查三份权重、两张 MNI 图像、三个先验、`mris_fix_topology_fnit` 和 `mris_place_surface`。开关关闭时沿用原路径。

[MNI152/辅助分割](MNI_AUX_CHAIN.md)、[Python finalsurfs](FINAL_SURFS_CHAIN.md)、[双侧预白质放置](WHITE_PREAPARC_CONDA_CHAIN.md)、[LH 候选前缀连通验证](../../validation/recon_all/python_gpu_port/white_connected_prefix_20260927/README.md)和[最终 smoothwm 同输入验收](SMOOTHWM_FINAL_PARITY.md)列出函数输入输出、官方命令与真实 T1 精度/时间。最终 `white` 仍由 `smoothwm` 近似复制，`pial` 仍由法线射线近似，皮层标签/脑区统计仍未通过整例验收。独立[双侧 Python pial.T1 函数](PYTHON_PIAL_PLACEMENT.md)在官方正确上游输入上逐顶点一致，但本开关不调用它。

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
    native_bin_dir="/path/to/recon-cpp-build/bin",
)
```

## 与官方结果比较

```bash
python -m fnit.recon_all.compare_native_free \
  /path/to/official_subject /scratch/subjects/sub01 \
  --output /scratch/comparison.json
```

[比较器](../../src/fnit/recon_all/compare_native_free.py)输出体素、表面、逐顶点图和逐脑区统计差异。最新完成的 v3 候选表面顶点数与官方不同，空间最近点误差只能用于定位差异，不能证明同源顶点一致。逐文件严格门槛和待完成步骤见[发布验收](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)；独立阶段的配对结果见[移植记录](../../validation/recon_all/python_gpu_port/README.md)。
