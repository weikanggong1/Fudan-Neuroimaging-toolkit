# Conda 编译可选的 FreeSurfer C++ 阶段

这一配置从 FreeSurfer 8.2 的固定源码提交编译六个程序，并通过 `fnit-recon-all` 的显式程序目录调用。它不安装 FreeSurfer 运行包；模型、图谱和许可证仍放在代码仓库之外。默认纯 Python 入口无需这些程序。

## 源码与 Conda 环境

参考版本为 `d932c45b7941662ea380a05efef580568b98d41a`。从 [FreeSurfer 官方仓库](https://github.com/freesurfer/freesurfer)取得源码后固定提交：

```bash
git clone https://github.com/freesurfer/freesurfer.git /path/to/freesurfer-source
git -C /path/to/freesurfer-source checkout d932c45b7941662ea380a05efef580568b98d41a
```

以下关键包组合已在 gpucw1 使用；Conda 从 `pytorch`、`nvidia`、`conda-forge` 安装编译器、ITK、CUDA 工具和 Python。gpucw1 的宿主 glibc 为 2.17，因此必须固定 Conda sysroot 2.17：使用 2.28 构建虽能链接，却会在该节点因缺 `GLIBC_2.27` 而无法启动。

```bash
RECON_PREFIX=/path/to/recon-conda
conda create -y -p "$RECON_PREFIX" --override-channels \
  -c pytorch -c nvidia -c conda-forge \
  python=3.11 pip 'numpy<2' scipy h5py nibabel pyyaml numba tifffile \
  pytorch=2.5.1 pytorch-cuda=11.8 cuda-version=11.8 cuda-nvcc=11.8 \
  gcc_linux-64=11 gxx_linux-64=11 gfortran_linux-64=11 \
  cmake ninja make pkg-config libitk=5.4.7 eigen libpng libtiff fftw hdf5 zlib
conda install -y -p "$RECON_PREFIX" --override-channels \
  -c pytorch -c nvidia -c conda-forge \
  libitk-devel=5.4.7 cuda-cudart-dev=11.8 imagecodecs=2024.12.30 \
  'numpy<2' pytest
conda install -y -p "$RECON_PREFIX" \
  https://conda.anaconda.org/conda-forge/noarch/sysroot_linux-64-2.17-h0157908_18.conda
conda activate "$RECON_PREFIX"
python -m pip install --no-build-isolation -e '.[recon-all-python-stages]'
```

在仓库根目录执行最后一行。这个固定 sysroot 包已在 [conda-forge](https://conda.anaconda.org/conda-forge/noarch/sysroot_linux-64-2.17-h0157908_18.conda) 查到；直接安装该包的 dry-run 没有改变其他依赖。Python 附加包经 `pip` 安装在同一个 Conda 环境；“纯 Conda 编译”指 C/C++/Fortran 编译器、ITK 开发库、CUDA 工具及依赖来自这个 Conda 环境，不表示 Python 包全部由 Conda 提供。

## 构建与外置数据

```bash
bash tools/build_recon_all_fs_cpp_conda.sh \
  /path/to/freesurfer-source /path/to/recon-cpp-build
fnit-setup-weights --model recon-all --dest /path/to/weights
fnit-setup-recon-all-assets --dest /path/to/assets
fnit-setup-weights --model recon-all --dest /path/to/weights --verify-only
fnit-setup-recon-all-assets --dest /path/to/assets --verify-only
```

构建脚本检查源码提交、Conda 工具位置及 gpucw1 上的 sysroot，随后构建 `mri_em_register`、`mris_fix_topology`、`mris_inflate`、`mris_sphere`、`mris_place_surface`、`mris_register`。它在独立工作副本中只修改上游 CMake 的 Python 路径选择，并记录该差异；图像与表面算法源码不作修改。输出目录保留配置/编译日志、各程序的 `ldd`、SHA-256、Conda 显式包列表和 FreeSurfer 软件许可证。用户的个人 `license.txt` 不复制到输出目录。

默认外置数据为 6 个权重文件（135,394,037 字节）和 15 个模板/图谱文件（217,497,770 字节），均逐文件检查大小和 SHA-256。15 个图谱中包括双侧球面配准所需的 folding atlas；不含 C++ 可执行程序。

## 整例调用

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

`--native-bin-dir` 启用 C++ GCA 配准；四个开关启用其余五个程序的相应阶段。`native_sphere` 和 `native_registration` 均要求 `native_topology`。单被试 Python API 与多被试 Python API 接受同名参数。C++ 阶段运行在 CPU，神经网络和部分体素/表面算子运行在 CUDA；这条路径不是全 GPU。个人 `FS_LICENSE` 只通过环境变量传入，不应写入 Git、日志或输出包。

在相同表面输入上的二进制输出一致性与同一 T1 的整例比较应分别验收。当前近似 `aseg/filled`、white/pial 几何和部分 atlas/统计步骤仍可能让最终网格与官方不同；请以[发布验收门槛](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)和整例报告判断是否可用于需要官方数值一致的研究。
