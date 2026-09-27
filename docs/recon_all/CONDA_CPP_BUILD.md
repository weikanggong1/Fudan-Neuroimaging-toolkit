# Conda 编译 recon-all 的 FreeSurfer C++ 阶段

当前整例从 FreeSurfer 8.2 固定源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 编译八个程序。`mri_em_register`、`mri_segment`、`mri_edit_wm_with_aseg` 是必需程序；五个表面程序由选项启用。无需安装 FreeSurfer 运行包；模型、模板和个人许可证单独提供。

## 安装与构建

从仓库根目录安装完整 Conda 环境：

```bash
CONDA_OVERRIDE_GLIBC=2.17 conda env create -f environment-recon-all-cpp.yml
conda activate fnit-recon-all-cpp
```

该 YAML 已在目标 glibc 2.17 条件下通过 Conda dry-run 求解；实际构建和整例使用等价的分步安装环境，尚未重复执行 YAML 完整创建。C/C++/Fortran 编译器、ITK 开发库和 CUDA 工具均来自 Conda；Python 附加包也安装在同一环境。gpucw1 的宿主 glibc 为 2.17，必须使用该 sysroot；用 2.28 构建的程序无法在该节点启动。

```bash
git clone https://github.com/freesurfer/freesurfer.git /path/to/freesurfer-source
git -C /path/to/freesurfer-source checkout d932c45b7941662ea380a05efef580568b98d41a
bash tools/build_recon_all_fs_cpp_conda.sh \
  /path/to/freesurfer-source /path/to/recon-cpp-build
fnit-setup-weights --model recon-all --dest /path/to/weights
fnit-setup-recon-all-assets --dest /path/to/assets
fnit-setup-weights --model recon-all --dest /path/to/weights --verify-only
fnit-setup-recon-all-assets --dest /path/to/assets --verify-only
```

脚本检查源码提交和洁净性、Conda 工具位置及 sysroot，在独立工作副本中只修改上游 CMake 的 Python 路径选择并记录差异。图像和表面算法源码不修改。输出保留构建日志、八个程序的 `ldd` 和 SHA-256、Conda 包列表及 FreeSurfer 软件许可证；个人 `license.txt` 不复制进去。[gpucw1 八目标构建记录](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md)退出 0，用时 185.01 秒。上游 `utils/version.cpp` 包含编译时间，所以重编会改变二进制哈希；目标节点运行前应记录实际哈希。

默认外置数据为 8 个权重文件（138,691,259 字节）和 16 个模板/图谱文件（217,498,051 字节），逐文件校验大小与 SHA-256。模板包含双侧 folding atlas 和填充所需的 `SubCorticalMassLUT.txt`。上述数据目录不含可执行程序。

## 整例调用

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

`--native-bin-dir` 必填并启用三个必需 C++ 程序；四个开关启用其余五个程序。`native_sphere` 和 `native_registration` 均要求 `native_topology`。单被试及多被试 Python API 接受同名参数。C++ 阶段在 CPU 上运行，神经网络和部分体素、表面算子在 CUDA 上运行。个人 `FS_LICENSE` 只经环境变量传入，不应写入 Git 或输出包。

[WM/filled 连续同输入验收](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md)中六张体积图逐体素一致。官方 `-ga` 拓扑模式、white/pial 几何、球面配准和后处理仍未通过连续整例验收；不应把已通过的局部阶段解释为最终指标一致。每个程序的功能、原生命令、准确度及耗时见[阶段报告](CONDA_CPP_STAGES.md)，完整门槛见[发布验收](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)。
