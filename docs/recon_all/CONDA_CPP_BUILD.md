# Conda 编译 recon-all 的 FreeSurfer C++ 阶段

当前整例从 FreeSurfer 8.2 固定源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 编译六个标准程序及一个 FNIT 拓扑变体。`mri_em_register`、`mri_segment`、`mri_edit_wm_with_aseg` 是必需程序；`mris_fix_topology_fnit`、`mris_inflate`、`mris_place_surface` 由表面开关启用，标准 `mris_fix_topology` 仅保留供诊断。标准球面和球面配准由 Python 实现。无需安装 FreeSurfer 运行包；模型、模板和个人许可证单独提供。

## 安装与构建

从仓库根目录创建 Conda 环境。已按这份 YAML 在 headcw 完整创建新环境，并在 gpucw1 验证 CUDA 与二进制启动；命令和结果见[安装实测](../../validation/recon_all/python_gpu_port/conda_yaml_install_20260927/README.md)：

```bash
CONDA_OVERRIDE_GLIBC=2.17 conda env create -f environment-recon-all-cpp.yml
conda activate fnit-recon-all-cpp
```

这次完整 YAML 环境已成功重新编译七个目标，但尚未用它重跑整例。C/C++/Fortran 编译器、ITK 开发库和 CUDA 工具均来自 Conda；Python 附加包也安装在同一环境。gpucw1 的宿主 glibc 为 2.17，必须使用该 sysroot；用 2.28 构建的程序无法在该节点启动。

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

脚本检查源码提交和洁净性、Conda 工具位置及 sysroot，在独立工作副本中修改上游 CMake 的 Python 路径选择，并从两份经 SHA-256 固定的 C++ 源文件进行三处有界数值补丁，构建 [FNIT 拓扑变体](TOPOLOGY_CONDA_GA.md)；原始源码和标准二进制保留。输出保留构建日志、六个标准程序及拓扑变体的 `ldd` 和 SHA-256、Conda 包列表及 FreeSurfer 软件许可证；个人 `license.txt` 不复制进去。以下[六目标 gpucw1 构建](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/six_target_build_20260927/REPORT.md)退出 0、用时 188.62 秒；六个标准程序均完成动态库检查和启动检查；拓扑变体为其后新增，计时不能沿用该旧构建记录。此前八目标构建保留为历史对照，不属于当前安装流程。上游 `utils/version.cpp` 包含编译时间，所以重编会改变二进制哈希；目标节点运行前应记录实际哈希。

默认外置数据为 10 个权重文件（145,283,019 字节）和 19 个模板/图谱文件（217,851,651 字节），逐文件校验大小与 SHA-256。模板包含双侧 folding atlas 和填充所需的 `SubCorticalMassLUT.txt`。上述数据目录不含可执行程序。

## 整例调用

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

`--native-bin-dir` 必填并启用三个必需 C++ 程序；四个开关启用拓扑变体、另外两个可选 C++ 程序和 Python 球面/配准。`native_sphere` 和 `native_registration` 均要求 `native_topology`。单被试及多被试 Python API 接受同名参数。C++ 阶段在 CPU 上运行，神经网络和部分体素、表面算子在 CUDA 上运行。个人 `FS_LICENSE` 只经环境变量传入，不应写入 Git 或输出包。

[WM/filled 连续同输入验收](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md)中六张体积图逐体素一致。官方 `-ga` 拓扑模式、white/pial 几何、球面配准和后处理仍未通过连续整例验收；不应把已通过的局部阶段解释为最终指标一致。每个程序的功能、原生命令、准确度及耗时见[阶段报告](CONDA_CPP_STAGES.md)，完整门槛见[发布验收](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)。
