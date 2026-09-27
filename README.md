# Fudan Neuroimaging Toolkit (FNIT)

`fudan-neuroimaging-toolkit` 是独立的脑 MRI 推理包，在 CPU 或 CUDA 上运行。单项推理无需安装 FreeSurfer、FSL、TensorFlow、VoxelMorph 或 Neurite。`fnit-recon-all` 使用 Python/CUDA 阶段、外置权重与模板，并要求从固定 FreeSurfer 源码在 Conda 中编译三个白质/GCA 程序；另外三个表面程序按开关启用，正式球面和球面配准调用在冻结官方输入上验证的 Python API；自产输入的球面尚未通过同输入验收。无需安装官方 FreeSurfer。当前整例尚未证实与官方数值一致。Python 导入名为 `fnit`，单项功能的命令行入口为 `fnit`。

| 模态 | 功能 | 输出与用途 | 用法、原版对照与验证 |
|---|---|---|---|
| sMRI | SynthStrip | 脑图、脑掩膜、有符号距离场 | [SynthStrip 文档](docs/synthstrip/README.md) |
| sMRI | SynthMorph | 刚性、仿射、非线性及联合配准；应用已有变换 | [SynthMorph 文档](docs/synthmorph/README.md) |
| sMRI | WMH-SynthSeg | 脑结构标签、白质高信号及软体积 | [WMH-SynthSeg 文档](docs/wmh_synthseg/README.md) |
| sMRI | 33 类 SynthSeg | T1 结构标签和软体积 | [独立 SynthSeg 文档](docs/synthseg/README.md) |
| sMRI | SynthSR | 从单幅 MRI 或 CT 合成 1 mm T1w | [SynthSR 文档](docs/synthsr/README.md) |
| sMRI | TorchFAST | T1 三组织分割、PVE 与偏置场校正 | [TorchFAST 文档](docs/fast/README.md) |
| sMRI | GPU FAST VBM | 原始 T1w 到 warped GM、Jacobian 和 modulated GM | [FastVBM 文档](docs/fast_vbm/README.md) |
| sMRI、fMRI、dMRI 通用 | PyTorch FLIRT | 12-DOF 仿射及 6-DOF normmi 刚性；reference-grid image 和 FSL scaled-mm `.mat` | [FLIRT 文档](docs/flirt/README.md) |
| sMRI、dMRI | PyTorch FNIRT | GM 或 TBSS FA 非线性配准；intent-2007 coefficients、warped image 和 Jacobian | [FNIRT 文档](docs/fnirt/README.md) |
| sMRI、fMRI、dMRI 通用 | GPU applywarp | 应用 FSL dense warp、FNIRT coefficient、premat 和 postmat | [applywarp 文档](docs/applywarp/README.md) |
| fMRI | BIDS FEAT 核心、ICA-AROMA 与可选 WM/CSF/motion 回归 | 4D BOLD、运动矩阵、脑掩膜、ICA 降噪；B0/BBR/GDC 尚未完整复现 | [fMRI 用法与真实数据对照](docs/fmri/README.md) |
| dMRI | PyTorch TOPUP | UKB AP/PA b0 选择、Hz 场估计、畸变校正和 FSL 输出 | [TOPUP 文档](docs/topup/README.md) |
| dMRI | PyTorch EDDY | 逐 volume 运动、二次 EC、TOPUP 场、Jacobian 和离群切片校正 | [EDDY 文档](docs/eddy/README.md) |
| dMRI | PyTorch DTIFIT | FSL 默认 OLS tensor、FA、MD、eigenvalue 和 eigenvector | [DTIFIT 文档](docs/dtifit/README.md) |
| dMRI | PyTorch AMICO-NODDI | AMICO 风格 NDI、ODI、FWF、方向和拟合误差；当前默认 Conda 精度见验证报告 | [AMICO-NODDI 文档](docs/amico_noddi/README.md) |
| dMRI | PyTorch MMORF (`run_mmorf`) | T1 scalar 与 DTI tensor 联合估计 reference-grid、reference-axis mm pull warp | [MMORF 文档](docs/mmorf/README.md) |
| dMRI | dMRI 参数图 pipeline | optional TOPUP → EDDY → DTIFIT/NODDI → TBSS 或 MMORF；统一九图输出 | [端到端文档](docs/dmri_pipeline/README.md) |
| dMRI | PyTorch BEDPOSTX | 估计体素内纤维方向及不确定性，供概率追踪使用 | [BEDPOSTX 文档](docs/bedpostx/README.md) |
| dMRI | PyTorch ProbtrackX | seed→voxel 密度、稀疏 voxel×voxel 矩阵、有向 ROI×ROI 连接矩阵 | [ProbtrackX 文档](docs/probtrackx/README.md) |
| dMRI | UKBConnectome | 已校正 DWI 和 T1w 到四张结构连接矩阵；追踪和 SIFT2 为近似 | [connectome 文档](docs/connectome/README.md) |
| sMRI | recon-all（Python/CUDA + Conda C++） | T1w 到核心分割、双侧皮层表面、顶点指标和脑区统计 | [recon-all 文档](docs/recon_all/README.md) |
| fMRI | MS-HBM 17 网络 | fsLR32k 静息态时序到个体网络划分，纯 CPU | [MS-HBM 文档](src/fnit/mshbm/README.md) |

fMRI 的 BIDS 单 run 入口为 `fnit-fmri` 或 Python API；真实 UKB FEAT 子步骤已单独对照，完整 FIX 前数值结果尚未达到等价，见功能页。

recon-all 提供单被试命令行与 Python API；多被试并行仅提供 Python API。其余功能只提供单被试 Python 和单被试命令行接口；需要处理多个病例时，由调用方在包外组织任务与设备。仓库提供 [T1w 样例](examples/README.md)和 [FLAIR 样例](examples/WMH.md)。

recon-all 的神经网络与部分体素、表面计算使用 PyTorch/CUDA；N4 使用仓库 C++ 与 Conda ITK 在 CPU 上运行，去噪和部分网格、统计使用 Python CPU。当前仍需从固定 FreeSurfer 源码用 Conda 编译三个必需、三个可选表面程序；无需安装 FreeSurfer 运行包。[构建和单被试调用](docs/recon_all/README.md)列出输入、外置数据及输出结构。

依赖迁移状态（2026-09-28）：当前 recon-all 的 N4、SynthStrip、33 类 SynthSeg、Talairach affine 与可选 MNI152 affine 注册已移除 Surfa、SimpleITK、ANTsPy、DIPY 的运行时调用；在目标 Conda 环境禁用这些模块后，单被试和批量入口可导入（[导入检查](validation/recon_all/python_gpu_port/runtime_imports_20260928.json)）。独立 WMH-SynthSeg 和 SynthSR 也已移除 Surfa 运行时导入。TorchFAST 和 FLIRT 的运行时 Surfa 依赖也已移除；FLIRT 在两组真实 T1 配准中与改写前 FNIT 输出逐字节一致（[验证记录](validation/flirt_no_surfa_20260928/README.md)）。Connectome 自动配准的 Surfa 调用也已移除；真实 b0/T1 的新旧配准矩阵逐值一致，完整追踪未重跑（[验证记录](validation/connectome_registration_no_surfa_20260928/README.md)）。通用 SynthMorph、FNIRT、FastVBM、TBSS 等功能仍有 Surfa 路径，故当前 pyproject 和 Conda 配置仍安装 Surfa，整个仓库尚未满足新的无 Surfa 安装要求。当前版未重跑完整 recon-all，不能由导入检查推断端到端等价。

[当前 recon-all 阶段状态与剩余工作](docs/recon_all/STAGE_STATUS_20260928.md)记录已完成的真实数据同输入核对、未完成的整例验收及停止边界。

最近一次从原始 T1 完成的 **v3 历史整例**耗时 2903.79 秒，严格比较 138 项中通过 19 项、缺失 47 项、不同 72 项。[整例原始报告](validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md)所列的 13 张上游体积图逐体素一致，仅适用于当时的 N4 实现。当前[Conda C++ N4 单阶段对照](docs/recon_all/N4_ITK_CONDA.md)在同一真实 T1 上使 `nu0.mgz` 有 9 / 16,777,216 个体素差 1，最终 `nu.mgz` 有 8 个体素差 1–2；本版尚未重新运行完整 138 项验收，不能沿用 v3 通过率或耗时作为当前结果。

后续[双侧拓扑冻结输入验证](docs/recon_all/TOPOLOGY_CONDA_GA.md)已得到逐点一致的 `orig`。[自产输入左侧标准球面](validation/recon_all/python_gpu_port/candidate_sphere_first_difference_20260927/full_stage/README.md)在相同候选输入下与官方有序顶点和面一致，Python 用时 884.56 秒，官方 328.61 秒；与归档官方球面的均差 2.959 mm 来自上游 `white.preaparc`。[ribbon 和最终 aseg 的 Python 同输入核对](validation/recon_all/python_gpu_port/ASEG_RIBBON_RUNNER_20260928.md)在冻结官方前序输入上使五张体积图逐体素一致。[脑区体积图的 Python 同输入核对](validation/recon_all/python_gpu_port/ATLAS_VOLUME_RUNNER_20260928.md)已在冻结官方表面、标签和 aseg 上使 aparc/a2009s/DKT 与 wmparc 四张图全部逐体素一致。左侧[白质面 Python 逐轮诊断](docs/recon_all/WHITE_PYTHON_FIRST_PASS.md)已核对前三轮的隔离步进；第三轮第 34 步最大坐标差 7.91e-6 mm，最终 white 仍未完成。最终 white/pial、右侧连通表面、配准球面及脑区指标仍需同一 T1 连续验收；当前不能声称等价重建加速。

相关 CUDA 路径默认允许 NVIDIA TF32 matmul 和 cuDNN 内核；为保持已验证的体素一致性，SynthStrip 和 SynthSeg 局部关闭 cuDNN TF32；recon-all 的 Talairach SynthMorph affine 前向局部关闭 matmul 和 cuDNN TF32 以缩小 LTA/eTIV 误差，随后恢复原设置。模型与影像张量仍保持 float32，本包不会自动改用 float16 或 bfloat16。各验证报告记录实际开关。

FastVBM 的两个分支共用仿射配准、FSL 坐标转换、GPU 重采样、仅非线性
Jacobian 和调制步骤；差别只在非线性形变由 SynthMorph 或 TorchFNIRT 估计。

FLIRT 的十例验证见 [FLIRT 报告](validation/flirt/report.public.json)；当前 TorchFNIRT/TBSS 的真实数据诊断见 [dMRI 验证页](validation/dmri_pipeline/README.md)。FastVBM 因注册核心已更新，旧端到端结果已移除；重新验证前以 [FastVBM 验证状态](validation/fast_vbm/README.md)为准。TOPUP、EDDY、DTIFIT、AMICO-NODDI、MMORF 和 dMRI 参数图 pipeline 的真实数据对照见上表各子页。BEDPOSTX、ProbtrackX 和 connectome 的验证边界也分别记录在功能页。

FLIRT、FNIRT、applywarp、TOPUP、EDDY、DTIFIT、MMORF、dMRI TBSS 分支、BEDPOSTX、ProbtrackX，以及 fMRI 的 FEAT/MELODIC 风格实现所依据的 FSL 算法受 [FSL Software Licence 6.0](licenses/FSL-6.0.txt) 的非商业使用条款约束。各功能的移植范围和验证边界见对应子页面。

## 安装

需要 Python ≥ 3.10。使用 GPU 时，请安装与本机驱动兼容的 CUDA 版 PyTorch。

```bash
git clone https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit.git
cd Fudan-Neuroimaging-toolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[recon-all-python-stages]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Conda GPU 环境

仓库提供独立的 [`environment.yml`](environment.yml)，固定本项目在 gpucw1 验证的 Python 3.11、PyTorch 2.5.1、CUDA 11.8 和 ProbtrackX GPU 所需的 Triton 3.1.0 组合，并包含 benchmark 绘图使用的 Matplotlib 和 Pillow。必须从仓库根目录创建环境，因为配置最后以 editable 模式安装当前源码：

```bash
git clone https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit.git
cd Fudan-Neuroimaging-toolkit
conda env create -f environment.yml
conda activate fnit
python -c "import fnit, torch; print(fnit.__version__, torch.__version__, torch.cuda.is_available())"
fnit --help
```

`environment.yml` 同时固定 `pytorch-cuda=11.8` 和 `cuda-version=11.8`，防止 Conda 求解器混入需要更新系统 glibc 的 CUDA 组件。该环境保持模型和影像张量为 float32，并允许 NVIDIA TF32 matmul 与 cuDNN 内核；不会自动启用 float16 或 bfloat16。修改配置后可用 `conda env update -n fnit -f environment.yml --prune` 同步环境。

如果联网登录节点的 glibc 比离线 GPU 节点更新，应按 GPU 节点版本在共享文件系统创建环境。例如 GPU 节点为 glibc 2.17 时，在联网节点运行：

```bash
FNIT_ENV_PREFIX=/path/on/shared-storage/fnit-conda
CONDA_OVERRIDE_GLIBC=2.17 conda env create -p "$FNIT_ENV_PREFIX" -f environment.yml
conda activate "$FNIT_ENV_PREFIX"
```

然后在 GPU 节点激活同一路径。这样 Conda 会按目标节点 ABI 选择二进制包；环境仍由同一份 `environment.yml` 完整构建。

### recon-all 的 Conda 环境与 C++ 阶段

[`environment-recon-all-cpp.yml`](environment-recon-all-cpp.yml) 包含上述 Python 功能及七个 FreeSurfer C++ 构建目标所需的 Conda 编译器、ITK 开发库、CUDA 工具和 glibc 2.17 sysroot；默认的 `environment.yml` 保持轻量。已在 headcw 从这份 YAML 创建完整环境，并在 gpucw1 验证 CUDA 和编译程序的 glibc 2.17 启动；[实测记录](validation/recon_all/python_gpu_port/conda_yaml_install_20260927/README.md)列出命令、哈希和真实 T1 阶段结果。以下命令创建环境，不编译或下载 FreeSurfer 程序：

```bash
CONDA_OVERRIDE_GLIBC=2.17 conda env create -f environment-recon-all-cpp.yml
conda activate fnit-recon-all-cpp
```

随后按 [C++ 编译与调用说明](docs/recon_all/CONDA_CPP_BUILD.md)固定 FreeSurfer 源码提交并构建六个程序：GCA、白质初分割、白质编辑三个为当前必需，拓扑、膨胀和顶点图三个按开关启用；另编译标准拓扑程序供诊断。标准球面与配准已改用 Python。每个阶段的功能、等价官方命令和精度/耗时验证见 [C++ 阶段说明](docs/recon_all/CONDA_CPP_STAGES.md)。模型和模板仍通过下述独立命令下载。

## 下载和部署权重

Git 仓库及 wheel 均不包含模型权重。下面从 FreeSurfer 官方地址下载各功能的默认权重，校验 SHA-256，并记录权重目录；推理时不会自动联网。GPU recon-all 的 33 类 SynthSeg 与 WMH-SynthSeg 是不同模型，分别用 `--model synthseg` 和 `--model wmh-synthseg` 安装。

fMRI 的 FEAT 核心、ICA、ICA-AROMA 与混杂回归均无模型权重。

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint \
  --model wmh-synthseg --model synthsr
```

FastVBM 的 SynthMorph 分支从原始 T1w 开始时需要 `synthstrip.1.pt` 和 `synthmorph.deform.3.h5`；`python tools/setup_weights.py --model fast-vbm` 安装这两个后端的权重超集。TorchFNIRT 分支只需 SynthStrip；已有脑 mask 时该分支无需 checkpoint。GM 模板由用户提供，不由配置脚本下载。TorchFAST、TorchFLIRT、TorchFNIRT、TorchApplyWarp、TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI、TorchMMORF、TorchBEDPOSTX 和 TorchProbtrackX 不使用预训练权重。dMRI pipeline 的 TBSS 分支不使用权重；MMORF 分支仅需 SynthStrip 权重。

recon-all 的默认下载组包含 10 个外置权重文件（145,283,019 字节）和 19 个模板/图谱文件（217,851,651 字节）；其中三个小型先验供可选的 MCA/dura、静脉窦阶段使用。该链另需按需下载 cropped/full 两张 MNI152 图像，目录中的另两份 MNI LTA 不参与该链。分别运行 `fnit-setup-weights --model recon-all --dest /path/to/weights` 和 `fnit-setup-recon-all-assets --dest /path/to/assets`；文件均按 SHA-256 校验。当前 recon-all 入口需在环境中设置外部 `FS_LICENSE`，并提供自行编译的程序目录；三个必需程序在 CPU 上运行。Git 仓库和 wheel 均不打包这些程序或许可证。调用与验收边界见[recon-all 文档](docs/recon_all/README.md)。

独立使用 33 类 SynthSeg 时只需 `python tools/setup_weights.py --model synthseg`，随后运行 `fnit synthseg --i T1.nii.gz --o seg.nii.gz --csv-vols seg.vol.csv`，或使用 Python 的 `SynthSeg` 类；详见[独立接口](docs/synthseg/README.md)。

`--all` 下载全部模型变体；`--dest /path/to/weights` 指定本地目录；`--verify-only` 检查已有文件。安装后也可使用 `fnit-setup-weights`。SynthStrip、SynthMorph、WMH-SynthSeg、33 类 SynthSeg 和 SynthSR 可通过 Python 的 `weights=`、CLI 的 `--weights` 或 `FNIT_WEIGHTS` 指定权重；FastVBM 分别使用 `synthstrip_weights=` / `--synthstrip-weights` 和 `synthmorph_weights=` / `--synthmorph-weights`。官方地址、文件大小、SHA-256、许可和离线部署方法见[权重说明](docs/WEIGHTS.md)。

## 项目资料

- 各功能的调用、输出、原版对应和数值比较见上表各子页；[代码结构](docs/ARCHITECTURE.md)说明共享接口。
- [功能 benchmark 索引](validation/README.md)、[FastVBM 验证](validation/fast_vbm/README.md)、[dMRI pipeline 验证](validation/dmri_pipeline/README.md)、[MMORF 验证](validation/mmorf/README.md)和[模型及源码来源](docs/provenance.json)。
- [第三方许可与引用](THIRD_PARTY_NOTICES.md)。
