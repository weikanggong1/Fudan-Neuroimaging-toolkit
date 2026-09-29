# recon-all 原生程序安装

标准单 T1 流程需要仓库的 `fnit_n4_itk`，以及从 FreeSurfer 8.2 固定源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 在 Conda 内编译的程序。它们不是系统预装 FreeSurfer 的可执行文件。脚本保留源码提交、CMake 补丁、构建日志、`ldd`、SHA-256 和 Conda 包清单；个人许可证不复制进安装目录。

## 用户安装

```bash
conda env create -f environment.yml
conda activate fnit
bash tools/setup_recon_all_native_conda.sh
fnit-setup-weights --model recon-all --dest /data/fnit-weights
fnit-setup-recon-all-assets --dest /data/fnit-assets
fnit-setup-weights --model recon-all --dest /data/fnit-weights --verify-only
fnit-setup-recon-all-assets --dest /data/fnit-assets --verify-only
```

从仓库根目录执行。安装脚本优先检出固定 Git 提交；Git 服务不可达时，改用同一提交的 codeload 压缩包，先核对压缩包 SHA-256，再核对解包源码树哈希。之后编译原生程序、复制到当前 `$CONDA_PREFIX/bin`，并检查运行时动态库。源码仅保存在本机 Conda 安装目录，不提交到 FNIT 仓库。下载和编译需要联网、磁盘空间和 C/C++/Fortran 工具链；这些工具链及 `curl` 已列入主页 `environment.yml`。在 glibc 2.17 节点创建环境时，可按本机需要设置 `CONDA_OVERRIDE_GLIBC=2.17`。首次安装后应核对 `share/fnit/recon_all_native/installed-bin.sha256` 和构建记录。2026-09-29 已用主页文件创建新环境，从已校验的固定源码归档安装 11 个原生程序，并确认 `ldd` 没有缺失库或系统 FreeSurfer 库；随后从全新下载的 codeload 源码树再次完成 12 个程序的构建，标准流程使用其中 11 个。[源码、程序哈希和构建结果](../../validation/recon_all/python_gpu_port/source_codeload_build_20260929.json)可复查。隔离整例仍需验收。

开发者已有固定源码时，可设置 `FNIT_RECON_ALL_SOURCE=/path/to/clean/source` 再运行同一脚本。源码必须是洁净的上述提交或已校验归档。直接构建入口仍是：

```bash
bash tools/build_recon_all_fs_cpp_conda.sh \
  /path/to/freesurfer-source /path/to/native-build
```

外部图谱下载服务器在本次 headcw 新目录安装中途拒绝连接；已有的 98 项资产经新环境 `--verify-only` 全部通过。遇到相同网络问题时，可将 `--dest` 指向已有的经校验资产目录运行 `--verify-only`，再把该目录传给 `--assets-dir`。这项本机复用不等于全新网络下载通过。

原生程序包括 `mri_em_register`、`mri_segment`、`mri_edit_wm_with_aseg`、`mris_fix_topology_fnit`、`mris_remove_intersection`、`mris_inflate`、`mris_place_surface`、`mrisp_paint`、`mris_curvature_stats`、`mri_label2vol`；N4 从 FNIT 源码另行编译。球面优化与配准调用 FNIT Python/Numba 实现。`mris_fix_topology_fnit` 使用构建脚本记录的有限源码补丁；未修改的上游 `mris_fix_topology` 仅留作诊断，不进入标准运行路径。

## 运行时检查

```bash
export FS_LICENSE=/private/license.txt
command -v fnit_n4_itk
command -v mris_fix_topology_fnit
ldd "$(command -v mris_place_surface)"
fnit-recon-all /data/sub01_T1w.nii.gz /data/subjects/sub01 \
  --weights-dir /data/fnit-weights \
  --assets-dir /data/fnit-assets \
  --device cuda:0 --threads 4
```

运行入口默认查找当前 Conda 环境的 `bin/`，检查必要程序可执行和资产哈希，并在报告中记录程序哈希。`FREESURFER_HOME` 在子进程中指向经过校验的 FNIT 数据资产目录；变量名不能用来推断是否调用了预装软件。严格隔离验收还须在没有系统 FreeSurfer/FSL 等软件的环境中核查进程、动态库和文件访问。

已有[2026-09-27 环境安装记录](../../validation/recon_all/python_gpu_port/conda_yaml_install_20260927/README.md)只对应当时的独立 `environment-recon-all-cpp.yml`。现版主页环境已完成安装和程序发现检查；它的完整整例运行尚待验证。各程序输入输出、官方命令和阶段数据见[阶段索引](CONDA_CPP_STAGES.md)。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 官方构建说明](https://freesurfer.net/fswiki/BuildGuide)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
