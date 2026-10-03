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

从仓库根目录执行。安装脚本优先检出固定 Git 提交；Git 服务不可达时，改用同一提交的 codeload 压缩包，先核对压缩包 SHA-256，再核对解包源码树哈希。之后编译原生程序、复制到当前 `$CONDA_PREFIX/bin`，并检查运行时动态库。源码仅保存在本机 Conda 安装目录，不提交到 FNIT 仓库。下载和编译需要联网、磁盘空间和 C/C++/Fortran 工具链；这些工具链及 `curl` 已列入主页 `environment.yml`。在 glibc 2.17 节点创建环境时，可按本机需要设置 `CONDA_OVERRIDE_GLIBC=2.17`。首次安装后应核对 `share/fnit/recon_all_native_full/installed-bin.sha256` 和构建记录。新版安装目录使用 `_full` 后缀，保留旧环境中不足以构建非线性程序的裁剪源码目录，不把两者混用。2026-09-29 已在主页 Conda 环境用已校验的完整固定源码树运行新版安装脚本，[14 个必需程序的构建、安装、哈希及动态库检查](../../validation/recon_all/python_gpu_port/build_full_14_20260929.json)通过；新目录从原站完成 [98 项标准资产的全新安装与复核](../../validation/recon_all/python_gpu_port/asset_fresh_install_20260930.json)；无预装软件环境的整例仍需验收。

开发者已有固定源码时，可设置 `FNIT_RECON_ALL_SOURCE=/path/to/clean/source` 再运行同一脚本。源码必须是洁净的上述提交或已校验归档。直接构建入口仍是：

```bash
bash tools/build_recon_all_fs_cpp_conda.sh \
  /path/to/freesurfer-source /path/to/native-build
```

外部图谱服务器曾在全新安装中途断开；再次运行同一安装器后从断点续传，当前源码 `--verify-only` 对全新目录的 98 项标准资产全部通过。下载失败时可重试同一 `--dest`；安装器仍核对大小及 SHA-256。

原生程序包括 `mri_em_register`、`mri_segment`、`mri_edit_wm_with_aseg`、`mris_fix_topology_fnit`、`mris_remove_intersection`、`mris_inflate`、`mris_place_surface`、`mris_place_surface_white_fast`、`mrisp_paint`、`mris_curvature_stats`、`mri_label2vol`、`mri_warp_convert`、`mri_ca_register` 和 `mri_convert`；N4从FNIT源码另行编译，共15项安装产物。`mris_place_surface_white_fast`仅移除white未消费的面哈希构建，pial仍调用原放置程序。球面优化与配准调用FNIT Python/Numba实现。`mris_fix_topology_fnit`使用构建脚本记录的有限源码补丁；未修改的上游 `mris_fix_topology` 仅留作诊断，不进入标准运行路径。当前CUDA recon-all的MNI后处理已调用FNIT GPU实现，独立API默认及CPU链仍保留Conda路径。MNI152非线性模型还需外置约3.5 GB的 `synthmorph.deform.3.h5`；项目安装器从固定FNIT Release分卷下载、合并并核对完整SHA-256，参见[该阶段说明](MNI_NONLINEAR_CHAIN.md)。

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

已有[2026-09-27 环境安装记录](../../validation/recon_all/python_gpu_port/conda_yaml_install_20260927/README.md)只对应当时的独立 `environment-recon-all-cpp.yml`。现版主页环境已完成安装、程序发现和[两例真实 T1 连续运行](../../validation/recon_all/python_gpu_port/current_full_runs_20260930.json)；138 项均存在且当时的标准mesh_validation通过（未覆盖后续发现的white/pial穿越），但严格数值比较仅通过 5/138 和 2/138 项。无预装 FreeSurfer/FSL 环境中的整例隔离验收仍待完成。各程序输入输出、官方命令和阶段数据见[阶段索引](CONDA_CPP_STAGES.md)。

## N4构建与各轮安装范围

五阶段串行优化新增N4能力查询、内部拟合/重建计时和可选重建线程，接口见[N4说明](N4_ITK_CONDA.md)。主页安装脚本已经调用同一build_n4_itk_conda.sh；显式CMake配置后关闭Ninja自动重生成，避免共享文件系统时间戳造成反复配置。实际新产物已在现有主页Conda环境独立编译，两例量化前后与旧程序零差异，1/4重建线程没有整段收益，默认保留1。生产原生程序束只连接源码构建产物；[清单](../../validation/recon_all/optimizations/20261001_serial/native_bundle_manifest.json)与[实际构建](../../validation/recon_all/optimizations/20261001_serial/stage4/build.json)绑定哈希。

2026-10-01串行轮复用了已有环境与13项原生程序，没有重新创建干净环境或重编全部程序。该轮整合版本ff372d7在同一主页Conda环境构建wheel、编译FNIT FastPD扩展、安装至独立目标目录并验证API导入与CLI帮助；[历史安装报告](../../validation/recon_all/optimizations/20261001_serial/whole/integrated_main/reports/installation/report.json)和[历史整例](../../validation/recon_all/optimizations/20261001_serial/FINAL_RESULTS.md)保留各自版本证据。

2026-10-02五任务运行源码8d750e2的wheel私有target安装通过，171项recon-all Python源码SHA一致；15项原生产物的来源、SHA及运行依赖核验通过。GCA本轮有限重编，white复用任务2独立固定源码Conda构建，其余13项复用既有独立构建；GCA/white另有专项能力查询。[本轮安装报告](../../validation/recon_all/optimizations/20261002_parallel/root_install/REPORT.md)及[当前两例原始T1整例](../../validation/recon_all/optimizations/20261002_parallel/FINAL_RESULTS.md)绑定实际产物。主页安装脚本已纳入这些优化，但完整全新Conda创建、全部15项从头构建和无预装脑影像软件环境的物理隔离尚未验证；哈希、启动和ldd检查不代替这些验收。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 官方构建说明](https://freesurfer.net/fswiki/BuildGuide)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
