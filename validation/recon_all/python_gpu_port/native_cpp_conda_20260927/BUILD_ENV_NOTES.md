# Conda 构建环境求解核验（2026-09-27）

本轮已在 headcw 的独立前缀 `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/conda_env` **实际安装** Python 3.11、PyTorch 2.5.1/CUDA 11.8、GCC/G++/GFortran 11.4、nvcc 11.8、ITK 开发库 5.4.7 和 glibc 2.17 sysroot，并从 FreeSurfer 固定源码编译六个程序。安装清单见同目录的 `conda-explicit.txt`。以下检查只验证文档命令的 Conda 求解；`--dry-run` 没有创建第二环境，也不等于整例重建通过。

在 headcw 对 `docs/recon_all/CONDA_CPP_BUILD.md` 的命令逐步检查：

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 原样 `conda create --dry-run`（Python、PyTorch/CUDA、GCC 11、ITK 运行库等） | exit 0；目标前缀未创建 | 远端 `logs/doc_conda_create_dryrun.log`；初始求解选中 sysroot 2.28，因此仍须执行文档的第三条固定命令 |
| 原样追加 `conda install --dry-run`（ITK 开发库、CUDART 开发库、imagecodecs、pytest） | exit 0；对实际已建环境报告全部已安装 | 远端 `logs/doc_conda_install_dryrun.log` |
| 单独执行官方 conda-forge URL 的 sysroot 2.17 `conda install --dry-run` | exit 0 | 远端 `logs/doc_conda_sysroot_dryrun.log` |
| 将全部关键包与 `sysroot_linux-64=2.17=h0157908_18` 作一次新的合并求解 | exit 0；目标前缀未创建；解析到 GCC 11.4、ITK 开发库 5.4.7、CUDA 11.8 和指定 sysroot | 远端 `logs/doc_conda_combined_dryrun.log` |
| 主页 `environment-recon-all-cpp.yml` 一键命令：`CONDA_OVERRIDE_GLIBC=2.17 conda env create --dry-run -f ... -p /tmp/fnit-recon-all-cpp-env-dryrun-20260927` | exit 0；`Solving environment ... done`，选中 sysroot 2.17、G++ 11.4、ITK-devel 5.4.7、nvcc 11.8；目标目录未创建 | headcw `/tmp/fnit-recon-all-cpp-env-dryrun-20260927.log`；输入 YAML SHA-256 `20086128cbccc530d97b4f4179317f230fb05c2bb6497b369f80eeccee3fe053`，与仓库文件一致 |

公开包地址为 `https://conda.anaconda.org/conda-forge/noarch/sysroot_linux-64-2.17-h0157908_18.conda`。分步的 `conda create` CLI 不混合包 URL 和命名规格，因此文档把该 URL 放在单独的 `install` 命令；一键 YAML 已按上表独立求解通过。实际已建环境使用字节对应的本地缓存包；公开 URL 已由 conda-forge 搜索确认，远端 `logs/conda_explicit_public.txt` 将显式清单中的本地缓存路径替换为这个 URL。

求解成功证明这些规格可获得并兼容；完整可用性另由本目录的源码构建记录和同输入指标报告验收。当前没有以这些 dry-run 命令再实际创建一个全新环境。
