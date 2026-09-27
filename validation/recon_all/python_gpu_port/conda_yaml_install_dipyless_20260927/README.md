# 移除 DIPY 后的 Conda 全量安装复核

这次只核验安装和入口，不运行完整 T1 重建。测试对象固定为提交 `140413d38b6cd45fdf0c970f65b4a40957cfdce2` 的源码快照；后续提交修改依赖时，需要另做安装核验。

| 固定项 | 值 |
| --- | --- |
| `environment-recon-all-cpp.yml` SHA-256 | `68808ef667afa46804deadfca269320ffdd230a6d6156ad560b9aaf99e8fd799` |
| `pyproject.toml` SHA-256 | `40600a1fe8408a50762f939f0a1a15cf7c36c3f85743e912912f104fcdc8af3e` |
| 节点、Conda | headcw；Conda 24.11.3 |
| 源码与新环境 | `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_yaml_install_dipyless_20260927/repo`、同目录下的 `env` |

源码由上述 Git 提交的 `git archive` 单独展开。新环境从空目录创建，运行命令为：

```bash
cd /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_yaml_install_dipyless_20260927/repo
CONDA_OVERRIDE_GLIBC=2.17 conda env create \
  -p /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_yaml_install_dipyless_20260927/env \
  -f environment-recon-all-cpp.yml
```

安装退出码为 **0**，墙钟时间 **3 分 47.21 秒**，安装进程最大 RSS **2,657,392 KiB**，环境占约 **11 GiB**。这些是共享存储上一次安装的观测值，不代表其他节点或网络条件的安装时间。

## 已检查的入口与依赖

| 检查 | 结果 |
| --- | --- |
| FNIT 与主要运行包 | FNIT 0.14.0、Python 3.11.16、PyTorch 2.5.1/CUDA 11.8、nibabel 5.4.2、Numba 0.67.0 |
| Conda 原生编译依赖 | GCC/G++/GFortran 11.4.0、ITK/ITK-devel 5.4.7、CUDA NVCC 11.8.89 均在新环境中 |
| DIPY | `importlib.util.find_spec("dipy")` 返回空，已安装发行包列表中也没有 DIPY |
| NODDI Python | `TorchAMICONODDI` 和 `fnit.amico_noddi.kernels` 导入成功；随包方向资产形状为 `(500, 3)`、`(500, 3)`、`(32761,)` |
| 单被试 recon-all Python API | `run_recon_all_python` 导入和函数签名读取成功；以完整命名参数传入不存在的 T1、权重及模板路径，按预期抛出 `FileNotFoundError` |
| 命令行 | `fnit-recon-all --help` 与 `fnit-amico-noddi --help` 均退出 0 |
| 依赖完整性 | `python -m pip check` 退出 0，输出 `No broken requirements found.` |

入口测试时清除了 `FREESURFER_HOME`、`FS_LICENSE`、`FSLDIR` 和 `PYTHONPATH`。Python API 测试只确认安装后的导入、参数绑定和缺失输入检查；**没有运行重建阶段**，因此不证明整例输出、精度或运行时间。此次未重新编译 FreeSurfer 来源的 Conda C++ 程序，也未用此新环境运行真实 T1。此前的源码编译及真实 `mri_segment` 对照记录见[原安装报告](../conda_yaml_install_20260927/README.md)，它采用旧的 YAML 与包版本，不能替代本次快照的编译实测。

本提交仍直接安装 **Surfa 0.6.3、SimpleITK 2.5.6、ANTsPyX 0.6.3**。这三个依赖尚未完成用户要求的仓库内替换，因此这次安装通过不代表最新依赖限制已全部满足。

本目录保留[新环境 Conda 显式包清单](conda-explicit.txt)、[pip 已安装包清单](pip-freeze.txt)和[入口检查结果](smoke.json)。前两者的 SHA-256 分别为 `b944bdea697a1da7bcd2298be6c645a68b202d7f681b5d8f27b895f61f63a42d`、`64ed1f0cd867e5faa69e832ae110e09b112d119c21e7097de988c8655764bfbc`；pip 清单记录安装来源，其中部分 Conda 构建路径不是可独立复用的下载地址。
