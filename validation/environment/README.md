# 独立 Conda 环境验证

2026 年 9 月 28 日以 `CONDA_OVERRIDE_GLIBC=2.17` 对当前仓库根目录的 `environment.yml` 重新执行 `conda env create --dry-run`，成功解析 221 个 Conda 包，包括 Python 3.11.16、PyTorch 2.5.1、CUDA 11.8、torchtriton 3.1.0 和 Connectome Workbench 2.1.0。基础依赖中已删除全仓未使用的 `trx-python`；本次解算结果也不含该包。

AMICO 数值等价所需的 NumPy 1.26.4 CPython 3.11 manylinux x86_64 wheel另行从 YAML 中固定的 PyPI URL 下载；文件大小 18,252,005 字节，SHA-256 为 `666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5`。该 wheel 已实际导入，BLAS 标识为 `openblas64 0.3.23.dev`，并用于当前 AMICO 真实数据对照。

机器可读记录见 [`report.public.json`](report.public.json)。dry-run 证明依赖可解析；它不等同于在另一台机器上已经完整创建环境。
