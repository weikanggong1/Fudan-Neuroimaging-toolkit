# PID 归属修复：临时模块安装与现场元数据补验

## 1. 功能简介

把已提交的 `profiling.py` 和同版 NVML 查询模块复制到新的临时 Python 导入目标，在实际旧内核上读取本进程身份、执行无可信映射的判断及 CPU 设备采样。只改变新核验进程的导入路径；原环境实体、冻结 raw GA 源码与报告保持原版本。

这是两个模块的安装和现场元数据验证，不是完整 FNIT wheel 或干净 Conda 安装。使用轻量独立命名空间 `fnit_sampler_install_probe`，避免导入整个生产调度。实际导入的模块文件与提交版本字节相同。

## 2. Python 调用、输入与输出

```python
from pathlib import Path
from verify_installed_sampler import run

installation_receipt = run(
    source_directory=Path("/data/sampler-source"),          # 本轮三文件冻结部署目录
    source_manifest_sha256="282db973ed18f098fec52bacd5a5cff45f1f7bbcc3d4da72f937eb81eea7370c",
    installation_directory=Path("/data/new-module-target"), # 尚不存在的临时导入目标
    output_directory=Path("/data/new-sampler-receipts"),     # 尚不存在的安全收据目录
)
```

四个参数全部必填，无影像、坐标或物理单位输入。`source_directory` 包含 `modules/profiling.py`、`modules/nvml_process_memory.py`、核验脚本及 `SOURCE_MANIFEST.json`；清单逐项绑定相对路径、大小和 SHA。`source_manifest_sha256` 是清单的完整 SHA-256。安装目标与收据目录须全新且相互独立；允许部署目录下具名的 `python_target`，不覆盖源码文件。

返回 JSON 兼容完成记录，包括状态、报告相对文件名、报告 SHA 和源码清单 SHA。`report.public.json` 保存实际模块 SHA、Python/Torch/kernel 版本、本进程可见 PID、NSpid、namespace inode 与启动 tick、映射可用性、CPU sampler 状态，以及 CUDA/TF32/线程/env 的前后检查。显存字段单位为字节；未执行查询保持 `null`。报告不包含 GPU UUID、服务器地址、账号、私有路径、cmdline 或日志。

源码大小/SHA 不符、实际导入位置不符、目标已存在时明确失败；运行检查失败保存安全失败报告并抛异常。整个核验无隐式 GPU 回退，也不自动提供宿主映射。实现见 [独立脚本](verify_installed_sampler.py)。

## 3. 命令行与复现

使用已有 Torch 环境，不下载或改写依赖。冻结部署布局及 SHA 见 [SOURCE_MANIFEST](reports/SOURCE_MANIFEST.json)。以下目录变量是本地示例，不是服务器路径。

```bash
sampler_source_directory=/data/sampler-source
sampler_installation_directory=/data/new-module-target
sampler_output_directory=/data/new-sampler-receipts
sampler_source_manifest_sha256=282db973ed18f098fec52bacd5a5cff45f1f7bbcc3d4da72f937eb81eea7370c

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python "$sampler_source_directory/verify_installed_sampler.py" \
  --source-directory "$sampler_source_directory" \
  --source-manifest-sha256 "$sampler_source_manifest_sha256" \
  --installation-directory "$sampler_installation_directory" \
  --output-directory "$sampler_output_directory"
```

现场通过 tmux 启动，`taskset` 指定四核。线程与 CUDA 可见性仅在新进程的启动环境中声明；脚本读取已有 Torch 配置，不调用全局线程/精度设置函数。它默认只做 proc 和 CPU 设备核验，不执行可选 GPU 查询。

## 4. 原软件对应关系

这是 FNIT 测量模块的部署核验，没有 FreeSurfer 算法命令。对应操作是读取自身 `/proc/<pid>/status` 中的 `Pid/NSpid`、`stat` 的启动 tick 及 `ns/pid` 链接。NVML 查询模块只安装源码；本次 CPU 采样没有构造 NVML 实例或启动 `nvidia-smi`。

## 5. 最新现场结果

| 项目 | 实际结果 |
| --- | --- |
| 模块提交 | `89e0da3e469295047d31636b9bba918f284af1cb` |
| 采样器源码及安装文件 | 28,023 字节；SHA `43fd54dd7217d789ff4eac0b6b9be702112444e9caf545f7f2f1c3f81acf09f8` |
| NVML 查询模块源码及安装文件 | 3,058 字节；SHA `1879606b4af4c10e088cff3c665255511dee872da91d4a9c8b2b1cb07550b348` |
| 解释器与计算库 | Python 3.11.16、Torch 2.5.1；kernel `3.10.0-1160.el7.x86_64` |
| 本进程身份 | PID 733；namespace/starttime 可读，身份错误为 `null`；NSpid 缺失 |
| 无可信映射 | `unavailable`、aliases 0；原因分别是 NSpid 缺失及未提供可信映射 |
| CPU sampler | `not_applicable`、query backend 为 `null`、样本为空；三类显存峰值均为 `null` |
| CUDA / 精度前后 | CUDA 未初始化；matmul TF32 为 false、cuDNN TF32 为 true，均未改动 |
| 线程与环境前后 | Torch intraop 4、interop 64，配置保持；环境键值无变化 |
| 安装校验 / Torch及模块导入 / 完整 probe | 0.004422602243721485 / 2.6741903368383646 / 2.679591305553913 秒 |
| 完成状态 | EXIT 0，10 项实际检查全部 true |

线程配置记录保留真实 interop 值，不能把启动时四核亲和性描述成修改了所有库线程配置。上述耗时是核验函数墙钟，包含验证、复制、Torch/真实模块导入和元数据查询，不含 SSH 上传或完整 CLI 启动；不是影像算法速度。

当前环境没有 `nvidia-ml-py` 分发包元数据，此项为 `null`。CPU 惰性行为不要求该包；本次没有验证 GPU 查询后端，也没有证明宿主/容器映射、FNIT 峰值或连续20GB预算。现场报告见 [report.public.json](reports/report.public.json)，完成及退出记录见 [COMPLETE](reports/COMPLETE.public.json)、[EXIT](reports/EXIT.public.json)。

收集的是脚本显式生成的五份安全 JSON 原字节，服务器 SHA 与本地 SHA 相同。后验源码/安装文件检查见 [collection_check](reports/collection_check.public.json)，逐文件原始 SHA 见 [EXPORT_MANIFEST](EXPORT_MANIFEST.json)。没有读取 MRI，因此本页不生成脑图或影像质量指标。

## 6. 版本记录与验证边界

- 旧源码现场审查：原模块 `17b12fdc…` 与历史 raw GA 显存记录保留在原报告，不回标为新代码。
- PID 修复：`43fd54dd…` 的 51 项本地合同及 12 个不变函数 AST 核对见 [修复专页](../20261011_gpu_pid_namespace/README.md)。本页不重复或改写它们。
- 本次新实测：新部署目录和临时模块 target，实际 proc/CPU 查询通过。原模块与安装文件的 SHA 完全相同；未更新旧 raw GA v1/v2。
- MRI/完整重建、GPU 查询、可信宿主映射和全包安装不在本次实测范围，保持未评估。

## 7. 参考资料与源码

- [FNIT ProcessTreeDeviceSampler](../../../../src/fnit/recon_all/profiling.py)、[NVML 查询模块](../../../../src/fnit/recon_all/nvml_process_memory.py)。
- [进程显存 API](../../../../docs/recon_all/GPU_PROCESS_MEMORY.md)。
- [Linux man-pages proc_pid_status](https://kernel.googlesource.com/pub/scm/docs/man-pages/man-pages/%2B/refs/tags/man-pages-6.11/man/man5/proc_pid_status.5)：NSpid 从 proc 挂载对应的 PID namespace 开始，单凭可见链不能证明 GPU driver 所用的命名空间。
