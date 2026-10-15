# recon-all 的 Triton 编译器选择

[重建入口](README.md) · [Python 安装与预检](PYTHON_RUNTIME_INSTALL.md) · [GPU 距离阶段](TOPOLOGY_GPU_DISTANCE_FITNESS.md)

## 1. 功能与范围

`inspect_triton_compiler()` 只读核验当前 Python 环境中用于 FNIT CUDA JIT 的 `ptxas`；`configure_triton_compiler()` 在首次科学 JIT 前固定该编译器，并检查进程内 Triton 已有缓存。它们不读写影像，不改变 TF32、autocast、CUDA 分配器或任何表面算法。编译器可用不证明 GPU 内核已运行，也不证明 recon-all 输出一致。

```mermaid
flowchart LR
  A[当前 Python 环境和目标 CUDA 设备] --> B[选择并核验 ptxas 版本及 SHA]
  B --> C[检查进程内 Triton 编译器缓存]
  C --> D[固定本次进程的编译器路径]
  D --> E[FNIT 后续 CUDA 核真实执行与单独验证]
```

## 2. Python 调用、参数与返回

```python
from fnit.recon_all.triton_runtime import (
    inspect_triton_compiler,
    configure_triton_compiler,
)

compiler_check = inspect_triton_compiler(
    device="cuda:0",  # 当前进程可见的目标 CUDA 设备；默认 cuda:0
    compiler_path=None,  # 无显式路径时先遵守 TRITON_PTXAS_PATH，再查当前环境
)
compiler_runtime = configure_triton_compiler(
    device="cuda:0",  # 与将执行科学 JIT 的设备一致
    compiler_path=None,  # 保持同一个已核验编译器选择
)
```

两个函数都只有 `device` 和 `compiler_path` 两个具名参数。`device` 默认 `"cuda:0"`；CPU 设备只返回 `{"status": "not_required", "device_type": "cpu"}`。`compiler_path` 默认 `None`，显式提供时应是当前账号可执行的 `ptxas` 文件；优先级为**显式参数 → 显式环境变量 `TRITON_PTXAS_PATH` → 当前 Python/Conda 前缀中的两个声明位置**。默认候选必须与 `torch.version.cuda` 的 release 匹配。显式指定的版本差异会记录在报告，不等于驱动或科学内核兼容性证明；失败不会搜索系统软件目录或静默换编译器。

返回值是可写入 JSON 的字典，没有影像、坐标空间或长度单位。CUDA 可用时包括 `status`、`selection_source`、`compiler_cuda_release`、`torch_cuda_release`、`matches_torch_cuda`、`compiler_version_text` 和 `compiler_sha256`；不包含私有绝对路径。配置函数另含 `configured`、`preexisting_path_cache`、`preexisting_version_cache`、`caches_cleared`、`idempotent_reuse`。`gpu_execution_verified=False` 明确表示只做编译器核验。编译器文件不存在、不可执行、版本无法解析或当前进程已缓存另一编译器时抛异常；后者应改用新解释器，不能清除第三方 JIT 缓存继续运行。

配置函数要在科学 JIT 之前串行调用。已初始化 CUDA 的 Python API 可以调用，但不能与另一个线程同时修改 Triton 编译设置。FNIT 完整入口会在 CUDA 科学计算前调用它，并把实际选择记录在 `fnit-native-free-run.json` 的 `triton_runtime` 字段。MRI 后续体积为 conformed voxel 网格，表面为 surface RAS/mm；这些空间约定与编译器选择本身无关。

## 3. 安装与命令行

主页 [`environment.yml`](../../environment.yml) 声明 PyTorch 2.5.1、CUDA 11.8、`cuda-nvcc=11.8` 和 Triton 3.1.0。预检入口只调用只读核验，不编译或执行科学内核：

```bash
conda env create -f environment.yml
conda activate fnit
fnit-verify-recon-all-python --assets-dir /data/verified_assets \
  --output /data/new_runtime_check.json
```

`--assets-dir` 为已授权的资源目录，可省略以只检查依赖；`--output` 为 JSON 结果路径。需要显式编译器时，应先在当前进程环境设置 `TRITON_PTXAS_PATH` 为已核验文件，再调用完整重建。这个 helper 没有独立的影像处理 CLI；`fnit-verify-recon-all-python` 成功也不是 CUDA 实际执行或干净环境整例证明。

## 4. 原软件对应关系

FreeSurfer `recon-all -all` 没有 Triton 编译器设置步骤。本组件只服务 FNIT 自有 CUDA 核，对照原软件时应在隔离 benchmark 中运行原命令，不能用其执行结果作为编译器预检的输入。需要比较的实际算法阶段见 [拓扑距离](TOPOLOGY_GPU_DISTANCE_FITNESS.md)、[球面注册](SPHERE_REGISTRATION_PERFORMANCE.md)及[完整重建](README.md)。

## 5. 真实验证与耗时边界

2026-10-15 在已授权的 H100 环境完成只读预检：PyTorch 2.5.1 / CUDA 11.8、Triton 3.1.0，当前 Conda `ptxas` 为 CUDA 11.8，SHA-256 为 `6fefc19360e1075dd7407102a69aca9a956c131fc574a440615d398dd9558dda`。预检没有初始化 CUDA 或编译 SM90a 内核，`gpu_execution_verified=false`。因此默认 11.8 工具链在该设备上的实际 JIT 仍待验证；旧整例使用过显式 12.2 编译器，不能据此推断当前源码已经通过。

现有[真实拓扑距离实验](TOPOLOGY_GPU_DISTANCE_FITNESS.md)给出当时冻结 CUDA 核与 CPU 完整第一缺陷配对的阶段精度、CLI 耗时和真实局部差图；它验证的是该阶段算法，**没有单独量出本组件的加速量**。本次文档对应的当前编译器选择、版本 SHA、JIT 失败与否须以新运行的 `triton_runtime` 报告核对。[旧 f016 整例](../../validation/recon_all/optimizations/20261014_gpucw1_f016_partial/README.md)一例完成、一例失败，也不是随后补丁的编译器或整例验收。当前两例空目录完整耗时、目标 GPU 上 FNIT 自身峰值及整体指标等效尚未评估；不会把预检成功或旧阶段秒数写成当前整例结果。

## 6. 更新与参考

- 2026-10-11：主页环境已有匹配 CUDA 工具链和 Triton 依赖；独立 Python 预检只核验依赖及编译器。
- 第二轮接线：完整入口显式选择当前环境编译器、检查 Triton 进程内缓存、记录版本与 SHA；本页补齐现有接口说明，没有新增 MRI benchmark。

FNIT 实现见 [`triton_runtime.py`](../../src/fnit/recon_all/triton_runtime.py)。上游编译器说明见 [NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) 与 [Triton 文档](https://triton-lang.org/main/index.html)；原版阶段源码参考见固定 [FreeSurfer 仓库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
