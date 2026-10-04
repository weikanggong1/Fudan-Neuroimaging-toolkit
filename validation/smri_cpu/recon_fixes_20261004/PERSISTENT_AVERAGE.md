# 球面平均：持续 OpenMP 线程组的隔离试验

## 1. 功能与范围

CPU 有序平均的每轮只读旧梯度、写新梯度，轮间必须同步。现有 NumBa 并行实现每轮进入并行区；本试验用自有 C++ 在一次 OpenMP 并行区内完成全部轮次，并交错计算 x/y/z。每个轴仍按原邻接顺序做 FP32 加法，未改变邻域、倒数、迭代数或舍入规则。

**目前是隔离试验，不是生产默认。** 已完成真实完整表面的算子 ABBA 和完整左半球配准轨迹核验；两者数值通过。完整耗时受未改刚体搜索的波动影响，本次尚不支持整段速度收益。没有新跑原始 T1 的 recon-all。

## 2. Python、输入与输出

[persistent_average_adapter.py](persistent_average_adapter.py) 是完整阶段的诊断适配器：加载冻结生产模块，只替换其 CPU NumPy 平均函数。它不注册为产品入口。

```python
import os
from pathlib import Path

frozen_average_source_path = Path("/data/frozen/src/fnit/recon_all/mris_register_average_numba.py")
compiled_average_library_path = Path("/data/pilot/persistent_average.private.so")
os.environ["FNIT_AVERAGE_BASE_SOURCE"] = str(frozen_average_source_path)
os.environ["FNIT_AVERAGE_PILOT_LIBRARY"] = str(compiled_average_library_path)
# 诊断模块所在目录须在 Python 搜索路径中；产品代码不导入此模块。
from persistent_average_adapter import RegistrationGradientAverager

gradient_averager = RegistrationGradientAverager(ordered_neighbor_indices, vertex_degrees, device="cpu")
averaged_gradient = gradient_averager(input_gradient, iterations=16384)
```

- `ordered_neighbor_indices`：int64 `(N,K)` 邻接矩阵，前 `vertex_degrees[v]` 列有效；未使用的列忽略。
- `vertex_degrees`：int64 `(N,)`，范围为 0 到 K；有效邻点须位于 `[0,N)`。
- `input_gradient`：float32 `(N,3)` CPU 张量，本例是既有距离和面积目标生成的真实非零梯度。
- `iterations`：非负整数；零轮由原公开类返回 clone。没有近似或提早停止。
- `device`：本试验指定 CPU；原类 CUDA 路径保持，未对它调用 C++。
- 返回：新 float32 `(N,3)` 张量，输入不被修改。CPU 线程数来自原模块已经设置的 NumBa 掩码，并按原规则恢复。
- 两个环境变量分别指定冻结 Python 源码和已编译自有库。二者均为私密验证产物；这份适配器不自动下载、编译或寻找软件。

## 3. 诊断 CLI 与 Conda 编译

编译器来自任务单独新建的 Conda-forge GCC/G++ 11 环境，没有修改正在运行的默认环境。创建 compiler prefix 的进程成功；编译本文件耗时 0.422 秒。这里只证明该算子可由纯 Conda 编译，不是整个 FNIT 全新环境安装证明。

```bash
conda create -y -p /data/pilot/compiler -c conda-forge --override-channels \
  gcc_linux-64=11 gxx_linux-64=11 eigen=3.4.0

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 PYTHONPATH=/data/frozen/src \
taskset -c 3,7,11,15,19,23,27,31 \
python benchmark_persistent_average.py \
  --sphere /data/subject/surf/lh.sphere \
  --smoothwm /data/subject/surf/lh.smoothwm \
  --compiler /data/pilot/compiler/bin/x86_64-conda-linux-gnu-c++ \
  --cpp-source persistent_average_prototype.cpp \
  --output /data/pilot/operator --threads 8
```

`--sphere/smoothwm` 为同一有序完整表面；`--compiler` 是可执行编译器；`--cpp-source` 指向[自有源文件](persistent_average_prototype.cpp)；`--output` 必须是新目录；`--threads` 设置同 CPU 预算及显式 OpenMP 线程组。Eigen 由该 prefix 的其他矩阵诊断使用，本算子不需要 Eigen 头文件。

编译固定 `-O3 -std=c++17 -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared`。C++ 库和完整梯度不提交仓库；报告保存 compiler、source、worker、gradient 和输入 SHA，以及运行库名称。完整阶段沿用 [benchmark_full_registration.py](benchmark_full_registration.py)，用 `--average-source persistent_average_adapter.py` 切换；[比较器](compare_persistent_registration.py)核对已成功保存的私密结果，公开报告见下节。

## 4. 原软件对应

有序平均属于原 `MRISaverageGradients`；完整原软件对照为：

```bash
mris_register -curv -threads 8 /data/subject/surf/lh.sphere \
  /data/assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
  /data/reference/lh.sphere.reg
```

本次复用此前同输入保存的官方结果和官方计时，不重复原软件。官方仅在独立 benchmark 环境执行；自有 C++ 算子不调用或链接 FreeSurfer。

## 5. 最新精度、时间及图

输入是公开 ds000114 v1.0.2 已有 FNIT 重建的完整左半球，119,451 点、238,898 面。nodecw7 同八个物理核心，使用原任务锁，CUDA 隐藏。实际编译器为 Conda GCC 11.4.0-13。

| 算子轮数 | 已接受 NumBa，中位数 | 持续线程组 C++，中位数 | 精度 |
|---:|---:|---:|---|
| 1 | 0.001238 s | 0.000823 s | 逐值、逐位相同，输入不变 |
| 16 | 0.012663 s | 0.005748 s | 同上 |
| 256 | 0.187765 s | 0.086488 s | 同上 |
| 16,384 | 9.182867 s | 3.832057 s | 同上 |

0 轮也逐位相同；独立 C 接口的零轮 clone 比原类稍慢，但原类零轮不会进入适配的函数。16,384 轮本组算子约 **2.40 倍**。全部 ABBA 时钟及首次编译/JIT 见[算子记录](results/persistent_average_prototype.public.json)。

完整阶段的有序面、全部 358,353 坐标分量、volume geometry 同此前已接受 NumBa 和官方结果；MAE/RMSE/max 均 0，保存负面积面 0。旧/新输入、共同源码、sulc seed、所有接受/停止轨迹和 cleanup history 相同。[完整比较报告](results/persistent_registration_pilot.public.json)绑定适配器、C++、库与比较器身份。

| 完整同输入阶段 | 先前已接受 NumBa | 隔离 C++ 试验 |
|---|---:|---:|
| 新进程墙钟 | 348.038 s | 451.127 s |
| API，含 IO | 345.714 s | 448.403 s |
| 108 次平均累计，含首次调用 | 48.069 s | 20.127 s |
| 未改 sulc 刚体搜索 | 49.294 s | 196.200 s |
| smoothwm 总阶段 | 150.018 s | 133.110 s |
| 进程树采样 RSS | 0.723 GB | 0.712 GB |

这是**非邻接、各一次**的共享节点记录。平均器约 2.39 倍，但完整流程更慢；刚体搜索发生在平均器之前，不能把其波动归因于 C++。先前官方为 261.199 秒。本试验没有达到完整速度验收，也不能用局部收益填补皮层拓扑、white/pial、统计或顶点对应的待验收项。

相同真实表面及配准球面脑图见[完整官方对照](FULL_REGISTRATION.md#5-最新精度完整时间和脑图)；本试验坐标与其中保存结果相同，没有另画相同数据。

## 6. 更新与下一步

- 已发布版本：NumBa 顶点并行，完整同输入左半球与官方坐标一致；CUDA 已有回归。
- 本次：自有 persistent OpenMP 算子、纯 Conda 编译、完整真实梯度 ABBA、完整配准坐标和接受轨迹核验；仅 validation。
- 后续生产候选：需要懒加载/缓存与并发构建、缺编译器回退、完整新进程对照、GPU 路径不加载 C++ 的实际回归。当前记录不能标为已落地或整段加速。

## 7. 原实现与参考

[FreeSurfer mris_register](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mris_register)及[utils](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/utils)。C++ 循环由 FNIT 编写，不复制上游代码、头文件或二进制；编译仅使用标准库和 OpenMP。

Fischl B, Sereno MI, Tootell RBH, Dale AM. *High-resolution intersubject averaging and a coordinate system for the cortical surface*. Human Brain Mapping 8(4):272–284, 1999。[论文](https://pubmed.ncbi.nlm.nih.gov/10619420/)。
