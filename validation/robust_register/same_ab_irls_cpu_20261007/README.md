# CPU robust regression：同一真实 A/b 的首差与 QR 修复候选

## 1. 功能简介

本页定位 CPU robust registration 的回归求解差异，并提供可审查的 Python 观察器与 CPU Float LINPACK QR 候选。

同一真实线性系统有 **9,196 行、6 列**。原 SDK 与当前 PyTorch 求解器的首轮中位数、MAD、sigma、归一化残差、平方根权重、加权 A 和加权 b 全部逐位相同，首差出现在 QR 返回的 6 个参数。使用保存的同一加权 A/b 单独检验新候选后，**6 个 Float32 参数全部逐位一致**。

这是一个回归算子和首个 QR 算子的实测结果。完整配准原有 **17/20** CPU 门保持原状态；新 QR 尚未接入默认求解器，GPU 代码未改，本阶段没有 GPU 性能测试。

文件用途：

- [solver_boundary_probe.py](solver_boundary_probe.py)：在固定求解器的数学 AST 中插入观测点，保留原中位数、求和、QR 和 IRLS 调用。
- [source/cpu_linpack_qr_candidate.py](source/cpu_linpack_qr_candidate.py)：自有 CPU Float QR 候选；实现无 pivot 的 SQRDC 与 JOB=100 的 SQRSL 所需步骤。
- [benchmark_summary.json](benchmark_summary.json)：此次真实数据的差异、分层时钟、失败历史和验收范围。

NumPy、PyTorch、Numba 均已列入项目现有 Conda 环境，不新增运行依赖。候选计算使用 Float 工作数组，显式保留 f2c 标量返回、除法和平方根的 Double 边界。

## 2. Python 调用、输入与输出

### 2.1 观察一次原求解器

`make_solver_tap(solver_module, *, source_binding, save_event)` 返回只能运行一次的 CPU 求解函数。调用者先用既有实验加载方式取得固定版本的 `solver_module`，再传入实际 A/b；它不是新的生产配准入口。

| 参数 | 含义与格式 |
|---|---|
| `solver_module` | 已加载的固定求解器模块，包含 `weighted_qr`、`robust_regression`、`median_even`。观察器检查两个求解函数的实际源码来源。 |
| `source_binding` | 固定源码身份字典，包含 `path`、`resolved_path`、`bytes`、`sha256`。这四项均须与模块实际文件一致。 |
| `save_event` | 回调 `save_event(label, state)`；接收阶段名称和真实局部变量。需要保存数组时，回调应在返回前复制并写入，避免后续状态改变影响记录。 |
| `design` | CPU `torch.float32` 二维 Tensor，形状 `(有效行数, 参数数)`，本例为 `(9196, 6)`。每列对应上游定义的一个变换参数，不改变列顺序或单位。 |
| `residual` | CPU `torch.float32` 一维 Tensor，形状 `(有效行数,)`。它是该线性系统原始右端项 b；观测中的迭代残差另行记录。 |
| `saturation` | 返回函数的具名参数，固定为 `50.0`，即此次实测 Tukey 截止参数。其他值或 GPU 输入被此验证观察器拒绝。 |

返回值继承原 `RegressionResult`：`parameters` 是参数向量，`sqrt_weights` 是逐行平方根权重，`report` 是迭代、停止原因和选择状态的字典。回调记录中位数、MAD、sigma、权重、加权 A/b、QR 参数、残差、误差和最终选择；不重复原中位数或原求和。

下面的函数接收已经加载并冻结的求解器模块；示例只记录事件名称和 Tensor 形状：

```python
from pathlib import Path
import hashlib
import torch
from solver_boundary_probe import make_solver_tap

def inspect_one_cpu_solve(solver_module, design_matrix, right_hand_side):
    # 固定实际模块文件；可额外与调用者预先保存的 SHA-256 清单比较。
    source_path = Path(solver_module.__file__)
    source_bytes = source_path.read_bytes()
    source_binding = {
        "path": str(source_path),
        "resolved_path": str(source_path.resolve()),
        "bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    observed_events = []

    def save_event(stage_name, state):
        # 本例记录元数据；真实数组持久化由验证运行器负责。
        tensor_shapes = {
            variable_name: list(value.shape)
            for variable_name, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        observed_events.append({"stage": stage_name, "shapes": tensor_shapes})

    one_cpu_solve = make_solver_tap(
        solver_module,
        source_binding=source_binding,
        save_event=save_event,
    )
    regression_result = one_cpu_solve(
        design_matrix,       # 原始未加权 A，CPU Float32
        right_hand_side,     # 原始 b，CPU Float32
        saturation=50.0,
    )
    return regression_result, observed_events
```

源码结构或身份改变会抛出 `ValueError`；第二次调用同一个观测函数会抛出 `RuntimeError`。观察器不接受任意求解器接口或任意参数组合。

### 2.2 单独调用新 QR 候选

`try_cpu_float_linpack(weighted_matrix, weighted_rhs)` 接收**已经完成 Float32 加权**的矩阵与右端项，不计算 MAD 或权重。

- `weighted_matrix`：普通、连续、无梯度的 CPU Float32 Tensor，形状 `(行数, 参数数)`；参数数范围为 1–12，行数至少等于参数数。
- `weighted_rhs`：同类型一维 Tensor，长度等于矩阵行数。行顺序必须与矩阵一致。
- 返回值：新建的 CPU Float32 参数向量，长度为参数数。候选内部复制工作数组，不覆盖输入。
- 不支持的 Tensor、CUDA、AD 包装、布局或形状返回 `None`，由调用者保留旧路径。检测到零对角或非有限解时抛出 `ValueError`；退化或非有限输入域尚未完成官方 API 等价验收。

```python
import numpy as np
import torch
from source.cpu_linpack_qr_candidate import try_cpu_float_linpack

# 使用上游真实线性系统已经生成的加权矩阵；不创建模拟 benchmark 数据。
weighted_matrix = torch.from_numpy(
    np.ascontiguousarray(np.load("weighted_design.npy"), dtype=np.float32)
)
weighted_right_hand_side = torch.from_numpy(
    np.ascontiguousarray(np.load("weighted_rhs.npy"), dtype=np.float32)
)
candidate_parameters = try_cpu_float_linpack(
    weighted_matrix,
    weighted_right_hand_side,
)
if candidate_parameters is None:
    raise ValueError("本次输入不属于 CPU Float QR 候选支持范围")
```

第一次调用包含 NumPy/Numba 导入和 JIT；本次实测将签名编译与唯一真实数值调用分开计时。观察器仅代理固定 helper 的 `solve_float`，其他 Numba 内部装饰器完整委托原 `njit`，退出时恢复全局装饰器。

## 3. 命令行调用

这两个内部验证函数没有独立生产 CLI。将第 2 节调用写入自己的诊断脚本后使用 `python your_diagnostic.py`。无需安装或调用 FreeSurfer 来运行 Python 候选。

本阶段不增加生产 `recon-all` 或配准命令行选项。

## 4. 原软件调用

完整配准的官方入口是 [`mri_robust_register`](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register)。本次函数对照使用独立 Conda SDK 中的原 `Regression<float>`：构造时传入同一 A/b，设置 `setFloatSvd(true)`，调用 `getRobustEstW(w, 50.0)`。

该内部函数没有独立官方 CLI。参考版本绑定 FreeSurfer 源码 `d932c45b7941662ea380a05efef580568b98d41a` 与 SHA 匹配的 ITK 5.4.0 VNL 源码；它是独立源码参考，不称为已安装 FreeSurfer 二进制的执行轨迹。SDK 源码和原生程序均未随本页发布。

## 5. 最新真实数据精度与耗时

### 5.1 当前 PyTorch 回归的首差

官方源函数与原 PyTorch 求解器各完成一次同 A/b 求解；两者均迭代 4 轮，选择第 3 轮并回滚，选择与回滚状态一致。

| 首轮边界 | 不同存储字数 | 最大绝对差 |
|---|---:|---:|
| 中位数、原始 MAD、sigma | 0 | 0 |
| 归一化残差、平方根权重 | 0 | 0 |
| 加权 A、加权 b | 0 | 0 |
| QR 返回的 6 个参数 | 6/6 | 8.22544 × 10⁻⁶ |
| QR 后残差 | 9088/9196 | 2.40326 × 10⁻⁴ |
| `sw` | 1/1 | 0.02783203125 |
| `swr` | 1/1 | 0.5078125 |
| 加权误差 | 1/1 | 1.41144 × 10⁻⁴ |

最终参数 6 个存储字均不同，最大差 **3.29018 × 10⁻⁵**；最终平方根权重有 **2473/9196** 个字不同，最大差 **1.41840 × 10⁻⁴**。

源代码和首差边界共同说明：原 VNL Float QR 使用 Float LINPACK Householder 工作数组，现 PyTorch 使用 reduced QR、`Q.T @ rhs` 和三角求解；相同加权输入上的结果首先在这里分开。首轮 `sw` 的权重输入相同，其独立差异符合原 Float 逐行累计与 Torch 归约的计算顺序不同。`swr` 和误差已经使用不同的 QR 残差，不能把两者差异全部归因于求和顺序。

### 5.2 同一加权 A/b 的新 QR 候选

复用已保存的首轮真实加权矩阵、右端项和原生 QR 参数，未重新执行原生程序或 MRI 准备。新候选执行一次数值求解：

| 指标 | 结果 |
|---|---:|
| 矩阵与右端项 | 9196 × 6；9196 |
| 参数 | 6 个 Float32 字 |
| 不同存储字 / 最大差 / P99 差 | **0 / 0 / 0** |
| 唯一 kernel | **0.595 ms** |
| 显式外层签名编译 | **0.919 s** |
| 冷导入 | 1.675 s |
| 冷完整 helper API | 4.045 s |
| 验证 entry / worker / outer | 18.441 / 25.216 / 25.396 s |

外层签名编译包含依赖函数的自动 lowering，不把它称为所有内部函数只编译一次。冷 API 包含约 3.117 秒的运行库身份检查点；各层时钟是嵌套关系，不能相加。

三次实际运行库检查点的映射数量为 104、116、116，均是已封印 123 个**字面文件身份**的子集，未发现未知或已删除映射。源码、输入、134 个资源、14 个冻结文件及进程树/锁/索引收尾检查通过。这是文件身份验证，不是动态 BLAS 调用轨迹。

### 5.3 回归函数耗时与整体范围

| 同 A/b 的整段回归函数 | 秒 |
|---|---:|
| 原 SDK，包含观测 I/O | 0.019434 |
| 当前 PyTorch，包含观测 I/O | 0.027005 |
| PyTorch 回调时间，包含在上一项 | 0.018959 |
| PyTorch 验证冷 entry | 7.786052 |

两种观测 I/O 实现不同，此表不代表无观测生产求解器的速度倍率。新候选 0.595 ms 是一个 QR kernel，不能与整段原回归 19.434 ms 作倍率比较。本阶段没有新的完整配准或 recon-all 端到端耗时。

精度结论来自服务器保存的原 COMPARISON 安全聚合，并绑定原件大小/SHA；矩阵、影像、A/b、日志和像素/参数 hex 没有传回本地，也未在本地重构原完整 JSON。此次对象是内部线性系统，不附缺少行到体素对应关系的脑图；上游准备图像和首 A/b 的真实对照见[首 A/b 报告](../first_ab_cpu_prefix_20261007/README.md)。

## 6. 最近版本与 benchmark 记录

| 阶段 | 结果与改动 |
|---|---|
| 首 A/b 对照 | 原始输入到 pyramid、halfway 和首个 A/b 的已观测边界一致；见上游报告。 |
| 当前 PyTorch 同 A/b 回归 | 首 7 个数值边界一致，首差 QR；4 轮、选择 3、回滚一致。 |
| 初次候选入口 | 原生求解已成功；候选在数值调用前因 4 字段身份与含 mode/uid 的 6 字段绑定整字典比较而停止。修复只投影共同字段，原 mode/uid 前后门保留；成功原生结果复用。 |
| Float LINPACK 候选 v2 | 补齐原 SAXPY 零系数直接跳过的语义；未知输入域尚未验收。 |
| QR 观察器首组 | 全局 `njit` 观察器误拦 Numba 内部装饰器，数值 0、显式 JIT 0；原失败保留。 |
| QR 观察器 v2 | 仅代理固定 helper 的 `solve_float`；一次真实 kernel 返回 6 字逐位一致，控制和收尾通过。 |

下一步在同一保存数据上分别验证：原 A/b 与**原生参数**生成的残差，以及原生残差/权重生成的加权误差。三个 CPU 基元验收后再检查自然 IRLS 全段和原有 20 项配准门。当前默认 API、GPU 路径和正式 17/20 结果未替换。

## 7. 原实现与参考文献

- FreeSurfer [`Regression.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register/Regression.cpp) 与 [`RobustGaussian.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register/RobustGaussian.cpp)：原回归、MAD、Tukey 权重和停止规则。
- ITK 5.4.0 [`vnl_qr.hxx`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/core/vnl/algo/vnl_qr.hxx)、[`sqrdc.c`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib/linpack/sqrdc.c)、[`sqrsl.c`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib/linpack/sqrsl.c)：Float QR 工作类型和原求解顺序。
- 同版本 [`sdot.c`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib/blas/sdot.c)、[`snrm2.c`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib/blas/snrm2.c)、[`saxpy.c`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib/blas/saxpy.c)、[`sscal.c`](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/ThirdParty/VNL/src/vxl/v3p/netlib/blas/sscal.c)：标量边界及向量操作。
- Reuter M, Rosas HD, Fischl B. *Highly accurate inverse consistent registration: a robust approach*. NeuroImage, 2010;53:1181–1196. [DOI](https://doi.org/10.1016/j.neuroimage.2010.07.020)。
- Dongarra JJ, Moler CB, Bunch JR, Stewart GW. *LINPACK Users' Guide*. SIAM, 1979. [DOI](https://doi.org/10.1137/1.9781611971811)。
- 项目 [robust registration 准备态说明](../../../docs/robust_register/PREPARATION.md)。
