# SynthSeg CPU原尺寸列缓冲复用：编译与接口核验（2026-10-06）

## 1. 功能与当前状态

**仅薄C++编译和接口加载通过，数值计算尚未执行，未接入生产。** 候选面向普通33类网络末端`up[3].conv0`：在一次layer调用内重复写入同一个列缓冲及小输出缓冲，保持原depth14分块、halo、FP32、NCDHW和矩阵尺寸。函数返回时释放缓冲，不保留跨层/跨调用缓存。

原目标环境的`Unfold3dCopyCPU`和FP32 ATen CPUBlas函数未动态导出，不能直接薄包装。当前方案为FNIT自有纯copy展开，再调用PyTorch已加载的公开MKL LP64 `sgemm_`；**不等同于隐藏的ATen wrapper**。目前未验证复制结果或SGEMM输出，与官方精度及速度均为`not_assessed`。

## 2. Python接口、输入和输出

`ColumnsReuse`是本报告的私有试验接口，不是公共FNIT API。当前冻结保留计算关闭状态，只能核验加载：

```python
from pathlib import Path
from prototype import ColumnsReuse

compiled_library_path = Path(private_compile_directory) / "columns_reuse.so"  # 私有编译产物
verified_provider_sha256 = frozen_plan["provider_sha256"]  # 现场Torch LP64库SHA
columns_reuse = ColumnsReuse(
    library=compiled_library_path,                # 自有胶水动态库
    provider_sha256=verified_provider_sha256,     # 不允许改用NumPy mkl_rt或另一BLAS
    allow_compute=False,                         # 当前只允许加载与ABI元数据核验
    allow_bounded_contracts=False,               # 未允许有界数值合同
)
```

| 参数/对象 | 意义 |
| --- | --- |
| `library` | 自有胶水`.so`；导出ABI元数据、copy、SGEMM三个函数。后两者只绑定，未调用。 |
| `provider_sha256` | 已加载Torch SGEMM实际provider文件的SHA；`RTLD_NOLOAD`、`dladdr`和SHA不符即失败。 |
| `allow_compute` | 默认`False`；此时任何`forward`请求立即抛错，不消费影像张量。 |
| `allow_bounded_contracts` | 默认`False`；未来经单独批准后才允许非真实shape的小合同。 |
| `layer` | 未来只允许当前CPU卷积的72→24、3³、pad1/stride1/dilation1/groups1、24项bias。 |
| `image` | 未来真实输入为1×72×192×224×256，FP32 CPU/eval/no-grad；为已存FNIT decoder检查点重join，不是官方输入。 |
| 预期输出 | 未来为1×24×192×224×256、连续NCDHW的pre-ELU FP32；当前未生成。 |

未来计算要求非oneDNN、无CPU autocast、无Module hooks，并保留训练/梯度/CUDA等原路径。当前原型的Tensor subclass和forward-AD dual拒绝尚待独立修订；不得把计划中的负例当成已通过合同。生产CPU卷积、1×1投影/低内存防崩保护和GPU数学完全没有修改。

## 3. CLI与Conda编译

没有新增公共命令。私有编译探针的三个必需参数如下，目标目录必须不存在：

```bash
# 仅编译和加载；需固定PLAN与头文件/源码/程序SHA，并由共同CPU锁调度。
python compile_probe.py \
  --root "$fnit_server_root" \
  --workspace "$frozen_columns_workspace" \
  --output "$new_private_compile_directory"
```

`--root`是固定FNIT入口；`--workspace`包含三个冻结源码及`PLAN.json`；`--output`保存编译标量、日志和私有`.so`。本次派发命令保存在[QUEUE.json](QUEUE.json)：外层300秒、等锁180秒、child130秒、编译120秒上限，地址空间4GB，8核亲和性、OMP/MKL1，CUDA不可见。编译器为**已有独立Conda GCC11.2工具链**；Torch runtime的`bin`里没有compiler。本轮没有安装、环境修改或同prefix清洁安装验收。

编译采用C++17、ABI0、`-fno-fast-math -ffp-contract=off`，链接已安装Torch/libc10及其OpenMP依赖，不链接新的BLAS。FNIT自有胶水不包含原软件源码或二进制。[INTERFACE.json](INTERFACE.json)记录现有包许可；MKL是Intel许可，libgomp为GPL加GCC exception，Torch为BSD3。上游header/库/编译器及`.so`不提交。未来若接入，必须先纳入主页Conda安装入口并完成清洁安装验证。

## 4. 原实现与保序方案

这是`mri_synthseg`内部CNN卷积，没有独立官方CLI。[PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)每块新建列矩阵；输出先填bias，GEMM使用beta1。[FP32 CPUBlas来源](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp#L160)先处理维度和可选bf32，再在LP64条件内调用SGEMM。当前生产非oneDNN，所有M/N/K>1且小于INT_MAX：原调用SGEMM的路径是源码与配置推断，本轮没有数值调用trace。

候选固定K1944=72×3³、N24、transpose NN、alpha/beta1。前13块M802816，末块depth10/M573440；lda/ldc均为当前实际M，ldb1944。末块用一维缓冲的紧凑prefix重排，不能沿用最大M行stride。copy按channel→kD→kH→kW和D/H/W只复制bits及边界+0，之后仍用Torch`copy_`填bias，再SGEMM。

最大列缓冲仍为**6,242,697,216B**，小输出77,070,336B，累计展开量不减少。期待减少大缓冲分配/释放和页面开销，但实际allocator对象、页面复用、fault归因未测，不能预先承诺提速。[上游默认CPUAllocator](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/core/CPUAllocator.cpp)调用alloc/free；这不证明现场allocator或libc页面行为。当前没有调用allocator getter/setter。

## 5. 实际结果与尚待验收的门

[COMPILE.json](COMPILE.json)是目标原始标量receipt的逐字节副本，SHA `74d1b96c…`；[RECEIPTS.json](RECEIPTS.json)保留转移前后bytes/SHA。[SUMMARY.json](SUMMARY.json)由[build_summary.py](build_summary.py)机械重算。

| 实际检查 | 结果 |
| --- | --- |
| Torch/ABI | 2.5.1/ABI0；float4/int4 metadata10404 |
| 编译 | rc0，1.329856s，compiler最大RSS317600KiB |
| 自有动态库 | 17,640B，SHA `e70100b6…`，仅留私有run |
| SGEMM provider | `libmkl_intel_lp64.so.2`，SHA `bd259733…`；Torch handle/global/LP64地址相同 |
| NumPy route | 不同SGEMM地址，禁止替代 |
| 调用范围 | copy0、SGEMM0、MRI0；只调用自有ABI元数据 |
| 14生产源码/flags | 前后相同；CUDA未初始化；无全局allocator/thread setter |
| 精度/耗时 | 未运行合同、MRI层或完整pipeline；无速度或准确性结论 |

`tensor_allocation_calls=0`计数指探针源码没有显式分配张量，**不是Torch import内部事件的动态审计**。编译时间也不是CNN时间。完整模型已有耗时见[真实profile](../seg_cpu_profile_20261006/README.md)，不得把本次编译或单层计划与官方完整时间相除。

冻结[PLAN.json](PLAN.json)的后续门尚未授权执行：copy与独立unfold bit oracle（signed zero/边界/strided）；同真实权重六例pre-ELU完整FP32逐位一致；重复缓冲poison全覆盖；input/weight/bias不变；shape/stride/finite/no-alias；GPU/训练/梯度/oneDNN/autocast/hooks/subclass/forward-AD fallback；异常与正常flags恢复；workspace生命周期。任何首差即停。

只有全部合同通过且再获审核批准，才能用原skip/value重join对**一个**`up[3].conv0`做旧新ABBA：A1只生成参考，B1/B2/A2做三次实际逐位门，RSS≤32GB、source14/权重/provider/flags保持，记录wall/进程CPU/faults。各arm180秒、外层23000秒，同8核/共同CPU锁；不执行tail/fullCNN/官方/GPU。

```bash
# 只重算已保存标量及SHA，不运行Torch、编译器或影像。
python validation/smri_cpu/seg_columns_reuse_20261006/build_summary.py --check
```

## 6. 更新与剩余工作

- 前一[64MiB slab候选](../seg_cpu_conv_slab_trial_20261006/README.md)改变M，在depth7首个逐位门失败；未做真实层。新候选保留原M以减少这一风险，不意味着新候选已逐位通过。
- 本轮仅完成读头文件/符号/provider/Conda工具链和薄compile/load。冻结原PLAN状态保留为编译前计划，最新状态在SUMMARY；不改写历史。
- 最初只读接口脚本存在嵌套字符串语法错误，发生在目标Torch import前，0远程修改/编译/数值；修复和原SHA保留[HARNESS_FAILURES.json](HARNESS_FAILURES.json)。
- 后续先独立v2修复严格Tensor类型/forward-AD守卫，再编写有界合同worker并审核；不得默改本次已编译的三文件/PLAN或重复当前探针。
- [INDEX_CLOSED.json](INDEX_CLOSED.json)记录六共享锁下的最终compile-only状态，其他任务和既有路径保留。当前生产速度门仍未过。

## 7. 来源

- [SynthSeg官方仓库](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis*, 2023。
- [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)、[CPUAllocator](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/core/CPUAllocator.cpp)。
- [PyTorch BSD许可证](https://github.com/pytorch/pytorch/blob/v2.5.1/LICENSE)。运行包许可按本次Conda metadata列于INTERFACE，未复制第三方源码或库。
