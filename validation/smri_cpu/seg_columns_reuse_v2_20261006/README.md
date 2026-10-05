# SynthSeg列缓冲复用v2：守卫修订与有界合同计划（2026-10-06）

## 1. 功能和状态

**仅本地准备和静态核对；0上传、0worker、0数学、0MRI。** 本轮保留[v1编译与接口证据](../seg_columns_reuse_20261006/README.md)，复用同一自有CPP源码和SHA绑定的私有`.so`，不重复编译。仅修订Python eligibility/provider守卫，并准备一次metadata-load与短合同队列；待root审查和明确授权后执行。

数值部分仍是原depth14、末块depth10、K1944/N24、NN、bias预填+beta1、实际M紧凑prefix。最大6.243GB列缓冲和累计展开量不变，局限单次layer调用。相同公开SGEMM地址和参数不是逐位一致的证明，速度尚未测量。

## 2. Python接口及输入输出

仍是私有`ColumnsReuse`，不是FNIT公共API。加载构造使用`allow_compute=False`；`forward`默认拒绝数值执行。v2正常Tensor策略为：

- `type(layer) is CPUInferenceConv3d`；派生layer保留原forward。
- image/weight/bias只能是**exact `torch.Tensor`或exact `torch.nn.Parameter`**；Tensor/Parameter子类保留原forward。
- 三个张量`unpack_dual(...).tangent`必须为None；前向AD dual保留原forward，不写AD开关。
- 保留CPU/FP32、72→24、3³、eval/no-grad、非oneDNN、无autocast/hooks等条件，CUDA及训练/梯度等原路径不变。
- 每次数值调用前后检查已加载Torch handle与global SGEMM地址相同、provider路径/SHA未变，不改用NumPy mkl_rt或另一BLAS。

```python
# 只加载已经编译的自有胶水，不调用copy/SGEMM。
columns_reuse = ColumnsReuse(
    library=previous_verified_library_path,       # v1私有run里的同一.so
    provider_sha256=verified_provider_sha256,     # 目标Torch LP64库SHA
    allow_compute=False,                         # 当前不允许数值执行
    allow_bounded_contracts=False,               # 当前不允许小shape合同
)
```

后续合同输入是六个固定种子的FP32 CPU张量（深度1/2/7/31/33/31，72通道，13×17平面），含signed zero、strided view和大幅正负抵消；卷积读当前真实`synthseg_2.0.h5`的C72→24权重及24项bias。它们是有意义的小合同，**不代替真实MRI benchmark**。预期输出是相同shape空间范围、24通道的完整pre-ELU值；不得仅测均值或标签。

权重、当前14个源、原CPP/PLAN/compile receipt/`.so`路径及完整SHA固定在[PLAN.json](PLAN.json)。[READONLY_TARGET.json](READONLY_TARGET.json)是现场只读标量转录，保留当前canonical main `f75e7246…`、INDEX和来源；不声称保存原stdout JSON的逐字节身份。源码一致性以SHA为门，不以报告-only Git头变化拒绝。

## 3. 私有CLI和调度

没有新增公共CLI。以下是**审查通过后**的一次私有metadata/合同派发形式，目前未执行：

```bash
# 外层硬界包含全部等待和两worker；必须显式使用root批准的合同flag。
timeout --signal=TERM --kill-after=30 23000 \
  python run_prepared.py \
    --root "$fnit_server_root" \
    --workspace "$frozen_columns_v2_workspace" \
    --run "$new_private_columns_v2_run" \
    --approved-contracts
```

`--root`固定FNIT入口；`--workspace`是独立`seg-columns-reuse-v2`冻结源码；`--run`必须是新建或空私有目录，receipt存在不重跑/不覆盖；`--approved-contracts`是审查后的显式阶段开关。metadata worker60秒、contracts180秒，共同CPU锁/8物理核、OMP/MKL/OpenBLAS/Numba/Torch8、interop8，CUDA不可见。外层23000秒从首次排队计，不重置；没有MRI/ABBA mode或代码路径。清除loader/core/PYTHONPATH overrides。

metadata加载`load_interface.py`只验证旧binary/ABI10404/provider/source/flags，0copy/SGEMM。其exit0后才运行`check_contracts.py`。旧CPP/动态库不变；依赖和许可证沿用v1，未新增包、安装或清洁Conda环境验收。

## 4. 对应原步骤及比较方向

这是`mri_synthseg`CNN内部卷积，没有独立官方命令。本次不运行FreeSurfer/FSL或原CNN，也不改变GPU数学。原方为当前成熟`convolution_slabs`；小合同显式cap=`16*C*H*W*4`，使旧方保持depth14，模拟目标层同一矩阵几何，**不是旧函数对小shape的默认cap**。

copy oracle从同一含深度halo的chunk做H/W padding、`unfold`/`permute`和连续复制，按C/kD/kH/kW/D/H/W排列，比较全部FP32 uint32 bits。它不调用SGEMM。随后真实权重的旧/新pre-ELU全部值逐位比较，不能解释成与官方误差更小或完整分割等价。

实现以相同32位`int32`view比较bit pattern；符号解释不改变逐位相等判定。非finite结果的最大差记`null`并失败，避免错误报告因JSON不接受NaN而丢失。

## 5. 预声明合同及停止规则

本节全部是**待测门**，无pass结论。

| 合同 | 范围与门 |
| --- | --- |
| 纯copy oracle | 六shape所有slab加H=W1/signedzero，共13slab；全prefix bit差0、实际M/stride正确，未消费tail的A5 poison保持。 |
| 实际权重卷积 | 六shape，旧新各14-plane；候选12次copy/12次SGEMM；完整pre-ELU bit差0、finite、shape/stride/dtype一致。 |
| poison/缓冲 | 每slab columns先填A5，消费prefix全部覆盖；观测三个显式`new_empty`，两工作区weakrefs在返回后释放，不加Module hooks。 |
| 23fallback | grad、oneDNN、training、dtype、meta device、input requires_grad、CPU autocast、local/global四hooks、image/weight/bias subclass及dual、unsupported shape、layer subclass；原forward sentinel身份相同，候选copy/SGEMM0。 |
| 正常/异常 | 默认compute拒绝；原fallback异常、provider错误、注入copy错误传播；正常及异常scope恢复flags，异常缓冲释放。 |
| 来源及资源 | source14/CPP/binary/weight不变，输入/权重/bias不变、输出不别名；hook表/参数恢复，CUDA未初始化，RSS≤32GB，合同AS cap8GB。 |

任何copy或pre-ELU首个bit差即保存已完成标量，非零退出并停止剩余合同。合同全部通过也不自动授权真实层。metadata/contract时间只作诊断观察，含导入、加载、hash、poison及比较，不是速度benchmark。

实际CUDA不执行：本阶段用meta设备保护合同与源码审查，不把它写成GPU输出或速度验收。autocast合同仅在sentinel路径检查开关，没有BF16/FP16卷积或SGEMM。

## 6. 更新记录和下一边界

- v1仅compile/load通过，旧freeze/receipt/私有binary保留，本轮不覆盖或重编译。
- v2新增strict type/forward-AD及global/Torch provider地址一致性保护；五个prepared源码SHA固定。当前AST/compile和JSON静态检查通过，未import Torch/执行worker。
- [PREPARATION_HISTORY.json](PREPARATION_HISTORY.json)保留只读relay中Git不在PATH的失败：source14预核后停止，0修改/0Torch；修复为直接读实际HEAD及loose/packed refs，随后只读核对成功。
- root审查之后才可上传/六INDEX锁登记v2，再派发一次metadata+合同。此次准备未登记一个虚假的运行状态。
- 下一真实阶段仍需root另行批准：原skip/value检查点重join，唯一`up[3].conv0`旧/新ABBA，完整pre-ELU逐位/shape/RSS/provider/flags门。当前没有完整CNN/官方/输出图/CSV或耗时结论。

## 7. 来源

- [v1已编译的自有CPP和依赖证据](../seg_columns_reuse_20261006/README.md)、[已拒绝64MiB方案](../seg_cpu_conv_slab_trial_20261006/README.md)、[当前真实CPU profile](../seg_cpu_profile_20261006/README.md)。
- [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)与[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。
- [SynthSeg官方代码](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis*, 2023。
