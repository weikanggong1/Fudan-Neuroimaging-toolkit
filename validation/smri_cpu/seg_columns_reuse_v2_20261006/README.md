# SynthSeg列缓冲复用v2：守卫修订与有界合同结果（2026-10-06）

## 1. 功能和状态

**一次有限合同已自然exit0；尚未真实MRI层或完整模型验收。** 本轮保留[v1编译与接口证据](../seg_columns_reuse_20261006/README.md)，复用同一自有CPP源码和SHA绑定的私有`.so`，不重复编译。仅修订Python eligibility/provider守卫。root审查冻结源码后批准一次metadata-load与短合同，已在目标Torch2.5.1完成；六个真实权重小输入的完整pre-ELU均逐位相同。生产源码和GPU路径没有修改。

数值部分仍是原depth14、末块depth10、K1944/N24、NN、bias预填+beta1、实际M紧凑prefix。最大6.243GB列缓冲和累计展开量不变，局限单次layer调用。六个小尺寸合同已验证逐位一致；真实M=802816/573440及单层耗时尚未测量，不能用小尺寸结果外推。

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
    allow_compute=False,                         # 构造默认仍关闭数值；合同另显式开启
    allow_bounded_contracts=False,               # 只有已审查的合同worker显式开启
)
```

本次合同输入是六个固定种子的FP32 CPU张量（深度1/2/7/31/33/31，72通道，13×17平面），含signed zero、strided view和大幅正负抵消；卷积读当前真实`synthseg_2.0.h5`的C72→24权重及24项bias。它们是有意义的小合同，**不代替真实MRI benchmark**。输出是相同shape空间范围、24通道的完整pre-ELU值；六例全部逐位比较，不仅比较均值或标签。

权重、当前14个源、原CPP/PLAN/compile receipt/`.so`路径及完整SHA固定在[PLAN.json](PLAN.json)。[READONLY_TARGET.json](READONLY_TARGET.json)是现场只读标量转录，保留当前canonical main `f75e7246…`、INDEX和来源；不声称保存原stdout JSON的逐字节身份。源码一致性以SHA为门，不以报告-only Git头变化拒绝。

## 3. 私有CLI和调度

没有新增公共CLI。以下是本次经审查授权使用的私有metadata/合同派发形式。原目录已有receipt，不能再次调用：

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

下表为冻结PLAN预声明门，已由唯一worker验证。原始标量记录保留在[CONTRACTS.json](CONTRACTS.json)；[build_summary.py](build_summary.py)仅用标准库机械重算，不导入Torch或执行卷积。

| 合同 | 范围与门 |
| --- | --- |
| 纯copy oracle | 六shape所有slab加H=W1/signedzero，共13slab；全prefix bit差0、实际M/stride正确，未消费tail的A5 poison保持。 |
| 实际权重卷积 | 六shape，旧新各14-plane；候选12次copy/12次SGEMM；完整pre-ELU bit差0、finite、shape/stride/dtype一致。 |
| poison/缓冲 | 每slab columns先填A5，消费prefix全部覆盖；观测三个显式`new_empty`，两工作区weakrefs在返回后释放，不加Module hooks。 |
| 23fallback | grad、oneDNN、training、dtype、meta device、input requires_grad、CPU autocast、local/global四hooks、image/weight/bias subclass及dual、unsupported shape、layer subclass；原forward sentinel身份相同，候选copy/SGEMM0。 |
| 正常/异常 | 默认compute拒绝；原fallback异常、provider错误、注入copy错误传播；正常及异常scope恢复flags，异常缓冲释放。 |
| 来源及资源 | source14/CPP/binary/weight不变，输入/权重/bias不变、输出不别名；hook表/参数恢复，CUDA未初始化，RSS≤32GB，合同AS cap8GB。 |

预声明规则是任何copy或pre-ELU首个bit差即保存已完成标量、非零退出并停止剩余合同；本次没有首差。合同全部通过不自动授权真实层。metadata/contract时间只作诊断观察，含导入、加载、hash、poison及比较，不是速度benchmark。

### 已完成的逐项结果

| 输入case | 深度 | pre-ELU不同bits | 最大绝对差 |
| --- | ---: | ---: | ---: |
| single_depth | 1 | 0 | 0 |
| two_depth | 2 | 0 | 0 |
| boundary_internal | 7 | 0 | 0 |
| many_slabs | 31 | 0 | 0 |
| strided_odd_depth | 33 | 0 | 0 |
| cancellation | 31 | 0 | 0 |

13个copy/slab（包括H=W1、signed zero及末块紧凑M）全部bit差0；未消费poison尾保持。候选12次copy/12次SGEMM、23个fallback均符合预声明；另一次注入copy错误为Python stub，不能记作C++复制。六例均观测3个显式缓冲、两工作区返回后释放；错误路径3个缓冲也在异常scope结束后释放。provider错误在分配前传播是源码顺序证据，该负例未安装分配计数器。

来源14文件、五合同源码、旧CPP/PLAN/compile receipt/17640B `.so`在两个worker前后相同。输入/参数/权重文件、hook表和flags不变；正常及异常恢复通过，CUDA未初始化。进程maxRSS为500,989,952B；这不是真实层6.243GB工作区的峰值。

共同锁等待0.0026秒；metadata进程2.3178秒、contract进程3.3188秒，contract内部观测2.9680秒。时钟包含导入、加载、hash、poison、比较和合同负例，只作诊断，不与官方速度或完整CNN比较。oneDNN初态/终态均为True，候选数值scope明确设False/no-grad并恢复；CPU autocast负例只走sentinel，没有低精度核。

[QUEUE.json](QUEUE.json)为原退出记录：timeout PID166878及controller166879已消失，两worker rc0。完整原字节通过base64传回：[INTERFACE_LOAD.json](INTERFACE_LOAD.json) SHA `ee513f8c1e84e81ae1bb240e9fcb6cd788133c1b2b18ec4b74acfd0e4b6262ef`，[CONTRACTS.json](CONTRACTS.json) SHA `00568b0c517afbcc9480e993c38f50caf2ae7d82e95b3ee479c2998a04ff784c`，[QUEUE.json](QUEUE.json) SHA `6c92189de04a4f4254ea63da59ac1c71b487cd021a3eac9f4db0c038e7faeba5`；[RECEIPTS.json](RECEIPTS.json)区分原远端file和relay stdout。没有下载影像、输入数组、权重、`.so`或环境。

实际CUDA不执行：本阶段用meta设备保护合同与源码审查，不把它写成GPU输出或速度验收。autocast合同仅在sentinel路径检查开关，没有BF16/FP16卷积或SGEMM。

## 6. 更新记录和下一边界

- v1仅compile/load通过，旧freeze/receipt/私有binary保留，本轮不覆盖或重编译。
- v2新增strict type/forward-AD及global/Torch provider地址一致性保护；五个prepared源码SHA固定。准备commit `2fa00c46` 的AST/compile和JSON静态检查通过；随后目标节点唯一metadata和contract完成，五源码/PLAN未改。
- [PREPARATION_HISTORY.json](PREPARATION_HISTORY.json)保留只读relay中Git不在PATH的失败：source14预核后停止，0修改/0Torch；修复为直接读实际HEAD及loose/packed refs，随后只读核对成功。
- root审查后固定六payload上传成功，六INDEX锁保留其他字段并登记；唯一合同结束后状态为`bounded_contracts_passed_pending_real_layer_review`。准备时的PLAN/status保留历史，不将它覆盖成as-run结果。
- [TRANSPORT_FAILURES.json](TRANSPORT_FAILURES.json)记录首次上传transport字符串parse错误（发生在解释器执行前，0远端修改/worker/math）。只修本地转义并完成AST核对；科学源码/PLAN未改，数值worker没有重跑。
- 下一真实阶段仍需root另行批准：原skip/value检查点重join，唯一`up[3].conv0`旧/新ABBA，完整pre-ELU逐位/shape/RSS/provider/flags门。当前没有真实层ABBA、完整CNN、官方、输出图/CSV、脑图或速度结论；准备下一计划不等于授权执行。

## 7. 来源

- [v1已编译的自有CPP和依赖证据](../seg_columns_reuse_20261006/README.md)、[已拒绝64MiB方案](../seg_cpu_conv_slab_trial_20261006/README.md)、[当前真实CPU profile](../seg_cpu_profile_20261006/README.md)。
- [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)与[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。
- [SynthSeg官方代码](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis*, 2023。
