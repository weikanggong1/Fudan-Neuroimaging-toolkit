# FNIRT：已保存平滑 moving 的单输入 RHS 控制

## 1. 功能与结论

在同一个真实 GM coarse-level solve3 参数点，只把当前已存 FNIT smoothed moving 换成已存官方 smoothed moving。原 field、scale、fixed、basis 和有效 λ 不变，只有一次采样。worker 自然 exit0：12个前置/输入门全部通过；完整 g 对保存官方 g 的 relL2 从8.3642e−7降至2.3589e−7。

总误差下降71.8%，其中混合参数向量的 **global_scale 主导总g误差**；三个392项系数 block 仅分别下降21.1%、29.9%、26.7%。这证明该输入差异有贡献。全1177项仍与官方不同，剩余原因未定位；没有 warp/分割改善、完整配准等价或生产 CPU 接入结论。

原始 [summary](run/moving/summary.public.json) 64571B，SHA `c0d8cd7cb545a7d69bf15369ca0903aa88368aa2f1ee533a9f8bae3d2c8ed9d8`。完整12门、实际调用和3exit见 [ACCEPTANCE](ACCEPTANCE.public.json)，全部 block 数据与 scalar 解释见 [INTERPRETATION](INTERPRETATION.public.json)。

## 2. Python、输入和输出

这是独立验证 harness，没有新增公共 FNIRT 参数。成熟函数来自 [registration.py](../../src/fnit/fnirt/registration.py) 与 [spline.py](../../src/fnit/fnirt/spline.py)；[gradient_body.py](source/gradient_body.py) 从原源码摘取 g-only 语句，在 normal 系统构造前截断，没有调用 evaluate/linearize/solver。

| 输入 | 格式、意义与使用范围 |
| --- | --- |
| accepted checkpoint | 私有NPZ，68项schema，只读取32项必要payload；恢复原逻辑值/byte stride。保存F32 field/warped/mask/residual/gradient、F64系数/scale/basis/Gram；原对象/alias graph未持久化 |
| solve3参数 | 已存FP64向量1177项，SHA21d339a4…；packed checkpoint点逐SHA相同；x/y/z系数各392项，另有global_scale1项 |
| current_fixed.npy | F32 `[24,28,24]`，8mm；基线固定图，与checkpoint逐bit相同 |
| official initial_fixed | F32 NIfTI同网格，`Ref()`在global scale之前；与checkpoint raw fixed逐bit相同，不做阶段转换 |
| official smoothed_moving | F32 NIfTI `[224,288,288]`；原存储方向、100/SPM-like mean强度、6mmFWHM。唯一替换输入，不重新flip、normalize、smooth或resample |
| 当前moving文件 | 仅SHA/大小绑定；基线使用原saved warp/derivative，不重复采样当前moving |
| current LM/FSL-order g | 两个已存full FP64 g，用作基线1177项逐bit门；没有读取H/diag/normal weights |
| saved native g/state | 同点FP64 fullg与原state.json；g仅在四个自产梯度算完后作事后比较，state核λ身份 |

17个FNIRT包文件、原assembly控制源码和三个实际导入依赖（_nib、_transforms、flirt.coordinates）共21源码；12个输入文件，完整身份见 [expected](source/expected.public.json)。旧 e67 saved moving 是基线输入来源，当前 caab 精确g/scalar桥通过；不能改称最新 caab 重新预处理的图。版本关系见 [SOURCE_AST_AUDIT](source/SOURCE_AST_AUDIT.public.json)。

输出只含 JSON scalar、12gate、布局/shape/dtype/hash、源码/输入前后绑定、调用和时钟。无影像、参数数组、H、缓存对象、官方代码或许可证内容。32项恢复值与派生只读 operands 在结束后不变；[原始 schema/采样合同](run/moving/summary.public.json) 保存完整实际信息。

## 3. CLI、参数和资源

下面展示本次已执行命令；运行器拒绝已有生命周期和输出，不可用它重派本任务。

```bash
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
# 独立冻结源与独立输出目录；没有替换生产模块。
FNIT_WORKSPACE="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-rhs-moving-control-v1"
FNIT_OUTPUT="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-rhs-moving-control-v1/moving"
"$FNIT_ROOT/envs/default/bin/python" "$FNIT_WORKSPACE/moving_control.py" \
  --root "$FNIT_ROOT" \
  --workspace "$FNIT_WORKSPACE" \
  --output "$FNIT_OUTPUT" \
  --approved-moving-control
```

`--root`定义已有canonical仓库/输入；`--workspace`保存15项已审冻结payload；`--output`新建700目录，只写原summary；`--approved-moving-control`是本独立科学范围的执行门。实际调用由 [run_moving.sh](source/run_moving.sh) 设置CPU8、common outer锁/inner锁、环境和180s child。

实际 nodecw7 亲和性32,36,40,44,48,52,56,60；Torch8/interop1，OMP/MKL/OpenBLAS/Numba8。Python3.11.16、Torch2.5.1、NumPy1.26.4、nibabel5.4.2、Numba0.61.2；真实prefix匹配canonical env/default，解释器SHA已记录。RLIMIT_AS20e9B是address space上限；maxRSS567760KiB另列。CUDA未初始化、TF32/default dtype/grad状态前后相同。DSO仅前后身份快照，不是全程syscall审计。

## 4. 原软件范围与 λ

FNIRT 内部 g/SSD/Ref/Jte 控制没有独立官方CLI。本次没有新官方进程、capture、H、优化器或完整配准；仅复用 [共享状态原报告](../fnirt_shared_followup_20261006/README.md) 的同点native g/state。完整原软件调用见 [FNIRT功能说明](../../docs/fnirt/README.md)。

native solve3保存的有效λ9049.463427795125与控制相同；installed `Lambda()`使用最新SSD×baseλ，cost更新latest_ssd，grad读取该状态。**native SSD未单独保存**，λ/150不能充作其精确记录。当前旧状态评价λ9049.463406053183被同点控制固定为9049.463427795125。替换moving后的新SSD不更新λ；cost共用已验收FNIT bending-energy scalar，不是新native SSD-weighted评价。

原定义身份与行号见 [NATIVE_STATE_SOURCE_AUDIT](source/NATIVE_STATE_SOURCE_AUDIT.public.json)。当前 LM 用sqrt(count)两次F32加权；FSL-order控制保留F32产品→Double adjoint/count，但仍使用FNIT separable adjoint和F32坐标投影，不等于官方每一项累加。官方该点的warp/mask/partial/Jte/bending组件未保存，不能给剩余误差指定唯一原因。

## 5. 实际精度、计数和时钟

全部12门通过：current/official rawfixed逐bit、baseline residual/count/SSD/cost/λ、两完整baseline g逐bit、替换valid mask/count、scaled_fixed不变。mask/count均14341；采样coords/warped/gradient_voxels F32，valid bool，CPU/no-grad；坐标/field/scale/basis/λ未变。

| 对同点保存native fullg的relL2 | 当前LM → moving替换LM | 当前FSL-order → moving替换FSL-order |
| --- | --- | --- |
| 完整1177混合项 | 8.364202e−7 → 2.358861e−7 | 7.365717e−7 → 2.483602e−7 |
| coefficient_x 392项 | 5.857033e−6 → 4.620911e−6 | 5.836769e−6 → 4.605807e−6 |
| coefficient_y 392项 | 2.956449e−6 → 2.071114e−6 | 2.978472e−6 → 2.079908e−6 |
| coefficient_z 392项 | 2.321334e−6 → 1.700764e−6 | 2.312338e−6 → 1.669660e−6 |
| global_scale 1项 | 8.340414e−7 → 2.309100e−7 | 7.338208e−7 → 2.436728e−7 |
| fullg maxabs | 8.717428e−6 → 2.413479e−6 | 7.669919e−6 → 2.546877e−6 |

LM full误差下降71.8%，FSL-order full下降66.3%；这两个总量由scale项主导，不能替代三系数block验收。所有1177项仍与native不逐bit相同；这些是混合参数单位，不是voxel位移误差。

当前/替换SSD分别60.329756040354546/60.329755786510965，固定λ cost61.81924365728813/61.81924340344455。当前与替换warped有1343 bits不同、max2.288818e−5；projected derivative有8413 bits不同、max3.910065e−5。这是两输入所引起的变化，缺少同点official warp参考，不能称warp精度提升。

实际 sampler1、coords1、projection1、state-prefix2、LM-prefix2、FSL-prefix2；成熟 bending.normal1、常量读取4。evaluate/linearize/gradient-method/dense-field-expansion/design-diagonal/PCG/SCG/H-callback/native全部0；无H/diag/normal cache、原MRI、全配准或GPU。science/controller/after-preflight退出码全部0。

worker至summary写前5.675026s，imports/binding3.186288s，restore/baseline/fixed门0.081031s；其中导入、哈希、读取和冷JIT包含在worker时钟，不能作为配准耗时或速度比。本诊断未做新配准/分割验收，因此不生成新的配准脑图；既有真实完整图与完整失败记录仍见 [FNIRT功能页](../../docs/fnirt/README.md)。

## 6. 版本、收集与下一步

实际执行freeze `354aa28b…`、worker `855d34e5…`；旧准备freeze `81eb5e25…` 没有执行，仅补3个import依赖和实际runtime记录后冻结。科学worker无失败或重派；日志中的OMP deprecation是库提示。历史准备、transport和调用边界见 [STATUS_AND_FAILURES](STATUS_AND_FAILURES.public.json)。

29项原metadata、190195B分5个≤45000B块收回，全文件SHA重验；0数组下载。controller10886为唯一派发；六INDEX锁登记prepared/queued/completed，保留其他任务字段及权限。原source/summary/3exit/INDEX收据均保留，见 [COLLECTION_BINDINGS](COLLECTION_BINDINGS.public.json)。生产和GPU源码不改。

下一步只准备自有CPU预处理匹配策略，核查SPM mean、kernel和各轴FP32乘积/Double累积/Float写回来源及旧savedmoving provenance。未授权新增科学、重跑moving控制/native/full，未接入生产。

## 7. 源码、许可与参考

FNIT成熟来源：[registration.py](../../src/fnit/fnirt/registration.py)、[spline.py](../../src/fnit/fnirt/spline.py)、[_sampling_cpu.py](../../src/fnit/fnirt/_sampling_cpu.py)。本leaf只公开自有验证代码和scalar/hash，未复制官方源或数据；既有FSL来源和许可见 [第三方声明](../../THIRD_PARTY_NOTICES.md) 及 [FNIRT功能页](../../docs/fnirt/README.md)。

原代码库：[FSL FNIRT](https://git.fmrib.ox.ac.uk/fsl/fnirt)。参考：Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007)。
