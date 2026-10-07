# FNIRT CPU 导数投影控制：总 RHS 未改善，候选不接入（2026-10-06）

## 1. 功能与结论

本轮定位当前 FP32 `gradient_fsl` 单位换算：当前 CPU 对 raw voxel derivative 乘已存 FP32 逆 pixdim，私有候选按实际几何符号除以正 FP32 pixdim。v2 的17个保存状态/位门通过，但对已保存原软件总 RHS 的 relative L2 **`2.483601764762327e-7 → 2.483864966670823e-7`**，没有整体改善。scale 元素逐位不变。**候选不接入生产默认；完整 FNIRT CPU 配准仍未通过。** [验收摘要](ACCEPTANCE.public.json)、[逐块表](BLOCK_METRICS.csv)、[解释](INTERPRETATION.public.json)。

v1 先因头信息表示守卫错误退出，采样/投影/RHS调用全部0；其结果不能判定投影算法。修正守卫后，在新 leaf 单独授权唯一 v2，没有重跑旧目录。两个原始 summary 与科学冻结均保留。

## 2. Python调用、输入和输出

[候选函数](v2/source/projection_adapter.py) `signed_axis_division(raw, moving_fsl2vox, voxel_sizes, axis_signs)` 属于 validation 私有控制，没有新增 FNIT 公共API。

```python
import torch
from projection_adapter import signed_axis_division

# raw_voxel_derivative_tensor: CPU Float32 [3,X,Y,Z]，requires_grad=False。
# moving_fsl2vox_matrix: 保存的CPU Float32 4x4矩阵；本控制要求上3x3为轴对角。
# moving_voxel_sizes_mm: 同一已绑定NIfTI header的三个正pixdim。
# resolved_axis_signs: 由双form方向及保存矩阵共同核验；本例(-1,1,1)。
with torch.no_grad():
    diagnostic_derivative_fsl = signed_axis_division(
        raw=raw_voxel_derivative_tensor,
        moving_fsl2vox=moving_fsl2vox_matrix,
        voxel_sizes=moving_voxel_sizes_mm,
        axis_signs=resolved_axis_signs,
    )
```

- `raw`：原成熟 sampler 单次返回的FP32导数 `[3,24,28,24]`；同一份raw数据同时用于两臂。
- `moving_fsl2vox`：已保存FP32几何Jacobian；对角逆值和符号必须与header逐位一致。rotation/shear、CUDA、训练/CPU autocast均拒绝。
- `voxel_sizes`：同已绑定header的正Float pixdim `.800000011920929,.7777777910232544,.7777777910232544`，不用affine列范数替代。
- `axis_signs`：原存储到NEWIMAGE radiological网格的链式导数符号。双form已知、一致、远离奇异后才能确定；正determinant给负X，YZ正，保存diagonal再核验。没有从native结果选择符号。
- 输出：同shape/dtype的FP32导数；worker将输出复制到原projection同stride，仅替换 `state.gradient_fsl`。除数为3元素FP32 Tensor、broadcast `[3,1,1,1]`，实际声明 `aten.div.Tensor`，没有0D/Python scalar reciprocal fastpath。

输入是既有 accepted checkpoint 的29/68项、既有 solve3第二accepted参数、官方已保存平滑moving及固定raw reference。parameters SHA `21d339a4…`，同点 count14341、固定λ `9049.463427795125`。[完整23源码/9文件绑定](v2/source/expected.public.json)和[PLAN](v2/source/PLAN.md)。不读原MRI、不重做平滑/normalize、不创建H或diag、不求解。

原软件总 g 的数值仅在自身两臂梯度完成且scale位门通过后加载；上传前/前后preflight只读其字节SHA。没有使用参考轨迹选择数学。输出只标量、schema、哈希、日志、exit；数组没有下载或发布。

## 3. CLI与真实执行边界

[实际runner](v2/source/run_projection.sh)、[worker](v2/source/projection_control.py)、[controller](v2/source/lifecycle_projection.py)。v2 leaf `fnirt-rhs-projection-control-v2` 已有结果，controller拒绝再次enqueue：

```bash
# 当时独立批准的一次派发；已有结果不可复用这个命令重跑。
python lifecycle_projection.py --phase enqueue \
  --freeze-sha256 780e109610e4f7f1f84bb639185b97950347e18de6fcd68e7f511e1bdead6a93 \
  --approved-lifecycle

# 只读本报告机械核验：标准库，无Torch/NumPy/模型或科学回放。
python verify_report.py
```

nodecw7八物理核 `32,36,40,44,48,52,56,60`；OMP/BLAS/Numba/Torch8，interop1。共同CPU8锁120s、module锁20s、controller360s、child180s+kill5s、AS20,000,000,000B；prepared/queued/completed六INDEX锁 sorted `LOCK_EX|LOCK_NB` 共同25s截止，失败关闭全部fd且未写INDEX，不自动retry。目录700、冻结payload和run输出600，worker umask077/output700。非冻结、仅哈希的UPLOAD_SERVER_VERIFY原文件为644，处于700 workspace内；实际权限记录保留。[派发](v2/run/launch.public.json)、[完成登记](v2/run/index_completed.public.json)。

v1 controller77738，原overall/science/post RC `1/1/0`；v2 controller89880，原三RC `0/0/0`。两阶段的canonical HEAD均 `7ff215ee…`。23源/9输入/科学冻结前后匹配，可用operands、flags全部不变；CUDA未初始化，FSL DSO前后快照空，仅代表记录时状态。[v1原summary](v1/run/projection/summary.public.json)、[v2原summary](v2/run/projection/summary.public.json)。

## 4. 对应原实现和定义差

这是官方 `fnirt` 内部 RobjDeriv→ObjVxs→Jte 分量，无独立投影CLI。对应已有原软件运行 `fnirt --config=GM_2_MNI152GM_2mm.cnf`；本轮未启动官方程序或调用/链接FSL库。

安装源码 `fnirt_costfunctions.cpp:600–603` 对volume<Float>直接 `/= ObjVxs`；`newimage.h:617–618`参数为Float，`newimage.cc:2270–2275`逐值除法。已安装DSO该operator符号含divps/divss。默认读取转换到radiological，sampling_mat以正pixdim对角；原存储负X导数由同一header的X翻转链式关系确定。[源码/ELF定义审计](v2/source/SOURCE_DEFINITION_AUDIT.public.json)。这是文件定义/指令证据，没有观察旧官方同点实际partial执行或缓存。

当前registration `caab8ffc…:942–946`在FP32中乘FP32逆矩阵（FP64 inverse后cast）。FSL-order控制保留成熟FNIT三轴separable Double adjoint和scale reduction；源码顺序/归约仍有其它差异。scale不依赖D投影，所以本控制必须保留其误差，不能把总残差全部归因projection。

## 5. 最新真实标量结果

v2 当前完整基线 SHA `b17ee043…` 与前次moving-only控制逐位桥接。17门只证明本控制状态、来源与单处替换合同；没有建立native gradient逐位等价。

| 对已保存native full g的分量 | 当前 relL2 | 候选 relL2 | 当前 maxabs | 候选 maxabs | 当前 RMSE | 候选 RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| full | 2.4836017647623272e-07 | 2.4838649666708229e-07 | 2.5468765265657112e-06 | 2.5468765265657112e-06 | 7.5679835288594313e-08 | 7.5687855526526678e-08 |
| coefficient_x | 4.6058066089470771e-06 | 4.601091849823152e-06 | 1.5309620278122511e-07 | 1.5293545226600404e-07 | 1.7136925461143896e-08 | 1.7119383153675539e-08 |
| coefficient_y | 2.0799084583552195e-06 | 2.0938693570456548e-06 | 1.4671044940711697e-07 | 1.4669704374511983e-07 | 1.5715545631953736e-08 | 1.5821032553530138e-08 |
| coefficient_z | 1.669660018220628e-06 | 1.6766947322234068e-06 | 8.1336665283190945e-08 | 8.2154684917859488e-08 | 1.0434201636586389e-08 | 1.0478163654937247e-08 |
| global_scale | 2.4367284016522854e-07 | 2.4367284016522854e-07 | 2.5468765265657112e-06 | 2.5468765265657112e-06 | 2.5468765265657112e-06 | 2.5468765265657112e-06 |

每个系数块392值，scale1值；两臂对native均1177位不同。三个系数块和scale是mixed参数单位，表中误差不是warp voxel位移。X略降，Y/Z略增；scale native绝对误差 `2.5468765265657112e-06` 完全保留，因此总RHS未改善。

两臂之间：D_fp32有12267/48384位不同、maxabs `1.9073486328125e-6`；full g有1051/1177位不同、maxabs `2.1484472863958493e-9`；scale位差0。候选完整g SHA `85480e64…`。所有baseline/fixed/mask/sourceflags门通过后才发生这项比较。

实际调用 coordinates1、sampler1、current projection1、candidate projection1、scalar prefix1、FSL-order prefix2、bending.normal1、cache reads2；evaluate/linearize/LM/H/diag/normal cache/PCG/native/GPU/完整配准0。λ不由替换moving的SSD更新；native SSD未独立保存。当前控制SSD `60.329755786510965`、cost `61.81924340344455`。

v2 import/binding `1.839485778s`，worker `4.438787737s`，self RSS `576780` KiB。单个进程含前置、JIT、两前缀和后门，不作速度比、完整注册计时或GPU性能结论。没有新warp/labels或新脑图；[旧完整orientation脑图](../smri_cpu/fnirt_cpu_orientation_20261004/cpu_brain.png)及负结果继续保留。

## 6. 更新、失败与后续范围

v1 首门在nib.load后header处失败，实际已读current fixed和official fixed两份保存数组；sampler/coordinates/projection/RHS/bending全0，控制未进入候选。原gzip348 SHA `fd2f1bfa…` exact；loaded binaryblock SHA `762ae128…` 只改bytes110/111/114/115/118/119，字段vox_offset/slope/inter由Nibabel5.4.2消费（analyze.py:914–915），14spatial字段逐字节一致。[只读证据](HEADER_REPRESENTATION_AUDIT.public.json)、[失败范围](FAILURES_AND_SCOPE.public.json)。

v2只将这一个守卫改为原gzip348 SHA加同loaded spatial bytes桥；header classify使用同一已绑定实际header。移除该guard/import后，worker完整AST与v1相同；数学adapter、23源、9输入、lambda和limits没改。新leaf独立授权，无自动科学重试。

旧orientation完整候选 coefficients RMSE `.084011→.127315`、warped `.020061→.031622`、Jacobian `.01327→.020135`，仍拒绝。此前[平滑整图桥](../fnirt_cpu_smoothing_bridge_20261006/README.md)和[moving-only RHS控制](../fnirt_cpu_rhs_moving_control_20261006/README.md)保留各自范围。本轮不能由projection定义一致恢复完整配准或warp/segmentation精度。

下一只读定位优先scale/Jte的组装累计及同状态中间缓存来源；不再重复相同division。本报告没有授权更多数学、求解、tighter tolerance、native capture或完整注册。准备中的4ce4元数据锁缺口已在实际25bf前修正；初次metadata precheck的cpu7 PATH缺git与只读取证命令失败保留原因，均未启动科学或上传数组。

## 7. 参考、许可与复现

[FNIRT源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)、[NEWIMAGE源码](https://git.fmrib.ox.ac.uk/fsl/newimage)、[FSL许可](../../licenses/FSL-6.0.txt)。Andersson、Jenkinson、Smith，*Non-linear registration, aka spatial normalisation*，FMRIB TR07JA2（2007）。复用已绑定的既有官方FAST GM及其保存平滑和level状态；输入出处与许可见[既有首差输入来源记录](../smri_cpu/fnirt_first_diff_20261004/README.md)。本leaf没有原始影像、实际header数组、官方代码/DSO/assembly或许可证内容。

计算使用项目Conda已有PyTorch/NumPy/Numba/nibabel，无新增依赖。原始metadata bytes与SHA均保存，57个原文件可机械复核；实际Conda prefix只记录basename/哈希。源中的统一FNIT目录是既有用户声明通用复现入口，未发布个人/临床输入路径、网络地址、认证路径或UKB字段。[payload范围](PUBLIC_PAYLOAD_SCOPE.public.json)。
