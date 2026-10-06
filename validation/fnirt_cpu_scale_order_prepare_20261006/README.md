# FNIRT CPU scale / SSD 累加来源与下一验证

## 1. 功能简介

本页核对同一保存状态中强度缩放梯度、SSD目标和系数梯度的计算顺序。当前是**只读源码分析及未执行计划**：没有新增数值运行、上传、求解或生产修改。已有导数轴除法控制未改善总梯度，scale误差保持 `2.546876526565711e-6`，因此下一验证只检查两项小标量。[源码定义审计](SOURCE_AUDIT.public.json)、[既有结果绑定](CONTEXT.public.json)、[有限PLAN](PLAN.public.json)。

## 2. Python 调用、输入与输出

本准备叶没有新增FNIT公共API。以下读取计划参数和保存状态schema，不读取NPZ像素：

```python
import json
from pathlib import Path

validation_directory = Path("validation/fnirt_cpu_scale_order_prepare_20261006")
validation_plan = json.loads((validation_directory / "PLAN.public.json").read_text())
expected_checkpoint = json.loads((validation_directory / "EXPECTED.public.json").read_text())
checkpoint_array_names = validation_plan["input_arrays"]  # 只允许下面四项成员
print(checkpoint_array_names)
print(expected_checkpoint["baseline_count"])  # 同点有效体素数14341
```

| 输入 | 格式与含义 |
| --- | --- |
| `fixed` | CPU FP32 `[24,28,24]`；缩放前raw Ref，并已与官方保存Ref逐位核对 |
| `state_residual` | CPU FP32同形状；已有current-moving状态的warp减缩放Ref，不重新采样 |
| `state_mask` | CPU bool同形状；同点有效mask，count14341 |
| `scale` | CPU FP64标量；保存点身份绑定，不优化或修改 |
| checkpoint文件 | 68成员私密NPZ；整体大小/SHA绑定，只恢复上面4成员；不会下载或公开 |
| 保存官方totalg | FP64 `[1177]`；仅未来自身两臂完成后读最后8字节作scale对照 |
| 保存Ref影像头 | gzip首348字节；核对FP32图像schema与radiological X-fast扫描方向，不读像素 |

输出仅包含标量、FP64 hex/ULP、schema、哈希、旗标和退出码。数组、完整官方源码和动态库不进入GitHub。[输入和参考绑定](EXPECTED.public.json)列出每项schema/stride/SHA。

`PLAN.public.json`参数：`canonical_leaf_proposed`指定未来独立源码/运行叶；`input_arrays`限定恢复成员；`input_shape`/`max_small_image_voxels`限定大小；`gate_order`固定前置门顺序；`maximum_calls`限定科学调用；`resource_limits`限定CPU8亲和性、锁等待120秒、worker60秒、controller210秒和8GB地址空间；`default_adoption_policy`规定失败即停、不放宽门槛、不变更GPU；`coefficient_Jte_trial`明确需另立计划。本文件尚未授权或派发科学worker。

## 3. 命令行调用

只读准备核验命令：

```bash
# 用主页已有Conda环境的Python；本命令只用标准库。
python validation/fnirt_cpu_scale_order_prepare_20261006/verify_prepare.py
```

没有用于数值控制的现成CLI。本次未生成或上传controller，也未登记新的服务器运行叶。未来数值控制需先冻结自有worker和独立输出目录，再按PLAN由共同CPU8锁运行一次。

## 4. 原软件调用与源码定义

这些内部标量没有独立官方命令行。完整官方调用见 [TorchFNIRT功能说明](../../docs/fnirt/README.md)。本页参考已安装FSL6.0.7.4定义，没有新增官方进程：

- `fnirt_costfunctions.cpp:885–898`：FP32 residual平方，按Z/Y/X扫描累入Double，最后除count；cost更新latest_ssd。
- `intensity_mappers.cpp:424–427`与`newimagefns.h:644–658`：FP32 Ref×residual，masked serial Double累加。
- `fnirt_costfunctions.cpp:992–993`：scale fullg的Double因子在累加后应用。
- `splinefield.cpp:766–877`：每系数在3D支持域按Z/Y/X累加；线程分不同系数，不拆同一系数归约。

源码SHA和行号见审计。源码定义不代表记录了旧solve3的机器指令或内部缓存。

## 5. 已有精度、时间与首差证据

这些数值全部引用已保存报告，**本轮没有重新计算**。

| 保存控制 | full scale g | SSD |
| --- | --- | --- |
| current-moving后除count | 10.452025165741516 | 60.329756040354546 |
| 官方保存moving＋FNIT采样后除count | 10.452035382537394 | 60.329755786510965 |

轴projection的总gradient相对L2 `2.483601764762327e-7 → 2.483864966670823e-7`，scale逐位不变。故该轴换算排除为此控制中scale残差的原因，候选没有接入默认。[已有真实控制与时钟](../fnirt_cpu_rhs_projection_control_20261006/README.md)保持原记录。

**已知源码首差**：生产direct/LM在FP32产品前较早归一化；现有FSL-order控制已改为乘积/Double累加之后除count。它仍有残差，因此下一步应分开测试Double归约扫描和最终factor2/count组合，不能继续把scale残差归因早除或轴projection。

**尚待数值验证**：官方串行X-fast与PyTorch归约的实际差值；官方预组合3D spline权重与FNIT Z→Y→X separable adjoint的差值。即使数学等价，乘法和相加分组也可能改变舍入。本次没有把这些可能性写成已证实原因。

同点原生Ref/ScaledRef/Robj/Mask/partial/Jte/bending缓存未在355项solve3/state文件中发现；该清单是当前叶的文件证据，不是全系统扫描。缺少这些同点值时不能断言唯一首差。λ/150也不能冒充已保存native SSD。本标量控制只用已有current-moving残差；先前替换moving的残差未单独存进checkpoint，不为复现它重新采样。

本次不生成脑图：未产生新的warp、分割或配准输出。既有真实脑图与完整失败结果见功能页。没有新的速度、端到端精度或安装验收。

## 6. 更新记录与后续边界

- 本轮：现场核对canonical main `7ff215ee`及registration/spline/assembly哈希，读取348字节Ref头，准备4成员标量计划；科学/上传/排队0。
- 上轮：导数轴除法控制完成17前置门，总gradient未改善；原负结果不变。
- 较早控制：仅替换保存moving输入，totalg残差下降，仍未逐位一致；原报告继续保留。

若未来serial sum和最终分组与baseline逐位相同，或没有改善，保留这一排除结论，转到输入缓存或独立有限Jte控制，不重复此试验。新worker尚未实现/冻结/执行，所有生产和GPU路径保持现状。

## 7. 原实现、许可与参考

成熟代码：[registration.py](../../src/fnit/fnirt/registration.py)、[spline.py](../../src/fnit/fnirt/spline.py)。仅公开自有解释和元数据，未复制FSL源、头、库或MRI。许可边界沿用 [第三方声明](../../THIRD_PARTY_NOTICES.md)。

原软件：[FSL FNIRT](https://git.fmrib.ox.ac.uk/fsl/fnirt)。参考：Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007)。
