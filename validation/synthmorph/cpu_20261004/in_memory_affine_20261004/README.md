# SynthMorph CPU：真实内存影像 API 与缩放精度定位

[返回 CPU 四模式报告](../README.md) · [功能与参数](../../../../docs/synthmorph/README.md)

2026-10-04，在既有真实 T1 上新增 **一次 affine、extent 256、完整双向 SpatialImage 调用**，复用原官方和路径候选输出。调用完整执行，但本次先解码成 float64 的对象输出与原路径候选不逐值相同。随后两个**无 CNN**控制把第一分歧定位到 NIfTI 的缩放 dtype；直接向 ArrayProxy 请求 float32 的物化路线与路径的张量和规范化网络输入逐值相同。第三路线没有重新运行网络，完整输出状态保持 `not_assessed`。

## 1. 固定来源与实际调用路径

- 输入为 OpenNeuro ds003138 v1.0.1、CC0 的原始 T1；moving/fixed 全图均为 `224×288×288`。SHA 分别为 `afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19`、`73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21`。没有裁剪或用派生脑图替代输入。
- 本次冻结源 `task5_candidate_cpu_v4`，head `6f1e2b38925a481df3fa622f925af076df5436a9`，实际归档 `ffda47a74376fbaec07c3e8aedaacdc30f2a60398b919c0d5feae38e3beba0d9`，包含当时已审阅的未提交变化。此次核对 9 个实际导入相关模块的 SHA；不称为重新核验全部归档文件。SynthMorph 模块字节与原任务 3 v4 相同。
- 现场重读 FNIT 统一 README/INDEX，INDEX SHA 为 `303f8879915c47523b94144827ef69820181559c057299fe667eecd5e07eeda2`；正式主仓库当时为 `1d31e7b`，它与本次冻结候选分别记录。
- affine 权重 `synthmorph.affine.2.h5` 为 51,455,312 字节，SHA `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6`，与已有固定 Release 校验清单一致；没有新增下载、再发布权重或改变环境。
- nodecw10，8 线程，同原任务 3 核组 `2,6,10,14,18,22,26,30` 和原共享锁串行执行。CUDA 不可见，无 autocast、FP16/BF16，CPU 构造和调用未改变已有 CUDA TF32 全局设置。默认 extent/hyper/steps 为 256/0.5/7，`compute_inverse=True`。

公开 `SynthMorph` 实例的 `__call__` 接受路径和 `nibabel.spatialimages.SpatialImage`。`load_image` 对路径调用 `nib.load`，对对象原样返回，随后共用 `_load → _tensor → network_space/transform/normalize → self.network → compose → 结果采样`。CLI 也直接调用同一个实例方法。裸 NumPy 数组或 Torch tensor 不是影像输入，需连同正确空间包装成 SpatialImage。没有公开的“注入保存的 network_transforms 后恢复整个 RegistrationResult”接口。

本次实际传入的两个对象是 `ndarray` 数据、没有文件名，数据/头/空间与默认解码的原图完全相同；不是仍含 ArrayProxy 的 `nib.load()` 对象。前后数据和几何未被调用修改。其他配准模式及对象参数分支没有追加执行。

## 2. 一次对象调用耗时

| 范围 | 秒 |
|---|---:|
| 读取 NIfTI 对象与头，尚未完全解码 | 0.002772 |
| 完全解码、复制和数据/头/几何校验 | 5.415408 |
| 初始化模型、加载权重 | 0.204851 |
| 完整双向配准 API，两个结果图及两个变换 | 12.764507 |
| 保存四项输出和数据校验 | 5.401096 |
| 新进程完整墙钟，含 Python 启动和前置来源校验 | 27.289485 |

两次现有输出比较墙钟为 12.770068 / 11.767889 秒；输入和完整头诊断为 29.538631 秒；第三路线无 CNN 控制为 12.769077 秒。它们都在事后执行，不能加入或替代上面的完整配准时间。节点共享负载、缓存和初始化范围不同，本次不计算对象 API 对官方冷 CLI 的加速比。

## 3. 保存的双向结果

对照既有同模式路径候选（v3 最终 affine 坐标修复；v4 仅改变 debug，正常路径字节相同）及独立官方默认参考。没有再次推理任一对照。

| 对照与方向 | 全 FOV NRMSE | 脑内 NRMSE | 上边界 NRMSE | 全 FOV max 强度差 |
|---|---:|---:|---:|---:|
| 对原路径候选：moving→fixed | 4.6741e−6 | 4.7589e−6 | 参考零动态范围，RMSE/非零点另列 | 0.191406 |
| 对原路径候选：fixed→moving | 5.1016e−5 | 4.4408e−6 | 6.3775e−6 | 283.968933 |
| 对官方：moving→fixed | 8.1584e−6 | 8.6089e−6 | 参考零动态范围，实际误差为 0 | 0.504150 |
| 对官方：fixed→moving | 1.7192e−4 | 6.5192e−6 | **0.00244438，超过 0.001 门槛** | 695.470337 |

NRMSE 分母为参考区域 `P99−P1`，沿用原 CPU 报告固定门槛；零动态范围不能除零。最大强度差位于填充的不连续边界附近，脑内小差异不能代替上边界验收。

相对原路径候选，forward/inverse 矩阵元素最大差为 `3.43915e−5 / 3.33928e−5`；同一完整源网格的最大世界位移为 `0.00010726 / 0.00010863 mm`。相对官方，完整网格最大世界位移为 `0.00023734 / 0.00012670 mm`，低于 `0.001 mm` 门槛。矩阵和两个图的数据均**不逐值相同**。

两个图相对原路径候选的 shape、dtype、affine、qform、sform、完整 header 字节及 extensions 相同。相对官方则保留 qform/pixdim、`regular`、`dim_info`、`descrip` 及 extensions 差异；不发布原描述字段内容，也不声称文件头完全匹配。

全部数值、逐项门槛、源码/输入/权重 SHA 与五项正式收据见 [report.public.json](report.public.json)。已有 [CPU 脑图](../figures/cpu_official_brains.png)仅属于此前四模式测试，本补测没有另生成图，也没有将其图像重标为本次对象输出。

## 4. 第一分歧：带缩放 NIfTI 的 dtype 路线

这两幅真实文件以 int16 存储，并带非默认 slope/intercept。NiBabel 5.4.2 的 `ArrayProxy._get_scaled` 会根据请求的 dtype 选择缩放计算精度；`__array__` 再完成该 dtype 的转换。[对应源码](https://raw.githubusercontent.com/nipy/nibabel/5.4.2/nibabel/arrayproxy.py)。因此，“同一文件、同一最终 float32”不保证两条缩放路线逐值相同。

| 路线 | 解码与传入数据 | `_tensor` | 规范化网络输入 |
|---|---|---|---|
| 既有路径输入 | `_tensor` 向 ArrayProxy 直接请求 float32 | 原参考路线 | 原参考路线 |
| 本次完整对象调用 | 先按默认精度解码成 float64，再由 `_tensor` 转 float32 | moving/fixed 分别 10,423,036 / 10,835,220 个体素不同，max `0.000244140625` | 两图 max `1.78813934e−7`；不逐值相同 |
| 第三路线，仅无 CNN 控制 | `np.array(ArrayProxy, dtype=np.float32, copy=True)` 直接物化 | 两图逐值相同 | 两图逐值相同 |

两种物化都保持原 Fortran 数组布局，原 `_tensor` 的 stride 均为 `(224,224,1,224,64512)`，规范化输入均为 contiguous；network-space 矩阵也逐值相同。这次第一分歧不是人为改变布局，而是缩放精度先后顺序。实际 ArrayProxy/张量/网络输入 SHA 和 stride 见 JSON；第三路线保持同一 header/affine，不修改 loader、CPU/GPU 运算顺序或冻结程序。

### 匹配路径的推荐物化方式

以下是已通过真实**前处理**控制的 NIfTI 调用形式。保持模型要求的 float32，同时在 ArrayProxy 解码时就指定 dtype；不是先 `get_fdata()` 的默认 float64 解码再转换。第三路线的完整网络输出尚未重跑，示例不表示新增端到端逐值验收。

```python
import nibabel as nib
import numpy as np
from fnit.synthmorph import SynthMorph

moving_source = nib.load(moving_t1_file)  # 原始单帧 NIfTI 路径
fixed_source = nib.load(fixed_t1_file)
moving_memory = nib.Nifti1Image(
    np.array(moving_source.dataobj, dtype=np.float32, copy=True),
    moving_source.affine.copy(), header=moving_source.header.copy(),
)
fixed_memory = nib.Nifti1Image(
    np.array(fixed_source.dataobj, dtype=np.float32, copy=True),
    fixed_source.affine.copy(), header=fixed_source.header.copy(),
)
registration_model = SynthMorph(weights=weight_directory, device="cpu", model="affine")
# 用户实际调用；本报告第三路线只验了上面的物化及网络输入，不再次执行此行。
registration_result = registration_model(moving_memory, fixed_memory)
```

对任意用户已计算的内存图像，数值本身就是输入，不从头字段重新添加 slope/intercept。上述建议用于从同一 NIfTI 构造与路径一致的模型输入，不要求其他函数一律采用这一 dtype 路线。

## 5. affine/joint 对官方剩余差异的定位范围

现有同一物理变换的独立 apply 已过既有门槛；CPU 的最终有效域 `[0,n)` 和 affine 直接乘加坐标修复均已接入。默认 affine/joint 的微小预测场差在填充不连续边界放大，不能把上边界失败简单写成尚未修复的 `[0,n−1]` 规则。

- affine 剩余逆向上边界未过，但世界坐标矩阵误差已在门内。相关 FNIT 路径为 `spatial.py:_prepare_transform` 的直接坐标（93–99）及有效域（118–123），`pipeline.py:_resampled_image` 决定 CPU 最终 Surfa 规则；网络预处理/积分继续保留 Neurite 规则。
- 默认 joint 的 inverse 位移分量 RMSE `0.00011265 mm` 已超过 `0.0001 mm`，是最终位移场层的分歧，发生在最终影像采样前。候选位置包括 `models.py:fit_affine`（91–98）、反对称仿射平均及求逆（163–174）、`matrix_sqrt`（131–145）、joint 场组合（261–291），但没有逐层数值记录证明第一处不同具体在哪一行。
- 官方 observer 只记录函数耗时，没有逐 Conv、barycenter、weighted fit、square root 的同输入数值轨迹。因此目前不能声称已证实“CPU 仿射分解 bug”或“square root 是唯一原因”。普通 affine 不走 mid-space square root，rigid 的 `_rigid` 分解也不属于普通 affine。
- 既有 joint192 FP32 Schur prototype 的 inverse 分量 RMSE 为 `0.00014436 mm`，仍未过门；未接入生产。它是替换平方根的隔离控制，不能证明只改平方根能消除整个流程差异。

官方实际注册模块 SHA `23ead41ec437f75b591e0347920bdd612e01d56406aa5758c103a8385c49d458` 与 [FreeSurfer v8.2.0 registration.py](https://github.com/freesurfer/freesurfer/blob/v8.2.0/mri_synthmorph/synthmorph/registration.py)一致，包含预处理、双向网络、组合和最终 Surfa 采样。实际 VoxelMorph 网络文件 SHA 为 `54fd354880811ff979e6b745355959a8b6906e19582d757528caad0590bc5df1`；其中 affine fit 位于 1410–1411、`tf.linalg.sqrtm` 位于 1424/1427。原作者 [固定 53d1b95 网络](https://github.com/voxelmorph/voxelmorph/blob/53d1b95fa734648c92fd8af4f3807b09cb56c342/voxelmorph/tf/networks.py)与[工具](https://github.com/voxelmorph/voxelmorph/blob/53d1b95fa734648c92fd8af4f3807b09cb56c342/voxelmorph/tf/utils/utils.py)是 FreeSurfer [该版 Dockerfile](https://github.com/freesurfer/freesurfer/blob/v8.2.0/mri_synthmorph/Dockerfile.cpu)固定的上游参考，但其网络 SHA 为 `a4483783…`，与现场安装不相同（包括 Lambda 包装差别），不冒称同一文件。FNIT 为 FP64 Denman–Beavers 后回到 float32；官方使用其 TensorFlow 平方根。这些实现位置给出下一步定位目标，不代替真实数值归因。

## 6. 复现脚本与边界

`object_api_worker.py` 是本次唯一新增 CNN 执行；`inspect_saved_outputs.py` 只读保存图/矩阵/完整头并比较真实前处理；`check_direct_float32_materialization.py` 只做第三路线前处理；`publish_report.py` 从已有 JSON 导出匿名结果。没有改变生产代码、权重、环境、比较阈值或旧产物。报告不发布影像、被试路径、凭据和原始描述字段，文件清单见 [manifest.public.json](manifest.public.json)。
