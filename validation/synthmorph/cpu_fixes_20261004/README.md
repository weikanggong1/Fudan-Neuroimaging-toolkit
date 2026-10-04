# SynthMorph CPU 精度续修：2026-10-04

本轮从已发布 `f1cbdab10fbfd573c3aa1b3461aa086dab220cfb` 继续修复 CPU。真实输入、资源许可及服务器统一索引均重新核对；冻结源码、环境 prefix 和先前结果保持原实体路径。生产仍只使用 nibabel、NumPy、SciPy 和 PyTorch，原版 FreeSurfer、Surfa、TensorFlow 仅在独立参考进程中运行。

## 1. 修复及输入输出契约

函数、参数、Python 示例、FNIT CLI 和原软件命令见[功能说明](../../../docs/synthmorph/README.md)。接口不增加参数。

1. **CPU 图像解码**：先以 ArrayProxy 的默认类型完成 slope/intercept 缩放，再转 float32。原版 Surfa 默认解码采用相同顺序；CPU 路径与默认解码后完全物化的 nibabel 对象使用相同体素值。真实 T1 的两条解码路线有约一千万个 float32 尾数差，单值最大差 `0.000244140625`。CUDA 保持直接请求 float32 的既有路线。
2. **CPU affine 最终图像**：按实际返回的 Affine 取逆生成 pull。原版最终 sampler 同样使用其关联的 Affine；另一方向的独立 float32 预测不保证精确互逆。无 CNN 的保存矩阵回放将 inverse 上边界 NRMSE 从 `0.00244438` 降至 `1.11984e−5`。
3. **CPU rigid 返回变换**：返回各自实际采样 pull 的 float64 精确逆，再由返回变换生成最终图像。直接反转另一方向的近似预测会在参考全零的上边界产生 12 个微值；实际 pull 的逆同时通过原版精度和返回变换重采样一致性门。
4. **CPU joint 首次归约**：按现场原版 Neurite/VoxelMorph 的两种归约形状分别计算置信 mass 和 barycenter。前者在物理 NHWDC 布局归约；后者使用 NCDHW 加末轴 XYZ 的交错 moments 与独立 denominator。旧实现复用同一 mass 并分别求 XYZ，浮点归约次序不同。仅 CPU joint 的 affine mid-space 阶段使用这项修复；CNN 布局、standalone rigid/affine 及全部 CUDA 保留原算术。
5. **World 链边界**：已声明的 `periodic` 指周期样条系数，并将 `[−1e−6,N−1+1e−6]` 内的舍入坐标夹回有效中心范围；更远的源 FOV 外位置为零。该政策已在既有 volume 文档中明确，本轮补充公共入口回归，没有改为无限循环采样。registration 的合法 `fill=0` 保持。

rigid/affine 的 `result.moved`、`result.fixed_moved` 与应用对应返回 Affine 的 CPU 结果逐值一致；图像头、frame 轴、TR、输入数组及输入几何分别核对。未改变权重、TF32、网络空间几何、速度积分或 CUDA 采样公式。

## 2. 真实数据与门槛

公开数据为 OpenNeuro `ds003138` v1.0.1，许可 CC0；两幅原始 T1w 完整网格 `224×288×288`。World 链另用真实采集的两帧 DWI `120×120×68×2`，不是复制 T1 造出的时间序列。输入、权重、各次实际源码、worker、保存输出和记录绑定 SHA-256；公共报告不包含影像、被试路径、认证或许可证内容。

沿用[提前固定的门槛](../cpu_20261004/acceptance.json)：仿射在完整 source 网格的最大世界位移误差不超过 `0.001 mm`；dense 场分量最大误差不超过 `0.001 mm`、RMSE 不超过 `0.0001 mm`；连续图像全 FOV、独立原版脑 mask 和坐标上边界分别用参考 `P99−P1` 归一化，NRMSE 不超过 `0.001`。零动态范围区域要求误差严格为零，nearest 要求逐体素相同。没有修改容差。

本轮新正式进程在 **nodecw7**，线程配置 8，绑定 `2,6,10,14,18,22,26,30` 八个物理核，采用同组文件锁。各 CLI 为新进程，包含加载、解压、双向计算及全部保存；页面缓存未清空，节点有其他任务，单次观测不等于稳定提速。本轮原版和基线也在 nodecw7 重跑；nodecw10 记录只作诊断，不能用于本轮时间比。

## 3. 当前真实验证

| extent 256 | 基线 main CLI（秒） | 原版 CLI（秒） | 修复版 CLI（秒） | 双向原版精度 |
|---|---:|---:|---:|---|
| rigid | 29.53 | 311.21 | 23.54 | 完整源网格世界误差最大 `0.000142410 / 0.000961941 mm`；全 FOV、脑内及上边界全部通过，正向零边界逐值相同 |
| affine | 20.53 | 65.10 | 26.53 | 世界误差最大 `0.000237338 / 0.000126700 mm`；inverse 上边界 NRMSE `1.11984e−5`，全部通过 |
| deform | 155.53 | 168.80 | 143.15 | 场 RMSE `1.02277e−5 / 1.01477e−5 mm`；两向场及全部影像区域通过 |
| joint | 148.24 | 174.97 | 148.17 | 场 RMSE `6.47982e−5 / 4.37923e−5 mm`；inverse 上边界 NRMSE `0.000389440`，全部通过 |

rigid 实测源码为 `final_rigid_v4`；affine/deform 为 `final_affine_v2`；joint、joint extent192 及最新 GPU affine 为最终默认源码 `final_joint_v7`。各 mode 单独记录实际源码和转移包哈希。最终源码的 CPU rigid/affine 保存 LTA 无 CNN 回放再次核对两向数组和完整 header；未受 joint 专属分支影响的 CNN、采样与积分逐函数 AST/文件哈希核对。上述记录为一次基线、一次原版、一次候选；没有将先前 nodecw10 的 R-C-C-R 计时改名为本轮实验。

所有默认 CPU 门均通过，逐项数值和实际 SHA-256 见[公共 JSON](report.public.json)。分量级误差不为零；通过提前固定的本例误差门不能推广为任意输入逐位等价。

| extent256 的 sampled process-tree RSS 峰值（GB，10⁹ bytes） | main | 原版 | 修复版 |
|---|---:|---:|---:|
| rigid | 4.852 | 9.828 | 5.059 |
| affine | 4.891 | 9.829 | 4.886 |
| deform | 11.674 | 19.670 | 11.678 |
| joint | 11.737 | 19.843 | 11.716 |

附加参数 `joint extent=192, hyper=0.75, steps=5` 在 nodecw7 同八核重跑：FNIT/原版完整 CLI `97.35 / 169.15 s`，FNIT RSS `5.440 GB`；两向场 RMSE `5.11062e−5 / 7.64936e−5 mm`，逆向上边界 NRMSE `0.000767640`，全 FOV/脑内/场通过。**正向参考全零的上边界有 2 个非零值，最大 `0.00159934`、RMSE `1.02073e−5`，严格零误差门失败**。该配置仍列为未完成项；不将 NRMSE 为 null 解释为通过。继续用实际原生 float32 pull 和规范化坐标定位，保留原门和 fill 政策。

完整物化对象 API：affine 模型加载 `0.345 s`、调用 `13.983 s`；rigid 加载和调用分别记录，其中调用 `10.684 s`。两向输出与对应 CLI 的数组、完整 header、返回 Affine 重采样数组和 header 全部一致，LTA 重读后的世界矩阵差不超过 `1e−12`；输入未被修改。joint 的实际物化对象与最终 CLI 两图、两场、完整 header/extensions/affine 逐值相同；模型加载 `3.471 s`、API `104.507 s`、NIfTI 保存 `20.674 s`，输入载入及物化 `0.972 s` 单列。observer、保存 npy 和进程启动包含在 worker 的 `148.885 s` 内，不并入已加载 API。

World 链的 nearest/linear/spline 经 SynthMorph、TorchApplyWarp 和共享 helper 三个入口组成 9 项完整 DWI 检查：入口间数组和完整 header 相同，TR 相同，越界非零值数量均为 0。nearest 与独立 SciPy oracle 逐值相同，linear/spline NRMSE 为 `1.94e−7 / 3.53e−7`。

![真实 T1 原版 CPU、FNIT CPU 与脑内差图](figures/cpu_official_brains.png)

默认 extent256 的显示前应用独立原版脑 mask，仅裁出脑部显示框；数值验收仍使用完整 FOV。各行差图色标为脑内绝对误差 P99，最低 0.01，色标外截断；完整 max 和边界门见 JSON，图及显示元数据另绑定 SHA-256。

## 4. GPU 回归及计时范围

H100 上按公共 GPU 锁串行执行完整 affine 旧/新 API，保存两向 Affine、两向图像及四个完整数组。四数组 SHA-256、两图完整二进制 header、extensions、shape 和 affine 全部相同；reserved 峰值均 `5,909,774,336 bytes`，低于 20 GB。已加载 API 与完整 worker 的旧/新精确时间见 JSON；当前最终源码候选完整 worker `13.0339 s`，保存的同锁 main 基线为 `13.7904 s`。当时存在外部 GPU 占用，这些时间仅是同场观测。

初次 harness 在模型构造前的 CUDA quota 初始化失败；失败 stderr、记录和时间保留。新 harness 先 `import torch → cuda.init → int0 quota`，再执行原 worker，生产计算不变。未将失败归因于模型显存，也未把未执行的 dense GPU 模式写为已通过。模型、积分、CUDA 直接 float32 解码及 paired sampler 路径保留；本轮实际完整 GPU 推理证据限 affine，历史四模式回归另见[前一轮报告](../cpu_20261004/README.md)。

## 5. 复现与分步骤记录

下列输入是原始两幅 3D T1w；输出影像分别位于 fixed/moving 网格，RAS pull 位移场为 `(X,Y,Z,3)`，单位 mm。完整参数及原软件对应关系见功能说明。

```python
import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph import SynthMorph

torch.set_num_threads(8)
weights_directory = "/path/to/verified_weights"  # 已校验大小和 SHA-256 的外置权重
moving_source = nib.load("moving_T1w.nii.gz")
fixed_source = nib.load("fixed_T1w.nii.gz")
# 先完成 NIfTI 默认缩放再物化，数据对象不保留源文件依赖。
moving_image = nib.Nifti1Image(np.array(np.asanyarray(moving_source.dataobj), copy=True),
                              moving_source.affine.copy(), moving_source.header.copy())
fixed_image = nib.Nifti1Image(np.array(np.asanyarray(fixed_source.dataobj), copy=True),
                             fixed_source.affine.copy(), fixed_source.header.copy())
registration = SynthMorph(weights=weights_directory, model="joint", device="cpu",
                          extent=256, hyper=0.5, steps=7)
registration_result = registration(moving_image, fixed_image)
registration_result.moved.save("moving_in_fixed.nii.gz")
registration_result.fixed_moved.save("fixed_in_moving.nii.gz")
registration_result.transform.save("forward_ras_pull.nii.gz")
registration_result.inverse.save("inverse_ras_pull.nii.gz")
```

```bash
# FNIT：新进程、完整双向输出；附加配置改为 -e 192 -r 0.75 -n 5。
weights_directory=/path/to/verified_weights
fnit synthmorph moving_T1w.nii.gz fixed_T1w.nii.gz \
  -m joint --device cpu --weights "$weights_directory" -e 256 -r 0.5 -n 7 -j 8 \
  -o moving_in_fixed.nii.gz -O fixed_in_moving.nii.gz \
  -t forward_ras_pull.nii.gz -T inverse_ras_pull.nii.gz
# 原版独立参考进程；生产 FNIT 不调用该命令。
mri_synthmorph register -m joint -e 256 -r 0.5 -n 7 -j 8 \
  -w "$weights_directory/synthmorph.affine.2.h5" \
  -w "$weights_directory/synthmorph.deform.3.h5" \
  -o original_moved.nii.gz -O original_fixed_moved.nii.gz \
  -t original_forward.nii.gz -T original_inverse.nii.gz \
  moving_T1w.nii.gz fixed_T1w.nii.gz
```

默认 joint 的同节点分步骤观测如下。边界有嵌套，原版构图还包含初始化调用，不能相加或替代第3节未插桩的完整 CLI。原版 observer 完整进程 `317.25 s`；其两图、两场及完整 header/extensions 与原版未插桩 CLI 相同。

| 观测边界（秒） | FNIT | 原版 |
|---|---:|---:|
| 输入读取与物化；原版只读取 | 0.972 | 0.430 / 0.467 |
| 模型加载；原版为 HyperVxmJoint 构网及两次 load_weights | 3.471 | 构网 30.085；载权重 13.209 / 18.380 |
| 顶层网络调用（inclusive） | 95.142 | 94.143 |
| 最终 image sampler；原版为捕获的四次 Surfa transform 调用 | 1.048 / 1.068 | 0.790 / 0.907 / 0.658 / 0.827 |
| 两图与两场 NIfTI 保存 | 20.674 | 本次 observer 未覆盖写入边界，完整 CLI 包含保存 |

完整 API `104.507 s` 还包含预处理、坐标组合及保存前 RAS 转换。嵌套 module/phase 的每次时间、原版实际源码和 worker 指纹全部保存在 JSON。

## 6. 诊断与版本记录

- [contract_api.py](contract_api.py)：完整物化对象 API、返回 Affine 及 CLI 两向一致性。
- [world_boundary.py](world_boundary.py)：真实 DWI，三插值、三公共入口、独立 oracle 和有效域控制。
- [full_affine_grid.py](full_affine_grid.py)：实际遍历完整 source 网格，补充世界位移误差。
- [final_replay.py](final_replay.py)、[source_compatibility.py](source_compatibility.py)：当前源码保存 LTA 无 CNN 回放、未改变的 CUDA 公式与共享文件 AST/哈希审计。
- [boundary_point_probe.py](boundary_point_probe.py)：附加192配置失败零边界点的实际原生 float32 pull、规范化和反规范化坐标捕获。
- [materialized_api_worker.py](materialized_api_worker.py)、[build_report.py](build_report.py)：真实物化对象的完整 joint API、CLI 全头/数组对照及公共记录导出。
- [gpu_initialized_worker.py](gpu_initialized_worker.py)、[compare_gpu.py](compare_gpu.py)：早初始化 benchmark harness 和完整 GPU 保存结果回归。
- [affine_stage.py](affine_stage.py)、[cpu_layout_worker.py](cpu_layout_worker.py)、[cpu_reduction_worker.py](cpu_reduction_worker.py)：独立参考、冻结 features 和 CPU 首次归约差异诊断；全网络通道布局候选没有进入生产。

joint 的 full channels-last 候选产生一个新的 forward 零边界差异点，因此不采用。单独改变 mass 布局也未通过 inverse 边界；返回 RAS warp 的无 CNN 重采样仍未解决它。最终依据现场原版首次归约的形状修复 CPU joint，冻结同一真实 feature 做差分后才运行完整候选。没有叠加未证实的 homogeneous-row reset、补零、有效域放宽或容差修改。

| 本轮版本 | 内容与验收 |
|---|---|
| decode v1 | CPU 默认解码对齐；记录 nodecw10 外部高负载及 timeout，不用于 nodecw7 时间比 |
| affine v2 | affine 返回变换与最终 sampler 一致，affine/deform 全部门通过；当时 rigid 的 12 点及 joint inverse 边界失败保留 |
| rigid v4 | 采用实际各方向 pull 的精确逆，rigid 原版和公共返回 Affine 契约同时通过 |
| joint v5 / layout v3 | 隔离归约布局或全网络布局诊断未通过严格边界，未进入生产 |
| joint v6 → final v7 | 冻结 feature 验证原版两种归约形状；最终 CPU 默认256、物化 API、GPU affine 保存数组回归通过；192 正向零边界2点未过，报告明确为 failed |

GPU 早初始化失败、World v1 JSON 的 NumPy bool 序列化失败、原版 observer 的 worker 路径及 shell/Python 入口启动失败均保留；后两项与生产计算无关。服务器现场源树完整 manifest 先校验，再导出相关文件、包与记录哈希；最终模型文件 SHA-256 `1a4cdfa3b348bcbb670a80db7003a20e9f1405dca9f7e63dc7d452aee691564b`，pipeline 为 `75044d4484b0c5a73efe31949ae609e5920b514583d1ee067ee440ff4cbc1bfc`。

本轮 `tests/synthmorph` 加 `tests/applywarp/test_world_transform.py` 共 160 项通过。新增缩放 dtype/真实 ArrayProxy、物化输入、双向返回仿射、CPU joint 分支与所有共享 World 插值边界的定向回归。

## 7. 原实现与参考文献

固定参考构建为 FreeSurfer 8.2.0-1、Surfa 0.6.3、TensorFlow 2.13.1，依据现场安装源码和 SHA-256。源码链接：[SynthMorph registration](https://github.com/freesurfer/freesurfer/blob/dev/mri_synthmorph/synthmorph/registration.py)、[Surfa reader](https://github.com/freesurfer/surfa/blob/master/surfa/io/framed.py)、[Surfa affine](https://github.com/freesurfer/surfa/blob/master/surfa/transform/affine.py)、[Neurite](https://github.com/adalca/neurite)。开发分支链接用于浏览，实际复现以固定构建哈希为准。

Hoffmann et al., *Anatomy-aware and acquisition-agnostic joint registration with SynthMorph*, Imaging Neuroscience (2024), [doi:10.1162/imag_a_00197](https://doi.org/10.1162/imag_a_00197)；Hoffmann et al., *SynthMorph: learning contrast-invariant registration without acquired images*, IEEE TMI (2022), [doi:10.1109/TMI.2021.3116879](https://doi.org/10.1109/TMI.2021.3116879)。
