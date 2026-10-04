# SynthMorph CPU 官方对照：2026-10-04

## 输入与测量范围

使用 OpenNeuro `ds003138` v1.0.1 的两幅原始 T1w，固定第一幅为 moving、第二幅为 fixed，完整网格为 `224×288×288`。数据许可为 CC0；本目录不发布原始影像。独立 apply 使用同数据集真实 AP diffusion 的前两帧，网格 `120×120×68×2`，分别为 b0 和实际 diffusion frame，保留原始体素值、顺序、affine 和 TR。两帧用于检查 4D 与 frame chunk 契约。

CPU 主对照在 nodecw10 的同一组 8 个物理核 `2,6,10,14,18,22,26,30` 上串行运行，线程配置均为 8。每种默认配准执行一次冻结旧版，再执行 `官方、FNIT、FNIT、官方`。进程每次重新启动；未清空操作系统缓存。服务器同时运行其他任务，各组使用独立核组，节点仍有共享负载。所有真实耗时应保留两次观测范围。模型已经加载的 API 时间、完整 CLI 时间及插桩 observer 时间分开记录。

FNIT 基线为 `1d31e7baaebbb644ab199471f7fe6282721455fd`。候选包含本目录对应的 CPU 修复，以及主任务共享 `_nib.new_image` 修复 `dc2fc052`；报告按源码 SHA-256 绑定测量版本。外置模型的大小和 SHA-256 已与 FNIT 固定 Release 清单核对，模型未复制进源码目录。官方参考使用已安装 FreeSurfer 8.2.0-1、TensorFlow 2.13.1、Surfa 0.6.3；这些包仅存在于独立参考环境。

## 预先固定的门槛

见 [acceptance.json](acceptance.json)。

- 世界坐标 dense 位移：最大分量绝对误差 ≤`1e-3 mm`，分量 RMSE ≤`1e-4 mm`。
- 仿射：在同一完整源网格计算世界位移，最大向量误差 ≤`1e-3 mm`。仿射误差范数是凸函数，8 个角点给出完整长方体网格的最大值。
- 连续影像：报告全 FOV、独立官方脑 mask、参考 pull 的上边界带；NRMSE 使用该区域参考影像的 `P99−P1`，门槛为 `1e-3`。同时报告 MAE、RMSE、P99 和最大误差。零动态范围区域单列 RMSE，不能用除零后的数值声称通过。
- 最近邻与离散标签：要求逐体素相同。
- CPU 布局候选：world field 的 `allclose(rtol=1e-5, atol=1e-4 mm)`，并要求稳定时间收益。
- GPU 回归：同一真实输入、权重和完整输出，数组 SHA-256、网格、dtype、pixdim、qform、sform、units 一致；进程 reserved 显存低于 20 GB。

## 功能矩阵

| 分组 | 真实数据覆盖 | 对照方式 |
|---|---|---|
| 默认 registration | rigid、affine、deform、joint；extent 256；正反向变换、正反向重采样影像 | 原版完整 CLI 与 FNIT 完整 CLI；R-C-C-R |
| 参数分支 | extent 192；rigid header-only 与 debug；affine initial LTA + mid-space；deform initial LTA、hyper 0.25、steps 5、debug；joint hyper 0.75、steps 5 | 每分支一对完整 CLI，官方异常单列 |
| Python registration | 完整结果、transform-only、显式不返回 inverse、实际 float32/TF32 状态、FSL warp 转换 | 保存完整数组；同一实例检查 forward/inverse 与完整调用结果一致；保留反对称双向网络前向 |
| apply | affine/dense；linear/nearest；fill、dtype、header-only；真实 3D/4D；frame chunk 1/2/default | 同一明确指定的物理变换，独立原版 Surfa apply API 顺序；包括解码和写盘 |
| WorldTransformChain 扩展 | linear/nearest/spline、grid-constant/periodic、逐帧 motion、mask、chunk、坐标精度 | 独立物理坐标组合和 SciPy 插值数值 oracle，单独分组；SynthMorph 官方无同名 World 链 CLI |
| GPU 安全 | 四模式，完整真实 T1 网格；joint B-C-C-B | 同设备旧版/候选完整数组及 metadata；API 前后同步；单列读取/保存和进程显存 |

## CPU 修复与布局实验

1. CPU 模型构造不再更改 CUDA 全局 TF32 设置；CUDA 构造维持原有默认值。
2. CPU 最终 linear 重采样接受官方的末层体素中心外区间 `[n−1,n)`。网络预处理、速度场积分继续使用原 Neurite 规则；CUDA 最终采样继续沿用此前规则。因此跨设备最终图像的边界行为需要单独解释。
3. CPU scaling-and-squaring 复用不变的 voxel grid，每次采样和加法运算保持原顺序，GPU 使用原路径。
4. CPU nearest 用整数部分与小数部分比较实现原版 half-up；避免 float32 的 `x+0.5` 在半 ULP 附近提前舍入。CPU 最终 affine linear 直接按原版乘加顺序计算坐标，避免 affine→displacement→coordinate 的消减误差。CUDA 和网络 sampler 保留旧求值路径。
5. 有初始仿射的 CPU debug moving 输入按实际固定网络网格保存：真实重放 affine 最大误差 `2.15e−17 mm`，输入 NRMSE `1.08e−6 / 1.25e−6`。CUDA legacy debug 数据和几何重放逐值相同，正常 CUDA 四模式回归独立完成。官方另外输出 `tra_1/2`、`out_1/2`；FNIT 的公开 debug 契约为两幅输入及 `network_transforms.npz`。
6. 独立通道布局实验未进入生产代码。affine 时间没有稳定改善；joint 约快 8%，但 world field 未通过固定的 `allclose` 门槛。具体数据见公开报告的布局分组。

## 官方初始仿射异常

未修改官方命令 `affine -i initial.lta -M` 在最终 `vxm.utils.compose` 抛出 TensorFlow `MatMul` 的 float64/float32 混合类型异常。失败退出码和耗时保留在原版分支记录中。`patched_reference.py` 仅用于诊断：给混合类型的 compose 输入显式转为 float32，再调用原函数；记录实际改变的两次输入 dtype。修补参考的精度结果有独立标记，不能计作未修改官方命令成功。

## 完整 CPU 时长与精度

默认参数为 extent 256、hyper 0.5、steps 7，均请求正反向场和正反向影像。`v1` 是有效域、精度隔离与 grid 复用；`v2` 修最近邻半 ULP；`v3` 修最终 affine 坐标；`v4` 只修 debug header。每次测量绑定源文件哈希，不能把 v1 的完整对照时间标成 v4。

| 模式 | 冻结旧版（秒） | 原版 R1 / R2（秒） | v1 C1 / C2（秒） | 最新 v3（秒） | 原版 / v1 采样 RSS（GB） |
|---|---:|---:|---:|---:|---:|
| rigid | 19.53 | 89.14 / 44.57 | 22.29 / 21.54 | 22.54 | 9.82–9.83 / 4.73–4.84 |
| affine | 21.04 | 35.81 / 104.90 | 23.79 / 20.53 | 23.54 | 9.82–9.83 / 4.49–5.12 |
| deform | 152.72 | 213.11 / 163.73 | 156.21 / 155.69 | dense 计算未变，使用 v1 记录 | 19.77–19.81 / 11.68 |
| joint | 161.22 | 220.79 / 261.35 | 164.24 / 169.25 | dense 计算未变，使用 v1 记录 | 19.90–19.97 / 11.75 |

`RSS` 为 runner 对进程树定时采样的最大值，非全程精确内存峰值。没有清缓存的独立新进程时间与 warm API 分列；共享节点的两次 linear 原版时间变化较大。本轮候选的观测范围均小于原版，仍需更多配对才能给出稳定加速倍数。候选相对冻结旧版主要增加正确的末层采样工作，而不是减少双向计算。

| 模式 | 正向 / 逆向场误差（mm） | 正向 / 逆向全 FOV NRMSE | 正向 / 逆向脑内 NRMSE | 全部预定门槛 |
|---|---|---|---|---|
| rigid v3 | 完整网格 max 向量 0.00013810 / 0.00026800 | 8.02e−6 / 1.54e−4 | 6.28e−6 / 4.86e−6 | 部分通过：正向零动态范围上边界 3 点非零，NRMSE 无定义 |
| affine v3 | 完整网格 max 向量 0.00032878 / 0.00017458 | 7.50e−6 / 1.79e−4 | 6.19e−6 / 9.18e−6 | 未全过：逆向上边界 0.00244442 > 0.001 |
| deform v1 | 分量 max 0.00007629 / 0.00007629；RMSE 0.00001023 / 0.00001015 | 1.22e−6 / 3.74e−6 | 1.77e−6 / 1.60e−6 | 两向场、定义区域及零误差边界通过 |
| joint v1 | 分量 max 0.00032043 / 0.00044250；RMSE 0.00005367 / 0.00011265 | 6.70e−6 / 1.64e−4 | 6.47e−6 / 8.64e−6 | 未全过：逆向场 RMSE 超过 0.0001；逆向上边界 0.00152807 > 0.001 |

全 FOV 的正向 / 逆向最大强度误差分别为 rigid `7.76 / 661.28`、affine `0.69 / 695.47`、deform `0.096 / 20.03`、joint `0.21 / 715.07`。微小坐标差在填充值的不连续边界可能产生大的单点强度差，脑内小误差不能替代边界验收。v3 直接坐标修复保留场值，未消除上述模型预测差；默认 joint 的 TensorFlow/PyTorch 线性计算、矩阵平方根与后续场合成仍需逐阶段定位。没有改变门槛或采用更低精度。

CPU 候选的输出 affine 与目标一致；原版 Surfa 与 nibabel 的 qform/pixdim 编码有约 `1e−5 mm` 的差异。FNIT dense 文件是 `(X,Y,Z,3)`、intent 1007；原版是 `(X,Y,Z,1,3)`、intent 1006，含 source/target extension。报告保存这些差异，CPU 文件字节/全部 header 不称相同。

## 参数、API 与独立 apply

| 分支 | 原版 / FNIT 完整运行（秒） | 数值与契约结果 |
|---|---:|---|
| rigid 192、header-only、debug | 36.81 / 14.52 | 原图数据保持不变；场 max 0.00009697 / 0.00031248 mm；debug 网格通过 |
| affine 192、init、mid-space | 原版失败 13.27 / FNIT 17.28；v3 FNIT 17.78 | 未修改原版混合 dtype 异常；诊断参考仍有逆向上边界 NRMSE 0.00270425 未过 |
| deform 192、init、hyper 0.25、steps 5、debug | 158.72 / 102.40 | 两向 field max 0.00010681 / 0.00013733、RMSE 0.00001232 / 0.00001175 mm，通过；v4 debug 几何重放修复 |
| joint 192、hyper 0.75、steps 5 | 131.68 / 87.63 | 逆向 RMSE 0.000118833 mm 未过；边界 NRMSE 0.00076796 通过 |
| affine Python functional | FNIT 全测试进程 22.03 | transform-only 场与完整调用相同；不返回图像 |
| joint Python functional | FNIT 全测试进程 224.35，完整 warm API 62.86 | 三次调用检查 transform-only、compute_inverse=False；不返回 inverse 时保留两次反对称前向、forward 逐值相同 |

独立 apply 固定同一真实物理变换，隔离模型预测误差。adapter 的对象构造不在单项时间内；各项包含解码、执行与保存，组运行总时间含 Python 启动和多项准备。

| 同场功能组 | 项数 | 原版组 / 候选组运行（秒） | 结果 |
|---|---:|---:|---|
| affine apply（含 header-only、fill−7/float64、brain nearest、3D/4D、chunk 1/2/auto） | 8 | 31.55 / v3 17.53 | 连续图像全部 NRMSE 过门槛；nearest 0 个不同；header-only 数据不变；4D TR 和各 chunk 相同 |
| dense apply（含 fill−7、brain nearest、实际 DWI 两帧、chunk 1/2/auto） | 6 | 原版 Surfa adapter 28.29 / v2 26.04 | T1 NRMSE≤1.90e−6，DWI NRMSE 0.00016222；nearest 0 个不同；全部通过 |
| WorldTransformChain 扩展 | 5 | 独立 NumPy/SciPy oracle 18.28 / FNIT 7.01 | linear、nearest、spline、逐帧 motion、mask、pull 合成通过；最近邻与实际混合链逐值同 |
| FSL intent-2006 同场 applywarp | 1 | 原版 48.84 / TorchApplyWarp 14.27 | 全 T1 NRMSE 1.21e−6、脑内 1.44e−6、全图 max 0.04810，metadata 相同 |

World 链没有 SynthMorph 官方同名入口；周期边界项为真实影像上的 identity/edge-clamp，本轮不验收超出边界的周期采样。部分 oracle 图像创建时未继承 units，其差异已保留；真实 4D TR 是 3.5。FSL 同场测试核验转换文件被正确消费，不能代替 SynthMorph 与官方生成的形变精度。初次 reader 参数和绘图环境失败、修正后的完成记录均保留；不将 reader 失败等同于算法输出失败。

## 分步骤热点与被拒绝的候选

同真实 T1，joint 192/hyper 0.5/steps 7 的独立 observer：

| 阶段 | 原版 observer（秒） | FNIT observer（秒） |
|---|---:|---:|
| 构造/读模型 | affine 构造 4.49、joint 构造 33.35；两次权重加载 14.27 / 19.95 | 构造及权重加载合计 3.91 |
| 完整 register / 已加载 API | 144.90（inclusive） | 61.06 |
| 主要网络调用 | 最后完整 joint 调用 45.60（inclusive） | 整体 network 48.29；其中 affine 2.14、双向 deform 21.09 / 22.70 |
| 影像读取 | 0.456 / 0.496 | 0.458 / 0.499 |
| 预处理 transform | 1.069 / 0.771 | 0.462 / 0.496 |
| 场合成 | 包含在官方 joint/注册内部，未独立拆开 | 2.455 / 2.472 |
| RAS 场转换 | 未独立拆开 | 0.643 / 0.630 |
| 最终重采样 | Surfa 外层 transform 1.011 / 0.915 | 1.730 / 1.761 |
| 保存完整结果 | 未从 reference 注册单独扣除 | 22.00 |

双方边界不是完全同一统计口径，网络构造会调用网络，原版装饰器产生嵌套计时；这些 inclusive 行不能相加，也不能据此给出阶段加速倍数。CPU 网络卷积、双向 deform 和 gzip 保存是主要后续热点，sampler/积分需保留当前数值策略。

通道布局诊断的 affine contiguous/候选为 `10.29 / 9.92 / 10.21 / 9.78 s`，无稳定收益；joint 为 `114.58 / 104.24 / 106.63 / 115.57 s`，约 8% 收益但场 max 差 `0.00022888 / 0.00023651 mm`，未过 allclose。CPU float32 Schur sqrt 原型的逆向 field RMSE `0.00014436 mm` 比旧值 `0.00011883 mm` 更差，未采用。生产模型、CUDA layout 和平方根算法均未替换。

## GPU 回归与版本记录

同 H100、同真实完整输入、float32/TF32、未启用 autocast；v1 原候选执行完整四模式并对 joint 做 B-C-C-B，v2/v3 为完整四模式单次回归。v3 因 GPFS `flock ENOLCK` 中断一次，未运行的任务保留 waiting 旧记录后在同一 GPU 锁下恢复；计时从获得锁后开始，未改用其他锁。实际恢复的三项与已完成 rigid 的 source SHA 均匹配 v3。

| 模式 | v3 warm API（秒） | CUDA reserved（GB） | 同设备冻结旧版→v3 |
|---|---:|---:|---|
| rigid | 3.411 | 5.910 | forward/inverse/moved/fixed_moved 的数组 SHA 与 image header 全相同 |
| affine | 3.393 | 5.910 | 数组与 image header 全相同 |
| deform | 5.939 | 17.740 | 数组、image/warp header 全相同 |
| joint | 5.547 | 17.836 | 数组、image/warp header 全相同 |

仅报告 allocator reserved，不是 NVML 进程总显存；这组证明当前 CPU 修复保留 GPU 结果，没有重新验收 GPU 与原版的历史差异。v4 只把已保存的真实网络输入交给 debug helper 重放，CPU grid 匹配原版、CUDA legacy grid/data 相同；无额外 GPU NN 推理。

| 本轮版本 | 改动 | 实际验证 |
|---|---|---|
| v1 `ff997455` | CPU 精度隔离、最终有效域、积分 grid 复用 | 四模式完整 R-C-C-R、参数和同场 apply；GPU 四模式 + joint B-C-C-B |
| v2 `bf791091` | CPU nearest 半 ULP | 真实 8 affine / 6 dense apply、CPU 单测、GPU 四模式数据与 metadata 相同 |
| v3 `d95123ac` | CPU final affine 直接坐标 | rigid、affine、init-mid 和 8 项 apply 完整重测；GPU 四模式相同 |
| v4 `6dd3b044` | CPU init debug header | 实际 normalized 输入重放；129 项 SynthMorph 测试通过 |

本轮公开 [report.public.json](report.public.json) 保留误差、逐项门槛、时长、资源采样、退出状态与 source hash；输入哈希可复核，但不发布被试目录或原始影像。

![官方、FNIT 与脑内强度差](figures/cpu_official_brains.png)

rigid/affine 为最新 v3，deform/joint 为 dense 计算相同的 v1。显示前应用官方脑 mask，仅显示范围裁成脑部框，未裁剪 benchmark 输入。每行差图色标为脑内绝对误差 P99，最低 0.01，色标外值截断；数值结论由完整输出给出。

## 复现入口

以下脚本接受用户自己的文件路径；输入、权重和输出目录用完整变量名定义。

```bash
export PYTHONPATH=/path/to/FNIT/src
moving_t1=/path/to/moving_T1w.nii.gz
fixed_t1=/path/to/fixed_T1w.nii.gz
weights_directory=/path/to/fnit_weights
output_directory=/path/to/new_result_directory

# 已加载 API、完整两向结果；构造、推理和保存分别计时。
python validation/synthmorph/cpu_20261004/api_worker.py \
  --moving "$moving_t1" --fixed "$fixed_t1" \
  --weights "$weights_directory" --output "$output_directory" \
  --device cpu --model joint --extent 256

# 对已经保存的两份完整输出逐网格比较，不再运行配准。
python validation/synthmorph/cpu_20261004/compare_registration.py \
  --candidate /path/to/fnit_cli_outputs \
  --reference /path/to/official_cli_outputs --model joint \
  --moving "$moving_t1" --fixed "$fixed_t1" \
  --moving-mask /path/to/official_moving_brain_mask.nii.gz \
  --fixed-mask /path/to/official_fixed_brain_mask.nii.gz \
  --output /path/to/comparison.json

python -m pytest tests/synthmorph -q
```

`--observer` 只用于分步诊断，会添加 Python 计时开销。`nested_module_observer` 与 `phase_observer` 是嵌套 inclusive 时间，不能直接相加作为端到端时间。GPU正式回归使用冻结的 tools_v1 worker；后续 tools_v2 只增加 CPU observer。

参考：[FreeSurfer 8.2.0 注册源码](https://github.com/freesurfer/freesurfer/blob/v8.2.0/mri_synthmorph/synthmorph/registration.py)、[原版 CLI](https://github.com/freesurfer/freesurfer/blob/v8.2.0/mri_synthmorph/mri_synthmorph)、[Surfa](https://github.com/freesurfer/surfa)、[SynthMorph 论文](https://doi.org/10.1162/imag_a_00197)。
