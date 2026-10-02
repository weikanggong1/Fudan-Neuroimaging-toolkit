# 冻结 benchmark 与最新 main 的实际依赖审查

审查基线为 `ac692bb4f7868a24ea4bd67180162e81726de9b4`，本次核对的 `origin/main` 为 `f436de588647a0de80735e4a98d53df5d88e502d`。本机 HEAD 仍是冻结基线；只读取 Git blob、AST 和已有真实运行元数据，没有修改源快照、现有报告或任务，没有读取 MRI 或启动回归。

## 实际调用链

当前十例 raw 使用默认 `structures="all"`，未提供 coarse/parc/wmparc：

`segment_4_subregions` → `SubregionContext.prepare(need_coarse=True, need_parc=True)` → `SynthSegPlus(keep_geometry=True)` → `SynthSegSegmenter.posterior(flip=True,smooth=True)` → coarse + cortical labels → wmparc proxy → `prepare_automatic_raw_input` → `TorchFAST` → 白质强度归一化 → `_coronal_grid`/`_coronal_affine` → 四个 GEMS recipe → `TorchGEMS`。

`fast` 是 GEMS solver 的配置；本轮没有给 SynthSegPlus 传 `fast=True`，所以它确实运行左右翻转集成。stage 使用本例 fresh recon-all 的 norm/aseg/wmparc，跳过 SynthSeg、FAST 与 automatic-raw conform；没有重新调用 FNIT recon-all。

## 改动是否影响本轮有效计算

| main 改动 | 本轮实际依赖 | 审查结论 |
| --- | --- | --- |
| GEMS pipeline/context/preprocessing/recipe/core/优化器等 | raw 和 stage | 全部 GEMS Python blob 不变；网格、atlas、优化 profile、mesh 拟合、标签命名和输出算法不变 |
| FAST | raw 的 restored image、WM median、brain mask | FAST Python blob 不变；Tensor 分支、默认参数、dtype 与确定性设置不变。它的输入来自 SynthSegPlus，因此仍需检查上游实际输出 |
| `_dmri.configure_device(device, configure_precision=True)` | GEMS pipeline/core、SynthSegParc；新 Segmenter 构造用 False | 默认分支保留原 CUDA TF32 设置。GEMS 入口仍显式设 matmul/cuDNN TF32=True；SynthSegParc 构造也调用默认 True。没有新 float16、autocast 或默认 dtype。对本轮 GPU 全流程的入口和退出状态，没有发现有效精度改变 |
| `SynthSegSegmenter.__init__` 和 scoped posterior | raw 的 SynthSegPlus 33-class head | 构造不再永久更改 TF32；posterior 内默认 matmul=True/cuDNN=True，并 finally 恢复调用方设置。本轮进入时就是 True/True，退出后仍 True/True；新 precision 查询只是记录真实 dtype/autocast，不启动 autocast |
| 翻转集成：`0.5*(original+flipped)` → `flipped.add_(original).mul_(0.5)` | 所有 raw T1 都经过这一处 | 模型、翻转轴/通道、Gaussian blur、postprocess 不变；仍是独立缓冲的逐元素相加后乘 0.5。相同有限 FP32 输入后验和后端条件下，元素计算理论上 bitwise 等价；没有证明整个 CUDA pipeline bitwise 相同。减少临时缓冲会改变 allocator/显存与耗时，后续卷积的真实后端执行仍须检查 |
| 新 `SynthSegSegmenter._forward` 精度记录 | raw 的两个前向 | 增加 CPU 元数据查询与记录，前向仍调用同一模型/输入。对数学计算没有发现新增精度变换；运行兼容性和实际耗时是实测项 |
| 独立 `SynthSeg` 构造、结果 `precision` 字段 | 本轮 raw/all 走 Plus；stage 不推理 | 独立 SynthSeg 类没有被实例化。Plus 从该模块导入的 `_segmentation_image` 与 `_official_soft_volumes` 函数 AST 不变；本轮 volumes=False 不执行后者。不能把独立 SynthSeg 的全部 API 行为改变扩展为本轮有效算法改变 |
| `conform_gpu._det3` 的 float32 乘法求值次序 | GEMS 只调用 `_coronal_affine` | `_coronal_affine` AST 不变，且没有调用 `_det3/_inverse32/conform_volume`。该次序改动会影响另一条 recon-all conform 路径的舍入，但本轮子分区的 coronal grid 不经过它 |
| `_transforms._lta_value` 解析、SynthStrip、SynthMorph、native recon-all 等 | 本轮没有调用这些拟合或 LTA 读取 | 不在有效调用链；不据文件共享或 runtime 导入数量认定它们影响本轮分割 |

`recon_all.__init__` 会导入 `batch`，但 batch 的模块顶层只有标准库导入，新增 recon-all/profiling/thread-budget 功能均不在本轮模块初始化中执行；未发现导入时改变 torch 精度、线程或默认 dtype 的新增副作用。`_coronal_affine` 不会触发完整 FNIT recon-all。

## 可直接确认与需要实测的边界

- **确认未改**：GEMS/FAST blob，SynthSegPlus 两方法、ParcUNet、preprocess/postprocess、Parc 前向、run_synthseg_parc_t1、Gaussian blur、_coronal_affine 的函数内容；同一 atlas、权重和配置仍可复用。
- **默认作用域行为等价**：当前 raw/all、stage/all 入口强制 TF32=True 的前提下，configure_device 新可选开关与 Segmenter finally 恢复不会把 FAST/TorchGEMS 改成另一种精度。独立调用 SynthSeg 的调用者全局状态契约确实改变，但不属于本轮默认子分区路径。
- **仅有理论元素等价**：翻转后验缓冲复用；需要同一 FP32 前向结果、有限数值与相同后端才能据此推导，不等于最新 main 已完成真实分割验证。
- **必须真实核对后才能称为最新 main 实测**：模型实际 TF32/dtype/autocast、coarse/parc/wmparc 输出、FAST restored/grid、四分区 native/HR/soft volume/mesh Jacobian、每 ROI 对官方精度，以及 own-PID 显存和 compute/process wall。

## 建议决定

当前十例完整实验继续按冻结 commit 完成，不替换进行中的源码或结果。它可以准确称为“本轮冻结版本 ac692bb”，不能把记录的源 SHA 改写为 f436 或称为 f436 已实测。

为满足“目前实现”的精度和速度报告，建议另建独立目录，固定 f436 的完整源 SHA，补一次相同十例的 **raw end-to-end 回归**。这是实际触及新 Segmenter 和缓冲策略的分支；保持同一数据、权重、atlas、GPU/20 GB 约束和优化参数，逐例与冻结 raw 及同一 fresh 官方结果比较。不得将新 source 或新输出混入本轮 manifest。

官方数据、软件、命令和参数没有随 FNIT main 改变，无需重复官方 CPU 十例。stage 的有效计算未发现变化，可保留原冻结版实测并标注来源；如果需要所有表格均严格写为 f436 实测，再将 stage 单列回归，不能仅靠此静态审查改写其计时来源。

精确逐文件和逐函数 base/main SHA、AST 范围与分类证据见 `upstream_dependency_review.json`。本审查未启动任何额外拟合，最终回归由父任务安排。
