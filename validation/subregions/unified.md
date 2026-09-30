# 统一 TorchGEMS 亚区：真实 T1 验证

[功能和参数](../../docs/subregions/README.md) · [历史脑干验证](README.md) · [旧 C++/ITK 核团验证](nuclei.md)

## 输入和验收

使用 OpenNeuro ds000114 的一个完整 T1。输入为仓库已去除面部的公开衍生影像，未做 recon-all；来源、CC0 许可及处理记录见 [SOURCES.json](../../examples/data/SOURCES.json)。文件为 `sub-01_T1w.nii.gz`，形状 `256×156×256`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。这一个开发病例不作为独立留出集。

两个完整运行分别报告：

1. **相同阶段输入**：FNIT 与 FreeSurfer 8.2 使用相同 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`，比较亚区拟合。
2. **原始 T1 全流程**：FNIT 从上述 T1 自行生成粗分割、皮层分区和白质代理，再拟合全部结构；与同一人的官方输出比较。

每区要求 Dice ≥0.95、硬体积差 ≤5%；不要求逐体素相同。未通过的标签保留数字；参考为空而候选非空的标签记为失败，双方均为空不计入 Dice 验收。软体积单独报告后验积分差。

两组运行均有 7 个标签的双方硬分割为空：丘脑的双侧 Pc、Pt、VM，以及右侧杏仁核 Anterior-amygdaloid-area-AAA。这些标签的软体积非零。两份 TSV 均包含全部 110 个标签的软体积，`hard_evaluation_status=both_empty` 的 7 行将 Dice、硬体积差和验收结果留空；其余 103 行参与硬分割验收。表中的最大软体积差覆盖全部 110 个标签。

运行在共享 H100 PCIe 80 GiB 上，四个 CPU 线程，FP32 张量、默认 TF32。图谱和权重已准备并核对大小、SHA-256；耗时不包含首次下载及图谱先验准备。阶段对照限制 PyTorch 分配器为显卡容量的 14%，原始 T1 全流程为 23%；每 10 秒记录全卡负载及本进程显存，本进程超过 20,480 MiB 时停止该进程。PyTorch 分配峰值与 `nvidia-smi` 进程显存分别统计。`wall_seconds` 从进入 Python API 到返回结果，包含读入、预处理、拟合与合并，不含返回后保存 NIfTI、报告和对照指标的时间。

## 共享预处理：完整真实 T1

| 设备 | SynthSeg/SynthSeg+、白质代理总时间 | PyTorch 显存峰值 |
|---|---:|---:|
| CPU，四线程 | 124.08 s | — |
| H100，默认分配器 | 8.70 s | 15.47 GiB |

[CPU 报告](unified_shared_preprocessing_cpu.json) · [GPU 报告](unified_shared_preprocessing_gpu.json) · [CPU/GPU 逐标签对照](unified_shared_preprocessing_cpu_gpu.json)

GPU 与 CPU 使用相同的完整影像和权重。原始网格粗分割相差 113 个体素、皮层分区相差 194 个体素、白质代理相差 156 个体素；三者最小逐标签 Dice 分别为 0.99875、0.99823、0.99873。六个海马白质采样标签均非空，传播结果全部限制在原有白质内。该计时只覆盖共享预处理，不能作为全部亚区分割耗时。

本次释放 U-Net 中已经使用的 skip 张量、未使用的解码器返回值和 softmax 前的末层特征，保持网络运算和权重不变。未启用 FP16 或 BF16。

## 栅格化组件检查

使用真实 BrainstemSS 图谱的 `48×48×48` 局部网格，对照更改前的 FNIT 栅格化函数。该项是数值和组件计时检查，不是完整被试 benchmark。

| 精度设置 | 最大先验绝对差 | 梯度相对 L2 差 | 四面体归属变化 | 原函数热运行 | 新函数热运行 |
|---|---:|---:|---:|---:|---:|
| 完整 FP32 | 4.77×10⁻⁷ | 1.77×10⁻⁷ | 0 | 0.430 s | 0.063 s |
| TF32 开启 | 4.77×10⁻⁷ | 2.39×10⁻⁴ | 0 | 0.496 s | 0.080 s |

[FP32 报告](unified_raster_fp32.json) · [TF32 报告](unified_raster_tf32.json)

新函数批量查找候选四面体，每个四面体只计算一次逆矩阵；候选查找不构造反向图，选中的四面体重心坐标仍由 PyTorch 自动求导。完整 FP32 对照通过 `1e-5` 先验差及 `1e-4` 梯度差检查，内部梯度还通过有限差分检查。TF32 下的梯度差单独记录。

本次局部数值检查对图谱加 `[0.137, 0.211, 0.317]` 体素的平移，避免检查点全部落在整数网格面；两份报告记录函数源码 SHA-256、PyTorch/CUDA 版本及首次和暖运行耗时。组件计时来自共享 GPU，各次负载不同，不据此计算整例提速。

另用同一真实丘脑的官方工作图像和已拟合粗网格检查[有效块裁剪实验](unified_masked_block_experiment.json)：有效体素 725,122 个，候选块从 3,160 减为 1,842，成本相同，先验最大差为 0，梯度相对差 `3.47×10⁻⁸`。三轮暖运行的中位数由 1.224 s 降为 0.944 s，但首次构建为 5.543 s，另一次组件检查中暖运行也未加速。默认路径保留现有查找实现；实验脚本为 [check_masked_blocks.py](check_masked_blocks.py)。

同一丘脑图像的[合并写入实验](unified_single_write_experiment.json)将 262 次输出张量写入合并为一次。三组对照的先验、梯度和成本均相同；暖运行中位数从 1.735 s 变为 1.634 s。Profiler 中输出写入的反向节点从 262 个减为 1 个，但合并张量增加了内存，整体组件耗时改善较小。本次保留实验结果，未修改完整运行使用的生产函数。

## 完整亚区运行结果

### 原始 T1 全流程

完整四结构已经运行结束。主输出为输入 T1 的 `256×156×256` 网格，affine 最大差为 0，磁盘数据类型为 `int32`，所有非零标签均有元数据。四项高分辨率标签全部保存：脑干和丘脑为 0.5 mm，海马/杏仁核为 0.33333 mm。执行包记录的 GEMS、SynthSeg 源码 SHA-256 与当前实现相同。

| 结构 | 前景 Dice | 逐标签 Dice 中位数 | 最小 Dice | 同时达标标签 | 最大硬体积差 | 最大软体积差 |
|---|---:|---:|---:|---:|---:|---:|
| 脑干 | 0.9359 | 0.9009 | 0.7737 | 0/4 | 9.8% | 10.8% |
| 丘脑 | 0.9100 | 0.7197 | 0.0000 | 0/44 | 100.0% | 36.3% |
| 左海马 | 0.8934 | 0.6964 | 0.4324 | 0/19 | 16.4% | 17.5% |
| 右海马 | 0.8881 | 0.5926 | 0.3563 | 0/19 | 26.0% | 23.8% |
| 左杏仁核 | 0.8882 | 0.6937 | 0.2581 | 0/9 | 100.0% | 49.5% |
| 右杏仁核 | 0.8854 | 0.4789 | 0.0000 | 0/8 | 200.0% | 61.0% |

本例观测到的 103 个非空标签均未同时达到 Dice≥0.95、硬体积差≤5%。另外 7 个双方硬分割为空的标签按既定规则不计入验收，软体积仍完整报告。完整数字见[逐标签 TSV](unified_raw_t1.tsv)、[完整报告](unified_raw_t1.json)和[网格、结构、资源摘要](unified_raw_t1_summary.json)。不能将整体前景 Dice 当作每个亚区已达标，也不能把此结果称为官方等价。

API 墙钟 **14,920.79 s（248.68 min）**；PyTorch 分配峰值 **15.47 GiB**，采样观测的本进程显存最大值为 **18,930 MiB**。GPU 1 利用率中位数为 100%，全卡使用显存为 8,774–73,308 MiB，包含其他任务。共取得 1,287 次负载记录；查询延迟造成 12 段大于 20 秒的间隔，最大间隔 661.26 秒，因此进程显存数字是采样最大值。CUDA 分配器的 23% 上限持续生效。本次完整运行没有可据以宣称官方等价提速的结果。

下图来自本例真实 T1。六种颜色表示结构，第三行按完整亚区 label ID 显示差异；每个结构内部的逐标签精度由上面的 TSV 报告。

![原始 T1 全流程：官方、FNIT 和完整亚区标签差异，六张轴位切片](unified_raw_t1_axial.png)

### 相同阶段输入

完整四结构运行也已结束，参考与候选使用同一组三个 `norm/aseg/wmparc` 文件。输出为 `256×256×256` 原始网格的 `int32` NIfTI，affine 最大差为 0，所有非零标签都有元数据，四项高分辨率输出齐全；记录的 GEMS/SynthSeg 源码哈希与当前实现一致。

| 结构 | 前景 Dice | 逐标签 Dice 中位数 | 最小 Dice | 同时达标标签 | 最大硬体积差 | 最大软体积差 |
|---|---:|---:|---:|---:|---:|---:|
| 脑干 | 0.9933 | 0.9893 | 0.9605 | 4/4 | 1.0% | 1.5% |
| 丘脑 | 0.9842 | 0.9164 | 0.4000 | 12/44 | 133.3% | 27.4% |
| 左海马 | 0.9644 | 0.8844 | 0.6234 | 1/19 | 14.8% | 12.0% |
| 右海马 | 0.9725 | 0.8878 | 0.7079 | 2/19 | 19.8% | 5.3% |
| 左杏仁核 | 0.9764 | 0.8182 | 0.6667 | 2/9 | 33.3% | 6.5% |
| 右杏仁核 | 0.9782 | 0.9021 | 0.5714 | 1/8 | 33.3% | 5.9% |

脑干四区全部达标，最小 Dice 为 0.9605，最大硬体积差为 1.04%。其他 18 区达标，丘脑、海马和杏仁核仍有 81 区未达到联合验收目标。两种输入条件的全部失败标签保留在[阶段逐标签 TSV](unified_stage.tsv)、[完整报告](unified_stage.json)和[结构/资源摘要](unified_stage_summary.json)，阈值保持不变。

API 墙钟 **16,657.89 s（277.63 min）**；PyTorch 分配峰值 **6.55 GiB**，本进程采样显存最大值 **10,470 MiB**。物理 GPU 0 利用率中位数为 100%，全卡使用显存 29,804–79,543 MiB，包含其他任务。1,461 次负载记录中，11 段间隔超过 20 秒，最大间隔 661.40 秒。CUDA 分配器持续限制在显卡容量的 14%。

![同阶段输入：官方、FNIT 和完整亚区标签差异，六张轴位切片](unified_stage_axial.png)

### 完整运行的性能结论

| 输入条件 | API 墙钟 | PyTorch 分配峰值 | 本进程采样显存最大值 |
|---|---:|---:|---:|
| 官方阶段输入 | 16,657.89 s | 6.55 GiB | 10,470 MiB |
| 原始 T1 全流程 | 14,920.79 s | 15.47 GiB | 18,930 MiB |

旧官方四线程记录为脑干 240.10 s、丘脑 596.84 s、左右海马/杏仁核合计 1,005.17 s，见[脑干历史报告](README.md)和[核团参考报告](nuclei.md)。这些是另一时段的官方命令计时。本次 GPU 全流程较慢，且共享负载与计时范围不同；没有完整官方等价的速度优势证据。共享预处理及栅格化组件的改进不能替代这两个完整运行的结果。

当前功能已经完成纯 PyTorch 迁移和统一输出；逐核团精度与完整性能尚未全部达到目标。相同阶段输入的结果确认脑干保留既有验收水平，新增丘脑/海马/杏仁核仍需改进拟合精度。原始 T1 比同阶段输入增加了粗分割、白质代理与强度预处理差异，本报告不把两组数字混为同一项精度。

前一个保留全部候选反向图的批处理版本，在限定显存的脑干拟合中发生 OOM；原始 T1 的旧 U-Net 也发生 OOM。失败日志保留在服务器的 `full_stage_v3.log`、`full_raw_v3.log`。新版已改为只对选中的四面体求梯度，并释放 U-Net 中已用完的张量；上述完整 T1 GPU 预处理已成功通过同一显存限制。失败版本的日志和取消记录不用于计算新版速度。

## 测试与计算位置

`tests/gems`、`tests/synthseg_parc`、`tests/test_setup_weights.py`：**77 passed，0 failed，0 skipped**，35.25 s。测试使用已校验的 SynthSeg/SynthSeg+ 权重；六项此前因未指定权重路径而跳过的检查，本次补齐路径后全部通过。测试中使用官方皮层标签顺序作参考，并验证其与 FNIT 内置元数据相同；默认用户运行不依赖 FreeSurfer 安装目录。

测试和完整运行的环境为 Python 3.11.7、PyTorch 2.5.1/CUDA 11.8、NumPy 1.26.4、SciPy 1.11.4、Nibabel 5.4.0。冻结测试包没有 `native_samseg` 和 `_vendor_fsl`；代码版本为 `0dfb8e7`。此前共享 GPU 导致的 CUDA 初始化 OOM 单独保留为资源失败，不计为功能通过。

SynthSeg 网络、仿射优化、先验栅格化、Gaussian EM 和网格形变优化使用指定的 PyTorch CPU/CUDA 设备。图谱读写、裁剪、三次插值、形态学处理、白质标签传播、海马部分容积超参数准备及最终 Nibabel 重采样仍在 CPU 执行。默认统一路径不导入 `native_samseg`，不调用 FreeSurfer/FSL 程序。

## 复核命令

```bash
# 相同阶段输入：--aseg/--wmparc 与官方参考使用同一网格
python validation/subregions/run_unified.py \
  --t1 /absolute/path/subject/mri/norm.mgz \
  --aseg /absolute/path/subject/mri/aseg.mgz \
  --wmparc /absolute/path/subject/mri/wmparc.mgz \
  --atlas-root /absolute/path/subregion_atlases \
  --structures all --device cuda:0 --output-dir /absolute/path/stage_result \
  --reference-brainstem /absolute/path/official/brainstemSsLabels.FSvoxelSpace.mgz \
  --reference-thalamus /absolute/path/official/ThalamicNuclei.FSvoxelSpace.mgz \
  --reference-left /absolute/path/official/lh.hippoAmygLabels.FSvoxelSpace.mgz \
  --reference-right /absolute/path/official/rh.hippoAmygLabels.FSvoxelSpace.mgz

# 原始 T1 全流程：省略 --aseg/--wmparc，--weights 指已校验的模型缓存
python validation/subregions/run_unified.py \
  --t1 /absolute/path/sub-01_T1w.nii.gz --weights /absolute/path/weights \
  --atlas-root /absolute/path/subregion_atlases \
  --structures all --device cuda:0 --output-dir /absolute/path/raw_result
```

### Reference

原实现、各图谱和 SynthSeg 的论文列于[功能说明末尾](../../docs/subregions/README.md#官方对照与参考文献)。
