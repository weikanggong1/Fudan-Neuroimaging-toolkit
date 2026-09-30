# 统一 TorchGEMS 亚区：真实 T1 验证

[功能和参数](../../docs/subregions/README.md) · [历史脑干验证](README.md) · [旧 C++/ITK 核团验证](nuclei.md)

## 输入和验收

使用 OpenNeuro ds000114 的一个完整 T1。输入为仓库已去除面部的公开衍生影像，未做 recon-all；来源、CC0 许可及处理记录见 [SOURCES.json](../../examples/data/SOURCES.json)。文件为 `sub-01_T1w.nii.gz`，形状 `256×156×256`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。这一个开发病例不作为独立留出集。

两个完整运行分别报告：

1. **相同阶段输入**：FNIT 与 FreeSurfer 8.2 使用相同 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`，比较亚区拟合。
2. **原始 T1 全流程**：FNIT 从上述 T1 自行生成粗分割、皮层分区和白质代理，再拟合全部结构；与同一人的官方输出比较。

每区要求 Dice ≥0.95、硬体积差 ≤5%；不要求逐体素相同。未通过的标签保留数字；参考为空而候选非空的标签记为失败，双方均为空不计入 Dice 验收。软体积单独报告后验积分差。

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

## 完整亚区运行状态

当前完整四结构的阶段对照和原始 T1 全流程仍在执行；尚未形成这版的全部逐区 Dice、体积和总时间，不宣称丘脑、海马或杏仁核已达到验收标准。

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
