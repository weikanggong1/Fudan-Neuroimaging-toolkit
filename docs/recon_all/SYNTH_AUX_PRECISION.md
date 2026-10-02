# Synth辅助网络的精度设置

## 功能及修复

`segment_sclimbic_image`是FNIT已有PyTorch LimbicUNet推理入口，用于EntoWM、MCA-dura和静脉窦分割。pipeline诊断发现原函数固定开启cuDNN TF32，使调用方显式关闭TF32的设置失效。现在前向继承调用方的cuDNN TF32设置，并记录实际设备、dtype、TF32和autocast；生产默认TF32保留，未启用半精度。设备选择与精度策略分开。

函数作用域仍启用cuDNN，返回或异常时恢复原cuDNN状态。TF32开关属于进程全局状态，有不同精度策略的前向不能在同一进程并发执行。

recon-all主调度对EntoWM及MNI辅助链采用局部`torch.backends.cudnn.flags(allow_tf32=False)`。matmul TF32保持原默认，作用域结束或失败时恢复cuDNN策略，不全局关闭TF32。MCA双侧模型缓存键包含实际cuDNN TF32设置，避免报告与缓存身份不一致。

## Python接口与数据结构

| 参数 | 输入及默认值 |
| --- | --- |
| `source` | nibabel影像，3D、体素边长约1 mm，affine单位mm；保留原生网格与方向 |
| `model_path` | 声明的HDF5权重路径；未传入已加载模型时读取 |
| `rows` | 有序`(整数label_id, 标签名)`序列，顺序与模型通道对应 |
| `fov` | RAS网络输入立方体边长，体素，默认160 |
| `device` | 明确的`cpu`或`cuda:N`，函数默认`cpu`；recon-all传入目标GPU |
| `model` | 已加载、相同权重和device的`torch.nn.Module`；默认None加载一次 |
| `stats_path` | 可选统计文件路径，默认None；计数及概率积分体积，单位mm³ |
| `etiv` | 可选颅内容积，mm³，默认None |
| `precision_report` | 可选列表，默认None；追加实际前向设备、dtype、cuDNN/matmul TF32和autocast记录 |

返回`nibabel.MGHImage`：shape与输入相同，int32标签，原生affine及方向。内部转为RAS并裁剪/填充网络输入，推理后映回原网格；不把网络RAS立方体当作输出空间。影像保留在内存，统计路径非None时写统计文件。

非约1 mm体素、标签数与模型通道不符、权重加载或推理失败时抛异常。空图像或无有效强度范围不作为有效输入。实际前向记录不能只由张量float32推断TF32是否启用。

```python
import nibabel as nib
import torch
from fnit.recon_all.sclimbic import segment_sclimbic_image, _ctab_rows

source_image = nib.load("subject/mri/nu.mgz")  # 自产1 mm影像，原生空间
model_weight_path = "weights/entowm.fsm31.t1.nstd00-30.nstd21-108.h5"  # 已校验权重
ordered_label_rows = _ctab_rows("weights/entowm.ctab")  # 权重匹配的完整颜色表，按模型通道排序
actual_forward_records = []  # 保存本次前向精度，不能由历史声明推断
with torch.backends.cudnn.flags(allow_tf32=True):  # 默认GPU策略；诊断可显式设False
    label_image = segment_sclimbic_image(
        source=source_image,  # nibabel原网格影像
        model_path=model_weight_path,  # HDF5模型路径
        rows=ordered_label_rows,  # 与模型通道一一对应的完整标签表
        fov=160,  # 网络RAS立方边长，体素
        device="cuda:0",  # 显式指定逻辑GPU
        model=None,  # 此次加载模型；同一运行可复用已加载模型
        stats_path=None,  # 本例不写统计文件
        etiv=None,  # 不提供颅内容积
        precision_report=actual_forward_records,  # 写入实际前向状态
    )
```

## 命令及验证

该函数属于`mri_sclimbic_seg`内部网络步骤，没有独立等价CLI；FNIT公开文件接口为`mri_sclimbic_seg`和`mri_entowm_seg`。完整recon-all CLI见[入口说明](README.md)。原软件完整入口为`mri_sclimbic_seg --help`，权重/颜色表随具体模型匹配，不能用一个通用命令假定不同网络输入相同。

固定自产真实T1检查点的精度对照脚本为`validation/recon_all/optimizations/20261001_serial/benchmark_synth_aux.py`。`--cudnn-tf32`和`--matmul-tf32`仅取0/1，默认1；计时包括模型加载、传输与输出写出，检查点复制不计入阶段时间。每次使用新目录，记录影像、程序、权重、资产哈希及实际前向。

首次61926c7对照的cuDNN关闭设置被覆盖，不能当作FP32测试结论；原始记录保留在[失效对照](../../validation/recon_all/optimizations/20261001_serial/whole/diagnostics/aux_fp32/summary.json)。修复后真实回归结果追加到[本轮报告](../../validation/recon_all/optimizations/20261001_serial/WHOLE_RESULTS.md)，未完成项明确标记。CPU语义单元测试覆盖调用方cuDNN开关及TF32开关四种组合、实际前向记录和状态恢复，不代替真实GPU benchmark。

### 冻结真实T1的三种精度策略

同gpucw1/H100、4线程、显式cuda:0、分配缓存开启；c757cc8源码。默认TF32与修复前GPU逐体素相同；以下不同体素数相对此前同输入CPU结果，顺序为EntoWM/MCA-dura/vsinus。实际前向均float32、autocast关闭。

| 设置 | sub-01不同体素 | sub-02不同体素 | sub-01阶段秒 | sub-02阶段秒 |
| --- | --- | --- | ---: | ---: |
| cuDNN TF32 / matmul TF32 | 1/0/4 | 0/0/4 | 5.705 | 6.285 |
| cuDNN FP32 / matmul FP32 | 0/0/0 | 0/0/0 | 6.211 | 6.301 |
| cuDNN FP32 / matmul TF32 | 0/0/0 | 0/0/0 | 5.242 | 5.783 |

计时包含模型加载、传输和输出写出，GPU已同步。各策略各测一次，执行先后和共享主机负载不同，不将最后一行较短的观察值宣传为FP32比TF32更快。此前这些少量辅助标签变化传播到毫米级white/pial变化，所以生产采用经验证的最小卷积精度例外；后续整例验证另报。

## 原实现与引用

复用项目既有PyTorch网络与HDF5权重转换，不引入系统FreeSurfer运行依赖。上游实现为[FreeSurfer mri_sclimbic_seg](https://github.com/freesurfer/freesurfer/tree/dev/mri_sclimbic_seg)。模型和资产的来源、许可证及固定版本以[资源清单](../../validation/recon_all/optimizations/20261001_serial/runtime_fingerprints_61926c7.json)为准。
