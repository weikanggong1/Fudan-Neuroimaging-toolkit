# TorchFAST 源码目录

对脑提取后的单帧 T1w 估计 CSF、GM、WM 部分体积和乘性偏置场。输入可为路径或 nibabel 影像，八幅输出保留输入网格；不需要权重，不调用 FSL。CPU 使用 PyTorch，CUDA 顺序扫描使用项目 Conda 环境中的 Triton。

```python
from fnit.fast import TorchFAST

fast = TorchFAST(
    device="cuda:0",  # 计算设备
    threads=1,  # CPU 线程数
    execution="fsl",  # 保留 FAST4 的原位扫描和内部 radiological X 方向
)
result = fast(
    image="T1_brain.nii.gz",  # 脑提取后的 T1w
    mask=None,  # 省略时以正输入体素为脑区；显式 mask 需同网格
)
result.pve_gm.save(path="T1_brain_pve_1.nii.gz")  # GM 部分体积
result.bias_field.save(path="T1_brain_bias.nii.gz")  # 乘性偏置场
result.restored.save(path="T1_brain_restore.nii.gz")  # input / bias_field
```

对应单被试命令：

```bash
fnit fast -i T1_brain.nii.gz -o results/T1_brain --execution fsl --device cuda:0 -b -B
fast -t 1 -n 3 -b -B -o results/T1_brain T1_brain.nii.gz
```

第一条调用本包；第二条是原 FSL 的对应命令。输入是脑图，输出前缀为 results/T1_brain；-b/-B 保存偏置场及校正图，同时生成三张 PVE 和分类图。独立接口默认 execution="tensor"，需要原 FAST 顺序时显式用 fsl。

| 文件 | 分工 |
|---|---|
| `pipeline.py` | nibabel 输入检查、内部 X 翻转、输出影像与网格 |
| `algorithm.py` | 三组织 HMRF-EM、bias、混合类型和 PVE |
| `_fsl_scan.py` | glibc 连续随机流、18 邻域原顺序波前、单次 ICM、逐项卷积 |

当前固定同一真实 T1 脑图的三张 PVE 对原 FAST Pearson 为 0.999999993 / 0.999999992 / 0.999999998，三张分类图逐体素相同。PVE 仍有 28 / 40 / 12 个体素的一档 0.01 差异；八图不能整体称为逐位相同。H100 返回完整八图的调用时间 10.95 s，allocated 峰值 872.5 MiB，排除输入读取和压缩写盘；原 CPU 独立进程 150.00 s。共享负载和计时范围见[验证记录](../../../validation/fast/README.md)。

全部参数、输出合同、当前模板空间图、原命令及文献见[专属说明](../../../docs/fast/README.md)。已有上游参考和改写受 [FSL 6.0 非商业许可](../../../licenses/FSL-6.0.txt)约束。
