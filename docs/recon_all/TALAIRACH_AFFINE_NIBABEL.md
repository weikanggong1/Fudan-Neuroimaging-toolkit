# Talairach SynthMorph affine 注册

recon-all 的这一步将 SynthStrip 脑图配准到 MNI305 模板，生成供后续强度校正、分割和 eTIV 计算使用的 Talairach 变换。FNIT 在此路径以 NiBabel 读取 MGH 几何，PyTorch 运行 SynthMorph affine 权重，NumPy 写 LTA 和 XFM；无需安装 FreeSurfer 或 Surfa。网络权重和 MNI305 模板由外部资产目录提供。

## Python 调用

```python
from fnit.recon_all.talairach_synthmorph import register_talairach

matrix = register_talairach(
    moving="subject/mri/synthstrip.mgz",  # 输入：单帧 3D 去脑外组织 T1，MGZ，移动图像
    template="assets/average/mni305.cor.stripped.mgz",  # 输入：MNI305 固定模板，MGZ
    weights="weights",  # 输入：含 synthmorph.affine.2.h5 的外置权重目录
    output_xfm="subject/mri/transforms/talairach.xfm",  # 输出：MNI XFM 文本，3×4 RAS 仿射
    output_lta="subject/mri/transforms/synthmorph.mni305/aff.lta",  # 输出：type 1 RAS_TO_RAS LTA，含两端体积几何；可设为 None
    device="cuda:0",  # 推理设备；无 GPU 时设为 "cpu"
    threads=4,  # PyTorch CPU 线程数
)
# matrix：NumPy float32 4×4 数组，与 output_xfm 中前三行相对应。
```

`moving` 应与 recon-all 的 SynthStrip 输出一致；`template` 是 MNI305 脑模板，二者都必须是单帧 3D 图。`weights` 为目录或该权重文件路径。`output_xfm` 是必须写出的 Talairach 变换；`output_lta=None` 时只写 XFM。函数不输出配准后的体积图。两处输出目录不存在时会创建。网络张量保持 float32；此阶段暂时关闭 matmul 和 cuDNN TF32 以维持已验收的矩阵精度，结束时恢复调用前标志。

## 命令行调用

```bash
python -m fnit.recon_all.talairach_synthmorph \
  subject/mri/synthstrip.mgz assets/average/mni305.cor.stripped.mgz \
  --weights weights \
  --xfm subject/mri/transforms/talairach.xfm \
  --lta subject/mri/transforms/synthmorph.mni305/aff.lta \
  --device cuda:0 --threads 4
```

前两个位置参数依次是移动的 SynthStrip T1 和固定的 MNI305 模板；其余命名参数与上面的 Python 参数一一对应。`--lta` 可省略。整例程序会在输入/Talairach 阶段调用同一 Python 函数。

## 官方对应命令与验收

FreeSurfer 8.2 的对应流程由以下命令组成；路径应替换成各自被试和模板位置：

```bash
mri_synthmorph -m affine -t aff.lta synthstrip.mgz mni305.cor.stripped.mgz -j 4
lta_convert --ltavox2vox --inlta aff.lta --outlta talairach.xfm.lta
lta_convert --inlta talairach.xfm.lta --outmni talairach.xfm
```

FNIT 的同输入实测覆盖三张真实 SynthStrip 脑图。每组新旧 FNIT 的 16 个矩阵元素、LTA 和 XFM 字节以及 eTIV 完全相同；新路径稳态中位数 **0.738–0.832 秒**，旧 Surfa 路径 **1.446–1.734 秒**，PyTorch 峰值显存分配从 4497 MiB 降至 4433 MiB。归档 FreeSurfer 包装脚本输出与新路径仍有 1.2812e-5 的 LTA 最大矩阵元素差和 −0.420220 mm³ 的 eTIV 差；归档脚本使用 PyTorch neural hook，不是独立 TensorFlow 网络对照。[输入哈希、每次耗时与逐项差异](../../validation/recon_all/python_gpu_port/talairach_affine_no_surfa_20260927/README.md)记录了完整范围。

此改动只覆盖 recon-all Talairach affine 调用。通用 SynthMorph 的 `joint`、`deform`、`rigid`、`__call__` 和 `apply_transform` 仍依赖 Surfa，尚未完成同输入替换。

标准 CUDA 输入链现在在独立子进程中执行此 affine 阶段，以释放其 CUDA 常驻内存。子进程沿用 SynthStrip 后的 cuDNN benchmark、deterministic、TF32 标志，affine 推理局部关闭 TF32。对一张真实 T1 的冻结 `synthstrip.mgz`，子进程的 XFM 和 LTA 与同输入直接调用逐字节相同；子进程 PyTorch allocated/reserved 峰值为 4,715,548,672/4,884,267,008 字节。此结果尚不代表整例显存达标；[原始哈希与测量记录](../../validation/recon_all/python_gpu_port/talairach_child_20260929.json)。CPU 路径仍在当前进程调用该函数。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
