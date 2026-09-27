# SynthStrip：移除运行时 Surfa 后的同输入核对

本次仅替换 SynthStrip 的图像几何、裁剪、重采样、距离扩展和连通域处理。网络结构及权重不变。运行时用 nibabel 读写影像，仓库内代码计算图像几何，用 NumPy/SciPy 处理 CPU 预后处理；推理仍由 PyTorch 执行。传入既有 Surfa 内存对象时保留该对象类型，供旧调用方过渡；从文件或 nibabel 对象调用不会导入 Surfa。

## 输入、输出与运行条件

- 输入：同一被试的 `mri/orig.mgz`，SHA-256 `d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f`，256³、uint8，来自真实 T1 的 recon-all 中间结果。
- 权重：`synthstrip.1.pt`，SHA-256 `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33`；`no_csf=True` 用另一个官方权重。
- 机器：`headcw`；官方为 FreeSurfer 8.2.0-1 的 `mri_synthstrip` / `fspython`（PyTorch 2.1.2+cpu、Surfa 0.6.3）；本包为独立 Conda Python 3.11（PyTorch 2.5.1），CPU 4 线程。官方脚本 SHA-256 `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590`。
- 输出：`image`/`-o` 是去颅骨影像，`mask`/`-m` 是二值脑掩膜，`distance`/`-d` 是毫米单位有符号距离图。三者均保留输入的 256³ 体素格及 RAS 几何。本次分别保存 MGZ，图像与掩膜为 uint8，距离图为 float32。

对应命令如下。`-i` 为相同输入，`-o`、`-m`、`-d` 分别指定三张输出；官方 `-t 4` 与本包 `-j 4` 都指定 4 个 CPU 线程。

```bash
module load freesurfer
mri_synthstrip -i /path/to/orig.mgz -o /path/to/official_image.mgz \
  -m /path/to/official_mask.mgz -d /path/to/official_distance.mgz -t 4

fnit synthstrip -i /path/to/orig.mgz -o /path/to/candidate_image.mgz \
  -m /path/to/candidate_mask.mgz -d /path/to/candidate_distance.mgz \
  --weights /path/to/synthstrip.1.pt --device cpu -j 4
```

## 数值核对

| 输出 | 体素不一致数 / 总数 | 最大体素误差 | 仿射矩阵最大误差 |
|---|---:|---:|---:|
| 去颅骨影像 | 0 / 16,777,216 | 0 | 0 mm（nibabel 重载） |
| 二值掩膜 | 0 / 16,777,216 | 0 | 0 mm（nibabel 重载） |
| 有符号距离图 | 0 / 16,777,216 | 0 mm | 0 mm（nibabel 重载） |

Surfa 重载 MGZ 后的仿射数值差为约 `5.68e-14` mm。候选脑图与既有官方 `mri/synthstrip.mgz` 也逐体素一致。完整哈希和逐文件数值见 [机器报告](report.json)。压缩文件**并非逐字节相同**：MGH 体素载荷逐字节一致，但 FreeSurfer/Surfa 写盘保留的附加尾部元数据比 nibabel 多；本次官方文件的尾部为 1,208 字节，本包为 20 字节。后续 recon-all 核对的是体素和几何，不能将上述数值一致写成整文件哈希一致。

同一真实 T1 的三个 API 参数分支在原始 `orig.mgz` 网格上，`border=1`、`border=8`、`no_csf=True` 各自的影像、掩膜和距离均逐体素一致。斜切原始扫描 `orig/001.mgz`（256×156×256）上，影像和掩膜仍逐体素一致；距离图存在 float32 运算顺序差异，最大 `9.54e-7` mm，**没有**体素超过 `1e-4` mm。4D 入口用该真实 T1 的两帧派生 NIfTI 检查功能，影像和掩膜逐体素一致，距离图最大 `9.54e-7` mm；这两帧不能当作两个独立被试的准确度证据。NIfTI 写盘后的体素一致，数值仿射与 Surfa 写盘最大相差 `7.33e-6` mm。

独立进程在导入钩子中明确阻止 `surfa` 及其子模块后，`from fnit import SynthStrip`、nibabel 内存影像输入、推理和 MGZ 保存均完成；统一 `fnit synthstrip` 入口的掩膜写盘也在相同导入禁用条件下完成，`sys.modules` 无 Surfa 条目。输入内存影像保持不变。旧调用方传入已经创建的 `surfa.Volume` 时，返回值仍为同类对象，已单独确认。

最初使用 float64 坐标重采样时，斜切影像边界上有 29,694 个距离体素误差超过 0.1 mm。原因是 Surfa 0.6.3 的[插值实现](https://github.com/freesurfer/surfa/blob/v0.6.3/surfa/image/interp.pyx)在插值前把体素变换与坐标转为 float32，边界坐标会被舍入到视野内或视野外。按该精度重算后，大误差消失，最大残差降至 `9.54e-7` mm。默认脑掩膜在修正前后均无差异，但距离图属于公开输出，因此保留此修正。

## 耗时与内存

计时包含完整 CLI 进程的解释器启动、读盘、加载权重、CPU 推理和三张 MGZ 写盘；`module load` 在官方计时之前完成。两次顺序运行如下，不据两次试验估计总体速度分布。

| 运行 | 第一次墙钟 | 第二次墙钟 | 第一次峰值 RSS | 第二次峰值 RSS |
|---|---:|---:|---:|---:|
| FreeSurfer 8.2 官方 CLI | 175.39 s | 79.30 s | 3,466,496 KiB | 3,469,408 KiB |
| 本包无 Surfa CLI | 8.92 s | 9.18 s | 3,619,952 KiB | 3,624,184 KiB |

官方 `fspython` 的框架版本与本包不同，首次进程启动也明显波动；这组数据说明该机器上的完整命令耗时，**不能把两者差值归因于移除 Surfa**。在相同 Conda/PyTorch 2.5.1 进程中，旧 Surfa API 与新 API 的一次无写盘调用分别为 5.92 s 和 6.11 s；新几何路径没有在这一次调用中提速，几何处理更多在 CPU 上执行。CUDA 网络与 TF32 默认策略未改，本次没有重新做 GPU 计时。

## 复核范围

本报告只证明 SynthStrip 子函数的这些输入和参数，未重新运行完整 recon-all，也未验证跨被试稳健性。最新独立调用说明见 [SynthStrip 文档](../../docs/synthstrip/README.md)。
