# Python N4 后处理与完整阶段对照

FreeSurfer 8.2 使用 `mri_nu_correct.mni --ants-n4` 生成 `nu.mgz`。Python 实现先运行 [`n4_sitk.py`](../../src/fnit/recon_all/n4_sitk.py)，再由 [`n4_wrapper.py`](../../src/fnit/recon_all/n4_wrapper.py) 完成全局均值比值的五位小数处理、`mris_calc` 的 float32 缩放、以 Talairach 为中心的 50 mm 强度直方图，以及 `mri_make_uchar` 的 1%/90% 映射。它还保留 `mri_add_xform_to_header` 后的 MGH 头和 XFORM 元数据。此阶段在 CPU 上使用 Python 和编译版 SimpleITK，已接入当前 `run_recon_all_python`。最新 v3 同 T1 连通运行的 `nu.mgz` 与归档官方结果在全部 16,777,216 个体素、数据类型、仿射和前 284 字节 MGH 头上相同，见 [v3 汇总](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/benchmark_summary.json)。

移植依据为固定的 FreeSurfer [wrapper 脚本](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/mri_nu_correct.mni)和 [`mri_make_uchar` 源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_convert/mri_make_uchar.cpp)。

## 输入、输出与调用

`make_nu(original_file, n4_file, tal_xfm_file, output_file)` 只执行 N4 之后的步骤；四个参数依次为原始 `orig.mgz`、已校正的 `nu0.mgz`、Talairach `transforms/talairach.xfm` 和输出 `nu.mgz`。它从原始影像保留几何及 XFORM 尾部，输出 uchar MGH/MGZ。

```python
from fnit.recon_all.n4_wrapper import make_nu

scale, bins = make_nu(
    original_file="orig.mgz",  # 原始输入影像及几何来源
    n4_file="nu0.mgz",  # N4 已校正的中间图
    tal_xfm_file="transforms/talairach.xfm",  # Talairach 变换
    output_file="nu.mgz",  # 最终 uchar 强度图路径
)
```

函数返回 `scale`（全局均值缩放系数）与 `bins`（直方图首个/白质分箱的二元组）；最终影像写入 `output_file`。若从已有 `nu0.mgz` 用命令行重放后处理：

```bash
python -m fnit.recon_all.n4_wrapper \
  --orig orig.mgz --nu0 nu0.mgz --tal transforms/talairach.xfm --out nu.mgz
```

`--orig`、`--nu0`、`--tal`、`--out` 分别对应上述四个 Python 参数，均为必填；命令打印 `scale` 和 `histogram_bins`。官方完整阶段使用 `mri_nu_correct.mni --ants-n4`，并按 recon-all 顺序运行 `mri_add_xform_to_header -c`。

## 冻结真实 T1 的逐字节验证

固定输入为主验证中已完成被试 sub01 的 `orig.mgz`。其压缩文件 SHA-256 为 `7dde820d02968c9fe18056a9cc5cc776c1a6395c48dc4d3a47eb6ba9c399f518`；256³ 解码 uchar 体素 SHA-256 为 `84da8a990ef60c6ba30a4ddfbaba597a90e32bca720f96c6d741fcb08c504825`。原生程序组标识为 FreeSurfer 8.2.0-1、构建 `d932c45`。headcw 的两次新运行使用同一输入、Talairach 变换及 wrapper 参数 `--uchar ... --n 2 --ants-n4`。原生命令之后按 recon-all 顺序运行 `mri_add_xform_to_header -c`。

| 新生成的输出 | Python 与官方对照 |
| --- | --- |
| 最终 `nu.mgz` 形状/类型 | 双方均为 256³ / uchar |
| 差异体素 | 0 / 16,777,216 |
| 仿射及前 284 字节 MGH 头 | 相同 |
| 包含 XFORM 尾部的完整解压 MGH 字节 | **完全一致** |
| 完整解压 MGH SHA-256 | `6370e1ab0c888ec4d574a244ed745c640442c6ff6a2b54093d6353e89af07dc7` |
| 均值缩放与直方图分箱 | 两者均为 `1.13755989346540527642`、`(4, 45)` |

在参照版 `orig.mgz` 上，[`normalize_n4_footer`](../../src/fnit/recon_all/n4_wrapper.py) 还独立核对了 `nu0.mgz` 的元数据：规范化后，完整解压 `nu0.mgz` 与官方 N4 输出逐字节一致。这份参照输入的体素 SHA 和前 284 字节 MGH 头与主输入相同，但 XFORM 尾部路径不同（1,346 与 1,398 字节）；不能跨两条输入路径做文件字节比较。规范化修正一处已知 FreeSurfer XFORM `UNKNOWN` 标签编码：源长度含末尾零字节为 8，原生 N4 输出长度为 7。

较早 gpucw1 整例存档的 `nu.mgz` 相对 headcw 的**两次新运行**均差 34 个体素（最大绝对差 2）；其中 9 个体素高 1、25 个高 2。存档日志的缩放和直方图映射与新官方运行相同。将新 N4 uchar 图在这 34 处各增加 1，可重现全部存档最终值。原 wrapper 已删除存档 `nu0.mgz`，现有文件不足以确定上游首因。因此，Python 输出的逐字节验证参照是同输入的新官方运行，而非旧存档 `nu.mgz`。

## 耗时与范围

headcw 上一次完整阶段运行：原生 `mri_nu_correct.mni` 157.84 s，加上 `mri_add_xform_to_header` 0.26 s；Python N4、尾部规范化和最终 `nu.mgz` 生成合计 125.56 s。两者各测一次墙钟，均未用 GPU。原生脚本涵盖 N4 和全部后处理；Python 时间包含 SimpleITK 导入与文件 I/O。[N4_SITK_VALIDATION.md](N4_SITK_VALIDATION.md) 的单独 N4 计时来自其他运行，不能与这里相加。

只计后处理时，headcw 上以同一张冻结 `nu0.mgz` 交替配对运行三次。原生中位数为 **14.71 s**，包含 `mri_binarize`、两次 `mri_segstats`、`mris_calc`、`mri_convert`、`mri_make_uchar`、`mri_add_xform_to_header`；常驻 Python 函数中位数为 **0.99 s**。三对输出的完整解压 MGH 均逐字节一致。原生各子步中位数依次为 1.34、2.32、3.10、2.02、2.19、3.34、0.26 s。Python 计时不含模块导入，原生计时包含每次子进程启动。原始[配对报告](../../validation/recon_all/python_gpu_port/n4_wrapper_headcw_report.json)保存每次耗时与输出哈希。

仅后处理的配对 benchmark 可用 [`benchmark_n4_wrapper.py`](../../validation/recon_all/python_gpu_port/benchmark_n4_wrapper.py) 复现；脚本不重新运行 N4。远端阶段日志和字节对照文件保存在 headcw 的 `/tmp/reconall_n4_sub01_20260925/`。[`test_n4_wrapper.py`](../../tests/recon_all/test_n4_wrapper.py) 的两项针对性测试已通过。
