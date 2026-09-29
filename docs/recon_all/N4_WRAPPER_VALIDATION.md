# N4 后处理：`nu0.mgz` 到 `nu.mgz`

FreeSurfer 8.2 的 `mri_nu_correct.mni --ants-n4` 在 N4 后恢复全局均值，再按 Talairach 中心 50 mm 球内的强度直方图映射到 uchar。FNIT 先用[仓库内 Conda C++ N4](N4_ITK_CONDA.md)生成 `nu0.mgz`，再由 [`n4_wrapper.py`](../../src/fnit/recon_all/n4_wrapper.py) 执行这一步。固定官方依据为 [`mri_nu_correct.mni`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/mri_nu_correct.mni) 和 [`mri_make_uchar.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_convert/mri_make_uchar.cpp)。

## 输入、输出与调用

`make_nu(original_file, n4_file, tal_xfm_file, output_file) -> tuple[float, tuple[int, int]]` 的输入与输出：

| 参数或返回值 | 含义 |
| --- | --- |
| `original_file` | 原始三维 `orig.mgz`；提供体素均值、几何和 MGH 尾部。 |
| `n4_file` | 已完成 N4 的三维 uchar `nu0.mgz`。 |
| `tal_xfm_file` | `transforms/talairach.xfm`；用于定位 50 mm 强度直方图球。 |
| `output_file` | 写入三维 uchar `nu.mgz` 的路径；几何和 MGH 尾部来自 `original_file`。 |
| 返回 `scale, bins` | `scale` 是输入/校正图全局均值比；`bins` 为直方图首个及白质分箱编号。 |

```python
from fnit.recon_all.n4_wrapper import make_nu

scale, bins = make_nu(
    original_file="/data/sub01/mri/orig.mgz",  # 原始 T1
    n4_file="/data/sub01/mri/tmp/nu0.mgz",  # N4 子步输出
    tal_xfm_file="/data/sub01/mri/transforms/talairach.xfm",  # Talairach 变换
    output_file="/data/sub01/mri/nu.mgz",  # 最终 uchar 图
)
```

同一步的命令行：

```bash
python -m fnit.recon_all.n4_wrapper \
  --orig /data/sub01/mri/orig.mgz \
  --nu0 /data/sub01/mri/tmp/nu0.mgz \
  --tal /data/sub01/mri/transforms/talairach.xfm \
  --out /data/sub01/mri/nu.mgz
```

官方完整阶段命令为 `mri_nu_correct.mni --ants-n4`，按 recon-all 顺序还运行 `mri_add_xform_to_header -c`。FNIT 完整入口自动调用 `make_nu`，并在 `fnit-native-free-run.json` 中记录 `nu` 阶段墙钟。

## 真实 T1 比较

固定 sub01 的 256³ `orig.mgz` 输入 SHA-256 为 `d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f`。当前 Conda C++ N4 输出的 `nu0.mgz` 有 9 个体素比官方低 1；送入本函数后，最终 `nu.mgz` 有 **8 / 16,777,216** 个体素与官方不同，最大差 2；`scale=1.1375598934654052`，`bins=(4, 45)` 与官方一致，仿射和 MGH 前 284 字节一致。详情及哈希见[当前 N4 报告](../../validation/recon_all/python_gpu_port/n4_itk_conda_20260927/README.md)。它尚未通过完整重建的下游皮层指标验收。

为单独验证后处理函数，既有真实 T1 试验把**同一张**已校正 `nu0.mgz` 分别送入官方后处理及 Python 函数，配对三次；输出的完整解压 MGH 逐字节相同。官方多命令流程墙钟中位数 14.71 秒，常驻 Python 函数中位数 0.99 秒，见[配对原始记录](../../validation/recon_all/python_gpu_port/n4_wrapper_headcw_report.json)。这项对照不包含 N4 计算，不应与上述完整 N4 时间相加或作为当前整例提速依据。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
