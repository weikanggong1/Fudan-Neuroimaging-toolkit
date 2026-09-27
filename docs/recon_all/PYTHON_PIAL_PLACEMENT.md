# Python pial.T1 表面放置

`fnit.recon_all.place_pial_python.place_pial_t1` 实现 FreeSurfer 8.2 `mris_place_surface --pial` 的四轮几何优化，以及内侧壁固定和表面相交修复。它在 CPU 上使用 NumPy、Numba 和 nibabel。该函数可单独调用；默认 `fnit-recon-all` 仍生成近似 pial，启用 `--native-white-preaparc` 后会在最终 white 放置后调用此函数。该可选串联尚未完成整例验收。此阶段没有 CUDA 验证，也没有整例重建等价的验收结果。

## 输入、输出与调用

`subject` 是 FreeSurfer 格式的被试目录，`hemisphere` 为 `lh` 或 `rh`。以下七个文件须由各自的前序阶段生成：

| 相对于 `subject` 的输入 | 用途 |
| --- | --- |
| `surf/{hemi}.white` | 提供有序顶点、面、体积几何和表面标签；pial 优化从其坐标开始。 |
| `surf/autodet.gw.stats.{hemi}.dat` | 被试的灰白质强度阈值。 |
| `label/{hemi}.cortex+hipamyg.label` | 放置与相交修复使用的顶点 rip 掩膜。 |
| `label/{hemi}.cortex.label` | 优化后将内侧壁固定回 white。 |
| `mri/brain.finalsurfs.mgz` | 放置所用强度图与体素几何。 |
| `mri/wm.mgz` | 强度预处理使用的白质类别。 |
| `mri/aseg.presurf.mgz` | 边界搜索所用分割。 |

```python
from fnit.recon_all.place_pial_python import place_pial_t1

report = place_pial_t1(
    subject="/path/to/subjects/sub01",  # 含七个前置文件的被试目录
    hemisphere="lh",  # 左半球；右半球用 rh
    output="/path/to/subjects/sub01/surf/lh.pial.T1",  # 输出表面路径
)
```

`output` 可省略，默认写入 `subject/surf/{hemi}.pial.T1`。函数输出一个 FreeSurfer 三角表面：保留输入 white 的**有序面、完整体积几何标签和辅助尾部**，以放置后的 pial 坐标替换顶点坐标。`report` 含 `output`（路径字符串）、`hemisphere`（半球）、`steps`（接受的优化步数）、`pass_ends`（四轮的累计步数）、`cleanup`（相交次数、轨迹和平滑次数）及 `seconds`（墙钟耗时）。`max_steps` 默认 200；未收敛时抛出异常，不写入未完成的表面。

函数不生成厚度、面积、曲率、体积或 atlas 统计；这些指标须由后续阶段计算。它也不生成所需的 white、标签或 MRI 输入。默认 runner 将 `smoothwm` 复制成 `white`，不能据此重现下述冻结同输入的逐点结果；可选高精度路径虽已接线，但还没有完成连续输入验收。按官方顺序，还需先完成 `white.preaparc` 放置、皮层和海马杏仁核标签、sphere 配准与 aparc 注释、最终 white 放置。本 pial 函数不直接读取 `white.preaparc` 或 `aparc.annot`，但它们属于上述上游流程。

本模块没有独立 Python CLI；上面的具名实参调用是使用入口。在 `subject/mri` 目录中，使用 FreeSurfer 8.2 和 `FS_LICENSE` 的对应官方命令为：

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --seg aseg.presurf.mgz --threads 4 --wm wm.mgz \
  --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white \
  --o ../surf/lh.pial.T1 --pial --nsmooth 0 \
  --rip-label ../label/lh.cortex+hipamyg.label \
  --pin-medial-wall ../label/lh.cortex.label \
  --aparc ../label/lh.aparc.annot \
  --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white \
  --restore-255
```

右侧将文件名和半球参数中的 `lh` 改为 `rh`。Python 函数复现该命令中固定的 pial 放置分支；该分支使用显式 rip 标签，因此 Python 函数不需要注释文件。

## 冻结真实 T1 输入的同阶段对照

参照被试是由真实 T1 完成的 FreeSurfer 8.2 `a_official`，其 `mri/orig.mgz` SHA-256 为 `c99c246200cc35479b6b8cd691457985b66f2062d4a37a0c3592ff1b678b985c`。原始被试文件保存在 headcw 的 `work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`；Git 中没有存放患者影像。候选函数读取该被试保存的 white、brain.finalsurfs、WM、aseg、标签和阈值文件，**不读取官方 pial 几何、优化日志、决策或内存检查点**。输入哈希见[清单](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/input_manifest.json)。这是一项冻结官方上游输入的阶段验证，不是当前候选上游的整例验证。

| 同输入阶段 | Python 输出与官方对照 | 观察耗时 |
| --- | --- | ---: |
| LH pial.T1 | 独立接受 41 步；四轮结束于 26/32/36/41；有序顶点 106,622/106,622、float32 坐标 319,866/319,866、有序面 213,240/213,240 完全一致。保留源码写入方式后，体积几何字段与完整尾部字节一致。 | Python 在 headcw 耗时 1,240.8 s；另一次 headcw 官方 C++ pial 耗时 116.05 s。并发负载不同，未据此计算受控速度比。 |
| RH pial.T1 | 独立接受 41 步；四轮结束于 26/31/35/41；有序顶点 105,541/105,541、float32 坐标 316,623/316,623、有序面 211,078/211,078 完全一致。体积几何字段与完整尾部字节一致。 | Python 耗时 1,154.2 s；另一次 headcw 官方 C++ RH 重放耗时 91.77 s，运行负载发生变化。 |

输出文件的 SHA-256 可能因首条表面注释记录生成来源而不同。[双侧几何、尾部与耗时报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/python_pial/) 保存了逐项对照。以相同的官方 white 和 cortex 标签，将该阶段输出接入现有 Python 顶点指标函数后，相对归档官方图的 p99 绝对误差：厚度不超过 2.38e-7 mm、pial 面积不超过 2.38e-7 mm²、顶点体积不超过 4.77e-7 mm³；两侧相应最大误差分别不超过 4.77e-7 mm、9.54e-7 mm² 和 1.91e-6 mm³。曲率函数仍有小幅算法误差：LH/RH 的 p99 为 2.71e-5/2.15e-5，最大值为 4.54e-4/2.03e-4。双侧共八张图均没有超过既定容差的顶点（面积：0.001 + 0.001 × |reference|；其他图：0.005 + 0.001 × |reference|）。[指标图比较脚本](../../validation/recon_all/python_gpu_port/compare_pial_metric_maps.py)与 [LH/RH JSON 报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/python_pial/)保留逐图计数。这些冻结输入结果尚不能代表当前默认 runner 的指标或吞吐。此项 CPU pial 优化在有限观测中明显较慢；待上游候选链匹配后，碰撞检测和 KDTree 是主要提速对象。

所需 Python 包为 `nibabel`、`NumPy`、`SciPy`、`PyTorch`、`Numba`，由仓库 `environment.yml` 的 `recon-all-python-stages` 扩展安装。`place_pial_t1` 不调用外部 FreeSurfer 可执行程序。
