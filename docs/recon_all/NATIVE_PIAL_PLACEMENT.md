# 标准流程的 pial 放置

标准单 T1 流程使用 Conda 环境中从固定 FreeSurfer 8.2 源码独立编译的 `mris_place_surface` 执行四轮 pial 放置。它读取 FNIT 已生成的最终 white、MRI、皮层标签和注释，不读取官方参考被试。独立的 [Python pial 函数](PYTHON_PIAL_PLACEMENT.md)仍可用于单阶段计算与算法核对；在自产上游的真实 T1 测试中，Python 路径明显较慢，因此不作为标准流程的默认阶段。

## 输入、输出与空间

内部函数 `_run_native_pial(binary, subject, hemi, assets, threads) -> dict` 接受：

| 参数 | 含义 |
| --- | --- |
| `binary` | 当前 Conda 环境从固定源码构建的 `mris_place_surface` 可执行文件。 |
| `subject` | 单被试目录，含 `mri/`、`surf/`、`label/` 和 `scripts/`。 |
| `hemi` | `lh` 或 `rh`；决定读取和写入哪个半球。 |
| `assets` | 经 SHA-256 校验的 FNIT 图谱目录；程序子进程的 `FREESURFER_HOME` 指向此目录。 |
| `threads` | 原生放置程序的 CPU 线程数。 |

每个半球需要 `surf/H.white`、`surf/autodet.gw.stats.H.dat`、`mri/aseg.presurf.mgz`、`mri/wm.mgz`、`mri/brain.finalsurfs.mgz`、`label/H.cortex+hipamyg.label`、`label/H.cortex.label` 与 `label/H.aparc.annot`。其中三个体积均是 1 mm conform 网格；表面顶点是该被试的 surface RAS 毫米坐标，三个标签与注释均按 `H.white` 的有序顶点编号解释。

函数写入 `surf/H.pial.T1` 和 `scripts/H.pial.log`。`pial.T1` 保留 white 的顶点数和有序三角面，只更新顶点位置；标准流程随后复制为 `surf/H.pial`，并从 white/pial 计算厚度、面积、体积和曲率。返回字典含 `implementation`、`output`、`log`、`vertices`、`faces`、`seconds`。若命令失败、输出缺失、坐标非有限，或有序面发生变化，函数抛出异常；主运行报告记录失败阶段。

## 调用

标准入口自动运行该阶段：

```bash
export FS_LICENSE=/private/license.txt
fnit-recon-all /data/sub01_T1w.nii.gz /data/subjects/sub01 \
  --weights-dir /data/fnit-weights \
  --assets-dir /data/fnit-assets \
  --device cuda:0 --threads 4
```

单独排查这一内部阶段时，可在已完成最终 white、注释和前置 MRI 的被试目录调用：

```python
from pathlib import Path
from fnit.recon_all.native_free import _run_native_pial

result = _run_native_pial(
    binary=Path("/opt/conda/envs/fnit/bin/mris_place_surface"),  # Conda 源码构建程序
    subject=Path("/data/subjects/sub01"),  # 已完成上述前置文件的单被试目录
    hemi="lh",  # 左半球；右半球填 rh
    assets=Path("/data/fnit-assets"),  # FNIT 外置图谱目录
    threads=4,  # 原生程序使用的 CPU 线程数
)
print(result["output"])  # surf/lh.pial.T1 的绝对路径
```

对应 FreeSurfer 8.2 的完整命令形式为：

```bash
mris_place_surface --adgws-in surf/autodet.gw.stats.lh.dat \
  --seg mri/aseg.presurf.mgz --threads 4 --wm mri/wm.mgz \
  --invol mri/brain.finalsurfs.mgz --lh --i surf/lh.white \
  --o surf/lh.pial.T1 --pial --nsmooth 0 \
  --rip-label label/lh.cortex+hipamyg.label \
  --pin-medial-wall label/lh.cortex.label --aparc label/lh.aparc.annot \
  --repulse-surf surf/lh.white --white-surf surf/lh.white --restore-255
```

## 真实 T1 对照

以下双引擎配对使用 v5 快照生成的同一组自产上游输入；v7 的连续整例另行记录。

在 `sub-01_T1w.nii.gz` 的 FNIT 自产输入上，Conda 源码构建程序的左侧、右侧 pial 分别耗时 291.5 秒和 274.14 秒；独立自相交检测均为零。两个半球与同输入 Python pial 都保持相同的有序面。左侧顶点位移平均 0.0417 mm、P99 0.2797 mm、最大 2.5910 mm；厚度相关性 **0.99845**、平均绝对差 0.0238 mm。右侧顶点位移平均 0.00781 mm、P99 0.1313 mm、最大 1.3418 mm；厚度相关性 **0.99964**、平均绝对差 0.00519 mm。局部最大误差仍需追踪。此处是**两个 FNIT 计算引擎的同输入比较**，不能替代与官方整例的数值验收；输入、二进制、输出哈希和完整数字在[机器可读报告](../../validation/recon_all/python_gpu_port/native_pial_candidate_20260929.json)。测试时共享节点另有任务，不据此发布稳定加速比。

此前在冻结官方上游的真实左侧输入中，另一版 Conda 源码构建 `mris_place_surface` 相对归档官方 pial 的平均位移为 0.0297 mm、P99 0.2294 mm、最大 1.6693 mm，详见[同输入记录](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/official_frozen_lh.json)。[Python 双侧同输入记录](PYTHON_PIAL_PLACEMENT.md)则达到有序坐标和面的逐值一致，但观察耗时约为 1,241 和 1,154 秒。不同日期与共享负载下的单次耗时不能直接作为受控速度比。当前源码构建的完整整例结果另以[发布门槛](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)为准。

## 2026-10-02：当前同输入三方比较

候选61926c7，gpucw1、4线程，输入为c248520的自产sub01左侧七项前置和aparc，
每项SHA见[完整报告](../../validation/recon_all/optimizations/20261001_serial/stage5/native_reference.json)。
官方8.2.0与FNIT Python完整有序坐标/面相同；当前Conda输出与冻结Conda几何
相同，但相对本次官方均值0.031369mm、P99 0.225255mm、最大1.482209mm。
程序版本分别为freesurfer 8.2.0和freesurfer gongwk-local，二进制SHA不同。
这是当前程序产物之间的差异，具体源码/编译原因仍未定位，不能笼统归因于随机性。

Conda176.787秒、官方133.863秒；Python同主机含冷JIT与读写1453.060→1232.634秒，
单次配对减少15.17%，优化前后文件SHA相同。Python仍明显较慢，生产保留Conda
路径并单列局部精度问题。此处只有一个半球完整三方测试，不代表双侧或整例
通过；另四个半球静态索引/法向组件已通过精确回归，整例验证另行记录。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
