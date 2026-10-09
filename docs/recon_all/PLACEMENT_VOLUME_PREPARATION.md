# White/pial 放置强度图准备

## 1. 功能简介

`prepare_placement_volume()` 是表面放置的内部步骤。它在既有 conformed
体素网格内裁剪过亮白质，识别非白质亮区及其边界，分别生成 white 或 pial
边界搜索使用的强度图。计算复用现有 NumPy/SciPy 实现，不进行配准或重采样。
本次修复了成熟子函数的阈值错误；没有添加被试或体素特例。

固定源代码的表达式 `((3*3*3-1)/2)` 使用整数除法，结果为 **13**；旁边的
注释误写14。此前 FNIT 按注释写成14，多选了恰好有13个白质邻居的亮区
种子。改成实际表达式后，两例真实 white/pial 准备图均与当前 Conda 源码
构建程序一致。此修复会影响现有 white 与 pial 共用的准备函数，坐标空间、
标签语义与调用接口保持一致。

## 2. Python 调用、输入与输出

```python
import nibabel as nibabel
import numpy as numpy
from fnit.recon_all.place_surface_volume import prepare_placement_volume

brain_image = nibabel.load("/data/fnit/sub07/mri/brain.finalsurfs.mgz")
white_matter_image = nibabel.load("/data/fnit/sub07/mri/wm.mgz")
if brain_image.shape != white_matter_image.shape or not numpy.array_equal(
    brain_image.affine, white_matter_image.affine
):
    raise ValueError("输入MRI网格不一致")

placement_volume, bright_region_labels = prepare_placement_volume(
    brain=numpy.asarray(brain_image.dataobj),  # 三维放置强度图，conformed体素网格
    wm=numpy.asarray(white_matter_image.dataobj),  # 同网格白质编辑图；强度>=5为白质
    surface="white",  # white或pial，决定亮区的强度替换规则
    mid_gray=70.0,  # 从本被试autodet.gw.stats读取MID_GRAY；示例值不能用于生产
    restore_255=True,  # white将原始强度255恢复为110；pial不执行此恢复
)
```

| 参数 | 类型、默认值与含义 |
|---|---|
| `brain` | 必填三维数值数组；应是已准备的0–255强度，内部转为uint8，不承担原始T1缩放 |
| `wm` | 必填，与brain相同shape；内部转为uint8，值>=5用于白质邻居计数 |
| `surface` | 必填具名字符串，`"white"`或`"pial"` |
| `mid_gray` | 必填具名灰质目标强度；pial边界按 `floor(value+0.5)` 写uint8，white不使用它 |
| `restore_255` | `True`；仅white恢复原brain中强度255的位置为110 |

输入是体素数组，函数没有affine参数；调用者必须核对网格。邻域为3×3×3，
边界按最近边值延伸。白质邻居阈值是固定算法常量，不能作为拟合被试的选项。
强度没有物理量单位，空间单位及体素大小由调用方影像头保持。

返回二元组：

| 输出 | 数据结构与语义 |
|---|---|
| `placement_volume` | 与输入相同shape的三维uint8数组；white将标签100/130设0并可恢复255；pial将标签100设MID_GRAY、130设255，是CBV放置图 |
| `bright_region_labels` | 同shape uint8数组，0为非亮区、100为亮区边界、130为亮区 |

函数不写文件，不修改输入，不改变调用方TF32或CUDA缓存设置，不启用半精度。
非法surface、非三维或shape不同抛`ValueError`；类型转换、内存或底层计算
异常原样传播。它不是原始T1强度归一化接口。

## 3. 命令行与复现

该内部函数没有独立生产CLI。真实MRI-only回归使用具名参数：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python validation/recon_all/python_gpu_port/benchmark_placement_volume.py \
  --subject /data/frozen/sub07 \
  --candidate-module /data/candidate/src/fnit/recon_all/place_surface_volume.py \
  --legacy-module /data/control/src/fnit/recon_all/place_surface_volume.py \
  --native-binary /data/benchmark-native/bin/mris_place_surface \
  --output-directory /data/runs/placement-volume-sub07-v2 \
  --threads 4 \
  --code-commit ACTUAL_TESTED_COMMIT
```

`subject`提供三张同网格MRI、左侧orig和灰白阈值；两个module明确冻结旧新
源码。`native-binary`只用于诊断参考；输出目录必须不存在。JSON保存五项
输入、模块、脚本和程序SHA、线程、实际命令、shape/affine/dtype和逐体素误差。
候选不读取原生输出。MRI-only与完整表面或原始T1整例必须分别报告。

## 4. 原软件调用

`MRIclipBrightWM`和`MRIfindBrightNonWM`没有独立CLI，属于
`mris_place_surface`内部步骤。固定源码已有`--outvol-only`，写完准备图
便退出，不执行white/pial优化：

```bash
mris_place_surface \
  --adgws-in /data/frozen/sub07/surf/autodet.gw.stats.lh.dat \
  --wm /data/frozen/sub07/mri/wm.mgz --threads 4 \
  --invol /data/frozen/sub07/mri/brain.finalsurfs.mgz \
  --lh --i /data/frozen/sub07/surf/lh.orig \
  --o /data/reference/unused.diagnostic-surface \
  --white --seg /data/frozen/sub07/mri/aseg.presurf.mgz \
  --restore-255 --nsmooth 0 --no-rip --no-pin-medial-wall \
  --outvol-only /data/reference/white-invol.mgz
```

pial模式将`--white`改为`--pial`并要求`--repulse-surf`。本次MRI-only诊断
使用同拓扑orig满足读取参数；坐标不参与亮区准备，且程序在放置前退出，
该辅助输入不能视为white或完整pial参考。pial的`--outvol-only`保存的是
标签100替换后的`invol`，没有把标签130设255；因此回归明确比较这个中间
对象，不把它与函数返回的CBV或另一个PS图混称。

固定源提交为`d932c45b7941662ea380a05efef580568b98d41a`；现场核验
`utils/mri2.cpp` SHA-256 为
`f6f01c07a2e44c127eb1b22bca61d0e6e87a6065f538fbafc8de343bcec84ed4`。
当前Conda参考程序SHA为
`78b64b7395aa0db0592ab6912fc026128b221c56d9f802db225fa59c18ceda44`。

## 5. 当前真实精度与时间

公开ds000114两例、同一A100节点、四线程预算，全部比较在同一主机进行。
本子函数为CPU计算，未使用GPU。最新v2结果如下：

| 输入/模式 | 旧版不同体素/最大误差 | 修复版不同体素/最大误差 | 旧/新数组计算 | 原生冷CLI含读写 |
|---|---:|---:|---:|---:|
| sub06 white | 51 / 163 | 0 / 0 | 2.903 / 2.871 s | 4.901 s |
| sub06 pial invol | 78 / 113 | 0 / 0 | 2.717 / 2.878 s | 4.995 s |
| sub07 white | 3 / 110 | 0 / 0 | 2.937 / 3.051 s | 4.246 s |
| sub07 pial invol | 6 / 105 | 0 / 0 | 3.011 / 3.003 s | 4.604 s |

旧新各项P99误差均为0，说明稀少差异不能只看P99；修复版不同体素数和
最大误差也为0。shape均256³，affine与uint8类型一致。数组预加载计算与
原生冷CLI范围不同，此表不是GPU、CLI或整例加速声明。

机器可读报告分别为
[sub06 v2](../../validation/recon_all/optimizations/20261009_placement_torch/prepare_volume_threshold13_a100_v2_sub06.json)与
[sub07 v2](../../validation/recon_all/optimizations/20261009_placement_torch/prepare_volume_threshold13_a100_v2_sub07.json)。
它们绑定本次实际模块SHA，不以旧记录代替当前结果。完整修正版white
另行运行；不能从MRI-only相同推出表面或最终统计量相同。

## 6. 更新与 benchmark 记录

- 2026-10-09：从源码整数表达式定位阈值13；六项边界/255恢复契约通过，
  加上marker/white控制共32项本地测试通过。
- v1：white两例已0差异；pial参考命令缺少repulse-surf而失败，记录保留。
- v2：补足MRI-only参考参数并标明辅助表面语义；两例两模式全部0差异。

这项变更是算法正确性修复，尚无整例资源或干净部署验收。没有增加依赖，
使用主页Conda环境已有NumPy/SciPy/nibabel。

## 7. 原代码与参考文献

- [固定源码 MRIfindBrightNonWM](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri2.cpp)
- [固定源码 mris_place_surface](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_make_surfaces/mris_place_surface.cpp)
- Dale AM, Fischl B, Sereno MI. Cortical surface-based analysis. I. Segmentation and
  surface reconstruction. *NeuroImage* 9, 179–194 (1999).
- Fischl B. FreeSurfer. *NeuroImage* 62, 774–781 (2012).
