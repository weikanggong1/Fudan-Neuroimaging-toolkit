# 真实 annotation 配对脑图

## 功能与输入输出

`plot_annotation_stage_pair.py` 用两臂真实 `smoothwm` mesh 和 `aparc` 标签生成双侧外侧视图：两行分别为 lh/rh，三列分别为 baseline、candidate、逐顶点原始 packed ID 差异。它只读已有 stage 结果，不使用模拟数据、不启动 GPU、不重跑算法、不读取官方结果。没有 T1 原图时，图像仍来自真实 surface RAS 坐标，单位毫米。

输入为两份已完成 `annotation.stage.json`，以及新建输出目录。复用同目录 `compare_annotation_stage_pair.py` 的回执校验：stage/六annot/15input/TF32/线程记录、annot与NPZ的SHA及数组SHA、真实mesh与有序面SHA。渲染前要求两臂四个 smoothwm/sphere.reg 坐标和ordered faces完全相同、15输入relative SHA相同、aparc色表及名称相同。任一不符写 `plot_invalid` 回执、退出2，不能按不对应的vertex index画差异。

输出：

- `aparc.stage_pair.png`：标准可导出的PNG。
- `aparc.stage_pair.svg`：标准SVG；脑表面面片按指定DPI栅格化嵌入，标题/说明保留矢量，避免几十万面片生成巨大SVG。不是纯矢量mesh。
- `aparc.plot.receipt.json`：输入15资源SHA、stage/annot/NPZ/mesh原路径和SHA、坐标空间、真实顶点/面数量、全部顶点差异数、绘图规则、程序SHA、版本及导出文件SHA。

## 颜色与视角

所有原始有序三角面全部进入 Matplotlib `Poly3DCollection`，不抽样、不平滑、不修改mesh坐标或面索引。标准3D renderer使用 `zsort=average` 做深度排序，正交投影，左半球 azimuth=180°、右半球 azimuth=0°，elevation均为0°。每一行三列坐标范围相同，保证可比较。深度排序是Matplotlib painter近似，不用于定量可见面积分析。

baseline/candidate每个面的颜色取该面**第一个顶点**的 `label_table_indices`，映射对应 color-table RGB/255；-1未标注为灰色。使用原色，不做任意对比度增强。差异图每个面的颜色也由首顶点是否改变决定：红色改变、灰色未改变。这个首顶点规则可能让很小的顶点差异在外侧视图中不可见，不能用图上红色面积替代JSON中的全顶点计数。

差异定义为同一有序顶点上的 `original_annotation_ids` 不相等，packed RGB ID不是解剖类别数字；色表名称须相同，才能将此图当作同语义标签比较。图上始终标明全部顶点的实际差异数量。全部为零时明确写 `0 label differences (no contrast amplification)`，面全部灰色，不自动改变色阶。原annot文件字节SHA不同不被算作标签变化，回执单列。

## Python与命令行调用

```python
# 同目录比较脚本必须一起提供；仅CPU真实数据渲染。
from pathlib import Path
from plot_annotation_stage_pair import plot_pair

plot_directory = Path('/absolute/new_annotation_brain_figure')
plot_directory.mkdir()  # 必须全新，与输入stage目录分离。
plot_receipt = plot_pair(
    baseline_path=Path('/absolute/baseline/annotation.stage.json'),  # 基线完整阶段回执。
    candidate_path=Path('/absolute/candidate/annotation.stage.json'),  # 同输入候选回执。
    output=plot_directory,  # 新图像和SHA回执目录。
    dpi=150,  # PNG及SVG嵌入面片的分辨率。
    formats=('png', 'svg'),  # 两种可独立导出的文件格式。
)
print(plot_receipt['hemispheres'])
```

```bash
FNIT_PLOT_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
"${FNIT_PLOT_PYTHON}" plot_annotation_stage_pair.py \
  --baseline-stage /absolute/baseline/annotation.stage.json \
  --candidate-stage /absolute/candidate/annotation.stage.json \
  --output /absolute/new_annotation_brain_figure \
  --dpi 150 --formats png svg
```

参数：`--baseline-stage`、`--candidate-stage` 必填，为已有stage JSON路径；`--output` 必填，必须不存在且与输入目录分离；`--dpi` 正整数默认150，控制PNG及SVG嵌入脑图分辨率；`--formats` 接受png/svg一个或两个，默认两者。退出0表示导出完成，2表示校验/绘图无效。没有精度、GPU或算法参数。

依赖 NumPy、Nibabel、Matplotlib；使用既有项目Conda环境，不安装新依赖。大量原始面片全部绘制，需正常CPU内存与导出时间，绘图时间不是annotation算法耗时。本轮未将真实配对数据在本子任务中运行；已做Python语法检查，协调者在原环境运行并核对图及回执。官方整体等效始终`not_assessed`。

## 原软件与参考

这是输出可视化，没有原软件算法命令，不调用 FreeSurfer/其他影像软件。

- [Nibabel FreeSurfer IO：read_geometry/read_annot](https://nipy.org/nibabel/reference/nibabel.freesurfer.html)
- [Matplotlib Poly3DCollection：面片与深度排序](https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.art3d.Poly3DCollection.html)
- [Matplotlib 栅格化导出说明](https://matplotlib.org/stable/gallery/misc/rasterization_demo.html)

2026-10-04首版：真实mesh全量绘图、六面板、零差异文字、SHA回执。协调者在原Conda环境完成真实sub-06 CPU导出，195.731秒（含回执校验和PNG/SVG），双侧零标签差异。PNG已目视核对；完整[图与回执](runtime/startup_stage_regression_gpu1_v2/figures_sub06_v1/aparc.plot.receipt.json)绑定真实输入、程序和输出SHA。该时间属于绘图，不属于annotation算法。
