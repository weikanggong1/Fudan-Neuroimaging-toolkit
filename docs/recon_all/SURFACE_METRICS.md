# 最终 white/pial 的厚度、面积与曲率

[返回 recon-all](README.md) · [阶段与命令](CONDA_CPP_STAGES.md)

CUDA 流程复用仓库已有的 `thickness_map`、`area_map` 和 `curvature_map`。CPU 流程保留 Conda 内从固定源码构建的 `mris_place_surface`。两条路径都读取最终放置后的 white/pial，不改变网格、标注或厚度定义。中间面积和顶点体积继续使用原有 PyTorch 实现。

## 输入、输出及参数

`_run_surface_metrics(binary, subject, hemi, assets, *, device) -> dict[str, float]` 是主调度内部步骤，没有独立 CLI。`binary` 是当前 Conda 的 `mris_place_surface` 路径；`subject` 是已生成最终 white/pial 的被试目录；`hemi` 为 `lh` 或 `rh`；`assets` 为 FNIT 资产目录；`device` 为主流程传入的 CPU 或 CUDA 设备。参数均显式传入。GPU 逐图同步计时，返回五张图各自的秒数，包含读取、传输、计算和写出。

输入 `surf/H.white` 和 `surf/H.pial` 为 surface RAS 坐标（mm）的三角网格，必须有相同顶点数和有序面。输出为同一顶点顺序的 float32 FreeSurfer morph 文件：

| 文件 | 定义与单位 |
| --- | --- |
| `H.thickness` | 双向表面顶点搜索的平均厚度，mm；20 跳可达性约束，每方向截断至 5 mm |
| `H.area`、`H.area.pial` | 每个相邻三角形面积的三分之一，mm² |
| `H.curv`、`H.curv.pial` | 二阶邻域曲面拟合后的平均曲率，10 次邻域平均，mm⁻¹ |

输入缺失、white/pial 不对应、原生程序失败或输出缺失时抛异常。CUDA 路径需要可用的 CUDA 设备，不会自动切换到低精度或放宽验收标准。厚度的空间邻点计算和法向计算使用 PyTorch；图上的可达性检查仍使用 CPU。这些指标不适用于没有顶点对应关系的两张表面。

```python
from pathlib import Path
from fnit.recon_all.native_free import _run_surface_metrics

seconds = _run_surface_metrics(
    binary=Path("/opt/conda/envs/fnit/bin/mris_place_surface"),  # CPU 路径的 FNIT 自编译程序
    subject=Path("/data/subjects/sub01"),  # 已完成最终 white/pial 放置的目录
    hemi="lh",  # 左半球；右半球为 rh
    assets=Path("/data/fnit-assets"),  # 已校验的模板和图谱
    device="cuda:0",  # CUDA 使用已有 PyTorch 函数；cpu 使用原生程序
)
# seconds 包含 thickness、area、area.pial、curv、curv.pial 的墙钟秒数。
```

对应官方命令，右侧和 pial 指标替换相应输入、输出路径：

```bash
mris_place_surface --thickness surf/lh.white surf/lh.pial 20 5 surf/lh.thickness
mris_place_surface --area-map surf/lh.white surf/lh.area
mris_place_surface --curv-map surf/lh.white 2 10 surf/lh.curv
```

## 当前真实 T1 同输入验证

2026-09-30 使用两例 FNIT `b8cd17b` 从原始 T1 连续重建出的最终网格，分别调用官方 FreeSurfer 8.2、当前 Conda 源码构建程序和已有 PyTorch CUDA 函数。官方程序只用于隔离诊断。CPU/Conda 与官方相同输入的结果完全一致；PyTorch 按现有 morph 门槛检查：面积绝对阈值 0.001 mm²，其余 0.005，加 0.001 相对项。

两例左侧厚度的 Conda/CUDA 耗时分别为 20.23/5.37 秒、22.70/6.54 秒，最大误差均为 4.77×10⁻⁷ mm，没有越界顶点。左侧 white 面积最大误差为 4.77×10⁻⁷ mm²，曲率分别为 1.01×10⁻⁶、1.37×10⁻⁶ mm⁻¹。该计时包含函数读写，排除脚本导入和 CUDA 上下文初始化；GPU 与其他任务共享，单次阶段差异不能代替整例提速。

两例双侧及 pial 的逐图报告、程序和输入 SHA-256、PyTorch allocated/reserved 见[性能验证目录](../../validation/recon_all/python_gpu_port/performance_20260930/README.md)。完整整例另比较 138 项及最终脑区指标，不用阶段结果宣称整例数值验收通过。

修正调度设备传递的 `279e09f` 另直接调用生产封装，冻结 sub-01 左侧最终 white/pial 并显式启用 TF32：五张图均无超限顶点，厚度 8.39 s、white/pial 面积 0.014/0.010 s、曲率 0.727/0.655 s；PyTorch allocated/reserved 为 482,308,096/517,996,544 字节。该探针启用 CUDA 分配缓存，不代表默认关闭缓存的整例显存。[封装回归报告](../../validation/recon_all/python_gpu_port/performance_20260930/surface_metrics_wiring_tf32_279e09f.json)绑定实际源码和输入 SHA-256。

## 三方比较脚本

`benchmark_surface_metrics.py` 的 `--subject` 是 FNIT 被试目录，`--hemi` 选择半球，`--metric` 为 area/curv/thickness，`--surface` 默认为 white，选择 area/curv 的输入表面；厚度固定读取 white 和 pial。`--native-binary`、`--reference-binary` 是候选和官方程序路径；`--assets` 为候选资产；`--output` 必须不存在，写出三张同序标量图、程序日志和 JSON。`--device` 默认 cuda:0，`--threads` 默认 4，`--code-commit` 记录实际候选源码提交。任一计算或文件检查失败时脚本报错。

```bash
# FNIT 自产表面；参考结果仅在这个隔离目录生成。
python validation/recon_all/python_gpu_port/benchmark_surface_metrics.py \
  --subject /data/subjects/sub01 \
  --hemi lh --metric thickness --surface white \
  --native-binary /opt/conda/envs/fnit/bin/mris_place_surface \
  --reference-binary /benchmark/freesurfer/bin/mris_place_surface \
  --assets /data/fnit-assets \
  --output /data/diagnostics/sub01-lh-thickness \
  --device cuda:0 --threads 4 \
  --code-commit "$(git rev-parse HEAD)"
```

报告的差异数和最大/P99/平均误差按同索引计算，因为这项测试冻结了同一网格。与官方整例网格顶点数不同时，这种配对不成立。

## 原实现与文献

- [固定版本的 mris_place_surface](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_place_surface/mris_place_surface.cpp)。
- Fischl B, Dale AM. Measuring the thickness of the human cerebral cortex from magnetic resonance images. *PNAS*. 2000;97:11050–11055. [doi:10.1073/pnas.200033797](https://doi.org/10.1073/pnas.200033797)。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
