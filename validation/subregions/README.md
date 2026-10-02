# segment_4_subregions：真实 T1 验证

[功能、参数和流程图](../../docs/subregions/README.md)

`segment_4_subregions` 从一张三维 T1 完成共享粗分割、DK68 皮层分区、白质代理，以及脑干、丘脑、左/右海马与杏仁核四项 recipe 的原生 PyTorch 拟合。统一输出原 T1 网格标签、110 项硬/软体积和四份高分辨率标签。

## 数据及比较范围

使用 OpenNeuro ds000114 的公开去面部 `sub-01_T1w.nii.gz`，CC0；[来源与处理记录](../../examples/data/SOURCES.json)。形状 `256×156×256`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。该例用于开发验证。

- 原始 T1 全流程：FNIT 内部运行一次 SynthSeg+，生成粗结构、DK68 分区与白质代理，拟合并保存全部四项结构。
- 同一阶段输入：读取与已保存 FreeSurfer 8.2 参考相同的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`，比较亚区拟合。

只读取官方已保存结果，FNIT 运行时不调用官方程序。逐标签计算硬 Dice、硬体积及工作网格后验积分软体积；空参考与空候选分别记录，不以背景准确率替代结构指标。标签、网格、affine、dtype、Jacobians、来源和显存另行检查。

## 当前完整精度验收

[低 Dice 优化、完整 benchmark 与脑图](segment_4_subregions/raw_precision_analysis/README.md)记录两次最终四结构完整运行。原始 T1 计算 **436.344 s（7.27 min）**，API 含保存 **436.867 s**；同阶段计算 **473.205 s（7.89 min）**，API 含保存 **473.775 s**。进程 wall 分别为 **461.450 s**、**515.738 s**；完整 CPU 整例未重测。

双侧海马复用丘脑稳定拟合，自动 raw 流程复用 TorchFAST、白质中位数 110 归一化及自身头信息建立的 1 mm 工作网格；白质代理改为全部同侧皮层竞争。保留标准处理网格类别图的导出顺序，再把标签、置信度和支持范围共同最近邻映射回输入网格。运行时不调用官方软件。

| 家族 | 同阶段更新前 → 最新加权 Dice | raw 更新前 → 最新加权 Dice |
|---|---:|---:|
| 丘脑细核 | 0.960523 → 0.960579 | 0.775793 → 0.915134 |
| 左海马 | 0.870763 → 0.888019 | 0.678531 → 0.833178 |
| 右海马 | 0.813436 → 0.844525 | 0.634352 → 0.742617 |
| 左杏仁核 | 0.941614 → 0.948179 | 0.790483 → 0.914368 |
| 右杏仁核 | 0.917690 → 0.934494 | 0.760340 → 0.872840 |

来源、源码、模型、图谱、输出几何和体积检查通过；两次各有110项有限非负软体积，四个最终最小 Jacobian 均为正。资源按大小/SHA-256核验，本进程显存上限19,073 MiB，约5秒记录一次共享GPU负载及本进程占用。[当前汇总](segment_4_subregions/raw_precision_analysis/benchmark_summary.json)及[逐标签功能说明](../../docs/subregions/README.md#最新精度运行时间与脑图)保留分母、严格通过条件与可视化。

[上一版丘脑修复](segment_4_subregions/stability_fix/README.md)、[入口整合历史](segment_4_subregions/README.md)与[v16 C6](speed_v16/README.md)保留各自原始身份。计算含预处理、全部拟合及合并，API另含保存；首次资源下载、图谱安装及独立官方运行不计入。

## 复核

```bash
# 原始 T1：--weights 为已核验的 SynthSeg+ 模型目录
python validation/subregions/run_unified.py \
  --t1 /absolute/path/sub-01_T1w.nii.gz \
  --atlas-root /absolute/path/subregion_atlases \
  --weights /absolute/path/weights \
  --structures all --device cuda:0 --optimization fast \
  --output-dir /absolute/path/segment_4_subregions

# 同阶段输入：另加 --aseg 和 --wmparc，将 --t1 改成同一 norm.mgz
# 对照：--reference-brainstem/--reference-thalamus/--reference-left/--reference-right
# 分别指定官方已保存的四份 FSvoxelSpace 标签。
```

### Reference

[原软件命令、实现链接及四项算法文献](../../docs/subregions/README.md#reference)。
