# segment_4_subregions：真实 T1 验证

[功能、参数和流程图](../../docs/subregions/README.md)

`segment_4_subregions` 从一张三维 T1 完成共享粗分割、DK68 皮层分区、白质代理，以及脑干、丘脑、左/右海马与杏仁核四项 recipe 的原生 PyTorch 拟合。统一输出原 T1 网格标签、110 项硬/软体积和四份高分辨率标签。

## 数据及比较范围

使用 OpenNeuro ds000114 的公开去面部 `sub-01_T1w.nii.gz`，CC0；[来源与处理记录](../../examples/data/SOURCES.json)。形状 `256×156×256`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。该例用于开发验证。

- 原始 T1 全流程：FNIT 内部运行一次 SynthSeg+，生成粗结构、DK68 分区与白质代理，拟合并保存全部四项结构。
- 同一阶段输入：读取与已保存 FreeSurfer 8.2 参考相同的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`，比较亚区拟合。

精度对照使用另外运行的官方结果；FNIT 运行时使用项目原生 PyTorch 实现。逐标签计算硬 Dice、硬体积及工作网格后验积分软体积；空参考与空候选分别记录，不以背景准确率替代结构指标。标签、网格、affine、dtype、Jacobians、来源和显存另行检查。

## 当前完整精度与重复性验收

[2026-10-02 最终全流程验收](reproducibility_20261002/README.md)使用三次全新 FreeSurfer 官方细分割，FNIT 对原始 T1 和同阶段输入各完成三次 `structures=all` 独立运行。110 个标签 × 两类输入 × 两种网格共 440 项：421 项非空标签的最低内部 Dice 为 1，19 项空标签保持 NA；全部硬标签零体素差异，软体积三次逐值相同、最大 CV 为 0，与本例官方的重复观察一致。

原始 T1 计算 **259.99–348.20 s（4.33–5.80 min）**，API 含保存 **260.65–348.87 s**；同阶段计算 **352.86–361.63 s（5.88–6.03 min）**，API **353.44–362.21 s**。共享 GPU 上记录其他进程占用，自身采样峰值 9,726–16,468 MiB，限制 19,073 MiB。官方同阶段三个子流程合计 **24.57–24.95 min**。

本轮消除脑干梯度累计波动，丘脑改为完整有效体素积分，同阶段原网格细标签加权 Dice **0.953076→0.970813**；海马/杏仁核用闭运算稳定连通域选择，保留原始细标签。稳定的跨软件差异及局部小幅回退逐区记录，见[完整 ROI 表](reproducibility_20261002/final_all_analysis/final_roi_metrics.tsv)、[家族精度与分步骤耗时](reproducibility_20261002/final_all_analysis/README.md)、[脑图](../../docs/subregions/README.md#官方对照脑图)。几何、有限值、硬/软体积及正 Jacobian 已核验。

## 历史版本

[4178a48 完整 benchmark](segment_4_subregions/raw_precision_analysis/README.md)、[丘脑回溯修复](segment_4_subregions/stability_fix/README.md)、[入口整合](segment_4_subregions/README.md)与[v16 C6](speed_v16/README.md)保留历史身份。旧官方存档缺少生成时输入/源码的完整记录，与本轮三次新官方结果不同；历史 Dice 不用于估计本轮随机范围。本轮前后比较均重新使用相同的新官方参考。

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
