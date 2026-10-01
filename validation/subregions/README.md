# segment_4_subregions：真实 T1 验证

[功能、参数和流程图](../../docs/subregions/README.md)

`segment_4_subregions` 从一张三维 T1 完成共享粗分割、DK68 皮层分区、白质代理，以及脑干、丘脑、左/右海马与杏仁核四项 recipe 的原生 PyTorch 拟合。统一输出原 T1 网格标签、110 项硬/软体积和四份高分辨率标签。

## 数据及比较范围

使用 OpenNeuro ds000114 的公开去面部 `sub-01_T1w.nii.gz`，CC0；[来源与处理记录](../../examples/data/SOURCES.json)。形状 `256×156×256`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。该例用于开发验证。

- 原始 T1 全流程：FNIT 内部运行一次 SynthSeg+，生成粗结构、DK68 分区与白质代理，拟合并保存全部四项结构。
- 同一阶段输入：读取与已保存 FreeSurfer 8.2 参考相同的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`，比较亚区拟合。

只读取官方已保存结果，FNIT 运行时不调用官方程序。逐标签计算硬 Dice、硬体积及工作网格后验积分软体积；空参考与空候选分别记录，不以背景准确率替代结构指标。标签、网格、affine、dtype、Jacobians、来源和显存另行检查。

## 当前新入口实测

[完整 benchmark、逐标签对照与脑图](segment_4_subregions/README.md)使用两次新的完整运行。原始 T1 计算 **425.862 s（7.10 min）**、含保存 API **426.354 s**；同阶段输入计算 **489.905 s（8.17 min）**、含保存 API **490.499 s**。监控进程总墙钟分别为 **452.074 s（7.53 min）**、**531.436 s（8.86 min）**。完整 CPU 整例尚未测量。

两次各 29 项输入、来源及几何检查通过，全部 110 项软体积有效、四个最终最小 Jacobian 为正。家族外形与细亚区精度分别列出；同阶段丘脑细核的官方体素加权 Dice 从 C6 的 0.9641 降为 0.9110。新函数继承 v16 拟合内核，同时合入 main 的配准与预处理更新；[历史 C6 基线](speed_v16/README.md)保留原报告与计时。

计算包含影像读入、共享准备、全部拟合与合并；API 另含自动保存。Python 进程计时还含导入、CUDA 初始化及验证记录。首次资源下载及一次性图谱安装不计入。图谱、权重大小和 SHA-256 按项目固定清单核对。

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
