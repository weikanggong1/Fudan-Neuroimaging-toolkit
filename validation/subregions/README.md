# segment_4_subregions：真实 T1 验证

[功能、参数和流程图](../../docs/subregions/README.md)

`segment_4_subregions` 从一张三维 T1 完成共享粗分割、DK68 皮层分区、白质代理，以及脑干、丘脑、左/右海马与杏仁核四项 recipe 的原生 PyTorch 拟合。统一输出原 T1 网格标签、110 项硬/软体积和四份高分辨率标签。

## 数据及比较范围

使用 OpenNeuro ds000114 的公开去面部 `sub-01_T1w.nii.gz`，CC0；[来源与处理记录](../../examples/data/SOURCES.json)。形状 `256×156×256`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。该例用于开发验证。

- 原始 T1 全流程：FNIT 内部运行一次 SynthSeg+，生成粗结构、DK68 分区与白质代理，拟合并保存全部四项结构。
- 同一阶段输入：读取与已保存 FreeSurfer 8.2 参考相同的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`，比较亚区拟合。

只读取官方已保存结果，FNIT 运行时不调用官方程序。逐标签计算硬 Dice、硬体积及工作网格后验积分软体积；空参考与空候选分别记录，不以背景准确率替代结构指标。标签、网格、affine、dtype、Jacobians、来源和显存另行检查。

## 当前新入口实测

[精度修复、完整 benchmark 与细核脑图](segment_4_subregions/stability_fix/README.md)使用两次新的四结构完整运行。原始 T1 计算 **410.721 s（6.85 min）**、含保存 API **411.237 s**；同阶段输入计算 **533.773 s（8.90 min）**、含保存 API **534.373 s**。监控进程总墙钟分别为 **435.155 s**、**578.304 s**；完整 FNIT CPU 整例尚未测量。

稳定丘脑全部拟合的 alpha 与顶点梯度归约、总代价和 L-BFGS 内部精度；使用有界 Armijo 回溯及正 Jacobian 接受守卫，按实际 FP32 位移判断下降。网格小矩阵使用完整 FP32，退出闭包后恢复默认 TF32。影像、几何、Gaussian 参数和顶点梯度仍为 FP32，其他 recipe 保留原数值路径。同阶段丘脑细核官方体素加权 Dice 从入口整合后的 **0.910995** 恢复至 **0.960523**，C6 基线为 **0.964068**。原始 T1 的丘脑细核加权 Dice 为 **0.775793**；不同上游输入分别报告。

两次输入、源码、来源及几何检查均通过，全部 110 项软体积有效，四个最终最小 Jacobian 为正。家族外形、细核 Dice、硬/软体积和脑图随报告保存。[修复前记录](segment_4_subregions/README.md)及[历史 C6 基线](speed_v16/README.md)保留原身份。

计算包含影像读入、共享准备、全部拟合与合并；API 另含自动保存。进程计时还含导入、CUDA 初始化及验证记录。资源按大小和 SHA-256 核验，首次下载与图谱安装不计入。共享 GPU 负载与本进程显存约每 5 秒记录，本进程上限 19,073 MiB。

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
