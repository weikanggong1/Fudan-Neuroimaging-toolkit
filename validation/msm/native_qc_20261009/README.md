# MSMSulc / MSMAll：查面修复与完整 surface 验证

最新 A100 修复版、同预算 CPU 官方配对和完整 surface 见 [A100 验证页](a100/README.md)；下方 H100 保留其测量版本和时间。

## 本轮修复

旧球面插值先找最近顶点，再搜索它的邻面。固定 newMSM 使用按原面顺序建立的 Octree 叶节点候选池，两种方法并不等价。真实配准第二轮的 2,562 个查询中有 7 个选面不同：6 个原版候选面不在最近顶点邻面中，另 1 个没有遵守叶节点限制。差异随后进入特征插值、控制点更新和细级配准。

FNIT 现在独立构建相同的有序候选树。GPU 处理可稳定判断的唯一内部点；共享边界、重叠候选和回退查询采用 FNIT 原生扩展中的逐项 double 运算。该扩展与现有 HOCR/FastPD 共用构建入口，不加载 FSL 或 newMSM 库。20,496 个真实检查点的选面及原生投影已与固定原版逐位核对；生成几何回归测试不保存个体坐标。

另修复了 surface pipeline 的 QC 缺口：MSMAll 求解球面与合成到 native 网格的最终球面分别检查。最终 QC 重读实际保存的 float32 GIFTI，保留初始 MSMSulc、MSMAll solver、最终 native 三阶段状态。`msmsulc_qc_policy` 和 `msmall_qc_policy` 独立支持 `report/repair/error`；修复失败或选择 `error` 时，在 BOLD 投影与发布前拒绝输出。

## H100 初次完整双侧 GPU 配准

[机器可读报告](gpu_cores.public.json)记录聚合指标、配置、测量源码及原生扩展 SHA-256；逐个影像的输入/输出指纹保留在私密审计记录。两项公开 API 依次运行，每项左右半球并行；H100 PCIe、CPU 总预算 8（左右各 4）、20 GB PyTorch allocator 上限。实际 TF32 开关开启，配准几何使用 float64，GIFTI 保存为 float32。

| 完整双侧 API | 配置 | 墙钟 / s | 峰值 allocation / GB | 与固定原版单线程保存坐标 | 最终翻折 L/R：FNIT / 原版 |
|---|---|---:|---:|---|---|
| `run_msmsulc` | HCP 四级；`qc_policy="report"` | 93.130 | 1.756 | 坐标及有序 faces 逐值一致；角差、弦差全部统计为 0 | 2/0 / 2/0 |
| `run_msmall` | 32k、21 列 C 特征、三级、λ=0.05 | 73.608 | 1.787 | 坐标及有序 faces 逐值一致；角差、弦差全部统计为 0 | 0/0 / 0/0 |

API 墙钟包含输入读取、完整配准、球面与报告写盘；排除 Python 导入、CUDA 初始化和事后精度比较，也不包含特征估计、native composition、BOLD 投影或 CIFTI。两次调用的源码、原生扩展、输入、原版球面及原版配置前后哈希均不变。共享节点的时间为各一次观测。

MSMSulc `report` 的左侧 2 个翻折与固定原版坐标完全相同，应记为几何 warning。显式 `repair` 会改变这些坐标，必须另记修复后的 QC，不能用其无翻折结果代替原版精度比较。

## 原版范围与图像公开范围

数值参照为固定 `fsl-newmsm 1.0 h442c261_5`，可执行文件 SHA-256 为 `af5c04246cfeea19233232168acbc1f31266f28b1a7cb6779b8f1bfa32eb9615`。该版本把刚性初始化的 `simval=3` 转为 Pearson；fMRIPrep 25.2.4 容器所带 2019 版 MSM 使用 NMI。本轮逐值一致结论绑定前者的实测输入和配置。

公开报告只含数值指标、配置及程序源码哈希；个体 CIFTI/GIFTI、影像指纹和新生成的个体脑图保留在私密运行目录。下图沿用已公开的 HCP 参考组件，说明 C 特征的空间结构。

![已公开的 HCP 参考连接特征](../../../docs/msm/images/reference_rsn.png)

图像资源与原许可见[已有来源记录](../../../docs/msm/images/reference_rsn.json)。Octree 算法来源为 [newMSM 固定源码](https://github.com/rbesenczi/newMSM/tree/260718953547743c028a45f8c885d163441df87a/libraries/msm-newresampler)，该目录的 MIT 许可及作者声明保留在 FNIT 新头文件与[第三方说明](../../../THIRD_PARTY_NOTICES.md)中。

功能、参数与调用分别见 [MSMSulc](../../../docs/msm/README.md)、[MSMAll](../../../docs/msm/msmall.md)和 [fMRI surface pipeline](../../../docs/fmri/surface.md)。
