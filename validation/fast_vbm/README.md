# FastVBM 验证状态

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [复现脚本](validate_real.py)

## 单例真实数据定位

在 gpucw1 上，以同一例真实 T1w 的 FSL GM、FSL 仿射矩阵、GM 模板和显式 mask 固定输入，只改变 GM FNIRT 的两个隐式掩膜参数。FSL `GM_2_MNI152GM_2mm.cnf` 和实际日志均指定 `--imprefm=0 --impinm=0`。下表是显式 mask 内与 FSL 输出的 Pearson r：

| implicit reference / input | warped GM | Jacobian | modulated GM |
|---|---:|---:|---:|
| 开 / 开（旧配置） | 0.495 | 0.172 | 0.491 |
| 关 / 开 | 0.525 | 0.233 | 0.506 |
| 开 / 关 | 0.908 | 0.955 | 0.905 |
| 关 / 关（本版配置） | 0.996 | 0.997 | 0.996 |

固定 FSL GM 输入时，掩膜设置是主要原因。上表诊断采用上一源码快照和与本版等效的配置注入完成；它不是本版固定输入配准器的重新运行。

## 本版完整链复测

在 gpucw1 的 H100 上，使用本版 `FastVBM` FNIRT 分支从同一例原始 T1w 运行至模板空间，并与该病例现存的 FSL 6.0.7.4 VBM 输出比较。输入为真实影像；两侧使用相同 GM 模板、显式 reference mask 和 292,019 个评价体素。运行源码 `fast_vbm/pipeline.py` 与 `fast_vbm/registration.py` 的 SHA-256 分别为 `2d0b6a27…` 和 `2a931f50…`，与本版文件一致。

| 输出 | Pearson r | MAE | RMSE | Dice，阈值 0.2 |
|---|---:|---:|---:|---:|
| warped GM | 0.782962 | 0.133026 | 0.245557 | 0.849506 |
| nonlinear-only Jacobian | 0.732829 | 0.165220 | 0.264968 | 1.000000 |
| modulated GM | 0.726428 | 0.167939 | 0.350389 | 0.847056 |

FNIT 冷启动单次 compute 为 110.72 s，13 幅 NIfTI 及报告写盘为 7.56 s，峰值 CUDA allocated 为 12.97 GB。compute 包含读取、首次权重加载和 GPU 结果回传。现有 FSL 计时仅覆盖固定 warp 的应用和图像相乘，缺少同边界的原始 T1w 完整链时间，不能计算加速比。FNIT 与 FSL 的上游 GM 网格及数值不同；本例完整链尚未达到数值等价。样本量为 1，不能推断群体结果。私有影像、逐体素输出及完整运行报告留在 gpucw1。

## 后续验收

本次更新使 FastVBM GM 配准关闭 implicit reference 和 input mask；通用 TorchFNIRT 默认值不变。旧配置生成的 FNIRT 报告、汇总、校验清单和图片已移除。`validate_real.py` 保留为当前接口的复现脚本，运行时须使用真实 T1w、匹配的 FSL 输出和相同模板 mask。下一轮需多例配对，核对精度、示例图和同计时边界的速度。
