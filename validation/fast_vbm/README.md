# FastVBM 验证状态

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [TorchFNIRT 验证](../dmri_pipeline/README.md) · [FLIRT 验证](../../docs/flirt/README.md)

当前目录不发布 FastVBM 数值 benchmark。TorchFNIRT 的 implicit mask、masked smoothing、process handoff 和优化过程已经修改；旧 FastVBM 报告、表格、图、构建摘要、source-equivalence 声明和校验清单不再对应当前源码，已从目录删除。

## 当前可声明的范围

- FastVBM 的单被试 Python 和 CLI 接口、13 幅 NIfTI 文件名、输入/模板网格角色由当前源码定义。
- 两个后端共用 SynthStrip/TorchFAST、TorchFLIRT、FSL 坐标转换、TorchApplyWarp、dense nonlinear-only Jacobian 和 modulation；只在 nonlinear estimator 上分支。
- SynthMorph 网络没有 reference-mask 输入。
- TorchFNIRT 每一级使用 fixed 非零 implicit reference mask，并用 moving 非零 implicit input mask 做 masked smoothing 和 warped-input 筛选；显式 reference mask 按默认 GM schedule 在最后一级与 implicit mask 取交集。
- TorchFNIRT 当前优化数值轨迹仍与 FSL 不同。FastVBM、TorchFNIRT 与 UKB/FSL 均不能声明逐体素或完整数值等价。

这些是源码行为说明，不是新的外部 benchmark 结果。组件级 FNIRT/TBSS 真实数据诊断见 [`validation/dmri_pipeline`](../dmri_pipeline/README.md)，但该结果不能替代 raw-T1w-to-VBM 端到端验证。

## Fresh benchmark 尚未完成

当前源码尚未重新完成 FastVBM 的真实 T1w 配对 benchmark。因此暂不提供以下内容：

- warped GM、nonlinear-only Jacobian 和 modulated GM 的 Pearson、MAE、RMSE 或 Dice；
- SynthMorph 与 TorchFNIRT 两分支的运行时间或显存；
- UKB/FSL 与 FNIT 输出示意图；
- 当前源码的 pass/fail gate 或 source-equivalence 证明。

完成 fresh benchmark 时，应固定并公开以下条件：

| 项目 | 必须记录的内容 |
|---|---|
| 数据 | 真实 T1w 病例数、去标识 ID 清单和纳入规则 |
| 输入 | 每例 raw T1w、同一 GM template 和同一 reference mask |
| FSL reference | FSL/UKB 版本、命令、config、线程和输出网格 |
| FNIT | commit、package version、源码哈希、checkpoint 哈希和后端参数 |
| 硬件 | CPU/GPU 型号、CUDA/PyTorch、TF32 设置和峰值显存定义 |
| 精度 | 三幅模板空间输出的 Pearson、MAE、RMSE、阈值 Dice 及逐例汇总 |
| 时间 | 首次加载与 warm run 分开；compute、写盘和端到端边界分开 |
| 图 | 同一切面、同一色阶、reference/FNIT/绝对差，并注明病例或组平均 |
| 结论 | 明确区分接口对应、功能相似和数值等价 |

新的报告只有在结果由当前源码生成、工件哈希可复核且图与报告来自同一批输出后，才能作为当前 FastVBM benchmark。完成前，[FastVBM 主文档](../../docs/fast_vbm/README.md)只说明接口、算法和证据边界。
