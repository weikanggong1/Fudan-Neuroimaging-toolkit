# TorchFNIRT 真实数据验证

[返回功能文档](../../docs/fnirt/README.md)

本目录保存 2026 年 9 月 28 日的 matched-input 验证。1 例去标识化真实 UKB 格式 FA
同时输入当前 TorchFNIRT 和 FSL 6.0.7.4。两侧固定相同的 preprocessed FA、
FMRIB58_FA_1mm、FSL scaled-mm affine、implicit zero mask 和 Oxford s1/s2/s3 配置。
FLIRT registration weight 不传给任一 FNIRT 实现。
报告中 `__init__.py` 的测量 hash `cc9aa4…` 与当前 0.16.0 的 `be1cab…` 之间差异包括
版本号、独立的 fMRI 懒加载导出（含 13 个 surface API），以及主动撤下五个内部实现名称。归一化前两项并从新旧源码中过滤
这五个名称后，保留 API 的新旧 AST SHA-256 均为 `fd295c…`；被撤下名称不主张 API 兼容。
[包入口源码等价证明](../runtime_dependencies/package_entry_source_equivalence.public.json)记录了完整 hash、适用调用路径和 `fresh=false` 边界。

- [`report.real.current.json`](report.real.current.json)：输入、配置、源码和脚本 SHA-256，四类输出合同与完整 3D 指标，计时、显存、负载及验收结论。
- [`run_current_matched.py`](run_current_matched.py)：当前 TorchFNIRT 的单例执行脚本。
- [`run_official_matched.sh`](run_official_matched.sh)：FSL 三阶段参考命令及 full-affine Jacobian 生成命令。
- [`validate_real_current.py`](validate_real_current.py)：比较、报告和图生成脚本。
- [`SHA256SUMS`](SHA256SUMS)：上述文件与公开图的校验值。
- [真实 FA 与 Jacobian 对照图](../../docs/fnirt/figures/fnirt_real_current.png)。

四类输出的 shape、affine、float32 和 coefficient intent-2007 合同通过。Pearson r 为：
coefficient `0.999893`，warped FA `0.999203`，nonlinear Jacobian `0.999064`，含 affine
Jacobian `0.994737`。MAE 分别为 `0.018804`、`0.003795`、`0.010260` 和
`0.037611`。这些误差超过浮点舍入，所以 `numerical_equivalence_passed=false`。

TorchFNIRT 的同步核心、外部 wall 和 peak CUDA allocation 分别为 `16.933 s`、
`25.16 s` 和 `3.598 GB`。FSL 三阶段外部 wall 合计 `1265.14 s`。GPU 上有空闲但常驻
的其他进程，FSL 启动时主机 load average 为 `94.86/89.93/88.62`；两侧进程边界也不同，
因此报告保留原始时间，不给出加速比。当前证据只有 1 例。
