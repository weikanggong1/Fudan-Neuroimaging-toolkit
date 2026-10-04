# FastVBM GM-FNIRT：已有真实数组的只读定位

## 输入与范围

2026-10-04 现场读取固定 FNIT README/INDEX；正式仓库 HEAD 为 `f1cbdab10fbfd573c3aa1b3461aa086dab220cfb`。
复用 task4 冻结 v4 与独立官方 CPU v2 已保存数组，没有重跑 pipeline、优化系数、调用原软件或测新 benchmark。
这些官方数组只进入隔离诊断。诊断结果不改变生产输入或默认配置。

- 原始公开 T1：OpenNeuro ds003138 v1.0.1，CC0，`224×288×288`，SHA-256 `afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19`。
- GM 模板：`91×109×91`，SHA-256 `ab933db7455d7c4b88624d54f41a3065be4ba4289d00b9230daec0cdb1597a77`。
- 显式 FSL dilated mask：SHA-256 `342d41e1c445d87a812ca288786a116306e96a5595a198e46ba7e7515d4f221c`。模板和权重留在既有服务器目录，不随报告发布。
- 实际导入冻结 `task5_candidate_cpu_v4`，head `6f1e2b38925a481df3fa622f925af076df5436a9`；源码归档 SHA-256 `ffda47a74376fbaec07c3e8aedaacdc30f2a60398b919c0d5feae38e3beba0d9`。
- 只读探针 v3：nodecw7，8 核 `32,36,40,44,48,52,56,60`，`nodecw7.gems.cpu8.lock`，CUDA 不可见。没有输出新影像或计时结论。

本轮 root 随后修复的 FAST 默认八图逐位一致属于另一冻结版本。此处仍分析既有 v4 的27个 GM 差体素，没有把旧结果标为已修复链的实测。

## GM 参数和调度核验

官方实际日志与 FSL 6.0.7.4 `GM_2_MNI152GM_2mm.cnf`（SHA-256 `3bc82d0ff4d8f53d89a741bd853a427607a5cb108e170b44d8835c4b542a4980`）均已读取。
FastVBM `_deform_model` 构造的 `GMFNIRTConfig` 有效参数匹配：

| 参数 | 两侧值 |
|---|---|
| subsampling / 最大迭代 | `4,2,1,1` / `5,5,10,5` |
| input / reference FWHM，mm | `6,4,2,2` / `4,2,0,0` |
| lambda / SSD 加权 | `150,75,50,30` / 开启 |
| 显式参考 mask 调度 / 隐式 mask | `0,0,0,1` / input、reference 均关闭 |
| 强度估计 / 模型 | `1,1,1,0` / `global_linear` |
| warp 分辨率 / spline 阶数 | `10,10,10 mm` / cubic |
| 正则 / Jacobian 范围 | bending energy / `0.2,5` |
| 优化精度 / 梯度 / 插值 | FP64 state，FP32 图像 / input derivative / linear |

官方 `applyinmask=1` 在本例没有显式 input mask、隐式 mask 关闭时不形成不同输入。
intensity order、bias resolution/lambda 的值也相同，但 `global_linear` 路径不拟合 bias。
这证明调用配置一致；没有证明 PCG 求和顺序、更新轨迹或 topology 投影数值一致。

## 首次明显空间误差的位置

沿用已有评分区域：完整模板902,629体素；脑内为显式 mask 与正模板交集257,125体素。没有调整网格、阈值或掩膜。
下表为脑内 RMSE；warped GM 的参考 `P99−P1=1`，故数值也等于原报告 NRMSE。

| 只读比较 | RMSE | 最大绝对误差 |
|---|---:|---:|
| 已存 v4 warped GM 对官方 | **0.0189811** | **0.716199** |
| 同一官方 coeff，经 FNIT TorchApplyWarp | 0.00000349217 | 0.0000405908 |
| 同一官方 dense，经 FNIT TorchApplyWarp | 0.00000363582 | 0.0000402927 |
| 同一官方 dense→RAS pull→实际 common 转换和重采样 | 0.00000438747 | 0.0000506639 |
| 只替换旧 v4 的 GM，同一官方 coeff | 0.0000178719 | 0.00725591 |
| 同一官方 GM，只替换已记录 FLIRT，affine-only | 0.000840451 | 0.00774723 |

旧 GM 差异经固定官方形变只影响18个脑内体素。FLIRT 原空间最大世界坐标差为既有报告的0.0172093 mm。
affine-only 一行使用独立 NumPy/SciPy trilinear constant-zero；它测直接作用，不是非线性优化对 affine 扰动的敏感度上界。

因此，大误差首次在 **FLIRT 之后的 FNIRT 非线性形变估计段**出现；相同场的 common 坐标转换和 ApplyWarp 没有复现这一量级。
两侧已存 modulated GM 都与各自 FP32 `warped GM × Jacobian` 逐位相同，最终乘法不是额外误差来源。
v4 没有保存 FNIT coefficients/pull 或官方逐轮状态，不能进一步断言首个错误在 PCG、level 交接还是 topology，也不能排除上游小差对优化轨迹的影响。

## 已证明的 Jacobian 定义差异

原官方 `--jout` 是 coefficient 解析 Jacobian；FastVBM 公共后处理计算稠密中心差分（边缘单侧）。
已有同一官方场诊断的脑内 RMSE为 **0.00593142**、最大差 **0.0488632**；FNIT dense 对同一官方 dense 的脑内 RMSE仍为 **0.0122751**、最大差 **0.265896**。
故 Jacobian 定义差异是一个实质来源，但不足以解释形变估计的剩余偏差。
FNIT 本身已有解析 `nonlinear_jacobian`，FastVBM 当前仅记录与 dense 的差异，没有将它写为公共 `T1_GM_JAC_nl`。

本次 roundtrip 还核对了真实 oblique T1 的 header zoom 与 affine 列范数：X轴分别0.8000000119与0.8000068162 mm。
若把 common 转换的场错误地配回原 header，脑内 RMSE为0.000193655；按实际 `_common_applywarp` 重建 header 成对计算则为上表0.00000438747。
此项没有被误报为当前生产链的坐标 bug。

## 最小下一步

1. 用固定 GM、固定 FLIRT、同模板/mask 做 FNIRT 独立首层或首步对照，保存初始归一化/平滑/有效 mask、目标值、梯度、第一次 PCG 更新和 accepted coefficients；首次差异定位后再扩大到 topology。先不重跑脑提取、FAST 或整个 VBM。
2. 独立评估 FNIRT 分支直接采用其已有解析 nonlinear Jacobian，对照同一 coefficient 场；保留 SynthMorph 的既有公共定义。不能仅切换 Jacobian 就宣称 FNIRT 等价。
3. 当前实际使用 `execution="optimized"`、coefficient-space Gram。项目另一个真实 T1 对照已证明 Gram 的 FP64 求和顺序可改变轨迹；它是后续同输入消融候选，此次 GM 数据尚没有 reference/optimized 系数对照，不将其当作已证明根因。

## 证据与复现

- [完整逐区/网格指标及输入输出 SHA](fnirt_posthoc_stage_isolation.public.json)
- [冻结 source SHA、实际配置、level 摘要与既有 Jacobian 分解](fnirt_posthoc_context.public.json)
- [只读脚本](fnirt_posthoc_probe.py)：`--candidate`、`--reference` 分别为既有 v4/官方输出目录；`--template`、`--reference-mask` 指向原资源；`--output` 为新 JSON，拒绝覆盖。只用既有 Nibabel、NumPy、SciPy、PyTorch/FNIT，不调用原软件。
- 服务器冻结探针：`FNIT/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-posthoc-v3`；输出同名 `runs/.../gems/fnirt-posthoc-v3/report.public.json`。
- [既有完整 FastVBM v4 报告](../../smri_cpu_20261004/task04/fast_vbm_cpu_20261004/final_v4/README.md)、[FNIRT 文档](../../../docs/fnirt/README.md)、[FSL FNIRT 原实现](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
