# CPU 最小接入计划（尚未实施）

本计划依据有限37步和保存路径审计；没有改动生产，也没有启动完整recipe。最终功能门是每区Dice≥0.95、硬体积差≤5%；37步内部轨迹差异需要保留，但不据此预判最终ROI失败。

## 1. 现场源码与缺口

当前root HEAD为 `4fe9be18…`；其GEMS六个文件与 `6f624040…` 时相同。实际SHA：core `0ded484d…`、raster `f99d6c45…`、Gaussian `67856df0…`、deformation `d0a69c20…`、optim `f25b7251…`、recipe base `5e8b4af9…`。冻结诊断的Gaussian／deformation／optim／recipe source相同，core与raster不同；额外的Double／raw closure和私有优化器也不在main运行链。

| 部位 | main实读行为 | CPU候选最小变化 |
| --- | --- | --- |
| point／reference | `GEMSAtlas`本已保存np.float64，但`TorchGEMS._tensors`转到统一默认Float32 | 独立CPU mesh precision policy使point／reference保持Double；image和alpha捕获值保留 |
| 投影 | boundary按`self.dtype`生成QR | CPU boundary升格后真正计算Double QR；不能仅将旧QR矩阵升格 |
| Gaussian | 函数可支持Double，core固定Gaussian／初始化状态按原dtype使用 | CPU Gaussian／likelihood显式保持Double；无需改GPU Gaussian函数；真实intensity EM不能由fixed-Gaussian门代替验收 |
| raw mixture | CPU fallback在normalised prior上执行`logsumexp`，没有原生mixture+1e-15语义 | 新内部CPU mesh-objective路径使用raw插值质量和class-mixture+1e-15；公共Atlas／概率／posterior的normalization保留 |
| owner与插值 | main `_compact_lookup`没有独立owner-geometry入口 | CPU owner沿原FP32 NumBa lookup；origins/inverse／梯度使用Double；默认／GPUlookup保持原实现 |
| deformation prior | 现有函数已支持Double与有序归约，但core输入为Float32 | CPU输入与reference geometry Double、按现有native-frame／有序归约执行，不改signed volume公式 |
| 优化器 | Cached/Precision/Armijo PyTorch L-BFGS | 新CPU限定模块接入已测max-node H0、12对history、curvature、Wolfe bracket/zoom与checkpoint restore；不覆盖GPU优化器 |
| stage边界 | `atlas.with_vertices`本已保持np.float64，`previous_params`直接传递 | 避免下一stage `_tensors`或fixed params再向Float32舍入；stage actualdtype写入报告 |

首个CPU候选应限制在compact单个3-D image的RHA调用；现有其他功能继续使用原路径，不能将这一次验收扩展到多模态dense／所有核团。候选开关先显式启用，当前默认不直接替换。拟增内部CPU module与受控core/raster入口；优先保持Gaussian、deformation、公共Atlas与GPU fused/Triton函数源码不变。

## 2. 验证顺序

1. **冻结新候选。** 从root当前源码新建独立任务目录；保存全部source／assets／input SHA和actual dtype。运行时仍只用仓库PyTorch／NumBa／nibabel，不依赖官方安装。
2. **接线同点门。** 从既有capture和保存的0／3／4／37候选坐标调用新生产CPU objective；原cost≤0.01、gradient relative L2≤1e-5、prior≤1e-6、coverage exact门不放宽。测试raw zero-mass／epsilon、Double QR、历史／拒绝trial恢复，确认源定义optimizer metadata和stage options。先验证新生产源码，不能用私有adapter结果代替。
3. **GPU保护。** CPU入口拒绝CUDA tensor；GPU原分支／默认dtype／TF32／fused函数和调用参数不改变，CPU module延迟导入。只补受影响GPU真实固定状态和短配对时间门，并检查CPU flag不会进入GPU；不重跑全部已完成GPU recipes。
4. **一个受影响完整RHA recipe。** 另获执行批准后仅运行一侧右HA，使用已绑定的同一norm／aseg／wmparc、atlas／LUT、mask／alignment输入、stage schedule和profile；已完成官方输出以SHA复用。冻结actual options逐stage核对，不将synthetic的1e-10选项硬套所有intensity阶段。记录全部stage的dtype、EM／mesh次数、停止原因、index rebuild／trial budget和prep／EM／mesh／postprocess／I/O时间。

若任一接线／恢复／stage有限性门失败，则停止该候选，不将诊断预算耗尽作为成功收敛。CPU现有有限数据说明性能仍需要优化；本阶段先测真实正确性，再按profile决定是否写仓库内的NumBa fused CPU analytic kernel。新依赖没有必要。

## 3. 最小完整RHA验收

- 官方最终hard-label map、每区硬体积与原有soft-volume统计分别按实际含义匹配；每个实际核团逐区Dice≥0.95、硬体积差≤5%，报告最差区、体积方向和所有零／缺失标签，不只报平均值。
- map必须在同一个native空间、同shape／affine／label LUT；输出分辨率、重采样和背景／support一致。完整final posterior另检查有限性与类和，不以mesh overlap代替最终分割指标。
- 同输入最终分割／统计和脑图才作为功能证据；本次37步轨迹、局部Jacobian与同点门继续作为内部诊断。内部非逐点一致单独披露。
- 8个相同physical core、共用CPU锁；CPU墙钟与分步骤时间、峰值RSS、GPU未受影响的证据分别报告。复用旧官方time只有其线程／机器／负载条件可对应时才标可比；否则只列历史观察，不发布新的速度倍数。
- RHA单侧达到功能门后，再依未验收列表安排其它侧／核团／模态。main默认切换与完整CPU速度目标由这些实际结果决定。
