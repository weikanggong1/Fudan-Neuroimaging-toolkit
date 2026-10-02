# 丘脑细核变化的只读定位

本页为修复前已有输出的定位记录。后续代码核查、标量精度契约测试、真实重复及完整流程复测见[数值修复与官方对照](stability_fix/README.md)；以下历史轨迹数值保持原记录。

本次 stage 与 C6 的 T1、aseg、wmparc SHA 相同。两者使用提供的粗标签和 wmparc，SynthSeg/SynthSegPlus 调用均为0；丘脑初始 atlas affine 逐项相同，alignment Dice 都为0.9035182595，工作格网、裁剪、分组日程、采样及停止阈值相同。核心12个数值模块的冻结 SHA 均相同。

最早明确记录到的差异发生在 synthetic 全积分拟合，早于后续 stride4 及 owner hints。sigma3 阶段 evaluation/step 从70/20变为79/20；sigma2 从20/6变为72/27。synthetic 最小 Jacobian 从0.097082变为0.019344，mean displacement 从0.009176变为0.017188 voxel。最终强度拟合 Jacobian 仍为正，0.150522→0.159386，但细核边界轨迹已不同。

当前初始化实际调用 GEMS 内部 estimate_mask_affine 的 soft Dice L-BFGS，不调用 PyTorchFLIRT。这条 stage 路径也不调用 main 中改变的 FAST/SynthStrip/SynthSeg。因此不能把这次细核下降解释成上游线性配准矩阵或粗分割变化。

源码可支持的候选机制是数值轨迹不稳定：融合核的 FP32 顶点梯度用 atomic_add 累加；synthetic data cost 仍用 FP32 sum；Strong-Wolfe 的梯度和变化容差为1e-10。微小归约差异可以改变线搜索接受步及早停。该机制是依据源码和早期分岔作出的推断，现有记录未保存每次 objective/gradient 和具体停止原因，未做额外 GPU 重跑，不能当成已隔离的因果证明或已确认的成熟子函数 bug。

synthetic Gaussian 在未改变的 recipe 中固定为分组均值、variance0.01。保存的报告没有最终 Gaussian means/covariances，因此无法从现有8项输出直接核验最终 Gaussian 是否一致。完整数据与文件身份见 numerical_variation.json。
